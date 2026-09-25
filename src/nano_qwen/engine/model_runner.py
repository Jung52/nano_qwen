import pickle
import socket
from contextlib import contextmanager

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nano_qwen.config import Config
from nano_qwen.engine.sequence import Sequence
from nano_qwen.engine.cuda_graph import CudaGraphManager
from nano_qwen.layers.gated_delta_net import GatedDeltaNet
from nano_qwen.layers.sampler import Sampler
from nano_qwen.models.qwen3_5 import Qwen3_5ForCausalLM
from nano_qwen.models.qwen3_5_mtp import Qwen3_5MTP, load_mtp_weights
from nano_qwen.utils.context import set_context, get_context, reset_context
from nano_qwen.utils.loader import load_model
from nano_qwen.utils.trace import trace_event

from .async_output import AsyncModelOutput
from .decode_init import prepare_decode as prepare_decode_kernel


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class InputBatch:
    """Persistent request-to-slot mapping (MRV2-style persistent batch).

    Requests live in stable slots; each step builds ``batch_slots`` so the
    GPU-side kernels can read per-request state (e.g. sampled tokens) through
    the ``batch_idx -> slot`` indirection without touching CPU state.
    """

    def __init__(self, max_num_seqs: int):
        self.seqs: list[Sequence | None] = [None] * max_num_seqs
        self.seq_id_to_slot: dict[int, int] = {}

    def update(self, seqs: list[Sequence]) -> tuple[list[Sequence], list[tuple[Sequence, int]]]:
        new_entries: list[tuple[Sequence, int]] = []
        for seq in seqs:
            slot = self.seq_id_to_slot.get(seq.seq_id)
            if slot is None:
                slot = next(i for i, item in enumerate(self.seqs) if item is None)
                self.seq_id_to_slot[seq.seq_id] = slot
                new_entries.append((seq, slot))
            self.seqs[slot] = seq
        return [self.seqs[self.seq_id_to_slot[seq.seq_id]] for seq in seqs], new_entries

    def slots_for(self, seqs: list[Sequence]) -> list[int]:
        return [self.seq_id_to_slot[seq.seq_id] for seq in seqs]

    def remove_finished(self):
        for seq_id, slot in list(self.seq_id_to_slot.items()):
            seq = self.seqs[slot]
            if seq is not None and seq.is_finished:
                self.remove(seq_id)

    def remove(self, seq_id: int):
        slot = self.seq_id_to_slot.pop(seq_id, None)
        if slot is not None:
            self.seqs[slot] = None

    def clear(self):
        self.seqs[:] = [None] * len(self.seqs)
        self.seq_id_to_slot.clear()


class ModelRunner:

    def __init__(
        self,
        config: Config,
        rank: int,
        event: Event | list[Event],
        port: int | None = None,
        use_prefill_cudagraph: bool = True,
    ):
        self._closed = False
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # Rendezvous on an ephemeral port: a fixed port lingers in TIME_WAIT
        # after exit and makes a rerun within ~60s fail with EADDRINUSE.
        if port is None:
            port = find_free_port()
        dist.init_process_group(
            "nccl",
            f"tcp://127.0.0.1:{port}",
            world_size=self.world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3_5ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.gdn_layers = [
            module
            for module in self.model.modules()
            if isinstance(module, GatedDeltaNet)
        ]
        self.sampler = Sampler()
        self.output_copy_stream = torch.cuda.Stream()
        self._output_pin_bufs = [
            torch.empty(
                (config.max_num_seqs, 3),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(2) # Double-buffered CPU output buffers for asynchronous D2H copy.
        ]
        self._output_events = [
            torch.cuda.Event(blocking=True)
            for _ in range(2)
        ]
        # Benchmark can disable the copy stream to provide a synchronous D2H
        # baseline. Keep async output as the production default.
        self.async_output = True
        # The server benchmark can isolate prefill CUDA Graph overhead while
        # keeping decode graphs enabled. Production defaults to prefill graphs.
        self.use_prefill_cudagraph = use_prefill_cudagraph
        if self.config.enable_mtp:
            # MTP uses its dedicated two-row verify graph. The ordinary
            # piecewise-prefill graph does not preserve its hidden/state
            # lifecycle during prompt ingestion.
            self.use_prefill_cudagraph = False
        self.cuda_graphs = CudaGraphManager(self)
        self.mtp = None
        self.mtp_kv_cache = None
        self._target_hidden = None
        self._mtp_last_hidden: dict[int, torch.Tensor] = {}
        self.mtp_stats = {
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "verify_steps": 0,
        }
        self._output_buf_idx = 0
        # MRV2-style input-prep protection: the prior step's async H2D
        # transfers must be consumed before this step reuses the same
        # CPU/GPU staging buffers. Mirrors vLLM's synchronize_input_prep.
        self.prepare_inputs_event = torch.cuda.Event(blocking=True)
        self.sampled_token_ids_gpu = torch.empty(
            config.max_num_seqs, dtype=torch.int64, device="cuda",
        )
        # MRV2-style idx_mapping: batch_idx -> persistent request slot.
        self.batch_slots_gpu = torch.empty(
            config.max_num_seqs, dtype=torch.int64, device="cuda",
        )
        self.allocate_decode_buffers()
        self.input_batch = InputBatch(config.max_num_seqs)
        # Single pending sampling state: one batch is dispatched per step
        # (depth-1 async), so a single slot holds the logits/temps/seqs the
        # immediately-following sample_tokens call will consume.
        self._pending: tuple | None = None
        self.allocate_gdn_state_pool()
        self.warmup_model()
        self.input_batch.clear()
        if self.config.enable_mtp:
            self.mtp = Qwen3_5MTP(hf_config).eval()
            load_mtp_weights(self.mtp, config.model)
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nano_qwen", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nano_qwen")
                self.loop()

    def exit(self):
        if self._closed:
            return
        self._closed = True
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()

        # CUDA Graphs retain captured allocations through their Python
        # containers. Clear every runner-owned GPU reference, including the
        # lazily-created piecewise graphs, before the engine is discarded.
        self._pending = None
        self._target_hidden = None
        self._mtp_last_hidden.clear()
        self.cuda_graphs.clear()
        for name in ("gdn_layers", "decode_gpu", "decode_cpu",
                     "_output_pin_bufs", "_output_events"):
            value = getattr(self, name, None)
            if value is not None:
                value.clear()
        if hasattr(self, "input_batch"):
            self.input_batch.clear()
        self.model = None
        self.mtp = None
        self.kv_cache = None
        self.mtp_kv_cache = None
        self.sampled_token_ids_gpu = None
        self.batch_slots_gpu = None
        self.prepare_inputs_event = None
        self.output_copy_stream = None
        torch.cuda.empty_cache()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.execute_model(seqs, True)
        async_output = self.sample_tokens()
        if async_output is not None:
            async_output.get_output()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        # Only full_attention layers hold K/V cache; GDN layers keep their own
        # conv/recurrent state pools instead. Count cache-bearing modules from
        # the model structure (robust to both ``layer_types`` and the
        # ``full_attention_interval`` fallback, and to Dense checkpoints where
        # every layer is attention).
        num_attn_layers = sum(
            1 for module in self.model.modules()
            if hasattr(module, "k_cache") and hasattr(module, "v_cache")
        )
        assert num_attn_layers > 0, "no attention layers found; cannot size KV cache"
        block_bytes = (
            2 * num_attn_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )
        mtp_block_bytes = 0
        if self.mtp is not None:
            mtp_block_bytes = (
                2 * self.block_size * num_kv_heads * head_dim
                * hf_config.dtype.itemsize
            )
        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current
        ) // (block_bytes + mtp_block_bytes)
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2, num_attn_layers, config.num_kvcache_blocks,
            self.block_size, num_kv_heads, head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == num_attn_layers, (
            f"attention layer drift: assigned {layer_id} caches but sized "
            f"for {num_attn_layers}"
        )
        if self.mtp is not None:
            self.mtp_kv_cache = torch.empty(
                2,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                dtype=hf_config.dtype,
                device=self.kv_cache.device,
            )
            mtp_attention = self.mtp.layers[0].self_attn.attn
            mtp_attention.k_cache = self.mtp_kv_cache[0]
            mtp_attention.v_cache = self.mtp_kv_cache[1]

    def allocate_gdn_state_pool(self):
        num_slots = self.config.max_num_seqs
        for layer in self.gdn_layers:
            layer.allocate_state_pool(num_slots)
            if self.config.enable_mtp:
                layer.allocate_speculative_snapshot()

    def allocate_decode_buffers(self):
        size = self.config.max_num_seqs
        self.decode_cpu = {
            "seq_lens": torch.empty(size, dtype=torch.int32, device="cpu", pin_memory=True),
            "last_block_ids": torch.empty(size, dtype=torch.int32, device="cpu", pin_memory=True),
        }
        self.decode_gpu = {name: torch.empty_like(tensor, device="cuda") for name, tensor in self.decode_cpu.items()}
        self.decode_gpu.update({
            "input_ids": torch.empty(size, dtype=torch.int64, device="cuda"),
            "positions": torch.empty(size, dtype=torch.int64, device="cuda"),
            "slot_mapping": torch.empty(size, dtype=torch.int32, device="cuda"),
            "context_lens": torch.empty(size, dtype=torch.int32, device="cuda"),
        })

    @contextmanager
    def synchronize_input_prep(self):
        """Ensure the prior step's async H2D transfers have been consumed
        before this step reuses the same staging buffers (MRV2
        synchronize_input_prep, gpu_model_runner.py:3942-3956).

        Safe under the current synchronous step loop (get_output() already
        forces D2H completion), but required once prep starts overlapping
        with a prior forward.
        """
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        # Chunked prefill and prefix-cache prefill both need paged KV reads.
        # A fresh request has already allocated its full block table before
        # this point; warmup intentionally has no cache and stays on the
        # packed K/V path.
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        # Keep the fast packed K/V path for a fresh full prefill. Only a
        # continuation chunk (or a prefix-cache hit) needs paged KV reads.
        needs_paged_prefill = any(seq.num_cached_tokens > 0 for seq in seqs)
        if needs_paged_prefill and seqs and all(seq.block_table for seq in seqs):
            block_tables = self.prepare_block_tables(seqs)
        prefill_slices = list(zip(cu_seqlens_q[:-1], cu_seqlens_q[1:]))
        prefill_chunk_indices = []
        for batch_idx, (start, end) in enumerate(prefill_slices):
            prefill_chunk_indices.extend(
                (batch_idx, chunk_idx)
                for chunk_idx in range((end - start + 63) // 64)
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        state_indices = self.batch_slots_gpu[:len(seqs)]
        is_verify = all(seq.is_speculative for seq in seqs)
        set_context(
            True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=cu_seqlens_k[1:] if is_verify else None,
            block_tables=block_tables,
            state_indices=state_indices,
            prefill_slices=prefill_slices,
            prefill_chunk_indices=torch.tensor(
                prefill_chunk_indices,
                dtype=torch.int32,
                device=input_ids.device,
            ),
            is_verify=is_verify,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        bs = len(seqs)
        cpu = self.decode_cpu
        for i, seq in enumerate(seqs):
            cpu["seq_lens"][i] = len(seq)
            cpu["last_block_ids"][i] = seq.block_table[-1]
        for name, tensor in cpu.items():
            self.decode_gpu[name][:bs].copy_(tensor[:bs], non_blocking=True)
        gpu = self.decode_gpu
        prepare_decode_kernel(
            self.sampled_token_ids_gpu, self.batch_slots_gpu[:bs],
            gpu["seq_lens"], gpu["last_block_ids"],
            gpu["input_ids"], gpu["positions"], gpu["context_lens"], gpu["slot_mapping"],
            bs, self.block_size,
        )
        input_ids = gpu["input_ids"][:bs]
        positions = gpu["positions"][:bs]
        slot_mapping = gpu["slot_mapping"][:bs]
        context_lens = gpu["context_lens"][:bs]
        block_tables = self.prepare_block_tables(seqs)
        state_indices = self.batch_slots_gpu[:bs]
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            state_indices=state_indices,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def prepare_inputs(self, seqs: list[Sequence], is_prefill: bool):
        with self.synchronize_input_prep():
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        return input_ids, positions, temperatures

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        is_verify: bool = False,
    ):
        # Model warmup runs before graph buffers are allocated. Keep that
        # bootstrap pass eager; graph replay starts after capture_cudagraph().
        if self.enforce_eager or not self.cuda_graphs.decode_graphs:
            hidden = self.model.model(input_ids, positions)
            self._target_hidden = hidden
            return self.model.compute_logits(hidden)
        if is_verify and self.cuda_graphs.verify_graph is not None:
            return self.cuda_graphs.run_verify(input_ids, positions)
        if is_prefill and self.use_prefill_cudagraph:
            return self.cuda_graphs.run_prefill(input_ids, positions)

        if is_prefill:
            hidden = self.model.model(input_ids, positions)
            self._target_hidden = hidden
            return self.model.compute_logits(hidden)

        bs = input_ids.size(0)
        context = get_context()
        # The current graph set uses exact batch sizes (1, 2, 4, ...). Do not
        # run a larger graph with an uninitialized padding row: GDN mutates
        # recurrent state, so a fake row could corrupt a real request slot.
        return self.cuda_graphs.run_decode(input_ids, positions)

    def execute_model(self, seqs: list[Sequence], is_prefill: bool) -> None:
        """MRV2 step: prepare inputs, enqueue the forward, return None.

        Only the kernels are queued onto the compute stream; the engine does
        not wait for them. Sampling is a separate call (sample_tokens), which
        reads the pending logits and performs the async D2H copy.
        """
        self.input_batch.remove_finished()
        seqs, new_entries = self.input_batch.update(seqs)
        slots = self.input_batch.slots_for(seqs)

        if self.rank == 0:
            for seq, slot in new_entries:
                self.sampled_token_ids_gpu[slot] = seq.last_token
        else:
            # Non-zero TP ranks do not run the sampler, so refresh their
            # per-request token slots from the synchronized Sequence state.
            for seq, slot in zip(seqs, slots):
                self.sampled_token_ids_gpu[slot] = seq.last_token

        slots_t = torch.tensor(
            slots,
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        self.batch_slots_gpu[:len(seqs)].copy_(slots_t, non_blocking=True) #copy current batch slots to GPU for kernel access

        if new_entries and self.gdn_layers:
            new_slots = torch.tensor(
                [slot for _, slot in new_entries],
                dtype=torch.int64,
                device=self.batch_slots_gpu.device,
            )
            for layer in self.gdn_layers:
                layer.reset_state(new_slots)

        with trace_event(
            "prepare_inputs", "runner",
            args={"prefill": is_prefill, "bs": len(seqs)},
        ):
            input_ids, positions, temperatures = self.prepare_inputs(seqs, is_prefill)
        with trace_event(
            "run_model", "runner",
            args={"prefill": is_prefill, "tokens": input_ids.size(0)},
        ):
            logits = self.run_model(
                input_ids,
                positions,
                is_prefill,
                is_prefill and all(seq.is_speculative for seq in seqs),
            )
        # Depth-1 async: store the sampling state for the immediately-
        # following sample_tokens() call. batch_slots_gpu is safe to reuse
        # here because no second batch can be dispatched before sampling.
        self._pending = (logits, temperatures, seqs, is_prefill)
        return None

    @torch.inference_mode()
    def _run_mtp_prefill(
        self,
        seq: Sequence,
        slot: int,
        target_hidden: torch.Tensor,
        sampled_token: torch.Tensor,
        final_chunk: bool,
    ) -> torch.Tensor | None:
        """Extend MTP state for one target prefill chunk and return a draft."""
        start = seq.num_cached_tokens
        end = start + seq.num_scheduled_tokens
        input_ids: list[int] = []
        positions: list[int] = []
        hidden_rows: list[torch.Tensor] = []

        previous_hidden = self._mtp_last_hidden.get(slot)
        if start > 0:
            assert previous_hidden is not None
            input_ids.append(seq[start])
            positions.append(start)
            hidden_rows.append(previous_hidden)
        for position in range(start + 1, end):
            input_ids.append(seq[position])
            positions.append(position)
            hidden_rows.append(target_hidden[position - start - 1])
        if final_chunk:
            input_ids.append(int(sampled_token.item()))
            positions.append(end)
            hidden_rows.append(target_hidden[-1])

        count = len(input_ids)
        if count == 0:
            return None
        input_ids_gpu = torch.tensor(
            input_ids, dtype=torch.long, pin_memory=True
        ).cuda(non_blocking=True)
        positions_gpu = torch.tensor(
            positions, dtype=torch.long, pin_memory=True
        ).cuda(non_blocking=True)
        hidden = torch.stack(hidden_rows, dim=0)
        boundaries = torch.tensor([0, count], dtype=torch.int32, device=hidden.device)
        slot_mapping = torch.tensor(
            [
                block_id * self.block_size + position % self.block_size
                for position in positions
                for block_id in (seq.block_table[position // self.block_size],)
            ],
            dtype=torch.int32,
            device=hidden.device,
        )
        block_table = torch.tensor(
            [seq.block_table], dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        self._mtp_last_hidden[slot] = target_hidden[-1].detach().clone()
        set_context(
            True,
            cu_seqlens_q=boundaries,
            cu_seqlens_k=boundaries,
            max_seqlen_q=count,
            max_seqlen_k=count,
            slot_mapping=slot_mapping,
            block_tables=block_table,
            state_indices=None,
            prefill_slices=[(0, count)],
            prefill_chunk_indices=torch.tensor(
                [(0, i) for i in range((count + 63) // 64)],
                dtype=torch.int32,
                device=hidden.device,
            ),
        )
        try:
            mtp_hidden = self.mtp(
                input_ids_gpu,
                positions_gpu,
                hidden,
                self.model.model.embed_tokens,
            )
            logits = self.model.compute_logits(mtp_hidden)
            return logits[-1].argmax() if final_chunk else None
        finally:
            reset_context()

    @torch.inference_mode()
    def _run_mtp_decode(
        self,
        seqs: list[Sequence],
        target_hidden: torch.Tensor,
        input_tokens: torch.Tensor,
        positions: list[int],
    ) -> torch.Tensor:
        """Run the one-layer MTP head as a batched decode step."""
        max_blocks = max(len(seq.block_table) for seq in seqs)
        block_tables = torch.tensor(
            [
                seq.block_table + [-1] * (max_blocks - len(seq.block_table))
                for seq in seqs
            ],
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            [
                seq.block_table[position // self.block_size] * self.block_size
                + position % self.block_size
                for seq, position in zip(seqs, positions)
            ],
            dtype=torch.int32,
            device=input_tokens.device,
        )
        positions_gpu = torch.tensor(
            positions, dtype=torch.long, device=input_tokens.device,
        )
        context_lens = positions_gpu.to(torch.int32)
        if self.cuda_graphs.mtp_decode_graph is not None:
            return self.cuda_graphs.run_mtp_decode(
                input_tokens, positions_gpu, target_hidden, slot_mapping,
                context_lens, block_tables,
            )
        set_context(
            False,
            slot_mapping=slot_mapping,
            # MTP positions are one-based: positions 1..p contain p cached
            # tokens after the current write, so context length is p.
            context_lens=context_lens,
            block_tables=block_tables,
            state_indices=None,
        )
        try:
            mtp_hidden = self.mtp(
                input_tokens,
                positions_gpu,
                target_hidden,
                self.model.model.embed_tokens,
            )
            logits = self.model.compute_logits(mtp_hidden)
            return logits.argmax(dim=-1)
        finally:
            reset_context()

    def sample_tokens(self) -> AsyncModelOutput | None:
        if self.rank != 0:
            # Non-zero TP ranks do not run the sampler: drop the pending
            # batch's state so rank 0 stays authoritative.
            self._pending = None
            reset_context()
            return None

        logits, temperatures, seqs, is_prefill = self._pending
        self._pending = None
        bs = len(seqs)
        slot_ids = self.input_batch.slots_for(seqs)
        speculative_batch = is_prefill and all(seq.is_speculative for seq in seqs)

        # Prefill normally samples one tail row. A speculative prefill has
        # two rows per request: row 0 verifies the MTP draft and row 1 is
        # valid only after acceptance.
        starts = []
        tail_indices = []
        offset = 0
        for seq in seqs:
            starts.append(offset)
            tail_indices.append(offset + seq.num_scheduled_tokens - 1)
            offset += seq.num_scheduled_tokens

        tail_logits = logits
        if is_prefill and not speculative_batch:
            tail_indices_gpu = torch.tensor(
                tail_indices,
                dtype=torch.int64,
                device=logits.device,
            )
            tail_logits = logits.index_select(0, tail_indices_gpu)

        slots = self.batch_slots_gpu[:bs]
        # Validation-only samplers may request row-aligned logits/GDN-state
        # diagnostics. The production sampler has no hooks, so its behavior
        # remains unchanged.
        set_batch_context = getattr(self.sampler, "set_batch_context", None)
        if set_batch_context is not None:
            set_batch_context(seqs, is_prefill)
        observe_gdn_state = getattr(self.sampler, "observe_gdn_state", None)
        if observe_gdn_state is not None:
            observe_gdn_state(self.gdn_layers, slots)

        with trace_event("sampler", "runner", args={"prefill": is_prefill, "bs": bs}):
            # A verify batch uses greedy top-1 on both target rows below.
            # The generic sampler's full-vocabulary softmax is unused here.
            normal_tokens = (
                torch.empty(bs, dtype=torch.int64, device=logits.device)
                if speculative_batch
                else self.sampler(tail_logits, temperatures)
            )

        output_gpu = torch.full(
            (bs, 3), -1, dtype=torch.int64, device=logits.device
        )
        sampled_values = normal_tokens.clone()
        mtp_seqs: list[Sequence] = []
        mtp_hidden_rows: list[torch.Tensor] = []
        mtp_inputs: list[torch.Tensor] = []
        mtp_positions: list[int] = []

        if speculative_batch:
            assert self.mtp is not None
            self.mtp_stats["verify_steps"] += 1

        for i, seq in enumerate(seqs):
            start = starts[i]
            if is_prefill and seq.is_speculative:
                first_token = logits[start].argmax()
                draft_token = seq.draft_token
                first_token_id = int(first_token.item())
                accepted = first_token_id == draft_token
                draft_is_eos = (
                    not seq.ignore_eos and draft_token == self.config.eos
                )
                second_token = None
                if accepted:
                    output_gpu[i, 0] = draft_token
                    self.mtp_stats["accepted_tokens"] += 1
                    for layer in self.gdn_layers:
                        layer.commit_speculative_state()
                    if not draft_is_eos:
                        second_token = logits[start + 1].argmax()
                        output_gpu[i, 1] = second_token
                        sampled_values[i] = second_token
                    else:
                        sampled_values[i] = draft_token
                else:
                    output_gpu[i, 0] = first_token
                    sampled_values[i] = first_token
                    self.mtp_stats["rejected_tokens"] += 1
                    for layer in self.gdn_layers:
                        layer.rollback_speculative_state(slot_ids[i])

                committed_before = seq.num_tokens - seq.num_prompt_tokens - 1
                produced = 2 if accepted and second_token is not None else 1
                hit_eos = (
                    not seq.ignore_eos
                    and (
                        first_token_id == self.config.eos
                        or (
                            second_token is not None
                            and int(second_token.item()) == self.config.eos
                        )
                    )
                )
                if not hit_eos and committed_before + produced < seq.max_tokens:
                    mtp_seqs.append(seq)
                    mtp_hidden_rows.append(
                        self._target_hidden[start + 1 if accepted else start]
                    )
                    mtp_inputs.append(sampled_values[i])
                    mtp_positions.append(
                        seq.num_tokens if accepted else seq.num_tokens - 1
                    )
                continue

            token = normal_tokens[i]
            output_gpu[i, 0] = token
            if not (is_prefill and self.mtp is not None and seq.temperature <= 1e-6):
                continue

            final_chunk = (
                seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens
            )
            draft = self._run_mtp_prefill(
                seq,
                slot_ids[i],
                self._target_hidden[start:start + seq.num_scheduled_tokens],
                token,
                final_chunk,
            )
            if final_chunk and draft is not None and seq.num_completion_tokens + 1 < seq.max_tokens:
                output_gpu[i, 2] = draft
                self.mtp_stats["draft_tokens"] += 1

        if mtp_seqs:
            assert self.mtp is not None
            self.mtp_stats["draft_tokens"] += len(mtp_seqs)
            next_drafts = self._run_mtp_decode(
                mtp_seqs,
                torch.stack(mtp_hidden_rows, dim=0),
                torch.stack(mtp_inputs, dim=0),
                mtp_positions,
            )
            batch_indices = torch.tensor(
                [seqs.index(seq) for seq in mtp_seqs],
                dtype=torch.int64,
                device=logits.device,
            )
            output_gpu[batch_indices, 2] = next_drafts

        self.sampled_token_ids_gpu.scatter_(0, slots, sampled_values)
        token_ids = output_gpu
        self._target_hidden = None

        buf_idx = self._output_buf_idx
        self._output_buf_idx ^= 1 #ping-pong buffer index, switch between 0 and 1 for double buffering
        output_buf = self._output_pin_bufs[buf_idx][:bs]
        ready_event = self._output_events[buf_idx]

        if self.async_output:
            self.output_copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.output_copy_stream):
                # Non-blocking enqueue; the real copy latency surfaces as
                # d2h_wait on the consuming step.
                with trace_event("d2h_copy", "runner", args={"bs": bs}):
                    output_buf.copy_(token_ids, non_blocking=True)
                ready_event.record(self.output_copy_stream)
        else:
            # Blocking D2H baseline: return only after the CPU buffer is ready.
            with trace_event("d2h_copy_sync", "runner", args={"bs": bs}):
                output_buf.copy_(token_ids, non_blocking=False)
            ready_event.record(torch.cuda.current_stream())

        reset_context()
        return AsyncModelOutput(token_ids, output_buf, ready_event)

    def remove_request(self, seq_id: int):
        self.input_batch.remove(seq_id)

    def capture_cudagraph(self):
        """Capture decode and piecewise-prefill graphs through the manager."""
        self.cuda_graphs.capture()

    @property
    def graph_bs(self):
        return self.cuda_graphs.decode_graph_sizes

    @property
    def graphs(self):
        return self.cuda_graphs.decode_graphs

    @property
    def graph_vars(self):
        return self.cuda_graphs.decode_graph_vars

    @property
    def graph_pool(self):
        return self.cuda_graphs.decode_graph_pool

    @property
    def prefill_graph_sizes(self):
        return self.cuda_graphs.prefill_graph_sizes

    @prefill_graph_sizes.setter
    def prefill_graph_sizes(self, value):
        self.cuda_graphs.prefill_graph_sizes = list(value)

    @property
    def prefill_piecewise_graphs(self):
        return self.cuda_graphs.prefill_graphs
