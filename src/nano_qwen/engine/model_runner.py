import pickle
from contextlib import contextmanager

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nano_qwen.config import Config
from nano_qwen.engine.sequence import Sequence
from nano_qwen.layers.gated_delta_net import GatedDeltaNet
from nano_qwen.layers.sampler import Sampler
from nano_qwen.models.qwen3_5 import Qwen3_5ForCausalLM
from nano_qwen.utils.context import set_context, get_context, reset_context
from nano_qwen.utils.loader import load_model

from .async_output import AsyncModelOutput
from .decode_init import prepare_decode as prepare_decode_kernel


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

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
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
                config.max_num_seqs,
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
        self.use_prefill_cudagraph = True
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
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

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
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
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

    def allocate_gdn_state_pool(self):
        num_slots = self.config.max_num_seqs
        for layer in self.gdn_layers:
            layer.allocate_state_pool(num_slots)

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
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            raise RuntimeError(
                "prefix-cache prefill is not supported by this runner"
            )
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
        set_context(
            True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            state_indices=state_indices,
            prefill_slices=prefill_slices,
            prefill_chunk_indices=torch.tensor(
                prefill_chunk_indices,
                dtype=torch.int32,
                device=input_ids.device,
            ),
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
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # Model warmup runs before graph buffers are allocated. Keep that
        # bootstrap pass eager; graph replay starts after capture_cudagraph().
        if self.enforce_eager or not hasattr(self, "graphs") or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        if is_prefill and self.use_prefill_cudagraph:
            return self.run_prefill_piecewise(input_ids, positions)

        if is_prefill:
            return self.model.compute_logits(self.model(input_ids, positions))

        bs = input_ids.size(0)
        context = get_context()
        # The current graph set uses exact batch sizes (1, 2, 4, ...). Do not
        # run a larger graph with an uninitialized padding row: GDN mutates
        # recurrent state, so a fake row could corrupt a real request slot.
        if bs not in self.graphs:
            return self.model.compute_logits(self.model(input_ids, positions))
        graph = self.graphs[bs]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"].fill_(-1)
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph_vars["state_indices"].zero_()
        graph_vars["state_indices"][:bs] = context.state_indices
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run_prefill_piecewise(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """Run prefill with vLLM-style token-bucketed graph segments.

        Token-wise dense work before and after attention is captured with a
        fixed padded token count. Variable-length GDN/full-attention remains
        eager and consumes the real packed rows and metadata. Consequently,
        different request layouts with the same padded total-token count can
        share these graphs without capturing a dynamic kernel launch grid.
        """
        context = get_context()
        if context.block_tables is not None:
            raise RuntimeError(
                "prefix-cache prefill is not supported by the CUDA Graph path"
            )
        if context.prefill_slices is None:
            raise RuntimeError("prefill graph requires Context.prefill_slices")

        num_tokens = input_ids.size(0)
        graph_size = next(
            (size for size in self.prefill_graph_sizes if size >= num_tokens),
            None,
        )
        if graph_size is None:
            return self.model.compute_logits(self.model(input_ids, positions))

        entries = self.prefill_piecewise_graphs.setdefault(graph_size, {})
        hidden = self.model.model.embed_tokens(input_ids)
        residual = None
        for layer_idx, layer in enumerate(self.model.model.layers):
            entry = entries.get(layer_idx)
            if entry is None:
                entry = self.capture_prefill_layer_segments(
                    layer,
                    graph_size,
                    has_residual=residual is not None,
                )
                entries[layer_idx] = entry

            pre = entry["pre"]
            pre["hidden_in"].zero_()
            pre["hidden_in"][:num_tokens].copy_(hidden[:num_tokens])
            if residual is not None:
                pre["residual_in"].zero_()
                pre["residual_in"][:num_tokens].copy_(residual[:num_tokens])
            pre["graph"].replay()
            attention_input = pre["hidden_out"][:num_tokens]
            layer_residual = pre["residual_out"]

            # This is the graph break: only real packed tokens enter the
            # variable-length operator, with the current request boundaries.
            attention_output = layer.forward_attention(
                positions,
                attention_input,
                context.prefill_slices,
            )

            post = entry["post"]
            post["hidden_in"].zero_()
            post["hidden_in"][:num_tokens].copy_(attention_output)
            post["residual_in"].copy_(layer_residual)
            post["graph"].replay()
            hidden = post["hidden_out"]
            residual = post["residual_out"]

        hidden, _ = self.model.model.norm(
            hidden[:num_tokens],
            residual[:num_tokens],
        )
        return self.model.compute_logits(hidden)

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

        input_ids, positions, temperatures = self.prepare_inputs(seqs, is_prefill)
        logits = self.run_model(input_ids, positions, is_prefill)
        # Depth-1 async: store the sampling state for the immediately-
        # following sample_tokens() call. batch_slots_gpu is safe to reuse
        # here because no second batch can be dispatched before sampling.
        self._pending = (logits, temperatures, seqs, is_prefill)
        return None

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

        # Prefill produces one logits row per scheduled token. Sampling only
        # uses the final scheduled position of each request.
        if is_prefill:
            last_indices = []
            offset = 0
            for seq in seqs:
                offset += seq.num_scheduled_tokens
                last_indices.append(offset - 1)
            last_indices_gpu = torch.tensor(
                last_indices,
                dtype=torch.int64,
                device=logits.device,
            )
            logits = logits.index_select(0, last_indices_gpu)

        token_ids = self.sampler(logits, temperatures)
        slots = self.batch_slots_gpu[:bs]
        self.sampled_token_ids_gpu.scatter_(0, slots, token_ids)

        buf_idx = self._output_buf_idx
        self._output_buf_idx ^= 1 #ping-pong buffer index, switch between 0 and 1 for double buffering
        output_buf = self._output_pin_bufs[buf_idx][:bs] 
        ready_event = self._output_events[buf_idx]

        if self.async_output:
            self.output_copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.output_copy_stream):
                output_buf.copy_(token_ids, non_blocking=True)
                ready_event.record(self.output_copy_stream)
        else:
            # Blocking D2H baseline: return only after the CPU buffer is ready.
            output_buf.copy_(token_ids, non_blocking=False)
            ready_event.record(torch.cuda.current_stream())

        reset_context()
        return AsyncModelOutput(token_ids, output_buf, ready_event)

    def remove_request(self, seq_id: int):
        self.input_batch.remove(seq_id)

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        state_indices = torch.zeros(max_bs, dtype=torch.int64)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                state_indices=state_indices[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            state_indices=state_indices,
            outputs=outputs,
        )
        max_prefill_tokens = min(config.max_num_batched_tokens, 512)
        self.prefill_graph_sizes = [
            size
            for size in ([1, 2, 4, 8] + list(range(16, max_prefill_tokens + 1, 16)))
            if size <= max_prefill_tokens
        ]
        if (
            max_prefill_tokens > 0
            and max_prefill_tokens not in self.prefill_graph_sizes
        ):
            self.prefill_graph_sizes.append(max_prefill_tokens)
        # Piecewise prefill graphs are captured lazily by padded total-token
        # count. Decode graphs above remain full graphs by exact batch size.
        self.prefill_piecewise_graphs = {}

    @torch.inference_mode()
    def capture_prefill_layer_segments(
        self,
        layer,
        graph_size: int,
        has_residual: bool,
    ):
        """Capture the two static pieces around one dynamic attention op."""
        hidden_size = self.config.hf_config.hidden_size
        model_weight = self.model.model.embed_tokens.weight
        pre_hidden = torch.zeros(
            graph_size,
            hidden_size,
            dtype=model_weight.dtype,
            device=model_weight.device,
        )
        pre_residual = (
            torch.zeros_like(pre_hidden) if has_residual else None
        )
        for _ in range(2):
            pre_outputs = layer.forward_input(pre_hidden, pre_residual)
        torch.cuda.synchronize()
        pre_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(pre_graph, self.graph_pool):
            pre_outputs = layer.forward_input(pre_hidden, pre_residual)
        torch.cuda.synchronize()

        post_hidden = torch.zeros_like(pre_hidden)
        post_residual = torch.zeros_like(pre_hidden)
        for _ in range(2):
            post_outputs = layer.forward_output(post_hidden, post_residual)
        torch.cuda.synchronize()
        post_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(post_graph, self.graph_pool):
            post_outputs = layer.forward_output(post_hidden, post_residual)
        torch.cuda.synchronize()

        return {
            "pre": {
                "graph": pre_graph,
                "hidden_in": pre_hidden,
                "residual_in": pre_residual,
                "hidden_out": pre_outputs[0],
                "residual_out": pre_outputs[1],
            },
            "post": {
                "graph": post_graph,
                "hidden_in": post_hidden,
                "residual_in": post_residual,
                "hidden_out": post_outputs[0],
                "residual_out": post_outputs[1],
            },
        }
