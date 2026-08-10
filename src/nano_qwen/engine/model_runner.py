import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.decode_init import prepare_decode as prepare_decode_kernel
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class AsyncOutput:
    """Handle for an in-flight or recently completed sample.

    ``get_output()`` blocks only until the runner's device-to-host copy
    finishes. The engine may hold this handle across multiple ``step()``
    calls to overlap D2H with the next forward.
    """

    def __init__(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        num_tokens: int,
        token_ids_pin: torch.Tensor,
        copy_event: torch.cuda.Event,
    ) -> None:
        self.seqs = seqs
        self.is_prefill = is_prefill
        self.num_tokens = num_tokens
        self._token_ids_pin = token_ids_pin
        self._copy_event = copy_event

    def get_output(self) -> list[int]:
        self._copy_event.synchronize()
        return self._token_ids_pin[: self.num_tokens].tolist()


class InputBatch:
    """Minimal persistent request-to-slot mapping for nano-vllm."""

    def __init__(self, max_num_seqs: int):
        self.seqs: list[Sequence | None] = [None] * max_num_seqs
        self.seq_id_to_slot: dict[int, int] = {}

    def update(self, seqs: list[Sequence]) -> list[Sequence]:
        for seq in seqs:
            slot = self.seq_id_to_slot.get(seq.seq_id)
            if slot is None:
                slot = next(i for i, item in enumerate(self.seqs) if item is None)
                self.seq_id_to_slot[seq.seq_id] = slot
            self.seqs[slot] = seq
        return [self.seqs[self.seq_id_to_slot[seq.seq_id]] for seq in seqs]

    def remove_finished(self):
        for seq_id in list(self.seq_id_to_slot):
            slot = self.seq_id_to_slot[seq_id]
            seq = self.seqs[slot]
            if seq is not None and seq.is_finished:
                self.seqs[slot] = None
                del self.seq_id_to_slot[seq_id]

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
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.output_copy_stream = torch.cuda.Stream()
        self._output_pin_bufs = [
            torch.empty(config.max_num_batched_tokens, dtype=torch.int64, pin_memory=True)
            for _ in range(2)
        ]
        self._output_buf_idx = 0
        self._forward_done = torch.cuda.Event()
        self.allocate_decode_buffers()
        self.input_batch = InputBatch(config.max_num_seqs) if rank == 0 else None
        self._pending_logits = None
        self._pending_temperatures = None
        self._pending_seqs = None
        self._pending_is_prefill = False
        self.warmup_model()
        if self.input_batch is not None:
            self.input_batch.clear()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
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
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def allocate_decode_buffers(self):
        size = self.config.max_num_seqs
        self.decode_cpu = {
            "last_token_ids": torch.empty(size, dtype=torch.int64, device="cpu", pin_memory=True),
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
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        bs = len(seqs)
        cpu = self.decode_cpu
        for i, seq in enumerate(seqs):
            cpu["last_token_ids"][i] = seq.last_token
            cpu["seq_lens"][i] = len(seq)
            cpu["last_block_ids"][i] = seq.block_table[-1]
        for name, tensor in cpu.items():
            self.decode_gpu[name][:bs].copy_(tensor[:bs], non_blocking=True)
        gpu = self.decode_gpu
        prepare_decode_kernel(
            gpu["last_token_ids"], gpu["seq_lens"], gpu["last_block_ids"],
            gpu["input_ids"], gpu["positions"], gpu["context_lens"], gpu["slot_mapping"],
            bs, self.block_size,
        )
        input_ids = gpu["input_ids"][:bs]
        positions = gpu["positions"][:bs]
        slot_mapping = gpu["slot_mapping"][:bs]
        context_lens = gpu["context_lens"][:bs]
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def prepare_inputs(self, seqs: list[Sequence], is_prefill: bool):
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        return input_ids, positions, temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def execute_model(self, seqs: list[Sequence], is_prefill: bool):
        if getattr(self, '_forward_in_flight', False):
            self._forward_done.synchronize()  #waiting for the previous forward to finish

        if self.input_batch is not None:
            self.input_batch.remove_finished()
            seqs = self.input_batch.update(seqs)
        input_ids, positions, temperatures = self.prepare_inputs(seqs, is_prefill)
        self._pending_logits = self.run_model(input_ids, positions, is_prefill)
        self._pending_temperatures = temperatures
        self._pending_seqs = seqs
        self._pending_is_prefill = is_prefill
        self._forward_done.record(torch.cuda.current_stream())
        self._forward_in_flight = True

    def sample_tokens(self) -> AsyncOutput | None:
        if self.rank != 0:
            self._pending_logits = self._pending_temperatures = self._pending_seqs = None
            reset_context()
            return None

        token_ids = self.sampler(self._pending_logits, self._pending_temperatures)
        num_tokens = token_ids.numel()
        sample_done = torch.cuda.Event()
        sample_done.record(torch.cuda.current_stream())

        output_buf = self._output_pin_bufs[self._output_buf_idx]
        self._output_buf_idx = (self._output_buf_idx + 1) % len(self._output_pin_bufs)

        with torch.cuda.stream(self.output_copy_stream):
            self.output_copy_stream.wait_event(sample_done)
            output_buf[:num_tokens].copy_(token_ids, non_blocking=True)
            copy_done = torch.cuda.Event()
            copy_done.record(self.output_copy_stream)

        seqs = self._pending_seqs
        is_prefill = self._pending_is_prefill
        self._pending_logits = self._pending_temperatures = self._pending_seqs = None
        reset_context()
        return AsyncOutput(seqs, is_prefill, num_tokens, output_buf, copy_done)

    def remove_request(self, seq_id: int):
        if self.input_batch is not None:
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
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
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
            outputs=outputs,
        )