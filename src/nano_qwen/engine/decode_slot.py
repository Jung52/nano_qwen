"""Temporary execution-slot buffers; distinct from persistent request slots."""
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class DecodeSlot:
    slot_id: int
    cpu: dict
    gpu: dict
    batch_slots_cpu: torch.Tensor
    batch_slots_gpu: torch.Tensor
    blocks_cpu: torch.Tensor
    blocks_gpu: torch.Tensor
    temperatures_cpu: torch.Tensor
    temperatures_gpu: torch.Tensor
    tokens_gpu: torch.Tensor
    tokens_cpu: torch.Tensor
    h2d_done: object
    sample_done: object
    d2h_done: object
    busy: bool = False
    pending: object = None
    output: object = None

    @classmethod
    def allocate(cls, slot_id, config):
        size = config.max_num_seqs
        blocks = (config.max_model_len + config.kvcache_block_size - 1) // config.kvcache_block_size
        def host(shape, dtype):
            return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        def gpu(shape, dtype):
            return torch.empty(shape, dtype=dtype, device="cuda")
        cpu = {name: host(size, torch.int32) for name in ("seq_lens", "last_block_ids")}
        device = {name: gpu(size, torch.int32) for name in cpu}
        device.update({name: gpu(size, dtype) for name, dtype in (
            ("input_ids", torch.int64), ("positions", torch.int64),
            ("slot_mapping", torch.int32), ("context_lens", torch.int32))})
        return cls(slot_id, cpu, device,
                   host(size, torch.int64), gpu(size, torch.int64),
                   host((size, blocks), torch.int32), gpu((size, blocks), torch.int32),
                   host(size, torch.float32), gpu(size, torch.float32),
                   gpu(size, torch.int64), host(size, torch.int64),
                   torch.cuda.Event(blocking=True), torch.cuda.Event(),
                   torch.cuda.Event(blocking=True))
