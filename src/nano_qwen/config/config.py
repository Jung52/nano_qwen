import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    # GDN recurrent state is not part of the prefix-cache block payload.
    # Keep prefix reuse opt-in until state snapshots are cached as well.
    enable_prefix_cache: bool = False
    hf_config: AutoConfig | None = None
    full_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    is_hybrid: bool = False
    max_state_slots: int = 0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8

        self.full_config = AutoConfig.from_pretrained(self.model)
        self.hf_config = getattr(self.full_config, "text_config", self.full_config)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        layer_types = getattr(self.hf_config, "layer_types", None)
        self.is_hybrid = isinstance(layer_types, list) and any(
            t != "full_attention" for t in layer_types
        )
