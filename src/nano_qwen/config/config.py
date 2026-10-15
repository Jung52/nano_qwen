import os
from dataclasses import dataclass, field
from transformers import AutoConfig

from nano_qwen.quantization import QuantizationConfig, get_quantization_config
from nano_qwen.quantization.checkpoint import read_quantization_metadata, read_hf_fp8_config
from nano_qwen.utils.rope import rope_config


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
    # MTP2 currently implements greedy one-draft speculative decoding. Keep it
    # opt-in: older checkpoints may not contain ``mtp.*`` weights and the
    # stochastic rejection-sampling path is not implemented yet.
    enable_mtp: bool = False
    # Converted FP8 targets may omit mtp.*. Load its BF16 head independently.
    mtp_model: str | None = None
    # Reuse is opt-in for the validated FP8 TP1/B1 path; serial is the reference.
    mtp_verify_mode: str = "serial"
    hf_config: AutoConfig | None = None
    full_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    is_hybrid: bool = False
    max_state_slots: int = 0
    quantization: str | None = None
    quant_config: QuantizationConfig | None = field(default=None, init=False)

    def __post_init__(self):
        if self.mtp_verify_mode not in ("serial", "reuse"):
            raise ValueError("mtp_verify_mode must be 'serial' or 'reuse'")
        if self.mtp_verify_mode == "reuse" and not self.enable_mtp:
            raise ValueError("mtp_verify_mode='reuse' requires enable_mtp=True")
        metadata = read_quantization_metadata(self.model)
        hf_quant = read_hf_fp8_config(self.model) if metadata is None else None
        if metadata is not None and self.quantization is None:
            self.quantization = metadata["quant_method"]
        if hf_quant is not None and self.quantization is None:
            self.quantization = "fp8"
        self.quant_config = get_quantization_config(self.quantization, hf_quant)
        if self.mtp_verify_mode == "reuse" and self.quant_config is None:
            raise ValueError("MTP reuse currently requires an FP8 target")
        if self.quant_config is not None:
            if self.tensor_parallel_size != 1:
                raise ValueError("FP8 currently requires tensor_parallel_size=1")
            if metadata is None and hf_quant is None:
                raise ValueError("FP8 requires a converted nano_qwen checkpoint or HF block FP8 config")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert not self.enable_mtp or self.tensor_parallel_size == 1, (
            "MTP2 production decoding currently requires tensor_parallel_size=1"
        )
        assert not self.enable_mtp or self.max_num_seqs == 1, (
            "MTP2 production decoding currently requires max_num_seqs=1"
        )

        self.full_config = AutoConfig.from_pretrained(self.model)
        self.hf_config = getattr(self.full_config, "text_config", self.full_config)
        if self.enable_mtp and self.mtp_model is not None:
            if not os.path.isdir(self.mtp_model):
                raise ValueError(f"MTP checkpoint directory does not exist: {self.mtp_model}")
            source = AutoConfig.from_pretrained(self.mtp_model)
            source = getattr(source, "text_config", source)
            for name in ("hidden_size", "intermediate_size", "vocab_size",
                         "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                         "head_dim", "rms_norm_eps", "mtp_num_hidden_layers",
                         "mtp_use_dedicated_embeddings", "layer_types", "hidden_act",
                         "attention_bias", "linear_num_key_heads", "linear_num_value_heads",
                         "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim"):
                if getattr(source, name, None) != getattr(self.hf_config, name, None):
                    raise ValueError(f"MTP source and target differ in {name}")
            if rope_config(source) != rope_config(self.hf_config):
                raise ValueError("MTP source and target differ in effective RoPE configuration")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        layer_types = getattr(self.hf_config, "layer_types", None)
        self.is_hybrid = isinstance(layer_types, list) and any(
            t != "full_attention" for t in layer_types
        )
