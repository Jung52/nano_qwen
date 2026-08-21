"""Text-only Qwen3.5 Dense model for nano_qwen.

This module implements the hybrid Qwen3.5 decoder used by the dense text
checkpoints: Gated DeltaNet linear-attention blocks interleaved with gated
GQA blocks. Vision, MoE and MTP are intentionally out of scope here.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3_5TextConfig

from nano_qwen.layers.activation import SiluAndMul
from nano_qwen.layers.attention import Attention
from nano_qwen.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nano_qwen.layers.gated_delta_net import GatedDeltaNet
from nano_qwen.layers.layernorm import GemmaRMSNorm
from nano_qwen.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from nano_qwen.layers.rotary_embedding import InterleavedMRoPE
from nano_qwen.utils.context import get_context


def _rope_config(config: Qwen3_5TextConfig) -> tuple[float, float, list[int]]:
    """Read both current and early Qwen3.5 RoPE config layouts."""
    rope = getattr(config, "rope_parameters", None)
    if not isinstance(rope, dict):
        rope = getattr(config, "rope_scaling", None)
    if not isinstance(rope, dict):
        rope = {}

    base = rope.get("rope_theta", getattr(config, "rope_theta", 10_000_000.0))
    partial = rope.get(
        "partial_rotary_factor",
        getattr(config, "partial_rotary_factor", 0.25),
    )
    section = rope.get("mrope_section", [11, 11, 10])
    return float(base), float(partial), list(section)


class Qwen3_5Attention(nn.Module):
    """Qwen3.5 gated GQA.

    Unlike Qwen3, ``q_proj`` produces both query and per-query-head gate.
    Q/K are Gemma-style RMS-normalized before partial MRoPE, and the
    attention output is multiplied by ``sigmoid(gate)`` before ``o_proj``.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None:
        super().__init__()
        del layer_idx  # Kept in the signature for parity with the HF model.

        tp_size = dist.get_world_size()
        if config.num_attention_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by TP size")
        if config.num_key_value_heads % tp_size:
            raise ValueError("num_key_value_heads must be divisible by TP size")

        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        attention_bias = getattr(config, "attention_bias", False)

        # Checkpoint q_proj layout is [head, query | gate], so a contiguous
        # column-parallel shard owns complete local heads.
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            2 * self.total_num_heads * self.head_dim,
            bias=attention_bias,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=attention_bias,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_theta, partial_rotary_factor, mrope_section = _rope_config(config)
        self.rotary_emb = InterleavedMRoPE(
            head_size=self.head_dim,
            partial_rotary_factor=partial_rotary_factor,
            mrope_section=mrope_section,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_gate = self.q_proj(hidden_states)
        #把1维拼接拆成[token数，头数，2x头维度]，在使用dim=-1拆成query和gate两个部分
        q_gate = q_gate.view(-1, self.num_heads, 2 * self.head_dim)
        query, gate = q_gate.chunk(2, dim=-1)

        key = self.k_proj(hidden_states).view(
            -1, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            -1, self.num_kv_heads, self.head_dim
        )

        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self.rotary_emb(positions, query, key)

        attn_output = self.attn(query, key, value)
        attn_output = attn_output.reshape(-1, self.q_size)
        gate = gate.reshape(-1, self.q_size)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Qwen3_5MLP(nn.Module):

    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"Qwen3.5 expects silu, got {config.hidden_act!r}")
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(hidden_states)))


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None:
        super().__init__()
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            interval = getattr(config, "full_attention_interval", 4)
            block_type = (
                "full_attention" if (layer_idx + 1) % interval == 0
                else "linear_attention"
            )
        else:
            block_type = layer_types[layer_idx]

        self.block_type = block_type
        if block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, layer_idx)
        elif block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {block_type!r}")

        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def _run_linear_attention(
        self,
        hidden_states: torch.Tensor,
        prefill_slices: list[tuple[int, int]] | None,
    ) -> torch.Tensor:
        """Adapt the runner's packed 2D tensors to GDN's [B, S, H] API.

        GDN state is per request, so requests are kept separate. This is a
        correctness-first bridge until the layer consumes packed varlen input
        and ``state_indices`` directly in one kernel launch.
        """
        context = get_context()
        state_indices = context.state_indices
        outputs: list[torch.Tensor] = []

        try:
            if context.is_prefill:
                if prefill_slices is None:
                    raise RuntimeError("prefill_slices are required for GDN prefill")
                for batch_idx, (start, end) in enumerate(prefill_slices):
                    context.state_indices = (
                        None
                        if state_indices is None
                        else state_indices[batch_idx:batch_idx + 1]
                    )
                    output = self.linear_attn(
                        hidden_states[start:end].unsqueeze(0)
                    )
                    outputs.append(output.squeeze(0))
            else:
                if state_indices is None:
                    raise RuntimeError(
                        "GDN decode requires Context.state_indices; "
                        "ModelRunner must pass persistent request slots"
                    )
                for batch_idx in range(hidden_states.shape[0]):
                    context.state_indices = state_indices[
                        batch_idx:batch_idx + 1
                    ]
                    output = self.linear_attn(
                        hidden_states[batch_idx:batch_idx + 1].unsqueeze(1)
                    )
                    outputs.append(output.squeeze(1))
        finally:
            context.state_indices = state_indices

        return torch.cat(outputs, dim=0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        prefill_slices: list[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states,
                residual,
            )

        if self.block_type == "linear_attention":
            hidden_states = self._run_linear_attention(
                hidden_states,
                prefill_slices,
            )
        else:
            hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5Model(nn.Module):

    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.has_linear_attention = any(
            layer.block_type == "linear_attention" for layer in self.layers
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        prefill_slices = None
        context = get_context()
        if self.has_linear_attention and context.is_prefill:
            if context.cu_seqlens_q is None:
                raise RuntimeError("GDN prefill requires cu_seqlens_q")
            # One small synchronization per model forward, not once per GDN
            # layer. The next optimization is a packed varlen GDN entry point.
            cu_seqlens = context.cu_seqlens_q.tolist()
            prefill_slices = list(zip(cu_seqlens[:-1], cu_seqlens[1:]))

        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                prefill_slices,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3_5ForCausalLM(nn.Module):
    # Only MLP projections are physically packed in nano_qwen. Qwen3.5 keeps
    # q/k/v projections separate because q_proj also contains the query gate.
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Return logits for every row; ModelRunner selects prefill tails.

        ``ParallelLMHead.forward`` performs that selection itself, but the
        current ModelRunner also does it in ``sample_tokens``. Computing the
        projection here keeps a single source of truth and matches the
        runner's existing pending-logits contract.
        """
        logits = F.linear(hidden_states, self.lm_head.weight)
        if self.lm_head.tp_size == 1:
            return logits

        gathered = (
            [torch.empty_like(logits) for _ in range(self.lm_head.tp_size)]
            if self.lm_head.tp_rank == 0
            else None
        )
        dist.gather(logits, gathered, dst=0)
        return torch.cat(gathered, dim=-1) if gathered is not None else None


__all__ = [
    "Qwen3_5Attention",
    "Qwen3_5MLP",
    "Qwen3_5DecoderLayer",
    "Qwen3_5Model",
    "Qwen3_5ForCausalLM",
]
