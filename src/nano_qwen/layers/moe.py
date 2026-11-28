"""Mixture-of-experts feed-forward layer for Qwen3.5 text models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from nano_qwen.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from nano_qwen.layers.moe_grouped import grouped_experts
from nano_qwen.utils.context import get_context

if TYPE_CHECKING:
    from transformers import Qwen3_5MoeTextConfig


class Qwen3_5RoutedExperts(nn.Module):
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        if (not get_context().is_prefill and hidden_states.is_cuda
                and hidden_states.dtype == torch.bfloat16 and hidden_states.numel()):
            return grouped_experts(
                hidden_states.contiguous(), expert_ids, expert_weights,
                self.gate_up_proj, self.down_proj,
            )
        return self.forward_eager(hidden_states, expert_ids, expert_weights)

    def forward_eager(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        # [expert, top-k position, token row], so each dispatched row has a unique weight.
        assignments = F.one_hot(expert_ids, num_classes=self.num_experts).permute(2, 1, 0)
        active_experts = assignments.any(dim=(1, 2)).nonzero(as_tuple=True)[0].tolist()
        for expert_id in active_experts:
            topk_positions, token_rows = torch.where(assignments[expert_id])
            gate, up = F.linear(hidden_states[token_rows], self.gate_up_proj[expert_id]).chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, self.down_proj[expert_id])
            weights = expert_weights[token_rows, topk_positions, None]
            output.index_add_(0, token_rows, (expert_output * weights).to(output.dtype))
        return output


class Qwen3_5SharedExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size, intermediate_size])
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Qwen3_5MoEMLP(nn.Module):
    """Route packed token rows [T, H] and produce the same shape."""

    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen3_5RoutedExperts(config.num_experts, config.hidden_size, config.moe_intermediate_size)
        self.shared_expert = None
        if config.shared_expert_intermediate_size:
            self.shared_expert = Qwen3_5SharedExpert(config.hidden_size, config.shared_expert_intermediate_size)
            self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)
        probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        weights, expert_ids = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(router_logits.dtype), expert_ids

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weights, expert_ids = self.route(hidden_states)
        output = self.experts(hidden_states, expert_ids, weights)
        if self.shared_expert is not None:
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            output = output + shared_gate * self.shared_expert(hidden_states)
        return output
