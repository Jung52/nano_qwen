import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile(dynamic=True)
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, all_greedy: bool = False):
        # The runner derives this hint from CPU request metadata. Branching on
        # a Python bool avoids temperature tensor reads or device synchronization.
        if all_greedy:
            return logits.argmax(dim=-1)
        # MTP verifies greedy proposals with argmax. Tiny temperature alone
        # still samples randomly among tied BF16 logits; keep one tie rule.
        greedy_tokens = logits.argmax(dim=-1)
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(temperatures <= 1e-6, greedy_tokens, sample_tokens)
