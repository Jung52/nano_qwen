"""CPU-side vocabulary row plan shared by eager and CUDA Graph paths."""
from dataclasses import dataclass
from .sequence import Sequence


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    row_indices: tuple[int, ...]
    request_indices: tuple[int, ...]

    @classmethod
    def for_requests(cls, seqs: list[Sequence]):
        rows, requests, offset = [], [], 0
        for i, seq in enumerate(seqs):
            count = seq.num_scheduled_tokens
            if not seq.is_prefill or seq.num_cached_tokens+count == seq.num_tokens:
                rows.append(offset+count-1)
                requests.append(i)
            offset += count
        return cls(tuple(rows), tuple(requests))
