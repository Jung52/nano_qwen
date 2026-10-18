"""CPU contracts for vocabulary projection and authoritative state commits."""

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
            if seq.is_speculative:
                rows.extend(range(offset, offset+count))
                requests.extend([i]*count)
            elif not seq.is_prefill or seq.num_cached_tokens+count == seq.num_tokens:
                rows.append(offset+count-1)
                requests.append(i)
            offset += count
        return cls(tuple(rows), tuple(requests))


@dataclass(frozen=True, slots=True)
class OutputSpec:
    num_committed: int
    accepted_count: int
    cache_advance: int


@dataclass(frozen=True, slots=True)
class RequestOutput:
    committed_tokens: tuple[int, ...]
    accepted_count: int
    next_draft: int | None
    cache_advance: int

    def __post_init__(self):
        if self.cache_advance < 0 or not 0 <= self.accepted_count <= len(self.committed_tokens) <= 2:
            raise ValueError("Invalid execution commit metadata")
