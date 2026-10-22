"""One-batch lookahead for a stable, already-prefilled decode cohort.

Only execution metadata is advanced optimistically. Committed Sequence token
history changes only when a result is consumed. Unknown next tokens stay on
the GPU; these views must never enter CPU prefill or prefix-cache hashing.
"""
from collections import deque
from copy import copy

from nano_qwen.engine.sequence import Sequence, SequenceStatus


class AsyncDecodeCohort:
    def __init__(self, scheduler, seqs: list[Sequence]):
        self.scheduler = scheduler
        self.live = {seq.seq_id: seq for seq in seqs}
        self.pending = {seq.seq_id: 0 for seq in seqs}
        self.dispatched: deque[list[Sequence]] = deque()
        self.draining = False

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.draining:
            return [], False
        views = []
        for seq in self.live.values():
            ahead = self.pending[seq.seq_id]
            if seq.is_finished or seq.num_completion_tokens + ahead >= seq.max_tokens:
                continue
            if ahead >= 2:
                raise RuntimeError("async decode supports at most two outstanding batches")
            view = copy(seq)
            view.num_tokens = seq.num_tokens + ahead
            view.num_cached_tokens = view.num_tokens - 1
            view.num_scheduled_tokens = 1
            view.is_prefill = False
            view.status = SequenceStatus.RUNNING
            view._async_decode_view = True
            # This contains committed history only. Decode reads its input ID
            # from sampled_token_ids_gpu, never from this CPU history.
            view.token_ids = seq.token_ids.copy()
            views.append(view)

        manager = self.scheduler.block_manager
        # Reserve the whole dispatch before changing block ownership. Do not
        # preempt a request while any GPU work can still reference its blocks.
        needed = sum(max(0, view.num_blocks - len(view.block_table)) for view in views)
        if not views or needed > len(manager.free_block_ids):
            self.draining = True
            return [], False
        for view in views:
            seq = self.live[view.seq_id]
            if len(seq.block_table) < view.num_blocks:
                assert len(seq.block_table) + 1 == view.num_blocks
                assert len(view) % self.scheduler.block_size == 1
                manager.may_append(view)  # initially shares the live block list
            # Freeze each dispatch's table; future reservations may append to
            # the owner's table while this dispatch remains queued.
            view.block_table = seq.block_table.copy()
            self.pending[view.seq_id] += 1
        self.dispatched.append(views)
        return views, False

    def postprocess(self, views, token_ids, is_prefill):
        if is_prefill or not self.dispatched or self.dispatched[0] is not views:
            raise RuntimeError("async decode results must be committed in dispatch order")
        if len(views) != len(token_ids):
            raise RuntimeError("sample result row count does not match the dispatch")
        self.dispatched.popleft()
        finished, released = [], []
        for view, token_id in zip(views, token_ids):
            seq = self.live[view.seq_id]
            self.pending[seq.seq_id] -= 1
            if not seq.is_finished:
                assert seq.num_cached_tokens == view.num_cached_tokens
                seq.num_cached_tokens += 1
                seq.num_scheduled_tokens = 0
                seq.is_prefill = False
                seq.append_token(token_id)
                if ((not seq.ignore_eos and token_id == self.scheduler.eos)
                        or seq.num_completion_tokens >= seq.max_tokens):
                    seq.status = SequenceStatus.FINISHED
                    finished.append(seq)
            # An EOS can leave one already-enqueued decode behind. Its token
            # is discarded above, but KV/GDN slots stay owned until it retires.
            if seq.is_finished and self.pending[seq.seq_id] == 0:
                self.scheduler.block_manager.deallocate(seq)
                self.scheduler.in_flight.discard(seq.seq_id)
                released.append(seq.seq_id)
        return finished, released

    def close(self):
        if self.dispatched or any(self.pending.values()):
            raise RuntimeError("cannot release an async decode cohort before draining")
        for seq in self.live.values():
            self.scheduler.in_flight.discard(seq.seq_id)
            if not seq.is_finished:
                seq.status = SequenceStatus.RUNNING
                seq.is_prefill = False
                seq.num_scheduled_tokens = 0
                self.scheduler.running.append(seq)
