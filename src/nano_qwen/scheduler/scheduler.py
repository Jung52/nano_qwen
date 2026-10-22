from collections import deque

from nano_qwen.config import Config
from nano_qwen.engine.sequence import Sequence, SequenceStatus
from nano_qwen.engine.block_manager import BlockManager
from .async_decode import AsyncDecodeCohort


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.enable_prefix_cache = config.enable_prefix_cache
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # In the normal path, dispatched owners leave waiting/running until
        # postprocess. During lookahead, the cohort holds those owners and
        # schedules execution views using GPU-resident token feedback instead.
        self.in_flight: set[int] = set()
        self.async_decode: AsyncDecodeCohort | None = None

    def begin_async_decode(self) -> bool:
        """Borrow the entire ready cohort; admission/mixed batches stay serial."""
        if (self.async_decode is not None or self.waiting or self.in_flight
                or not self.running or self.enable_prefix_cache
                or len(self.running) > min(self.max_num_seqs, self.max_num_batched_tokens)):
            return False
        seqs = list(self.running)
        assert all(not seq.is_finished and seq.num_cached_tokens == len(seq) - 1 for seq in seqs)
        self.running.clear()
        self.in_flight.update(seq.seq_id for seq in seqs)
        self.async_decode = AsyncDecodeCohort(self, seqs)
        return True

    def end_async_decode(self):
        self.async_decode.close()
        self.async_decode = None

    def is_finished(self):
        return not self.waiting and not self.running and not self.in_flight

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.async_decode is not None:
            return self.async_decode.schedule()
        scheduled_seqs = []
        num_batched_tokens = 0
        # A sequence is either awaiting dispatch (waiting/running) or in
        # flight. Overlap would let the engine dispatch the same request
        # twice and corrupt its KV/GDN state slots.
        ready_ids = {
            seq.seq_id for seq in self.waiting
        } | {
            seq.seq_id for seq in self.running
        }
        assert not (self.in_flight & ready_ids), (
            "scheduler state overlap: seq is both in_flight and ready"
        )

        # State slots live across dispatches, partial prefill and preemption.
        # A new request cannot take a slot just because its owner is in flight
        # or waiting for another prompt chunk. Finished owners leave all queues.
        active_ids = self.in_flight | {seq.seq_id for seq in self.running} | {
            seq.seq_id for seq in self.waiting
            if seq.num_cached_tokens or seq.num_completion_tokens
        }

        # Reserve token budget for ready decode before prompt work.
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                self.in_flight.add(seq.seq_id)
                scheduled_seqs.append(seq)
                num_batched_tokens += 1

        # Ordinary decode may share remaining budget with prompt chunks.
        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            if seq.seq_id not in active_ids and len(active_ids) >= self.max_num_seqs:
                # Continue an admitted prompt behind new arrivals, rather than
                # starving all partial prompts behind a request needing a slot.
                admitted = next((s for s in self.waiting if s.seq_id in active_ids), None)
                if admitted is None:
                    break
                self.waiting.remove(admitted)
                self.waiting.appendleft(admitted)
                seq = admitted
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # Always check capacity. can_allocate() doubles as the
                # free-block admission check; skipping it when prefix caching
                # is disabled let allocate() pop from an empty deque once the
                # KV pool was exhausted (benchmark bs=112 crash).
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # A preempted replay may need blocks held by a partial
                    # prompt behind it. Finish that already allocated prompt
                    # before declaring exhaustion or waiting indefinitely.
                    continuation = next((s for s in self.waiting if s.block_table), None)
                    if continuation is not None:
                        self.waiting.remove(continuation)
                        self.waiting.appendleft(continuation)
                        continue
                    if not scheduled_seqs and not self.running and not self.in_flight:
                        raise RuntimeError(
                            "KV cache exhausted before admission: "
                            f"need {seq.num_blocks} blocks, "
                            f"have {len(self.block_manager.free_block_ids)} free"
                        )
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            seq.is_prefill = True
            num_batched_tokens += seq.num_scheduled_tokens
            self.waiting.popleft()
            self.in_flight.add(seq.seq_id)
            active_ids.add(seq.seq_id)
            scheduled_seqs.append(seq)

        any_prefill = any(seq.is_prefill for seq in scheduled_seqs)
        scheduled_ids = [seq.seq_id for seq in scheduled_seqs]
        assert len(set(scheduled_ids)) == len(scheduled_ids), (
            "scheduler produced a duplicate request in one batch"
        )
        return scheduled_seqs, any_prefill

    def preempt(self, seq: Sequence):
        if seq.seq_id in self.in_flight:
            raise ValueError("Cannot preempt an in-flight request before consuming its result")
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.in_flight.discard(seq.seq_id)
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        if self.async_decode is not None:
            return self.async_decode.postprocess(seqs, token_ids, is_prefill)
        for seq, token_id in zip(seqs, token_ids):
            self.in_flight.discard(seq.seq_id)  # sample consumed, seq schedulable again
            if self.enable_prefix_cache:
                self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                self.waiting.append(seq)  # chunked prefill: back to waiting for the next chunk
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
            else:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)  # back to schedulable

