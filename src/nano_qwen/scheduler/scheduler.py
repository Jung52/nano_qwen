from collections import deque

from nano_qwen.config import Config
from nano_qwen.engine.sequence import Sequence, SequenceStatus
from nano_qwen.engine.block_manager import BlockManager
from nano_qwen.engine.execution import RequestOutput


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.enable_prefix_cache = config.enable_prefix_cache
        self.enable_mtp = config.enable_mtp
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # MRV2 zombie equivalent: seqs dispatched to execute_model but not yet
        # consumed by postprocess. They are moved OUT of waiting/running while
        # in flight, so schedule() can never re-dispatch them (their next
        # decode step depends on the in-flight sample).
        self.in_flight: set[int] = set()

    def is_finished(self):
        return not self.waiting and not self.running and not self.in_flight

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
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

        # Reserve token budget for ready decode before admitting prompt work.
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # Verification owns a two-time-step state/snapshot layout. Keep
            # it separate from prompt ingestion and ordinary final rows.
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break
            seq = self.running[0]
            committed_completions = seq.num_tokens - seq.num_prompt_tokens
            can_verify = (
                self.enable_mtp
                and seq.draft_token is not None
                and seq.temperature <= 1e-6
                and committed_completions + 2 <= seq.max_tokens
                and remaining >= 2
            )
            if self.enable_mtp and scheduled_seqs and can_verify != scheduled_seqs[0].is_speculative:
                break
            self.running.popleft()
            if can_verify:
                seq.is_speculative = True
                # Target verify and shifted MTP catch-up both fit within the
                # same two-row extent; RoPE positions do not add a cache row.
                if self.block_manager.ensure_capacity(seq, 2):
                    seq.num_scheduled_tokens = 2
                    seq.is_prefill = True
                    self.in_flight.add(seq.seq_id)
                    scheduled_seqs.append(seq)
                    num_batched_tokens += 2
                    continue
                # Rare KV-pressure fallback: discard the draft and replay the
                # request as a normal prefill after preemption.
                seq.is_speculative = False
                seq.draft_token = None
                self.preempt(seq)
                continue

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
        decode_owns_batch = self.enable_mtp and bool(scheduled_seqs)
        # prefill
        while (self.waiting and len(scheduled_seqs) < self.max_num_seqs
               and not decode_owns_batch):
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
        seq.is_speculative = False
        seq.draft_token = None
        self.in_flight.discard(seq.seq_id)
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        results: list[RequestOutput],
        is_prefill: bool,
    ):
        if len({s.seq_id for s in seqs}) != len(seqs):
            raise ValueError("Duplicate request in commit batch")
        if len(seqs) != len(results):
            raise ValueError("Execution result count differs from request count")
        # Validate the whole batch before changing authoritative state.
        for seq, result in zip(seqs, results):
            if not isinstance(result, RequestOutput) or seq.seq_id not in self.in_flight:
                raise ValueError("Commit requires an explicit result for an in-flight request")
            tokens = result.committed_tokens
            partial = seq.is_prefill and not seq.is_speculative and (
                seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens)
            if seq.is_speculative:
                expected_advance = len(tokens)
                valid = (len(tokens) in (1, 2) and result.accepted_count in (0, 1)
                         and (result.accepted_count == 1 or len(tokens) == 1)
                         and (result.accepted_count == 0 or tokens[0] == seq.draft_token)
                         and (result.accepted_count == 0 or len(tokens) == 2
                              or (not seq.ignore_eos and tokens[0] == self.eos)))
            else:
                expected_advance = seq.num_scheduled_tokens
                valid = result.accepted_count == 0 and len(tokens) == int(not partial)
            if not valid or result.cache_advance != expected_advance:
                raise ValueError("Invalid committed token count or cache advance")
            if seq.num_completion_tokens + len(tokens) > seq.max_tokens:
                raise ValueError("Commit exceeds remaining output budget")
            if not seq.ignore_eos and self.eos in tokens[:-1]:
                raise ValueError("Commit continues past EOS")
            if result.next_draft is not None and (partial or not self.enable_mtp):
                raise ValueError("Unexpected draft for this execution")

        for seq, result in zip(seqs, results):
            cached_before = seq.num_cached_tokens
            seq.num_cached_tokens += result.cache_advance
            seq.num_scheduled_tokens = 0
            seq.is_speculative = False
            for token in result.committed_tokens:
                seq.append_token(token)
            seq.draft_token = result.next_draft
            self.in_flight.remove(seq.seq_id)
            if self.enable_prefix_cache:
                self.block_manager.hash_blocks(
                    seq, start_token=cached_before, end_token=seq.num_cached_tokens)
            if not result.committed_tokens:
                self.waiting.append(seq)
                continue
            hit_eos = not seq.ignore_eos and self.eos in result.committed_tokens
            if hit_eos or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                seq.draft_token = None
                self.block_manager.deallocate(seq)
            else:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
