from collections import deque

from nano_qwen.config import Config
from nano_qwen.engine.sequence import Sequence, SequenceStatus
from nano_qwen.engine.block_manager import BlockManager


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

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
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
                    if not scheduled_seqs and not self.running:
                        raise RuntimeError(
                            "KV cache exhausted before admission: "
                            f"need {seq.num_blocks} blocks, "
                            f"have {len(self.block_manager.free_block_ids)} free"
                        )
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            seq.is_prefill = True
            num_batched_tokens += seq.num_scheduled_tokens
            self.waiting.popleft()
            self.in_flight.add(seq.seq_id)
            scheduled_seqs.append(seq)

        # Fill any remaining token budget with decode rows. This produces a
        # mixed prefill+decode batch when a prefill request leaves budget
        # available, and remains a pure decode batch when no prefill was
        # scheduled.
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break
            seq = self.running.popleft()
            committed_completions = seq.num_tokens - seq.num_prompt_tokens
            if (
                self.enable_mtp
                and seq.draft_token is not None
                and seq.temperature <= 1e-6
                and committed_completions + 2 <= seq.max_tokens
                and remaining >= 2
            ):
                seq.append_token(seq.draft_token)
                seq.is_speculative = True
                # Two rows are verified by the target and the accepted-row
                # sampler immediately writes one MTP KV row beyond them.
                if self.block_manager.ensure_capacity(seq, 3):
                    seq.num_scheduled_tokens = 2
                    seq.is_prefill = True
                    self.in_flight.add(seq.seq_id)
                    scheduled_seqs.append(seq)
                    num_batched_tokens += 2
                    continue
                # Rare KV-pressure fallback: discard the draft and replay the
                # request as a normal prefill after preemption.
                seq.token_ids.pop()
                seq.num_tokens -= 1
                seq.last_token = seq.token_ids[-1]
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

        any_prefill = any(seq.is_prefill for seq in scheduled_seqs)
        scheduled_ids = [seq.seq_id for seq in scheduled_seqs]
        assert len(set(scheduled_ids)) == len(scheduled_ids), (
            "scheduler produced a duplicate request in one batch"
        )
        return scheduled_seqs, any_prefill

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.in_flight.discard(seq.seq_id)
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[list[int]],
        is_prefill: bool,
    ):
        for seq, token_id in zip(seqs, token_ids):
            self.in_flight.discard(seq.seq_id)  # sample consumed, seq schedulable again
            next_draft = seq.draft_token
            cached_before = seq.num_cached_tokens
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                self.waiting.append(seq)  # chunked prefill: back to waiting for the next chunk
                continue

            outputs = token_id if isinstance(token_id, list) else [token_id]
            was_speculative = seq.is_speculative
            seq.is_speculative = False
            if was_speculative:
                # The draft was appended before execution. On rejection the
                # target token overwrites that slot; on acceptance the second
                # target output is the only newly appended token.
                if len(outputs) == 2:
                    seq.append_token(outputs[1])
                else:
                    seq.replace_last_token(outputs[0])
                    # The rejected draft was executed but never committed.
                    # Keep its KV slot scheduled again with the corrected
                    # target token.
                    seq.num_cached_tokens = cached_before + 1
            else:
                seq.append_token(outputs[0])

            hit_eos = not seq.ignore_eos and any(token == self.eos for token in outputs)
            if hit_eos or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
            else:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)  # back to schedulable
                seq.draft_token = next_draft
            if self.enable_prefix_cache:
                self.block_manager.hash_blocks(seq)
