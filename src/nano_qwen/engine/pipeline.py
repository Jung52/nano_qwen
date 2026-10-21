"""Bounded, commit-before-sample decode pipeline; no CUDA dependencies here.

Execution slots (0/1) own temporary work. Request slots, KV and GDN state
remain leased until every referencing step has committed. Prefill is a
draining boundary in this first implementation, including partial prompts.
"""
from collections import deque
from copy import copy
from dataclasses import dataclass


@dataclass(slots=True)
class RequestLease:
    seq: object
    frontier: int
    inflight_refs: int = 0
    pending_samples: int = 0


@dataclass(frozen=True, slots=True)
class StepRow:
    lease: RequestLease
    view: object
    start: int
    count: int
    produces_token: bool


@dataclass(frozen=True, slots=True)
class StepPlan:
    rows: tuple[StepRow, ...]
    is_prefill: bool


@dataclass(slots=True)
class PendingStep:
    slot_id: int
    plan: StepPlan
    output: object


class DualSlotPipeline:
    """Drive the existing scheduler's queues and block pool in opt-in mode.

    Default Scheduler.schedule/postprocess are deliberately not called: their
    exclusive in_flight membership is the depth-1 scheduling contract.
    """

    def __init__(self, scheduler, runner, config):
        self.scheduler, self.runner, self.config = scheduler, runner, config
        self.requests: dict[int, RequestLease] = {}
        self.pending: deque[PendingStep] = deque()
        self._prefill_turn = True
        self._poisoned = False
        self.stats = dict(decode_forwards=0, decode_rows=0, zombie_rows=0)

    def _wants_prefill(self):
        s = self.scheduler
        admitted_prompt = any(seq.seq_id in self.requests for seq in s.waiting)
        has_capacity = len(self.requests) < s.max_num_seqs
        return bool(s.waiting) and (admitted_prompt or has_capacity) and (
            self._prefill_turn or not s.running
        )

    def _row(self, lease, count, prefill):
        seq, start = lease.seq, lease.frontier
        # No launch/commit ever mutates this private execution view. In
        # particular block_table[-1] and scheduled counts cannot drift when
        # a subsequent step allocates the next KV page.
        view = copy(seq)
        view.token_ids = list(seq.token_ids)
        view.block_table = list(seq.block_table)
        view.num_cached_tokens = start
        view.num_scheduled_tokens = count
        view.is_prefill = prefill
        if not prefill:
            view.num_tokens = start + 1  # GPU supplies the uncommitted token.
        produces = not prefill or start + count == seq.num_tokens
        lease.frontier += count
        lease.inflight_refs += 1
        lease.pending_samples += int(produces)
        assert 0 < lease.inflight_refs <= 2
        self.scheduler.in_flight.add(seq.seq_id)
        return StepRow(lease, view, start, count, produces)

    def _plan_prefill(self):
        s, rows, budget = self.scheduler, [], self.scheduler.max_num_batched_tokens
        for _ in range(len(s.waiting)):
            if not budget or len(rows) == s.max_num_seqs:
                break
            seq = s.waiting.popleft()
            lease = self.requests.get(seq.seq_id)
            if lease is None:
                if len(self.requests) == s.max_num_seqs:
                    s.waiting.append(seq)
                    continue
                if len(seq) + seq.max_tokens - 1 > self.config.max_model_len:
                    raise ValueError("dual-slot request exceeds max_model_len")
                cached = s.block_manager.can_allocate(seq)
                if cached == -1:
                    s.waiting.appendleft(seq)
                    break
                # Prefix reuse is rejected by Config in this branch.
                s.block_manager.allocate(seq, 0)
                lease = RequestLease(seq, seq.num_cached_tokens)
                self.requests[seq.seq_id] = lease
            assert lease.inflight_refs == 0
            count = min(seq.num_tokens - lease.frontier, budget)
            rows.append(self._row(lease, count, True))
            budget -= count
        return StepPlan(tuple(rows), True) if rows else None

    def _plan_decode(self):
        s, rows = self.scheduler, []
        budget = min(s.max_num_seqs, s.max_num_batched_tokens)
        for _ in range(len(s.running)):
            if len(rows) == budget:
                break
            seq = s.running.popleft()
            s.running.append(seq)
            if seq.is_finished:
                continue
            lease = self.requests[seq.seq_id]
            if seq.num_completion_tokens + lease.pending_samples >= seq.max_tokens:
                continue
            length = lease.frontier + 1
            if length > self.config.max_model_len:
                raise RuntimeError("dual-slot compute frontier exceeds model limit")
            if not s.block_manager.reserve_tokens(seq, length):
                continue  # drain pending results before declaring capacity failure
            rows.append(self._row(lease, 1, False))
        return StepPlan(tuple(rows), False) if rows else None

    def _retire(self, lease):
        seq = lease.seq
        if lease.inflight_refs == 0:
            self.scheduler.in_flight.discard(seq.seq_id)
            if seq.is_finished:
                self.scheduler.block_manager.deallocate(seq)
                self.runner.call("remove_request", seq.seq_id)
                del self.requests[seq.seq_id]

    def _commit_oldest(self):
        step = self.pending[0]  # keep slot reserved throughout get_output/commit
        tokens = step.output.get_output()
        if len(tokens) != len(step.plan.rows):
            raise RuntimeError("dual-slot output row count mismatch")
        finished, work = [], 0
        for row, token in zip(step.plan.rows, tokens):
            lease, seq = row.lease, row.lease.seq
            if seq.is_finished:
                self.stats["zombie_rows"] += 1
            else:
                if seq.num_cached_tokens != row.start:
                    raise RuntimeError("out-of-order dual-slot commit")
                seq.num_cached_tokens = row.start + row.count
                work += row.count if step.plan.is_prefill else -1
                if row.produces_token:
                    if token < 0:
                        raise RuntimeError("missing sample for a live request")
                    seq.append_token(token)
                    if ((not seq.ignore_eos and token == self.config.eos)
                            or seq.num_completion_tokens >= seq.max_tokens):
                        # Use the existing enum, without importing CUDA-facing
                        # package initialization into this pure state machine.
                        seq.status = type(seq.status).FINISHED
                        self.scheduler.running = deque(
                            s for s in self.scheduler.running if s.seq_id != seq.seq_id)
                        finished.append((seq.seq_id, seq.completion_token_ids))
                    else:
                        seq.status = type(seq.status).RUNNING
                        if step.plan.is_prefill:
                            self.scheduler.running.append(seq)
                else:
                    self.scheduler.waiting.append(seq)
            lease.inflight_refs -= 1
            lease.pending_samples -= int(row.produces_token)
            assert lease.inflight_refs >= 0 and lease.pending_samples >= 0
            self._retire(lease)
        self.runner.call("release_slot", step.slot_id)
        self.pending.popleft()
        self._prefill_turn = not step.plan.is_prefill
        return finished, work

    def step(self):
        if self._poisoned:
            raise RuntimeError("dual-slot pipeline failed; close this engine")
        try:
            wants_prefill = self._wants_prefill()
            if wants_prefill and self.pending:
                return self._commit_oldest()  # explicit prefill barrier
            plan = self._plan_prefill() if wants_prefill else self._plan_decode()
            if plan is None and wants_prefill and self.scheduler.running:
                # A new prompt can need more free pages than an already
                # admitted decode. Keep the latter progressing to retirement.
                plan = self._plan_decode()
            if plan is None:
                if self.pending:
                    return self._commit_oldest()
                if self.scheduler.waiting or self.scheduler.running:
                    raise RuntimeError(
                        "dual-slot KV capacity exhausted; reduce max_num_seqs "
                        "or request length (preemption is not enabled)")
                return [], 0
            used = {step.slot_id for step in self.pending}
            slot = next(i for i in range(2) if i not in used)
            views = [row.view for row in plan.rows]
            self.runner.call("launch_slot", views, plan.is_prefill, slot)
            if not plan.is_prefill:
                self.stats["decode_forwards"] += 1
                self.stats["decode_rows"] += len(views)
            # t+1 is already on the shared compute stream while the CPU
            # waits for and commits t. Finalize sampling only afterwards.
            result = self._commit_oldest() if self.pending else ([], 0)
            live_ids = [row.lease.seq.seq_id for row in plan.rows
                        if not row.lease.seq.is_finished]
            output = self.runner.call("sample_slot", slot, live_ids)
            self.pending.append(PendingStep(slot, plan, output))
            if plan.is_prefill:
                # First version keeps fresh/chunked prefill serialized.
                assert not result[0] and result[1] == 0
                return self._commit_oldest()
            return result
        except BaseException:
            # A failed GPU launch may already have modified KV/GDN. Never
            # roll back/free/reuse those resources and continue execution.
            self._poisoned = True
            raise

    def clear(self):
        # Caller must synchronize the runner first (LLMEngine.exit does so).
        self.pending.clear()
        self.requests.clear()
