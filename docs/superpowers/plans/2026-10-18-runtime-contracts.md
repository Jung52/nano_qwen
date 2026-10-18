# Execution, commit and scheduling boundaries — implementation plan

> **For agentic workers:** Execute inline with superpowers:executing-plans; use TDD and one final read-only review. Keep changes local; no automatic commit/push.

**Goal:** Make committed tokens, consumed state and selected sampling rows explicit, then evaluate real MTP end-to-end cost.

**Architecture:** Complete worker metadata and reject unvalidated state-restoration modes. Keep full target hidden outputs; carry a CPU sampling plan and named per-request results across runner → async output → scheduler. Draft stays outside authoritative token history; scheduler prioritizes ready decode work and records exact graph dispatch behavior without padding.

**Tech Stack:** Python, PyTorch/CUDA, existing Triton FP8/GDN kernels and CUDA Graphs.

**Spec:** User code review dated2026-10-18; preserved in logs/runtime-contracts/plan.md with scope/rulings.

## Global Constraints

- Preserve existing user README/linear comments, DEVELOPMENT_LOG/craft and local records.
- main receives worker-serialization and unsupported-mode guards only; all main edits remain local.
- TP>1 explicitly rejected until real multi-GPU validation; serialize all worker metadata nonetheless.
- Hybrid prefix cache rejected; pure-attention prefix hashes describe executed authoritative rows before block release.
- No MTP depth/batch expansion, unsafe graph padding, drift waiver, global GEMM-tuner mutation or automatic push.
- Formal timing exactly2 samples/configuration, profiler off; diagnostic cohorts independent.

## Review Focus

- Accepted draft EOS consumes only first state but accepted_count remains1; bonus EOS retains second state.
- Nonfinal chunk produces no sampled token/head, while MTP shifted cache and previous hidden continue correctly.
- Cancelled/preempted/inflight slots never permit new requests to modify live state or authoritative token history.
- All accepted subsets/live batch sizes1..8 keep per-request cache lengths and exact graph mapping.
- Independent-off engine uses default prefill Graph and no MTP weights/cache; resident off is only an ablation.

### Task1: Safety (main and mtp2)

- [x] RED tests worker pickle prefill/decode, hybrid prefix guard, TP guard; verify full metadata through InputBatch.
- [x] Serialize complete fields; disable TP>1 explicitly; hybrid prefix guard in both Config paths.
- [x] RED pure-attention full/chunked prefill prefix registration; hash explicit consumed range before deallocation.
- [x] Targeted safety GREEN; run main full suite.

### Task2: Sampling rows

Files: new engine/execution.py; model_runner.py; cuda_graph.py; CPU sampling-plan tests and real model gates.

- [x] Define SamplingPlan(row_indices, request_indices). Keep the private pending tuple; no unrelated pending-storage migration.
- [x] RED nonfinal chunk no head/no sampler; final tails and mixed decode rows select right hidden rows; full target hidden retained.
- [x] All eager/prefill/decode graph dispatch returns compact target logits through one projection interface. Verify retains two rows/request and ordinary M=B head arithmetic.
- [x] Migrate test-only forced logits to compact rows; compare finite tail/top1 and strict completions/state.

### Task3: Explicit commit

Files: execution.py, async_output.py, sequence.py, scheduler.py, model_runner.py, llm_engine.py and lifecycle tests.

- [x] RequestOutput(committed_tokens, accepted_count, next_draft, cache_advance); CPU OutputSpec carries explicit counts while async transport stays private.
- [x] Scheduler creates two execution rows without appending draft; prepare_prefill appends pending draft only to GPU input plan.
- [x] Update MTP positions from authoritative length L: accepted cache-only L, bonus L+1, rejection L; CPU metadata0/1/2 committed and explicit1/2 cache advance.
- [x] Engine consumes named results; only postprocess appends tokens, updates cache count and next draft. Validate counts/EOS/budgets/inflight state.
- [x] Strict forced accept/reject/alternate/EOS/slot/chunk lifecycle; adapt historical benchmark-only legacy reference explicitly, do not infer acceptance in production.

### Task4: Scheduling and graph telemetry

- [x] RED sustained-arrival long-prefill starvation and engine queue tests. Reserve ready decode budget first; MTP verify separated; keep preemption and admission capacity progress.
- [x] Exact graph hits/eager fallbacks by batch/mode; B63 remains eager at capacity64, never pads into active GDN slots.
- [x] CPU control-flow reproductions include decode wait counters and dynamic64→63→… distribution.

### Task5: Validation and performance

- [x] Full suites main/mtp2; numerical and state gates against serial and frozen pre-change behavior where applicable.
- [x] 4B PC B1/B2 input64/1024 output128. Same-engine resident off/reuse interleaved2/config, unchanged tuner; independent true-off engine with prefill Graph and no MTP allocation2/config.
- [x] Record actual committed tokens, acceptance, TTFT/TPOT/e2e, memory, graph hit rate/batch distribution; preserve negatives.
- [x] Final independent review and data audit; report limitations and remaining decisions, no automatic commit/push.
