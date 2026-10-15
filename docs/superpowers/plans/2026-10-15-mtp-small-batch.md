# Small-batch MTP implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. Check off evidence as work completes.

**Goal:** Establish B1 workload/trace baselines, then support independently accepted/rejected B2 requests in production reuse.

**Architecture:** Keep external verify rows request-major; execute two time steps of B rows internally. Dense projections share 2B rows using ordinary M=B GEMM configurations, while norms, recurrent updates, attention and vocabulary projection preserve B-row arithmetic. Capture exact B1/B2 graphs, including MTP subsets, without padded state writes. Schedule homogeneous verification batches.

**Tech stack:** Python, PyTorch, Triton, CUDA Graph, pytest.

**Spec:** User's 2026-10-15 request (B1 matrix/trace before B2; AA/AR/RA/RR, unequal contexts, EOS, completion and slot reuse).

## Global constraints

- FP8 TP1 greedy decoding; serial remains the correctness reference.
- B1 inputs 64/256/1024 × outputs 128/512, plus Chinese/code prompts; off/reuse, two formal repeats each, no profiler during timing.
- Strict completion equality; failures retained, no numerical-drift waiver.
- B2 first; B4/8 and unavailable 5090 are subsequent work.
- Preserve existing ordinary decode B64 baseline and unrelated dirty files.

## Review focus

- Mixed verify/ordinary/prefill dispatch must never bypass rollback snapshots.
- Per-request KV lengths are differences of cumulative lengths, not their sums.
- One request's rejection/EOS cannot restore another request's slot.
- Exact-size graphs handle B2→B1 transitions and MTP accepted subsets.
- Reference M=B math applies to norms and BF16 exclusions as well as FP8 tiles.

### Task 1: B1 baseline and reuse trace

Files: `benchmarks/bench_mtp_small_batch.py`, `tests/test_mtp_small_batch.py`, retained `benchmarks/reports/mtp_small_batch/`.

- [x] Add strict comparison/length-case tests; observe failures, implement harness around production `run_mode`.
- [x] Run matrix and representative content on both FP8 models, two samples/config; retain completions and all metrics.
- [x] Profile short warmed reuse decode separately; attribute non-overlapping GPU work and synchronization/idle; decide whether a local optimization is justified.

### Task 2: B2 request/time mapping and selective state

Files: `models/mtp_verify.py`, `quantization/{triton_gemm,unquantized}.py`, `engine/{model_runner,cuda_graph}.py`, `scheduler/scheduler.py`, `config/config.py`, B2 integration tests.

Interfaces: request-major `[A0,A1,B0,B1]` in/out; time groups `[A0,B0]`, `[A1,B1]` internally; `gemm_reference_m=B`. Independent snapshot slots and cache lengths; verify and MTP graphs indexed by actual B=1/2, no padding.

- [x] Add tests for four accept masks and exact serial/reuse dense/hidden/logits/GDN/KV equality in eager/Graph; observe current B1 guard failure.
- [x] Generalize dispatch and B-row norm/head execution; implement homogeneous scheduling, per-request lengths, exact graph banks and subset MTP execution.
- [x] Verify unequal contexts, EOS/remaining budgets, B2→B1 completion and slot reuse; preserve B1 gates.

### Task 3: Real-model acceptance and documentation

Files: benchmark/report, `docs/mtp-verify-reuse.md`, development log, tests.

- [x] Strict B2 real-model eager/Graph completion check against ordinary decode, then two timing samples per mode/model when correct.
- [x] Run full pytest and lint, fresh code review, fix material findings.
- [x] Record limitations and results, commit focused changes and update the existing branch without touching unrelated files.

## Final evidence

- B1: 64 formal samples after deterministic greedy tie fix, strict completions; historical failed matrix retained.
- Fresh reuse eager/Graph trace: target verify ~63–64%, all LM head ~32–33% of graph GPU kernel sum; snapshot event span ~1–2%, no snapshot optimization before B2.
- B2: both FP8 models, 6 full128 eager gates and24 two-repeat Graph samples; strict off/serial/reuse and eager/Graph equality on unequal contexts.
- Full451 tests passed63.77s; scoped Ruff and git diff --check passed. Review found KV-release/continuation boundary issues, both RED→GREEN and independently rechecked.
- Ordinary nonstandard decode capacities3/5/17 covered without changing B1/B2/B64 capture sizes. B4/8 and5090 remain guarded/unmeasured.
