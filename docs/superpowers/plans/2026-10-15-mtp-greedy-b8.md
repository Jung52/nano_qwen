# Greedy fast path and B4/B8 MTP implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline; preserve RED/GREEN and benchmark evidence in the plan ledger.

**Goal:** Remove unnecessary greedy sampling work, then validate production MTP through B8 and measure batch break-even on one GPU.

**Architecture:** Pass an all-greedy boolean derived from CPU Sequence temperatures into Sampler, without reading GPU temperature values. Extend the existing request-major/time-major verify implementation to B1–8. Exact graph banks cover every live/subset count, and ordinary small-batch decode uses the same actual M=B numerical reference.

**Tech stack:** PyTorch, Triton, CUDA Graph, pytest, standalone plotting tools.

**Spec:** User's 2026-10-15 request: greedy fast path → representative interleaved remeasurement → B4/8 full lifecycle → B1/2/4/8 throughput/TPOT curves. No B16 without evidence; no unavailable 5090 claims.

## Constraints and review focus

- Every formal configuration has exactly two samples; warmup/diagnostics are separate.
- Off/reuse completions must match strictly; preserve serial as independent state reference.
- All-greedy skips softmax/RNG; non-greedy/mixed distributions and tie handling remain unchanged.
- CPU temperature routing introduces no GPU synchronization; custom validation samplers follow the same optional hint interface.
- Every accepted subset size 0..B, noncontiguous slots and masks, variable draft sizes, B4→3→2→1 and B8→1, EOS/budgets, completion/reuse/new arrivals.
- M=2B uses local M=B selection for all new shapes; no global cache transplant. Small ordinary decode graphs must have exact B3/5/6/7 rather than padded reference arithmetic.
- Model, GPU, lengths and timing protocol fixed across curve points. Throughput counts real committed tokens, never projected values.
- Preserve user's unrelated README/linear/craft and untracked historical development documents.

### Task 1: Greedy dispatch and representative remeasurement

Files: `layers/sampler.py`, `engine/model_runner.py`, sampler consumers/tests, `benchmarks/bench_mtp_interleaved.py`, retained report.

- [x] Run clean 7dc09b9 baseline tests; compare B2 same64/chinese-code using the unchanged full-sampling reference and fast path in one engine, interleaving off/reuse with two repeats.
- [x] RED: all-greedy returns deterministic argmax without softmax/RNG or device scalar read, and runner chooses using CPU requests; stochastic/mixed behavior retained.
- [x] Implement `Sampler.forward(logits, temperatures, all_greedy=False)` and propagate the optional hint to internal validation samplers.
- [x] GREEN full suite, repeat identical interleaved protocol after change; report off/reuse ratios before/after without mixing old unpaired data.
- [x] Review and commit this independently before B4/B8 changes.

### Task 2: B4/B8 exact shapes and full lifecycle

Files: `config/config.py`, `models/mtp_verify.py`, `engine/{model_runner,cuda_graph}.py`, generalized test utilities and new lifecycle tests.

- [x] RED: public capacity1..8, each M=B reference, exact decode/verify/draft/cache-only banks including3/5/6/7.
- [x] Generalize guards only after tests exist; increase fixture KV blocks to support real B8 contexts.
- [x] For both FP8/eager/Graph, compare serial/reuse hidden/logits/state at every valid prefix, subset sizes0..B, arbitrary slot subsets, dynamic live counts, EOS, budgets, slot reuse/new requests and chunked prefill.
- [x] Confirm actual graph dispatch sizes, dense M=2B, norm/head M=B and inactive state preservation.

### Task 3: Fixed-workload batch curves and final integration

Files: generalized interleaved benchmark, JSON/CSV/Markdown and standalone throughput/TPOT plots, API docs and local development log.

- [x] Real-model B4/B8 strict eager/Graph off/serial/reuse gates, including unequal contexts; investigate any numerical failure before accepted timings.
- [x] Same GPU/models, input64/output128 at B1/2/4/8; off/reuse interleaved, exactly two formal samples each. Preserve any negative result and measured acceptance.
- [x] Plot per-request TPOT and aggregate throughput, distinguish supported points from a proven continuous break-even threshold.
- [x] Full pytest/lint and fresh review, fix material findings and retain reports. Final combined suite:619 passed,32 skipped; B16 guarded and5090 unmeasured.
- Integration target: two separate commits, fast-forward and push existing `feat/qwen35-mtp2`; record remote hash in local execution ledger.

Evidence: first greedy commit `4c83dac`; real4B gates24/24 strict and curve samples32/32 strict. Per-channel B1/2 +10.7/+10.4%, B4/8 −3.3/−6.2%; block-wise +0.8/+3.9/−1.1/+1.3%, no stable monotonic threshold. B16 deferred; new B2/B4 trace still needed before the next kernel change.
