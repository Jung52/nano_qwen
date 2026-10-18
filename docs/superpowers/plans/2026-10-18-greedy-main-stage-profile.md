# Standalone greedy dispatch and B2/B4 MTP cost attribution

> Execute inline using superpowers:executing-plans; one fresh review before integration.

**Goal:** Put deterministic CPU-routed greedy sampling on main without MTP dependencies, and explain per-channel B2→B4 behavior using new off/reuse traces on mtp2.

**Design:** Port only sampler, runner hint and validation sampler signatures to main. Keep mixed sampling and its RNG algorithm; document the deterministic tiny-temperature tie rule. Extend the diagnostic profiler with explicit graph-aware timing for target body/head, draft body/head, cache-only, snapshots/rollback and CPU decision transfer. Preserve actual submitted token counts, per-step active request/slot mappings, and physical GPU activity union/gaps without double counting. No kernel optimization, B16 expansion, 5090 or9B claims in this change.

**Spec:** User's 2026-10-18 main greedy → mtp2 B2/B4 profiling request.

**Isolation:** main worktree `/home/wei/code/nano_qwen-main-greedy`; mtp2 profiling worktree `/home/wei/code/nano_qwen-mtp-cache` on `perf/mtp-batch-stage-profile`. User's primary checkout and untracked development records remain intact.

## Task1 — main standalone greedy

- [x] Clean baseline; RED deterministic ties, all-greedy no RNG/softmax/device scalar, CPU runner routing and mixed sampling.
- [x] Port sampler and CPU hint without any MTP code or weights; adapt benchmark validation samplers.
- [x] Complete main suite; real-model MTP-free controlled reference/fast measurements, two samples/config, full-completion equality to deterministic reference; record tied-token behavior explicitly.
- [x] Retain report and commit independently on main.

## Task2 — mtp2 B2/B4 profile

- [x] Regression for disjoint phase attribution, GPU interval union/gaps, graph-bank event selection and active batch/committed token bookkeeping.
- [x] Generalize existing profiler to B2/B4 and off/reuse; benchmark-only graph recapture instruments all actual sizes without affecting production execution semantics.
- [x] Same input64/output128, per-channel production Graph B2/B4 traces; both FP8/eager/Graph tiny-model state gates. Record accepted/rejected rounds and dynamic active batch; separate profiler-free timing, two samples/config. Eager/other-FP8 CLI retained without claiming a full real-model matrix.
- [x] Report target verify vs head; draft vs draft head; cache-only; snapshot/rollback; decision-table wait and overlapping GPU work; actual submitted tokens and per-token costs. Keep instrumentation effects visible.
- [x] Explain acceptance ≈84% vs B2/B4 gain using measured stage amortization; select one next optimization direction, but do not implement it in this commit.
- [x] Full suite/lint and fresh review: main102passed, mtp2627passed32skipped, all changed files Ruff/diffclean. Main e936f58 pushed. Integrate the profiling commit and update the local development log as the final delivery step.

Decision: select only batch-slot rollback for the next implementation. Current fresh formal B2/B4 −2.6%/−5.8%; do not claim stable old B2 positive gain. Event spans, physical GPU sums and host scopes have distinct units/overlap. The independent host probe has different tuning choices and is not a third formal sample. Main independent gains0.10..2.31%, not the old MTP runner's36..39%.
