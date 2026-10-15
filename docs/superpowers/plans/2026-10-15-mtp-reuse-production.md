# Production MTP verify reuse implementation plan

> Execute inline with superpowers:executing-plans. The user's supplied design
> defines the scope; no change to the serial default or B1/TP1 boundary.

**Goal:** Enable `mtp_verify_mode="reuse"` through formal APIs and reproduce
the prior B1 FP8 result with strict token and state gates.

**Architecture:** Share GDN dense/decode-core/output interfaces between ordinary
decode and reuse; retain two rowwise norm reductions, sequential state updates
and single-query attention. A call-local Context policy selects the current
device's M1 GEMM tile without overwriting the M2 autotuner entry. Both eager and
Graph call the same reuse implementation. Serial remains the reference.

**Tech stack:** Python, PyTorch/CUDA, Triton, existing FP8 and paged KV/GDN pools.
**Spec:** User request of 2026-10-15; prior experimental evidence is recorded in
`benchmarks/reports/mtp_verify_reuse/results.md`.

## Constraints and review focus

- Reject unknown mode, reuse without MTP/FP8, TP>1 and B>1. Default is serial.
- Keep hidden/residual, logits, GDN final/snapshot and effective KV strict.
- M1 selection is keyed by device, shape and dtypes; never transplant configs
  between cards. Cold eager warms M1; cold Graph must not tune or silently pick
  a different reference tile. Existing M2 entries remain unchanged.
- Preserve arithmetic for unquantized projections in a mixed FP8 checkpoint
  by applying them rowwise during reuse. FP8 dense/MLP GEMMs retain two rows.
- Preserve root README/development records, linear.py comments and craft.py.
- This commit covers API integration and the original input64/output128 test.
  The length/content matrix, new trace, B2/4/8 and 5090 are subsequent steps.

## Task 1: Formal execution path and strict gates

Files: `config/config.py`, `utils/context.py`, `quantization/triton_gemm.py`,
`quantization/unquantized.py`, `layers/gated_delta_net.py`,
`models/qwen3_5.py`, new `models/mtp_verify.py`, `engine/model_runner.py`,
`engine/cuda_graph.py`, and existing MTP/FP8 tests.

- [x] Add failing tests for mode validation; M2 execution using the M1 tile
  without cache mutation; cold eager/capture; and production serial/reuse
  equality under eager/Graph and forced lifecycle decisions.
- [x] Implement `forward_decode_core(dense, state_indices=None)` returning
  the GDN pre-output-projection tensor; keep ordinary decode using this core.
  Expose the equivalent attention core without output projection.
- [x] Implement `forward_verify_serial` and the selected production reuse
  dispatch using the same core APIs, rowwise norms, first-row snapshot and
  Context restoration. No runtime patching or autotuner-cache rewriting.
- [x] Run targeted RED/GREEN and the whole test suite. Retain first-difference
  diagnostics through the production reuse API rather than the prototype.

## Task 2: Production benchmark and evidence

Files: `bench_qwen3_5_mtp_production.py`, `bench_mtp_verify_reuse.py`, benchmark
docs/reports; append local `DEVELOPMENT_LOG.md` after integration.

- [x] Add the explicit CLI mode and optional existing-engine timing interface;
  remove experiment projection/method/cache patches from the retained harness.
- [x] Run real 4B layer/state gates and complete eager/Graph token gates for
  both FP8 models. Reuse the authoritative serial trajectory for diagnostics.
- [x] Run off/serial/reuse, input64/output128, each exactly two formal samples,
  no profiler; count 127 actual decode tokens after prefill's first token.
- [x] Review code and data; commit API, tests and report; integrate into mtp2.
  Report measured gains even if they fall short of the previous +8.8%.

## Evidence ledger

- Start: `1d5f04a`; isolated workspace `/home/wei/code/nano_qwen-mtp-cache`.
- Baseline: 286 pass. Configuration/dispatch RED: 15 fail (new APIs absent);
  selected execution path RED: four fail (two M1 calls instead of M2).
- Mixed unquantized projection RED: one fail; graph-mode guard RED: one fail.
  Targeted GREEN: 64 pass. Full suite: 318 pass in 42.60s.
- Logs: root `logs/mtp-reuse-production/`; real PC eager/Graph first-window
  layer/state diagnostics each pass 522 comparisons through production APIs.
- Final review: production execution accepted; benchmark borrowing needed
  idle/source/capacity checks and request-specific exception cleanup. Fixed
  with nine CPU ownership regressions RED (9 fail) to GREEN (9 pass).
- Review stats finding treated as measurement correctness: return per-call
  counter deltas, retaining caller cumulative totals instead of clearing them.
  The same ownership regression verifies repeat-call acceptance statistics.
- Real FP8 serial trajectories: PC66 and BW70 windows, 522 strict comparisons
  per window; first-window eager and Graph both pass for each model.
- Expanded production-mode lifecycle gates: both FP8 schemes and eager/Graph,
  all nine prompt block boundaries, accepted/rejected/bonus EOS and remaining
  output budgets1/2/3. Full suite 383 pass in 51.42s.
- Two formal samples per mode: PC off83.39/serial69.84/reuse94.06 tok/s;
  BW off83.26/serial65.92/reuse88.31. Reuse/off +12.8%/+6.1% for this one
  workload. All full128-token gates exact. Reports and source hashes retained.
