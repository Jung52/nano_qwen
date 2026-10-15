# MTP cache lifecycle implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation. Complete each correctness gate before performance measurement.

**Goal:** Repair MTP cache indexing, accepted-row continuity, chunked prefill and request lifecycle before measuring FP8+MTP.

**Architecture:** Keep RoPE positions one-based and map them to zero-based MTP cache rows. Track valid rows separately from target KV, append accepted draft and bonus rows in order, and preserve the previous target hidden across every prefill chunk. Keep current TP=1/B=1 support.

**Tech stack:** PyTorch, FlashAttention, FlashInfer, Triton, pytest.

**Spec:** User's four-step review on 2026-10-15: repair cache boundaries, fill accepted draft rows, test full lifecycle, then evaluate real committed-token performance.

## Global constraints

- Preserve the user's unrelated `linear.py` comments and `craft.py`.
- No projected throughput or numerical-drift flag may qualify a correctness pass.
- Test prompts 127/128/129, 255/256/257, 511/512/513.
- Compare valid cache prefixes, not uninitialized rows or states at different consumed prefixes.
- Keep FP8 target and MTP weight loading independent; do not relax multi-request guards without implementing and testing batching.

## Review focus

- One-token first chunk must preserve its target hidden even when no MTP row is emitted.
- Accepted draft EOS must keep GDN and logical target KV at the authoritative prefix.
- A two-row verify fitting the last block must not reserve a phantom third row.
- Finished and preempted slots must not inherit old MTP valid lengths or hidden metadata.
- Graph replay must update physical slots independently from RoPE positions.

## Task 1: Cache metadata and capacity

Files: `engine/model_runner.py`, `engine/cuda_graph.py`, `scheduler/scheduler.py`, `tests/test_mtp_cache_lifecycle.py`.

- [x] Add failing tests using real runner methods and a metadata recorder: full prompt boundaries, chunk accumulation, first chunk length one, decode row mapping, exact-block capacity.
- [x] Run `python -m pytest -q tests/test_mtp_cache_lifecycle.py` and preserve the failures.
- [x] Map RoPE p to row p-1, record cumulative valid lengths, preserve last hidden before early returns, and reserve two verify rows.
- [x] Run the metadata tests and existing regressions.

## Task 2: Acceptance, EOS and reuse

Files: `engine/model_runner.py`, `tests/test_mtp_cache_lifecycle.py`.

- [x] Add failing real-sampler tests for accepted/rejected rows, consecutive and alternating decisions, EOS, final output budget and slot reuse.
- [x] Append the accepted draft pair before the bonus pair; roll back an uncommitted EOS verify row; clear per-slot MTP metadata on removal.
- [x] Run the lifecycle tests and full pytest suite.

## Task 3: Real kernel lifecycle validation

Files: `tests/test_mtp_cache_integration.py`, reusable test fixtures under `tests/`.

- [x] Compare eager/graph, full/chunked prefill and forced acceptance patterns on a small real hybrid checkpoint.
- [x] Compare completion and GDN/target/MTP valid state against canonical consumed prefixes, including EOS, remaining 1/2 outputs and slot reuse.
- [x] Run strict local 4B baseline/MTP checks; report any failure without accepting a drift flag.

## Task 4: Independent weights and gated FP8 performance

Files: `config/config.py`, `engine/model_runner.py`, `models/qwen3_5_mtp.py`, `benchmarks/bench_qwen3_5_mtp_production.py`, checkpoint tests and retained report.

- [x] Add an explicit MTP model path and validate its architecture against the target before loading.
- [x] Test loading original BF16 MTP weights with both converted FP8 targets.
- [x] Recheck FP8 with MTP disabled, then strict B1 eager/graph correctness.
- [x] Measure two formal runs only after correctness passes; use committed completion tokens and retain diagnostics separately.
- [x] Review the final diff and run the full suite before integration.

## Implementation findings

- Added a real preemption replay reference: an existing slot replaying from zero must reset GDN as well as MTP metadata. Both eager and graph failed before the reset fix and pass afterward.
- Shared effective RoPE normalization between model construction and independent-source validation; covered top-level theta/partial fallback and hybrid GDN architecture mismatches.
- Real 4B strict comparison exposed B1/B2 arithmetic differences independently of cache correctness. Verification now captures two sequential ordinary B1 target forwards and uses per-row LM head projection. This preserves the baseline's arithmetic, at a measured throughput cost.
- Two formal production runs for each FP8 variant used actual127 decode tokens after prefill, with all128 completion tokens equal to baseline. No projected result or drift override was accepted.
- The old path_audit wrapper could not accept the runner's fourth is_verify argument; the off-path control aborted before recording timing. Fixed transparent argument forwarding and added three invocation-form tests. Both archived8322 and current off controls use this identical harness patch; archived production src stays unchanged.
- B4/B8, TP>1 and stochastic sampling remain unsupported behind existing guards. They require a separate batching implementation rather than removing the guard.
