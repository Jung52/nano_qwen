# Main ports and paired MTP amortization measurements

User scope: independently commit generic improvements on main, keep all draft/commit/rollback semantics on mtp2; freeze mtp2 production code for one purposeful 5-pair performance experiment and separate B4/B8 traces.

Existing isolated main worktree is explicitly authorized for four local commits; no push requested. Preserve unrelated user files and all existing mtp2 changes. Execute inline with one final fresh read-only review.

1. main safety commit: full worker execution metadata, TP1 restriction, reject hybrid prefix; run main full regression before commit.
2. main sampling-row commit: standalone SamplingPlan; retain model hidden internally; skip nonfinal head/sample, final tails and decode rows only. Keep main scalar token wire and sentinel for nonfinal; no MTP APIs. CPU and actual GPU full/selected projection gates, then full main suite.
3. main scheduling commit: decode-first reservation, remaining prompt chunks; preserve admitted slots/inflight ownership/preemption progress and decode coalescing. CPU continuous-arrival/pressure tests plus GPU mixed lifecycle regression, main full suite.
4. main Graph stats commit: exact graph hits/eager fallback and batch/token histogram only; no padding changes. CPU63/64 dispatch gate and main full suite.
5. Frozen mtp2 benchmark:4B per-channel, true independent off engine with prefill Graph vs reuse engine without prefill Graph. B1 I64/1024 O128/512; B2 I64 O512. User confirmed B4/B8 I64 O128/512 only. Exactly5 pairs/case, pair order alternates off→reuse / reuse→off; rebuild engines sequentially for VRAM/state isolation, fixed untimed warmup and persistent/frozen tuner selections. No profiler during formal timing. Record strict128/512 tokens, e2e/TTFT/TPOT/acceptance/Graph/memory, actual commit totals and source hashes; preserve failures/negatives.
6. Separate short B4/B8 I64 O512 traces off/reuse on their own independent engines; stage events/host spans apart from untouched graph timeline. Count target body/head, draft/head/cache-only, snapshot/rollback, decision transfer, GPU kernel groups/gaps; stages do not imply additive CPU/GPU wall time. Initial-prefill trace diagnostic separate from steady decode.
7. Summarize paired distributions/ratios, output-length amortization, batch boundary and new hotspots; audit raw records. One final fresh code/report review, complete local commits/main and retain evidence; no push.

Review focus: main has no MTP config/import/protocol; nonfinal slots untouched; selected row/sampler alignment; slot reservation under partial prompts; in-flight requests not dispatched twice; no B63 padding; paired order and engine isolation; actual submitted-token denominator; Graph-event instrumentation excluded from formal timing; BF16 shape changes retain top1 and completion gates.
