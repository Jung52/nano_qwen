# MTP cache-only and two-token reuse

Use inline execution; the user's design supplies the implementation scope.

- [x] Add real eager/graph cache-only tests: exact KV including untouched rows,
  boundaries and non-contiguous blocks, plus a subsequent full prediction.
- [x] Implement one-layer MTP prefix through K/V store only, an independent
  CUDA Graph and acceptance catch-up dispatch. Preserve serial verification.
- [x] Run the existing forced-decision/cache lifecycle suite and real FP8 B1
  correctness gates. Measure input64/output128, two runs, profiler disabled.
- [ ] Build an independent two-token layer experiment: batch dense/MLP work,
  sequential GDN updates and first-row snapshot, single-query attention.
  Compare each layer's hidden/residual, logits and final/snapshot/KV state to
  two ordinary B1 forwards; record the first mismatch without drift waivers.
- [ ] Integrate only proven changes; retain serial reference and all existing
  B1/TP1 guards. Leave snapshots/synchronization and B4/8/5090 for later.

Preserve root README/DEVELOPMENT_LOG/evidence changes, linear.py and craft.py.
Keep correctness diagnostics separate from exactly two formal samples.
