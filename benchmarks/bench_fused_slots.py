#!/usr/bin/env python3
"""One command: GPU math tests, norm microbench, four-way B8/I128/O256 + traces.

Each configuration runs in a fresh process: reference/fused x single/dual.
One warmup, three unprofiled timing rounds, three separate profiler windows.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def compare_tokens(reference, candidate):
    failures = []
    if len(reference) != len(candidate):
        return [dict(reason="Different round counts")]
    for ri, (a, b) in enumerate(zip(reference, candidate), 1):
        if a["seed"] != b["seed"] or a["prompt_sha256"] != b["prompt_sha256"]:
            failures.append(dict(round=ri, reason="Different seeds/prompts"))
            continue
        if len(a["token_ids"]) != len(b["token_ids"]):
            failures.append(dict(round=ri, reason="Different request counts"))
            continue
        for request, (xs, ys) in enumerate(zip(a["token_ids"], b["token_ids"])):
            first = next((i for i, (x, y) in enumerate(zip(xs, ys)) if x != y), None)
            if first is not None or len(xs) != len(ys):
                failures.append(dict(round=ri, request=request, first_token=first,
                    reference=xs[first] if first is not None else None,
                    candidate=ys[first] if first is not None else None,
                    reference_length=len(xs), candidate_length=len(ys)))
    return failures


def summarize(reports):
    errors, rows, comparisons = [], {}, []
    names = ["reference_single", "fused_single", "fused_dual", "reference_dual"]
    for name in names:
        r = reports.get(name, {})
        if r.get("status") != "ok" or r.get("returncode", 0):
            errors.append(dict(config=name, reason="Worker failed", error=r.get("error")))
            continue
        ms = r["measurements"]
        rows[name] = {k: statistics.median(m[k] for m in ms) for k in
                      ("ttft_ms", "decode_ms", "tpot_ms", "decode_tok_s", "e2e_ms", "output_tok_s")}
        groups = r["trace_stats"]["gpu_grouping"]
        rows[name].update(kernels=groups.get("kernel", 0), fused_norm_kernels=groups.get("fused_norm_kernels", 0),
            graph_replays=groups.get("graph_launches", 0),
            graph_kernels_per_replay=groups.get("graph_kernel_events", 0)/groups["graph_groups"]
                if groups.get("graph_groups") else None)
    if len(rows) == 4:
        base = reports["reference_single"]
        for name in names[1:]:
            r = reports[name]
            if (r["kv_blocks"] != base["kv_blocks"] or
                r["source"]["source_sha256"] != base["source"]["source_sha256"] or
                r["model"] != base["model"]):
                errors.append(dict(config=name, reason="Different model/KV capacity/source"))
            mismatch = compare_tokens(base["measurements"], r["measurements"])
            if mismatch:
                errors.append(dict(config=name, reason="Greedy output differs", mismatches=mismatch))
        for a, b, meaning in [("reference_single", "fused_single", "fusion_single"),
                ("reference_dual", "fused_dual", "fusion_dual"),
                ("reference_single", "reference_dual", "slots_reference"),
                ("fused_single", "fused_dual", "slots_fused")]:
            comparisons.append(dict(comparison=meaning, reference=a, candidate=b,
                decode_speedup=rows[a]["decode_ms"]/rows[b]["decode_ms"],
                e2e_speedup=rows[a]["e2e_ms"]/rows[b]["e2e_ms"],
                correctness_accepted=not errors))
    return dict(status="failed" if errors else "ok", errors=errors, medians=rows, comparisons=comparisons,
        notes=["Formal timings have no torch.profiler; prefill/decode phase boundaries synchronize both variants.",
               "Decode includes pipeline prime/drain; no per-step cuda synchronize.",
               "Three rounds are descriptive, not a statistical performance guarantee.",
               "Any greedy mismatch fails acceptance; numerical tolerance does not waive token mismatch."])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/home/wei/code/models/4b-channel-fp8")
    p.add_argument("--output-dir", default=f"reports/fused_slots_{time.strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--mem-util", type=float, default=0.65)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--trace-steps", type=int, default=16)
    p.add_argument("--trace-start", type=int, default=64)
    p.add_argument("--skip-unit-tests", action="store_true", help="Use only after passing the GPU unit tests")
    a = p.parse_args()
    if not 0 < a.mem_util < 1 or a.trace_steps < 1 or a.trace_start < 2 or a.trace_start+a.trace_steps+2 > 255:
        p.error("Invalid memory fraction or trace window")
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Check CUDA BEFORE allowing pytest skip markers to produce a false pass.
    check = subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA required'"], cwd=ROOT)
    if check.returncode:
        return check.returncode
    if not a.skip_unit_tests:
        test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_fused_norm.py", "tests/test_slot_trace.py"], cwd=ROOT)
        if test.returncode:
            return test.returncode
    micro = subprocess.run([sys.executable, str(ROOT/"benchmarks/bench_fused_norm.py"),
        "--output", str(out/"norm_micro.json")], cwd=ROOT)
    if micro.returncode:
        return micro.returncode
    reports = {}
    # Counterbalance backend order across the two slot modes.
    for name, fused, mode in [("reference_single", "0", "single_slot"), ("fused_single", "1", "single_slot"),
                               ("fused_dual", "1", "dual_slot"), ("reference_dual", "0", "dual_slot")]:
        dest = out/name
        dest.mkdir(exist_ok=True)
        path = dest/f"{mode}.report.json"
        path.unlink(missing_ok=True)  # Never accept a stale successful worker report.
        cmd = [sys.executable, str(ROOT/"benchmarks/profile_slots_b8.py"), "--worker", mode,
            "--model", str(Path(a.model).expanduser().resolve()), "--output-dir", str(dest),
            "--mem-util", str(a.mem_util), "--fused-ops", fused, "--trace-steps", str(a.trace_steps),
            "--trace-start", str(a.trace_start)]
        if a.eager:
            cmd.append("--eager")
        result = subprocess.run(cmd, cwd=ROOT)
        reports[name] = json.loads(path.read_text()) if path.exists() else dict(status="failed")
        reports[name]["returncode"] = result.returncode
    summary = summarize(reports)
    summary["report_paths"] = {name: str(out/name/f"{r.get('mode', 'unknown')}.report.json") for name,r in reports.items()}
    (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print("config | decode ms | decode tok/s | graph kernels/replay", flush=True)
    for name, r in summary["medians"].items():
        print(f"{name} | {r['decode_ms']:.3f} | {r['decode_tok_s']:.2f} | {r['graph_kernels_per_replay']}")
    for row in summary["comparisons"]:
        print(f"{row['comparison']}: {row['decode_speedup']:.3f}x (accepted={row['correctness_accepted']})")
    print(f"{summary['status']}: {out}/summary.json", flush=True)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
