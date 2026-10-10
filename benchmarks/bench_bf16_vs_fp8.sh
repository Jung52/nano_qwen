#!/usr/bin/env bash
# Place this file and validate_fp8_model.py in nano_qwen/benchmarks/.
# Usage: bash benchmarks/bench_bf16_vs_fp8.sh [eager|decode_graph] [batch|context|legacy]
set -euo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-decode_graph}"
PROFILE="${2:-batch}"
[[ "$MODE" == eager || "$MODE" == decode_graph ]] || { echo "Invalid mode: $MODE" >&2; exit 2; }
[[ "$PROFILE" == batch || "$PROFILE" == context || "$PROFILE" == legacy ]] || { echo "Invalid profile: $PROFILE" >&2; exit 2; }

BF16_MODEL="${BF16_MODEL:-/root/autodl-tmp/nano_qwen/9b}"
FP8_MODEL="${FP8_MODEL:-/root/autodl-tmp/nano_qwen/fp8}"
GPU_ID="${GPU_ID:-0}"
MEM_UTIL="${MEM_UTIL:-0.65}"
WARMUP="${WARMUP:-5}"
ROUNDS="${ROUNDS:-10}"
DECODE_TOKENS="${DECODE_TOKENS:-256}"
STAMP="$(date -u +%Y%m%dT%H%M%S)_$$"
OUT="${OUT:-benchmarks/reports/${MODE}_${PROFILE}_${STAMP}}"
mkdir -p "$OUT"

# Physical GPU index for nvidia-smi; it becomes CUDA device 0 inside Python.
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
current_python=""
current_poll=""
cleanup() {
    if [[ -n "$current_python" ]]; then kill "$current_python" 2>/dev/null || true; fi
    if [[ -n "$current_poll" ]]; then kill "$current_poll" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

extra_args=()
[[ -z "${MAX_NUM_SEQS:-}" ]] || extra_args+=(--max-num-seqs "$MAX_NUM_SEQS")
[[ -z "${MAX_MODEL_LEN:-}" ]] || extra_args+=(--max-model-len "$MAX_MODEL_LEN")
if [[ -n "${PERF_CASES:-}" ]]; then
    read -r -a requested_cases <<< "$PERF_CASES"
    extra_args+=(--perf-cases "${requested_cases[@]}")
fi

run_perf() {
    local name="$1" model="$2" rc=0
    echo "=== [$name] mode=$MODE profile=$PROFILE mem_util=$MEM_UTIL ==="
    nvidia-smi -i "$GPU_ID" > "$OUT/${name}.gpu_before.txt" 2>&1 || true
    python benchmarks/validate_fp8_model.py \
        --model "$model" --mode "$MODE" --suite performance \
        --perf-profile "$PROFILE" --gpu-memory-utilization "$MEM_UTIL" \
        --warmup-rounds "$WARMUP" --rounds "$ROUNDS" --decode-tokens "$DECODE_TOKENS" \
        --json-out "$OUT/${name}_perf.json" "${extra_args[@]}" \
        > "$OUT/${name}.console.log" 2>&1 &
    current_python=$!
    (
        echo 'timestamp,index,memory_used_mib,gpu_util_percent,power_w,sm_clock_mhz,memory_clock_mhz,temperature_c'
        while kill -0 "$current_python" 2>/dev/null; do
            nvidia-smi -i "$GPU_ID" \
                --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu \
                --format=csv,noheader,nounits || true
            sleep 0.5
        done
    ) > "$OUT/${name}.gpu.csv" 2> "$OUT/${name}.poll.log" &
    current_poll=$!
    wait "$current_python" || rc=$?
    current_python=""
    wait "$current_poll" || true
    current_poll=""
    echo "$rc" > "$OUT/${name}.exit_code"
    echo "=== [$name] exit=$rc; log: $OUT/${name}.console.log ==="
    # Keep running the second precision even if the first process fails.
    return 0
}

case "${ORDER:-bf16_first}" in
    bf16_first) run_perf bf16 "$BF16_MODEL"; run_perf fp8 "$FP8_MODEL" ;;
    fp8_first) run_perf fp8 "$FP8_MODEL"; run_perf bf16 "$BF16_MODEL" ;;
    *) echo 'ORDER must be bf16_first or fp8_first' >&2; exit 2 ;;
esac

python - "$OUT" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
reports = {}
exit_codes = {}
for name in ("bf16", "fp8"):
    exit_codes[name] = int((out / f"{name}.exit_code").read_text())
    path = out / f"{name}_perf.json"
    if not path.exists():
        print(f"[{name}] no JSON report; inspect {name}.console.log")
        continue
    reports[name] = json.loads(path.read_text())
    d = reports[name]
    print(f"\n[{name}] status={d.get('status')} exit={exit_codes[name]}")
    if "parameters_mib" in d:
        print(f"weights={d['parameters_mib']:.1f} MiB "
              f"KV={d['kv_mib']:.1f} MiB GDN={d['gdn_state_mib']:.1f} MiB")
    for w in d.get("performance", {}).get("workloads", []):
        label = f"p{w['prompt_tokens']}_bs{w['batch']}"
        if w["status"] != "completed":
            print(label, w["status"], w.get("reason", w.get("error", "")))
            continue
        print(f"{label:14s} prefill_ready={w['prefill_ready_ms']:8.2f}ms "
              f"step={w['decode_step_ms']:7.2f}ms "
              f"decode={w['decode_tokens_per_s']:9.2f}tok/s "
              f"e2e={w['e2e_ms']:9.2f}ms "
              f"torch_peak={max(r['torch_peak_allocated_mib'] for r in w['runs']):.0f}MiB")
    peaks = []
    with (out / f"{name}.gpu.csv").open() as f:
        for row in csv.DictReader(f):
            try:
                peaks.append(float(row["memory_used_mib"]))
            except (KeyError, TypeError, ValueError):
                pass
    print(f"GPU-wide sampled peak: {max(peaks):.0f} MiB" if peaks else "GPU sampling unavailable")

if "bf16" in reports and "fp8" in reports:
    a, b = reports["bf16"], reports["fp8"]
    configs_match = a.get("config") is not None and b.get("config") is not None
    for key in ("max_num_seqs", "max_model_len", "gpu_memory_utilization",
                "prefix_cache", "queue_depth", "async_output", "kv_dtype"):
        x, y = a.get("config", {}).get(key), b.get("config", {}).get(key)
        if x != y:
            configs_match = False
            print(f"CONFIG MISMATCH {key}: {x} vs {y}")
    if a.get("performance", {}).get("timing_version") != 2 or b.get("performance", {}).get("timing_version") != 2:
        configs_match = False
    left = {(w["prompt_tokens"], w["batch"]): w for w in a.get("performance", {}).get("workloads", [])}
    right = {(w["prompt_tokens"], w["batch"]): w for w in b.get("performance", {}).get("workloads", [])}
    print("\n=== FP8 vs BF16: comparable completed cases ===")
    for key in sorted(left.keys() | right.keys()):
        x, y = left.get(key), right.get(key)
        if not x or not y or x["status"] != "completed" or y["status"] != "completed":
            print(f"p{key[0]}_bs{key[1]}: not comparable; retain as capacity/failure evidence")
            continue
        if (not configs_match or x["decode_forwards"] != y["decode_forwards"]
                or x["prompt_sha256"] != y["prompt_sha256"]):
            print(f"p{key[0]}_bs{key[1]}: incompatible configuration/protocol/prompt")
            continue
        speed = (y["decode_tokens_per_s"] / x["decode_tokens_per_s"] - 1) * 100
        latency = (y["e2e_ms"] / x["e2e_ms"] - 1) * 100
        print(f"p{key[0]}_bs{key[1]}: decode throughput {speed:+.2f}% / e2e latency {latency:+.2f}%")

print(f"\nREPORT DIRECTORY: {out}")
if any(exit_codes.values()):
    sys.exit(1)
PY
