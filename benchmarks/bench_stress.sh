#!/usr/bin/env bash
# BF16 vs FP8 压力测试:大 batch / 长 prompt / 轮数少
# 用法: bash bench_stress.sh [eager|decode_graph] ["2048x8,4096x8,8192x4,16384x2"]
set -euo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-decode_graph}"
CASES="${2:-2048x8,4096x8,8192x4,16384x2}"
BF16_MODEL=/root/autodl-tmp/nano_qwen/9b
FP8_MODEL=/root/autodl-tmp/nano_qwen/fp8
OUT=benchmarks/reports
mkdir -p "$OUT"
GPU_ID=0

run_case() {
    local name="$1" pid="$2"
    local vram_log="$OUT/${name}_stress.vram.log"
    ( while kill -0 "$pid" 2>/dev/null; do
          nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID"
          sleep 0.5
      done ) > "$vram_log" &
    local poll_pid=$!
    wait "$pid"
    wait "$poll_pid" 2>/dev/null || true
    echo "== $name 峰值显存: $(sort -n "$vram_log" | tail -1) MiB (log: $vram_log)"
}

run_stress() {
    local name="$1" model="$2"
    echo "=== [$name] 开始 ($MODE, cases=$CASES) ==="
    PYTHONPATH=src python benchmarks/stress_bench.py \
        --model "$model" \
        --mode "$MODE" \
        --cases "$CASES" \
        --json-out "$OUT/${name}_stress.json" \
        --warmup 1 --rounds 2 --decode-tokens 128 &
    run_case "$name" $!
    echo "=== [$name] 完成: $OUT/${name}_stress.json ==="
}

run_stress bf16 "$BF16_MODEL"
run_stress fp8  "$FP8_MODEL"

echo
echo "===== 压力对比 ====="
python - "$OUT" <<'EOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = {}
for name in ("bf16", "fp8"):
    d = json.loads((out / f"{name}_stress.json").read_text())
    rows[name] = d
    print(f"\n[{name}] {d['quantization'] or 'bf16'}  weights={d['parameters_mib']:.0f} MiB"
          f"  kv={d['kv_mib']:.0f} MiB  kv_blocks={d['num_kvcache_blocks']}")
    for w in d["workloads"]:
        print(f"  p{w['prompt_tokens']:>5} bs{w['batch']:>2}  ttft={w['ttft_ms']:>8.1f}ms"
              f"  prefill={w['prefill_tok_per_s']:>8.0f} tok/s"
              f"  decode={w['decode_tok_per_s']:>8.2f} tok/s")

print("\n--- decode 加速比 (fp8/bf16) ---")
for wb, wf in zip(rows["bf16"]["workloads"], rows["fp8"]["workloads"]):
    if wb["prompt_tokens"] != wf["prompt_tokens"] or wb["batch"] != wf["batch"]:
        continue
    r = wf["decode_tok_per_s"] / wb["decode_tok_per_s"]
    print(f"  p{wb['prompt_tokens']:>5} bs{wb['batch']:>2}  {r:.3f}x")
EOF
echo
echo "峰值显存: $(sort -n "$OUT"/bf16_stress.vram.log | tail -1) MiB (bf16) vs $(sort -n "$OUT"/fp8_stress.vram.log | tail -1) MiB (fp8)"
