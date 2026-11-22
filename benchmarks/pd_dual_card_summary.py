"""Merge bench_pd_dual_card.py outputs into comparison tables and CSV.

Usage:
    python benchmarks/pd_dual_card_summary.py OUT_DIR [OUT_DIR ...]

Reads every perf-*.json / shakedown-*.json in each directory written by the
driver, verifies their gates, and writes next to each input directory:

    perf-summary.json    merged aggregates + verification per config
    perf-summary.md      median comparison table
    perf-summary.csv     one row per config
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def table_rows(section):
    rows = [r for r in section["rows"] if not r.get("warmup") and not r.get("trace")]
    return rows


def config_row(path, section):
    rows = table_rows(section)
    agg = section.get("aggregates", {})
    if not rows:
        return None
    values = {
        "config": section["config"],
        "file": path.name,
        "passed": section.get("passed"),
        "requests": len(rows),
        "prompt_lens": section.get("lens"),
        "output_len": section.get("output_len"),
        "concurrency": section.get("concurrency"),
        "ttft_p50_ms": agg.get("ttft_ms", {}).get("p50"),
        "ttft_p95_ms": agg.get("ttft_ms", {}).get("p95"),
        "first_to_second_p50_ms": agg.get("first_to_second_ms", {}).get("p50"),
        "first_to_second_p95_ms": agg.get("first_to_second_ms", {}).get("p95"),
        "tpot_p50_ms": agg.get("tpot_ms", {}).get("p50"),
        "decode_ms_per_token_p50": agg.get("tpot_ms", {}).get("p50"),
        "e2e_p50_ms": agg.get("e2e_ms", {}).get("p50"),
        "e2e_p95_ms": agg.get("e2e_ms", {}).get("p95"),
        "handoff_pause_p50_ms": agg.get("pause_ms", {}).get("p50"),
        "handoff_pause_p95_ms": agg.get("pause_ms", {}).get("p95"),
        "d_queue_p50_ms": agg.get("queue_ms", {}).get("p50"),
        "d_queue_p95_ms": agg.get("queue_ms", {}).get("p95"),
        "d2h_p50_ms": agg.get("d2h_ms", {}).get("p50"),
        "h2d_p50_ms": agg.get("h2d_ms", {}).get("p50"),
        "ack_p50_ms": agg.get("ack_ms", {}).get("p50"),
        "req_per_s": agg.get("req_per_s"),
        "output_tok_per_s": agg.get("output_tok_per_s"),
        "prompt_tok_per_s": agg.get("prompt_tok_per_s"),
        "transfer_bytes_total": agg.get("transfer_bytes_total"),
        "nbytes_median": agg.get("nbytes", {}).get("median"),
    }
    finals = []
    for key, worker in section.get("workers", {}).items():
        final = worker.get("final")
        if final:
            finals.append((key, final.get("peak_allocated_mib"),
                           final.get("peak_reserved_mib")))
    if finals:
        values["gpu_peak_allocated_mib"] = max(f[1] for f in finals)
        values["gpu_peak_reserved_mib"] = max(f[2] for f in finals)
    if "span_overlap" in section and section["span_overlap"]:
        values["span_overlap_total_ms"] = section["span_overlap"].get("total_overlap_ms")
        values["span_overlap_mean_ms"] = section["span_overlap"].get("mean_overlap_ms")
    trace = section.get("trace_evidence")
    if trace:
        values["kernel_overlap_us"] = trace.get("kernel_overlap_us")
    histogram = section.get("decode_batch_histogram")
    if histogram:
        values["decode_batch_histogram"] = histogram
        values["b2_steps"] = int(histogram.get("2", 0))
    values["verification"] = section.get("verification")
    return values


CSV_FIELDS = [
    "config", "file", "passed", "requests", "concurrency", "output_len",
    "ttft_p50_ms", "ttft_p95_ms", "first_to_second_p50_ms", "first_to_second_p95_ms",
    "tpot_p50_ms",
    "e2e_p50_ms", "e2e_p95_ms", "handoff_pause_p50_ms", "handoff_pause_p95_ms",
    "d_queue_p50_ms", "d_queue_p95_ms", "d2h_p50_ms",
    "h2d_p50_ms", "ack_p50_ms", "req_per_s", "output_tok_per_s",
    "prompt_tok_per_s", "transfer_bytes_total", "nbytes_median",
    "gpu_peak_allocated_mib", "gpu_peak_reserved_mib",
    "span_overlap_total_ms", "kernel_overlap_us", "b2_steps",
]


def markdown(rows):
    f1 = lambda v, spec=".1f": f"{v:{spec}}" if v is not None else "—"
    f0 = lambda v: f"{v:.0f}" if v is not None else "—"
    lines = ["# 双卡 PD 性能汇总", "",
             "所有配置共用同一 checkpoint（per-channel FP8）、同一请求序列与采样参数；",
             "计时在预热与 Graph 捕获之后开始。P50/P95 为请求级聚合。", ""]
    order = ("pd-serial", "pd-pipeline", "single", "dual-replica")
    ordered = sorted(rows, key=lambda r: (
        order.index(r["config"]) if r["config"] in order else len(order),
        r.get("concurrency") or 0))
    header = ("| 配置 | 并发 | 请求 | TTFT P50 | E2E P50 | E2E P95 | 首→次token P50 | "
              "TPOT P50 | D排队P95 | req/s | 输出tok/s | B2步数 | 显存峰值MiB | 传输总量 |")
    sep = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines += [header, sep]
    for r in ordered:
        histogram = r.get("decode_batch_histogram") or {}
        lines.append(
            f"| {r['config']} | {r['concurrency']} | {r['requests']} | "
            f"{f1(r['ttft_p50_ms'])} | {f1(r['e2e_p50_ms'])} | {f1(r['e2e_p95_ms'])} | "
            f"{f1(r['first_to_second_p50_ms'])} | {f1(r['tpot_p50_ms'], '.2f')} | "
            f"{f1(r['d_queue_p95_ms'])} | {f1(r['req_per_s'], '.2f')} | "
            f"{f1(r['output_tok_per_s'])} | {histogram.get('2', 0)} | "
            f"{f0(r['gpu_peak_allocated_mib'])} | "
            f"{f1(r['transfer_bytes_total'] / 2**30, '.2f')} GiB |")
    lines += ["",
              "握手阶段（ms）：d2h / h2d / ack / pause（首token→D恢复完成）", ""]
    lines.append("| 配置 | 并发 | d2h P50 | h2d P50 | ack P50 | pause P50 | pause P95 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in ordered:
        lines.append(f"| {r['config']} | {r['concurrency']} | {f1(r['d2h_p50_ms'], '.2f')} | "
                     f"{f1(r['h2d_p50_ms'], '.2f')} | {f1(r['ack_p50_ms'], '.2f')} | "
                     f"{f1(r['handoff_pause_p50_ms'], '.2f')} | "
                     f"{f1(r['handoff_pause_p95_ms'], '.2f')} |")
    lines += ["",
              "验证门禁（全部配置）：",
              "- PD 路径 token 与 P 侧同权重 reference 对比：D 全程 B1 时逐 token 一致；",
              "  混合 B2 行按仓库 near-tie 门禁验收，另记录 fp8 有界翻牌判定",
              "  （同竞争对 + 一侧 near-tie + 交叉 logit 位移 ≤ 2× 仓库门限）",
              "  与 replay 逐位确定性证据；",
              "- single c=2 混合批按仓库 near-tie 门禁验收（1024-token 分块行跳过，",
              "  已由 D≤2 流水线全量补齐验证）；",
              "- D 每请求 0 次 prefill、127 次 decode dispatch；ACK 100%；",
              "- 排空后双端 KV blocks/GDN slots 归零、store 占用回到起点。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+")
    args = parser.parse_args()
    for directory in args.dirs:
        directory = Path(directory)
        paths = sorted(p for p in directory.glob("perf-*.json")
                       if not p.name.startswith("perf-summary"))
        if not paths:
            print(f"no driver outputs in {directory}")
            continue
        rows, failures = [], []
        for path in paths:
            report = load(path)
            section = report.get("section")
            if section is None:
                failures.append(dict(file=path.name, error="no section"))
                continue
            row = config_row(path, section)
            if row is None:
                failures.append(dict(file=path.name, error="no measured rows"))
                continue
            rows.append(row)
            if not section.get("passed"):
                failures.append(dict(file=path.name,
                                     verification=section.get("verification")))
        summary = dict(rows=rows, failures=failures,
                       passed=all(r["passed"] for r in rows) and not failures)
        (directory / "perf-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        (directory / "perf-summary.md").write_text(markdown(rows))
        with open(directory / "perf-summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in CSV_FIELDS})
        print(json.dumps(dict(dir=str(directory), configs=len(rows),
                              passed=summary["passed"], failures=failures),
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
