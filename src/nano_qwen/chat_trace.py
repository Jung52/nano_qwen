"""Optional per-turn torch.profiler capture for the interactive chat.

Imported lazily by chat.py only when --profile-steps is set, so a normal
chat startup never touches Kineto. Trace analysis is pure Python and also
runs without torch/GPU on an exported trace.
"""
from __future__ import annotations

import gzip
import html
import json
from pathlib import Path
from time import strftime

LAUNCH_NAMES = ("cudaGraphLaunch", "cuGraphLaunch")
# A decode graph replay owns the model's ~2k kernels. FP8 per-channel GEMM
# micro-graphs and other epilogue work own a handful and must not create
# step pairs; they still count toward GPU busy time.
DECODE_LAUNCH_MIN_KERNELS = 50


def init_kineto(torch) -> None:
    """Initialize Kineto before CUDA Graph capture (documented workaround).

    Must run before the engine captures decode graphs; never start or stop
    profiling inside a captured graph.
    """
    try:
        from torch.profiler._utils import _init_for_cuda_graphs
    except ImportError:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]):
            torch.empty(1, device="cuda").zero_()
            torch.cuda.synchronize()
    else:
        _init_for_cuda_graphs()


def start_profiler(torch):
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
    )
    prof.start()
    return prof


def finish_profiler(prof, args, turn_no: int) -> str:
    """Stop, export a gzipped Chrome trace and analyze it; never raises."""
    prof.stop()
    out = Path(args.profile_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"turn{turn_no:03d}_{strftime('%Y%m%d_%H%M%S')}.trace.json.gz"
    prof.export_chrome_trace(str(path))
    try:
        return summarize(path, turn_no=turn_no, profile_steps=args.profile_steps)
    except Exception as exc:  # Analysis must never break the chat session.
        return f"[trace] exported {path} (analysis failed: {exc!r})"


def summarize(path: Path, turn_no: int = 1, profile_steps: int | None = None) -> str:
    """Analyze a trace, write summary.json + timeline.svg, return a verdict line."""
    summary = analyze_trace(path)
    summary["turn"] = turn_no
    summary["trace"] = str(path)
    if profile_steps is not None:
        summary["profile_steps"] = profile_steps
    stem = path.name.replace(".trace.json.gz", "")
    summary_path = path.with_name(stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    draw_svg(path.with_name(stem + ".timeline.svg"), summary,
             f"nano_qwen chat turn {turn_no}: async decode overlap")
    pairs = summary["pairs"]
    overlap = sum(1 for p in pairs if p["verdict"] == "OVERLAP")
    idle = summary["gpu_idle"]
    tail = (
        f"overlap={overlap}/{len(pairs)}"
        if pairs else f"overlap=n/a decode_launches={summary['decode_graph_launches']}"
    )
    return (f"[trace] decode_steps={summary['decode_graph_launches']} "
            f"(graph replay x{summary['decode_graph_kernels_per_replay']} kernels) {tail} | "
            f"step={summary['step_interval_ms_median']:.1f} ms | "
            f"gpu_idle max {idle['max_gap_us']/1000:.2f} ms, total {idle['idle_percent']:.1f}% | "
            f"plot {path.with_name(stem + '.timeline.svg')}")


def analyze_trace(path: Path) -> dict:
    """Measure launch/forward overlap and GPU idle directly from raw events.

    A forward window is derived from CUPTI correlations: kernels whose
    launching API was issued between consecutive graph launches belong to
    that decode step (graph replay plus the out-of-graph LM head/sample).
    """
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    kernels = sorted((e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"),
                     key=lambda e: e["ts"])
    apis = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")
            and e.get("ph") == "X"]
    launches = sorted((a for a in apis if a.get("name") in LAUNCH_NAMES),
                      key=lambda e: e["ts"])
    api_by_corr = {}
    for a in apis:
        corr = (a.get("args") or {}).get("correlation")
        if corr is not None:
            api_by_corr[corr] = a

    owned = []
    kernels_per_launch = {}
    for k in kernels:
        corr = (k.get("args") or {}).get("correlation")
        owner = api_by_corr.get(corr) if corr is not None else None
        if owner is not None:
            owned.append((k["ts"], k["ts"] + k["dur"], owner["ts"]))
            if owner["name"] in LAUNCH_NAMES:
                kernels_per_launch[owner["ts"]] = kernels_per_launch.get(owner["ts"], 0) + 1
    owned.sort()

    decode_launches = [l for l in launches
                       if kernels_per_launch.get(l["ts"], 0) >= DECODE_LAUNCH_MIN_KERNELS]

    # A per-step CPU scope that issued the whole enqueue work of one decode
    # step bounds the window correctly in BOTH scheduling modes; the plain
    # launch-interval fallback would also catch the NEXT step's staging
    # kernels, whose GPU tail then fakes overlap on a serial pipeline.
    # nqc/step/N: chat consumer scope; nqp/execute/bN: bench-script scope.
    step_scopes = sorted((e for e in events
                          if e.get("cat") == "user_annotation" and e.get("ph") == "X"
                          and (e.get("name", "").startswith("nqc/step/")
                               or e.get("name", "").startswith("nqp/execute/"))),
                         key=lambda e: e["ts"])

    def scope_span(launch_ts):
        for sc in step_scopes:
            if sc["ts"] <= launch_ts < sc["ts"] + sc["dur"]:
                return sc["ts"], sc["ts"] + sc["dur"]
        return None

    def forward_window(i: int):
        """GPU span of kernels owned by this step's own APIs."""
        span = scope_span(decode_launches[i]["ts"])
        lo, hi = span if span else (
            decode_launches[i]["ts"],
            decode_launches[i + 1]["ts"] if i + 1 < len(decode_launches) else float("inf"))
        ks = [(s, e) for s, e, at in owned if lo <= at < hi]
        if not ks:
            return None
        return (min(s for s, _ in ks), max(e for _, e in ks))

    steps = []
    pairs = []
    for i, L in enumerate(decode_launches):
        steps.append({"n": i, "launch_us": [L["ts"], L["ts"] + L["dur"]],
                      "forward_us": forward_window(i)})
    for i in range(len(decode_launches) - 1):
        nxt = decode_launches[i + 1]
        window = steps[i]["forward_us"]
        if window is None:
            pairs.append({"n": i, "verdict": "UNKNOWN"})
            continue
        end = window[1]
        overlap = max(0.0, min(nxt["ts"] + nxt["dur"], end) - nxt["ts"])
        pairs.append({
            "n": i,
            "launch_next_start_us": nxt["ts"],
            "launch_duration_us": nxt["dur"],
            "forward_n_end_us": end,
            "launch_after_forward_end_us": nxt["ts"] - end,
            "overlap_us": overlap,
            "verdict": "OVERLAP" if nxt["ts"] < end else "AFTER_FORWARD",
        })

    # Idle is measured over the decode phase only: from the first decode
    # graph launch to the last kernel, so tokenizer/prefill preparation gaps
    # before the first replay stay out of the pipeline numbers.
    phase_start = decode_launches[0]["ts"] if decode_launches else (kernels[0]["ts"] if kernels else 0.0)
    phase_kernels = [k for k in kernels if k["ts"] >= phase_start]
    busy = []
    for k in phase_kernels:
        s, e = k["ts"], k["ts"] + k["dur"]
        if busy and s <= busy[-1][1]:
            busy[-1][1] = max(busy[-1][1], e)
        else:
            busy.append([s, e])
    gaps = [(busy[i][1], busy[i + 1][0]) for i in range(len(busy) - 1)]
    span = (busy[-1][1] - busy[0][0]) if busy else 0.0
    steps_ms = sorted(decode_launches[i + 1]["ts"] - decode_launches[i]["ts"]
                      for i in range(len(decode_launches) - 1))
    replay_kernels = sorted(kernels_per_launch[l["ts"]] for l in decode_launches)
    return {
        "window_method": ("step_scopes" if step_scopes else "launch_intervals"),
        "graph_launches": len(launches),
        "decode_graph_launches": len(decode_launches),
        "decode_graph_kernels_per_replay": replay_kernels[len(replay_kernels) // 2] if replay_kernels else 0,
        "kernels": len(kernels),
        "unattributed_kernels": len(kernels) - len(owned),
        "step_interval_ms_median": (steps_ms[len(steps_ms) // 2] / 1000.0) if steps_ms else 0.0,
        "pairs": pairs,
        "steps": steps,
        "gpu_idle": {
            "max_gap_us": max((b - a) for a, b in gaps) if gaps else 0.0,
            "gaps_over_100us": sorted(((b - a, a, b) for a, b in gaps if b - a > 100),
                                      reverse=True),
            "total_idle_us": sum(b - a for a, b in gaps),
            "span_us": span,
            "idle_percent": (100.0 * sum(b - a for a, b in gaps) / span) if span else 0.0,
        },
    }


def draw_svg(path, summary: dict, title: str) -> None:
    """Two-lane timeline: GPU forward windows vs cudaGraphLaunch on one axis.

    Orange launch bars drawn inside a green forward bar are the visible
    proof of the async one-batch lookahead.
    """
    steps = [s for s in summary.get("steps", []) if s["forward_us"]]
    if not steps:
        return
    lo = min(s["launch_us"][0] for s in steps)
    hi = max(s["forward_us"][1] for s in steps)
    width, left, plot = 1800, 170, 1560
    scale = plot / max(hi - lo, 1.0)

    def x(t):
        return left + (t - lo) * scale

    overlap = sum(1 for p in summary["pairs"] if p["verdict"] == "OVERLAP")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="360" viewBox="0 0 {width} 360">',
           '<rect width="100%" height="100%" fill="#11151c"/>',
           '<style>text{fill:#e5eaf3;font:14px sans-serif}</style>',
           f'<text x="25" y="30" font-size="21">{html.escape(title)}</text>',
           f'<text x="25" y="55">decode steps={summary["decode_graph_launches"]}  '
           f'launch covered by previous forward: {overlap}/{len(summary["pairs"])}  '
           f'step={summary["step_interval_ms_median"]:.1f} ms  '
           f'gpu_idle={summary["gpu_idle"]["idle_percent"]:.1f}%</text>',
           '<text x="25" y="112">GPU forward</text>', '<text x="25" y="212">CPU GraphLaunch</text>']
    for i in range(9):
        gx = left + plot * i / 8
        out += [f'<path d="M{gx:.1f} 78 V248" stroke="#354052" stroke-dasharray="3 5"/>',
                f'<text x="{gx:.1f}" y="270" text-anchor="middle">{(hi-lo)*i/8000:.2f} ms</text>']
    for g, a, b in summary["gpu_idle"]["gaps_over_100us"]:
        out.append(f'<path d="M{x(a):.3f} 92 V128" stroke="#e05252" stroke-width="{max(1.0,(b-a)*scale):.2f}">'
                   f'<title>gpu idle {g:.0f} us</title></path>')
    for s in steps:
        fs, fe = s["forward_us"]
        out.append(f'<rect x="{x(fs):.3f}" y="92" width="{max(0.8,(fe-fs)*scale):.3f}" height="36" '
                   f'fill="#31bf80" fill-opacity="0.55" stroke="#31bf80">'
                   f'<title>forward n={s["n"]}: {fs-lo:.0f}..{fe-lo:.0f} us ({fe-fs:.0f} us)</title></rect>')
        ls, le = s["launch_us"]
        out.append(f'<rect x="{x(ls):.3f}" y="192" width="{max(1.2,(le-ls)*scale):.3f}" height="36" '
                   f'fill="#ff9a4a" fill-opacity="0.85" stroke="#ff9a4a">'
                   f'<title>cudaGraphLaunch n={s["n"]}: {ls-lo:.0f}..{le-lo:.0f} us ({le-ls:.0f} us)</title></rect>')
    out += ['<text x="25" y="308">Green: GPU forward window per decode step (graph + LM head + sample kernels).</text>',
            '<text x="25" y="330">Orange: cudaGraphLaunch API call. Async works when each orange bar starts before the green bar above it ends.</text>',
            '<text x="25" y="352">Red ticks: GPU idle gaps &gt;100 us. Diagnostic profiler run; step time includes CUPTI overhead.</text>',
            '</svg>']
    Path(path).write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    """Re-render summary.json + timeline.svg for an existing chat trace.

    Usage: python -m nano_qwen.chat_trace logs/chat_trace/turnNNN_*.trace.json.gz
    """
    import argparse
    import re
    import sys

    cli = argparse.ArgumentParser()
    cli.add_argument("trace", type=Path)
    argv = cli.parse_args()
    m = re.search(r"turn(\d+)", argv.trace.name)
    print(summarize(argv.trace, turn_no=int(m.group(1)) if m else 1))
    sys.exit(0)
