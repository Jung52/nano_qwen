"""Shared, CUDA-free analysis of explicitly labelled inference dispatches."""
from collections import Counter, defaultdict
import gzip
import html
import json
from pathlib import Path
import re
import statistics

MARKER = re.compile(r"^nq/([a-z_]+)/b(\d+)(?:/slot([01]))?$")
LAUNCH_NAMES = {"cudaGraphLaunch", "cuGraphLaunch"}
MIB = 1024 * 1024


def end(event):
    return event["ts"] + event.get("dur", 0)


def merge(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if result and lo <= result[-1][1]:
            result[-1][1] = max(result[-1][1], hi)
        else:
            result.append([lo, hi])
    return result


def overlap(interval, windows):
    return sum(max(0, min(interval[1], hi) - max(interval[0], lo))
               for lo, hi in merge(windows))


def median_ms(values):
    return statistics.median(values) / 1000 if values else None


def _attribute(data):
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    scopes, by_thread = defaultdict(dict), defaultdict(list)
    for e in events:
        match = MARKER.fullmatch(e.get("name", ""))
        if e.get("cat") == "user_annotation" and e.get("ph") == "X" and match:
            stage, bid, slot = match[1], int(match[2]), int(match[3] or 0)
            scopes[bid].setdefault(stage, e)
            if stage in ("launch", "lm_head", "sample_call"):
                by_thread[e.get("pid"), e.get("tid")].append((e, stage, bid, slot))
    by_correlation, launches = defaultdict(list), defaultdict(list)
    for e in events:
        if e.get("cat") not in ("cuda_runtime", "cuda_driver") or e.get("ph") != "X":
            continue
        candidates = [x for x in by_thread[e.get("pid"), e.get("tid")]
                      if x[0]["ts"] <= e["ts"] and end(e) <= end(x[0]) + .001]
        tag = None
        if candidates:
            scope, stage, bid, slot = min(candidates, key=lambda x: x[0]["dur"])
            if stage != "launch" or e.get("name") in LAUNCH_NAMES:
                tag = (stage, bid, slot)
                if stage == "launch":
                    launches[bid].append(e)
        corr = e.get("args", {}).get("correlation")
        if corr is not None:
            by_correlation[corr].append((tag, e))
    kernels = defaultdict(lambda: defaultdict(list))
    ambiguous = unassigned = total = 0
    for e in events:
        if e.get("cat") != "kernel" or e.get("ph") != "X":
            continue
        total += 1
        corr = e.get("args", {}).get("correlation")
        candidates = [(tag, api) for tag, api in by_correlation.get(corr, [])
                      if api["ts"] <= e["ts"] + .001]
        owners = {(api.get("pid"), tag) for tag, api in candidates}
        # Reused correlations across dispatches/processes are ambiguous.
        # Never silently assign them to the most recent host API.
        if len(owners) != 1 or any(tag is None for _, tag in owners):
            unassigned += 1
            ambiguous += int(len(owners) > 1)
            continue
        _, (stage, bid, _) = next(iter(owners))
        kernels[bid][stage].append(e)
    frames = []
    for bid in sorted(scopes):
        s, k = scopes[bid], kernels[bid]
        calls = [e for e in launches[bid] if e.get("cat") == "cuda_runtime"] or launches[bid]
        if len(calls) != 1 or not all(name in s for name in ("schedule", "execute", "sample_call", "commit")):
            continue
        if not all(k[name] for name in ("launch", "lm_head", "sample_call")):
            continue
        launch = calls[0]
        if s["schedule"]["ts"] > launch["ts"]:
            raise ValueError("Invalid plan/launch ordering")
        def span(stage):
            return [min(e["ts"] for e in k[stage]), max(end(e) for e in k[stage])]
        graph, head, sample = span("launch"), span("lm_head"), span("sample_call")
        frame = dict(batch_id=bid, plan=[s["schedule"]["ts"], launch["ts"]],
                     launch=[launch["ts"], end(launch)], graph=graph, lm_head=head,
                     forward=merge([graph, head]), sample=sample,
                     commit=[s["commit"]["ts"], end(s["commit"])],
                     graph_kernel_count=len(k["launch"]),
                     graph_correlations=sorted({e["args"]["correlation"] for e in k["launch"]}),
                     forward_kernel_intervals=merge([(e["ts"], end(e))
                         for stage in ("launch", "lm_head") for e in k[stage]]))
        if "result_wait" in s:
            frame["result_wait"] = [s["result_wait"]["ts"], end(s["result_wait"])]
        frames.append(frame)
    return frames, dict(kernels=total, unassigned_kernels=unassigned,
                        ambiguous_kernels=ambiguous, labelled_dispatches=len(scopes))


def analyze(data):
    """Return only fully attributed dispatches, independent of kernel count."""
    return _attribute(data)[0]


def summarize_frames(frames):
    adjacent = [(a, b) for a, b in zip(frames, frames[1:]) if b["batch_id"] == a["batch_id"] + 1]
    result = dict(complete_batches=len(frames), adjacent_pairs=len(adjacent),
                  graph_kernel_counts=[f["graph_kernel_count"] for f in frames])
    for name in ("plan", "launch"):
        result[name + "_next_overlaps_forward"] = sum(overlap(b[name], a["forward"]) > 0 for a, b in adjacent)
    result["launch_next_overlaps_physical_kernel"] = sum(
        overlap(b["launch"], a["forward_kernel_intervals"]) > 0 for a, b in adjacent)
    result["commit_overlaps_next_forward"] = sum(overlap(a["commit"], b["forward"]) > 0 for a, b in adjacent)
    result["commit_overlaps_next_physical_kernel"] = sum(
        overlap(a["commit"], b["forward_kernel_intervals"]) > 0 for a, b in adjacent)
    result["sample_before_next_graph"] = sum(a["sample"][1] <= b["graph"][0] for a, b in adjacent)
    for key, values in {
        "graph": [f["graph"][1]-f["graph"][0] for f in frames],
        "forward_active_span": [sum(hi-lo for lo, hi in f["forward"]) for f in frames],
        "plan": [f["plan"][1]-f["plan"][0] for f in frames],
        "launch_api": [f["launch"][1]-f["launch"][0] for f in frames],
        "sample": [f["sample"][1]-f["sample"][0] for f in frames],
        "result_wait": [f["result_wait"][1]-f["result_wait"][0] for f in frames if "result_wait" in f],
        "gpu_step_interval": [b["graph"][0]-a["graph"][0] for a, b in adjacent],
        "sample_to_next_graph": [b["graph"][0]-a["sample"][1] for a, b in adjacent],
    }.items():
        result[key + "_ms_median"] = median_ms(values)
    pairs = []
    for a, b in adjacent:
        ls, le = b["launch"]
        fs, fe = a["forward"][0][0], a["forward"][-1][1]
        physical = overlap(b["launch"], a["forward_kernel_intervals"])
        span = overlap(b["launch"], a["forward"])
        verdict = ("OVERLAP" if physical > 0 else "PREQUEUED" if le <= fs else
                   "AFTER_FORWARD" if ls >= fe else "WITHIN_FORWARD_SPAN")
        pairs.append(dict(batch_n=a["batch_id"], batch_next=b["batch_id"],
                          verdict=verdict, launch_overlap_physical_kernel_us=physical,
                          launch_overlap_forward_us=span, launch_after_forward_end_us=ls-fe))
    result.update(pairs=pairs, diagnostic_only=True)
    return result


def device_activity(data, frames):
    """Count copy/memset engines too; kernel gaps alone are not GPU idle."""
    if not frames:
        return dict(span_us=0, device_idle_us=None, device_idle_percent=None)
    lo = frames[0]["graph"][0]
    hi = max(f["sample"][1] for f in frames)
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    busy = merge([(max(lo, e["ts"]), min(hi, end(e))) for e in events
                  if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
                  and e["ts"] < hi and end(e) > lo])
    idle = max(0, hi-lo-sum(b-a for a, b in busy))
    return dict(span_us=hi-lo, device_idle_us=idle,
                device_idle_percent=100*idle/(hi-lo) if hi > lo else None)


def draw_svg(path, frames, summary):
    """Display true CPU API intervals and measured GPU spans separately."""
    if not frames:
        Path(path).write_text('<svg xmlns="http://www.w3.org/2000/svg" width="700" height="80">'
                             '<text x="12" y="40">UNKNOWN: no complete labelled dispatches</text></svg>')
        return
    shown = frames[:16]
    lo = min(f["plan"][0] for f in shown)
    hi = max(max(f["sample"][1], f["commit"][1]) for f in shown)
    scale = 1300/max(hi-lo, 1)
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="250">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="12" y="25">Dispatch-labelled CPU/GPU timeline (diagnostic only)</text>',
           '<text x="12" y="85">GPU forward</text>', '<text x="12" y="145">CPU launch</text>']
    for f in shown:
        for spans, y, color in [(f["forward"], 65, "#339966"), ([f["launch"]], 125, "#ee9933")]:
            for a, b in spans:
                out.append(f'<rect x="{160+(a-lo)*scale:.3f}" y="{y}" width="{max(.6,(b-a)*scale):.3f}" '
                           f'height="25" fill="{color}"><title>batch {f["batch_id"]}: {a}..{b} us</title></rect>')
    text = " | ".join(f"{p['batch_n']} to {p['batch_next']}: {p['verdict']}" for p in summary.get("pairs", [])[:8])
    out += [f'<text x="12" y="200">{html.escape(text)}</text>',
            '<text x="12" y="230">Forward spans may contain gaps; OVERLAP requires a physical kernel intersection.</text>', '</svg>']
    Path(path).write_text("\n".join(out), encoding="utf-8")


def report_trace(path, output_dir=None):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    frames, attribution = _attribute(data)
    summary = summarize_frames(frames)
    summary.update(attribution=attribution, device_activity=device_activity(data, frames),
                   status="ok" if frames else "unknown", schema="nq-dispatch-v1")
    if not frames:
        summary["reason"] = "Missing/partial dispatch markers or CUDA correlations; no consumer-scope fallback"
    directory = Path(output_dir) if output_dir else path.parent
    directory.mkdir(parents=True, exist_ok=True)
    name = path.name.removesuffix(".gz").removesuffix(".json").removesuffix(".trace")
    target = directory / (name + ".summary.json")
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_svg(directory / (name + ".timeline.svg"), frames, summary)
    return summary, target


def derived_events(frames, pid, mode):
    result = [{"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
               "args": {"name": f"{mode}: plan includes prepare; launch is CPU; forward/sample are GPU"}}]
    tracks = {1: "CPU: plan / launch / commit", 2: "GPU: forward / sample", 3: "GPU: cuda_graph"}
    for tid, label in tracks.items():
        result.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": label}})
    for i, frame in enumerate(frames):
        def add(name, tid, span):
            result.append({"ph": "X", "cat": "nq_derived", "pid": pid, "tid": tid,
                           "name": name, "ts": span[0], "dur": span[1] - span[0],
                           "args": {"batch_id": frame["batch_id"],
                                    "source": "native CUDA correlation / CPU markers", "may_contain_kernel_gaps": True}})
        for name in ("plan", "launch", "commit"):
            add(f"{name} {i}", 1, frame[name])
        for interval in frame["forward"]:
            add(f"forward {i}", 2, interval)
        add(f"sample {i}", 2, frame["sample"])
        add("cuda_graph", 3, frame["graph"])
    return result


def bounded_trace(data, frames, mode, max_steps=6, max_bytes=20*MIB):
    """Keep a continuous time window and complete correlation groups."""
    # Start after the first profiled execution, allowing pipeline startup at
    # the profiler boundary to stay outside the saved comparison window.
    candidates = frames[1:1+max_steps]
    if len(candidates) < 4:
        raise RuntimeError(f"Only {len(candidates)} complete steady batches; need at least 4. Check CUPTI/markers")
    original = data["traceEvents"]
    numeric_pids = [e["pid"] for e in original if isinstance(e.get("pid"), int)]
    pid = max(numeric_pids, default=0) + 100
    for count in range(len(candidates), 3, -1):
        selected = candidates[:count]
        lo = min(f["plan"][0] for f in selected)
        hi = max(max(f["commit"][1], f["sample"][1], f["graph"][1], f["lm_head"][1]) for f in selected)
        # Keep original X events that intersect the window, with unmodified
        # timestamps/durations. A few edge context events may extend past it.
        kept = [e for e in original if e.get("ph") not in ("M", "s", "f", "t")
                and "ts" in e and e["ts"] <= hi and end(e) >= lo]
        flows = defaultdict(list)
        for e in original:
            if e.get("ph") in ("s", "f", "t"):
                flows[(e.get("cat"), e.get("name"), str(e.get("id")), str(e.get("id2")))].append(e)
        for group in flows.values():
            if all(lo <= e["ts"] <= hi for e in group) and any(e["ph"] == "s" for e in group) and any(e["ph"] == "f" for e in group):
                kept.extend(group)
        active_pids = {e.get("pid") for e in kept}
        kept = [e for e in original if e.get("ph") == "M" and e.get("pid") in active_pids] + kept
        # A selected graph must retain EVERY native kernel from the recording.
        corr = {c for f in selected for c in f["graph_correlations"]}
        expected = Counter(e["args"]["correlation"] for e in original if e.get("cat") == "kernel" and e.get("args", {}).get("correlation") in corr)
        actual = Counter(e["args"]["correlation"] for e in kept if e.get("cat") == "kernel" and e.get("args", {}).get("correlation") in corr)
        if expected != actual:
            raise RuntimeError("Window dropped kernels from a selected graph")
        kept.extend(derived_events(selected, pid, mode))
        output = {k: v for k, v in data.items() if k != "traceEvents"}
        output.update(traceEvents=kept, nq_window={"mode": mode, "complete_batch_ids": [f["batch_id"] for f in selected],
                      "start_us": lo, "end_us": hi, "edge_context_may_be_partial": True,
                      "policy": "continuous window; all selected graph kernels; unchanged timestamps and durations",
                      "requested_complete_batches": max_steps, "max_uncompressed_bytes": int(max_bytes)})
        encoded = json.dumps(output, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) <= max_bytes:
            return encoded, selected, output["nq_window"]
    raise RuntimeError("Even four complete batches exceed --max-trace-mb; increase the size cap. No kernels were sampled")
