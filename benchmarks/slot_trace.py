"""Bounded-memory CUPTI correlation -> GPU slot ranges, used during export.

No torch dependency. GPU ranges use physical GPU timestamps, never CPU scope
durations. They are display envelopes, not additional work or fused kernels.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import re

PHYSICAL = {"kernel", "gpu_memcpy", "gpu_memset"}
APIS = {"cuda_runtime", "cuda_driver"}
FLOWS = {"s", "f", "t", "S", "T", "F", "b", "e", "n"}
SCOPE = re.compile(r"^nq/cpu/(launch_forward|forward|sample_slot|sampler|lm_head|d2h_enqueue)/slot(\d+)$")


def scan_slot_groups(source, records):
    scopes, apis = [], defaultdict(list)
    gpu = defaultdict(dict)
    active = set()
    counts = Counter()
    for kind, event in records(source):
        if kind != "event" or event.get("ph") != "X":
            continue
        cat = event.get("cat")
        if cat not in PHYSICAL | APIS | {"user_annotation", "cpu_op"}:
            continue
        active.add((event.get("pid"), event.get("tid")))
        ts, end = event["ts"], event["ts"] + event.get("dur", 0)
        corr = event.get("args", {}).get("correlation")
        if cat in PHYSICAL:
            counts[cat] += 1
            if cat == "kernel" and "_norm_kernel" in event.get("name", ""):
                counts["fused_norm_kernels"] += 1
            if corr is None:
                counts["physical_without_correlation"] += 1
                continue
            name = event.get("name", "")
            subtype = "DtoH" if cat == "gpu_memcpy" and ("DtoH" in name or "D2H" in name) else cat
            key = event["pid"], event["tid"], subtype
            row = gpu[corr].get(key)
            if row is None:
                gpu[corr][key] = [ts, end, 1]
            else:
                row[0], row[1], row[2] = min(row[0], ts), max(row[1], end), row[2]+1
        elif cat in APIS and corr is not None:
            apis[(event["pid"], event["tid"])].append((ts, end, corr, event.get("name", "")))
        elif cat == "user_annotation":
            match = SCOPE.match(event.get("name", ""))
            if match:
                scopes.append((match[1], int(match[2]), event["pid"], event["tid"], ts, end))
    for rows in apis.values():
        rows.sort(key=lambda a: a[0])
    starts = {track: [row[0] for row in rows] for track, rows in apis.items()}

    def contained(scope):
        _, _, pid, tid, lo, hi = scope
        rows = apis.get((pid, tid), [])
        times = starts.get((pid, tid), [])
        return [a for a in rows[bisect_left(times, lo):bisect_right(times, hi)] if a[1] <= hi+0.001]

    groups = []
    def envelope(name, slot, correlations, allowed=None):
        tracks = {}
        for corr in set(correlations):
            for (pid, tid, kind), row in gpu.get(corr, {}).items():
                if allowed is not None and kind not in allowed:
                    continue
                key = pid, tid
                old = tracks.get(key)
                if old is None:
                    tracks[key] = list(row)
                else:
                    old[0], old[1], old[2] = min(old[0], row[0]), max(old[1], row[1]), old[2]+row[2]
        for (pid, tid), (lo, hi, n) in tracks.items():
            groups.append(dict(ph="X", cat="gpu_user_annotation", name=f"{name}_slot_{slot}",
                pid=pid, tid=tid, ts=lo, dur=hi-lo, args=dict(slot=slot, physical_events=n,
                derived_from="CUPTI correlation and physical GPU timestamps",
                note="Display envelope; do not add its duration to child kernel time")))

    has_launch = any(s[0] == "launch_forward" for s in scopes)
    has_sample = any(s[0] == "sample_slot" for s in scopes)
    graph_seen = set()
    for scope in scopes:
        kind, slot = scope[:2]
        calls = contained(scope)
        corrs = [a[2] for a in calls]
        if kind == ("launch_forward" if has_launch else "forward"):
            envelope("forward", slot, corrs, {"kernel", "gpu_memcpy", "gpu_memset"})
        if kind in ("launch_forward", "forward"):
            for api in calls:
                if "cudaGraphLaunch" not in api[3] or api[2] in graph_seen:
                    continue
                graph_seen.add(api[2])
                envelope("cuda_graph", slot, [api[2]], {"kernel"})
        elif kind == ("sample_slot" if has_sample else "sampler"):
            envelope("sample", slot, corrs, {"kernel"})
        elif kind == "lm_head":
            envelope("lm_head", slot, corrs, {"kernel"})
        elif kind == "d2h_enqueue":
            envelope("D2H", slot, corrs, {"DtoH"})
    all_graphs = {a[2] for rows in apis.values() for a in rows if "cudaGraphLaunch" in a[3]}
    counts["graph_launches"] = len(all_graphs)
    counts["unassigned_graph_launches"] = len(all_graphs - graph_seen)
    counts["graph_groups"] = sum(g["name"].startswith("cuda_graph_slot_") for g in groups)
    counts["graph_kernel_events"] = sum(g["args"]["physical_events"] for g in groups
                                        if g["name"].startswith("cuda_graph_slot_"))
    counts["gpu_ranges"] = len(groups)
    return groups, active, dict(counts)
