"""Lossless operator export with explicit CPU launches and GPU Graph slots.

Only GPU *annotations* get a separate display lane. CPU operators, CUDA APIs,
physical GPU kernels/copies and their original timestamps/pids/tids are kept.
No torch import: export and its integrity tests also run without CUDA.
"""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re

PHYSICAL = {"kernel", "gpu_memcpy", "gpu_memset"}
APIS = {"cuda_runtime", "cuda_driver"}
REPLAY = re.compile(r"^CUDAGraph\.replay/slot(\d+)/bs(\d+)$")
LEGACY = re.compile(r"^nq/cpu/(?:launch_forward|forward)/slot(\d+)$")


def trace_records(path):
    """Stream a Chrome trace, rejecting incomplete JSON (including footer)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    decoder = json.JSONDecoder()
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        buffer = ""
        while True:
            chunk = stream.read(1024*1024)
            if not chunk:
                raise ValueError(f"No traceEvents array: {path}")
            buffer += chunk
            match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
            if match:
                prefix = buffer[:match.end()-1]
                header = json.loads(prefix + "[]}")
                header.pop("traceEvents", None)
                yield "header", header
                buffer = buffer[match.end():]
                break
            if len(buffer) > 2*1024*1024:
                raise ValueError("Trace header exceeds 2MiB")
        position = 0
        need_comma = after_comma = False
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position == len(buffer):
                buffer = stream.read(1024*1024)
                position = 0
                if not buffer:
                    raise ValueError("Truncated traceEvents array")
                continue
            if buffer[position] == "]":
                if after_comma:
                    raise ValueError("Trailing comma in traceEvents")
                footer = buffer[position+1:] + stream.read(2*1024*1024+1)
                if len(footer) > 2*1024*1024:
                    raise ValueError("Trace footer exceeds 2MiB")
                json.loads(prefix + "[]" + footer)
                return
            if need_comma:
                if buffer[position] != ",":
                    raise ValueError("Missing event separator")
                position += 1
                need_comma, after_comma = False, True
                continue
            try:
                event, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                more = stream.read(1024*1024)
                if not more:
                    raise ValueError("Truncated or invalid event") from None
                buffer = buffer[position:] + more
                position = 0
                continue
            if not isinstance(event, dict):
                raise ValueError("Trace events must be objects")
            yield "event", event
            position, need_comma, after_comma = end, True, False
            if position > 1024*1024:
                buffer, position = buffer[position:], 0


def inspect_trace(source):
    scopes, legacy, launches = defaultdict(list), defaultdict(list), []
    gpu = defaultdict(dict)
    counts = Counter()
    tracks, gpu_tracks, annotation_tracks = set(), set(), set()
    for kind, e in trace_records(source):
        if kind != "event":
            continue
        track = e.get("pid"), e.get("tid", 0)
        tracks.add(track)
        cat = e.get("cat", "uncategorized")
        counts[cat] += 1
        if e.get("ph") != "X":
            continue
        if "ts" not in e:
            raise ValueError("Complete trace event has no timestamp")
        lo, hi = e["ts"], e["ts"] + e.get("dur", 0)
        if cat == "gpu_user_annotation":
            annotation_tracks.add(track)
        if cat in PHYSICAL:
            gpu_tracks.add(track)
            corr = e.get("args", {}).get("correlation")
            if corr is not None and cat == "kernel":
                row = gpu[corr].get(track)
                if row is None:
                    gpu[corr][track] = [lo, hi, 1]
                else:
                    row[0], row[1], row[2] = min(row[0], lo), max(row[1], hi), row[2]+1
        elif cat in APIS and "GraphLaunch" in e.get("name", ""):
            launches.append(e)
        elif cat == "user_annotation":
            match = REPLAY.match(e.get("name", ""))
            old = LEGACY.match(e.get("name", ""))
            if match:
                scopes[track].append((lo, hi, int(match[1]), int(match[2])))
            elif old:
                legacy[track].append((lo, hi, int(old[1]), None))
    for table in (scopes, legacy):
        for rows in table.values():
            rows.sort()
    matched, missing = [], []
    for api in sorted(launches, key=lambda e: e["ts"]):
        track = api["pid"], api["tid"]
        rows = scopes.get(track) or legacy.get(track, [])
        candidates = [r for r in rows if r[0] <= api["ts"] and
                      api["ts"]+api.get("dur", 0) <= r[1]+0.001]
        owner = min(candidates, key=lambda r: r[1]-r[0]) if candidates else None
        corr = api.get("args", {}).get("correlation")
        children = gpu.get(corr, {}) if corr is not None else {}
        if owner is None or not children:
            missing.append(dict(name=api["name"], ts=api["ts"], correlation=corr,
                                reason="missing replay slot scope" if owner is None else "missing GPU kernels"))
            continue
        matched.append(dict(api=api, slot=owner[2], batch=owner[3], gpu=children))
    # Runtime/driver APIs may refer to the same launch: do not double group it.
    unique, seen = [], set()
    for row in matched:
        key = row["api"].get("args", {}).get("correlation")
        if key not in seen:
            unique.append(row)
            seen.add(key)
    order = [row["slot"] for row in unique]
    return dict(counts=dict(counts), matched=unique, missing=missing, tracks=tracks,
        gpu_tracks=gpu_tracks, annotation_tracks=annotation_tracks, slot_order=order,
        slot_alternating=all(a != b for a,b in zip(order, order[1:])))


def export_full_trace(source, destination, *, mode, strict=True):
    """One automatic export. Never filters CPU ops, kernels, copies or APIs."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    info = inspect_trace(source)
    slots = set(info["slot_order"])
    if strict and (info["missing"] or not info["matched"]):
        raise ValueError(f"Cannot associate every Graph launch with slot + GPU kernels: {info['missing'][:3]}")
    if strict and mode == "dual" and (slots != {0, 1} or not info["slot_alternating"]):
        raise ValueError(f"Expected alternating slot0/1 launches: {info['slot_order']}")
    if strict and mode == "single" and slots != {0}:
        raise ValueError(f"Expected slot0 only: {info['slot_order']}")

    tracks = set(info["tracks"])
    next_tid = max([int(t) for _,t in tracks if isinstance(t, int)] + [1000000])+100
    def display_tid(pid):
        nonlocal next_tid
        next_tid += 1
        while (pid, next_tid) in tracks:
            next_tid += 1
        tracks.add((pid, next_tid))
        return next_tid
    annotation_lanes = {t: display_tid(t[0]) for t in sorted(info["annotation_tracks"], key=str)}
    cpu_lanes = {pid: display_tid(pid) for pid in {r["api"]["pid"] for r in info["matched"]}}
    renames = {t: f"CUDA stream {t[1]} | kernels + Graph slots" for t in info["gpu_tracks"]}
    renames.update({(r["api"]["pid"], r["api"]["tid"]): "CPU engine | torch ops + CUDA runtime"
                    for r in info["matched"]})
    metadata = []
    def name(pid, tid, title, order):
        metadata.extend([dict(ph="M", name="thread_name", pid=pid, tid=tid, args=dict(name=title)),
            dict(ph="M", name="thread_sort_index", pid=pid, tid=tid, args=dict(sort_index=order))])
    for (pid, tid), title in renames.items():
        name(pid, tid, title, 10)
    for pid, tid in cpu_lanes.items():
        name(pid, tid, "CPU cudaGraphLaunch | display lane", 0)
    for (pid, old_tid), tid in annotation_lanes.items():
        name(pid, tid, f"GPU profiler scopes | original stream {old_tid}", 100)

    added = []
    graph_kernel_events = 0
    for row in info["matched"]:
        api, slot = row["api"], row["slot"]
        args = dict(api.get("args", {}), slot=slot, batch=row["batch"], display_only=True,
                    original_pid=api["pid"], original_tid=api["tid"])
        added.append(dict(ph="X", cat="nq_display_cuda_api", name=f"cudaGraphLaunch / slot{slot}",
            pid=api["pid"], tid=cpu_lanes[api["pid"]], ts=api["ts"], dur=api["dur"], args=args))
        for (pid, tid), (lo, hi, count) in row["gpu"].items():
            graph_kernel_events += count
            added.append(dict(ph="X", cat="gpu_user_annotation", name=f"cuda_graph_slot{slot}",
                pid=pid, tid=tid, ts=lo, dur=hi-lo, args=dict(slot=slot, kernels=count,
                correlation=api.get("args", {}).get("correlation"), display_only=True,
                derived_from="CUPTI GPU kernel timestamps; not CPU replay duration")))
    tmp = destination.with_name(destination.name+".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    moved, original_events, first = 0, 0, True
    try:
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as out:
            def emit(e):
                nonlocal first
                if not first:
                    out.write(",")
                out.write(json.dumps(e, ensure_ascii=False, separators=(",", ":")))
                first = False
            for kind, e in trace_records(source):
                if kind == "header":
                    e["nano_qwen_full_trace"] = dict(operators_filtered=False, mode=mode,
                        native_gpu_scopes="retained on separate annotation lanes",
                        graph_ranges="CUPTI GPU timestamps; display envelopes, not extra kernels",
                        cpu_launch_lane="copies real CUDA API intervals for visibility; not another CPU thread")
                    out.write(json.dumps(e, ensure_ascii=False, separators=(",", ":"))[:-1]+',"traceEvents":[')
                    # Parent ranges must precede children when their start
                    # timestamps are identical. Otherwise a Chrome/Perfetto
                    # importer can place the first kernel above its parent.
                    for display in metadata + sorted(added, key=lambda x: (x["ts"], -x["dur"])):
                        emit(display)
                    continue
                original_events += 1
                track = e.get("pid"), e.get("tid", 0)
                if e.get("ph") == "M" and track in renames and e.get("name") in ("thread_name", "thread_sort_index"):
                    # Replaced by equivalent metadata with descriptive names.
                    continue
                if e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X":
                    e = dict(e, tid=annotation_lanes[track], args=dict(e.get("args", {}),
                        original_pid=track[0], original_tid=track[1], display_track="annotation_only"))
                    moved += 1
                emit(e)
            out.write("]}")
        tmp.replace(destination)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dict(trace=str(destination), compressed_bytes=destination.stat().st_size,
        original_events=original_events, original_counts=info["counts"],
        cpu_graph_launches=len(info["matched"]), gpu_graph_ranges=sum(len(r["gpu"]) for r in info["matched"]),
        graph_kernel_events=graph_kernel_events, slot_counts=dict(Counter(info["slot_order"])),
        slot_order=info["slot_order"], slot_alternating=info["slot_alternating"],
        moved_gpu_annotations=moved, missing=info["missing"], operators_filtered=False)
