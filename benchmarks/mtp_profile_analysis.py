"""CPU-only accounting for MTP diagnostic traces and captured event banks."""

from collections import Counter, defaultdict


class GraphEventBank:
    def __init__(self):
        self.events = defaultdict(list)

    def clear(self, key):
        self.events[key].clear()

    def add(self, key, label, start, end):
        self.events[key].append((label, start, end))

    def collect(self, replays):
        if any(n > 1 for n in Counter(replays).values()):
            raise ValueError("Repeated graph replay overwrites event timestamps within one step")
        result = defaultdict(float)
        for key in replays:
            for label, start, end in self.events[key]:
                result[label] += start.elapsed_time(end)
        return dict(result)


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def overlap(intervals, windows):
    return sum(max(0, min(b, d)-max(a, c)) for a, b in union(intervals) for c, d in union(windows))


def signature(event):
    args = event.get("args", {})
    return event["name"], tuple(args.get("grid", [])), tuple(args.get("block", []))


def summarize_trace(events):
    physical = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    kernels = [e for e in physical if e["cat"] == "kernel"]
    labels = {"target", "target_head", "draft", "draft_head", "cache_only", "snapshot", "rollback"}
    scopes = [e for e in events if e.get("cat") == "gpu_user_annotation" and e.get("name") in {"mtp."+s for s in labels}]

    def contains(scope, event):
        return (scope.get("pid") == event.get("pid") and scope.get("tid") == event.get("tid")
                and scope["ts"]-.1 <= event["ts"]
                and event["ts"]+event["dur"] <= scope["ts"]+scope["dur"]+.1)

    # Target heads execute outside the graph in this very trace. Restrict the
    # fingerprint match to the draft graph; no cross-run/eager kernel transplant.
    heads = [e for e in scopes if e["name"] == "mtp.target_head"]
    head_signatures = {signature(e) for e in kernels if any(contains(s, e) for s in heads)}
    groups, categories, calls = defaultdict(float), defaultdict(lambda: defaultdict(float)), Counter()
    from bench_one_batch import kernel_category
    for event in kernels:
        parents = [s for s in scopes if contains(s, event)]
        label = min(parents, key=lambda s: s["dur"])["name"][4:] if parents else "other"
        if label == "draft" and signature(event) in head_signatures:
            label = "draft_head"
        groups[label] += event["dur"]
        categories[label][kernel_category(event["name"])] += event["dur"]
        calls[label] += 1
    intervals = union((e["ts"], e["ts"]+e["dur"]) for e in physical)
    busy = sum(b-a for a, b in intervals)
    span = intervals[-1][1]-intervals[0][0] if intervals else 0
    waits = [(e["ts"], e["ts"]+e["dur"]) for e in events
             if e.get("ph") == "X" and e.get("cat") == "user_annotation" and e.get("name") == "mtp.decision_wait"]
    wait = sum(b-a for a, b in union(waits))
    gpu_overlap = overlap(intervals, waits)
    return dict(kernel_sum_us=sum(e["dur"] for e in kernels), kernel_groups_us=dict(groups),
                kernel_categories_us={k: dict(v) for k, v in categories.items()}, kernel_calls=dict(calls),
                device_busy_union_us=busy, device_span_us=span, device_internal_gap_us=span-busy,
                decision_wait_cpu_us=wait, decision_wait_gpu_overlap_us=gpu_overlap,
                decision_wait_without_gpu_us=max(0, wait-gpu_overlap),
                decision_wait_calls=len(waits), physical_device_events=len(physical),
                head_signatures=[dict(name=s[0], grid=s[1], block=s[2]) for s in sorted(head_signatures)])
