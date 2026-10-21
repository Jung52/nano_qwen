#!/usr/bin/env python3
"""Fixed-batch baseline/dual-slot A/B; put this file in nano_qwen/benchmarks.

Each variant/case/pair gets a fresh process and one engine. Timings include
pipeline startup/drain, exclude model load, tokenization, profiler and JSON I/O.
Prefill graphs/prefix caching are off; decode graphs use exact batch sizes.
The existing workload KV policy reserves identical page capacity for both modes.
--trace adds short CPU+CUDA diagnostic runs, labelled slot scopes, Perfetto
traces, overlap analysis and an offline trace_compare.html timeline viewer.
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16])
    p.add_argument("--inputs", nargs="+", type=int, default=[128])
    p.add_argument("--outputs", nargs="+", type=int, default=[256])
    p.add_argument("--modes", nargs="+", choices=["graph", "eager"], default=["graph"])
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--pairs", type=int, default=1, help="Fresh-process pairs; order alternates AB/BA")
    p.add_argument("--temperature", type=float, default=1e-6, help="1e-6: greedy; 1: stochastic production sampler")
    p.add_argument("--mem-util", type=float, default=0.65)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=1800, help="Seconds per child process")
    p.add_argument("--trace", action="store_true", help="Additional diagnostic processes; never timed as A/B samples")
    p.add_argument("--trace-only", action="store_true", help="Only collect diagnostic traces; no formal speedup")
    p.add_argument("--profile-decode-steps", type=int, default=32,
                   help="Trace at most this many decode tokens per request; includes prime/drain")
    p.add_argument("--record-shapes", action="store_true", help="Diagnostic profiler only; adds overhead")
    p.add_argument("--with-stack", action="store_true", help="Diagnostic profiler only; adds overhead")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--worker", choices=["baseline", "dual"], help=argparse.SUPPRESS)
    p.add_argument("--phase", choices=["timing", "trace"], default="timing", help=argparse.SUPPRESS)
    p.add_argument("--result", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if min(a.batches + a.inputs) < 1 or min(a.outputs) < 2:
        p.error("batches/inputs must be positive; outputs must be >= 2")
    if a.warmup < 1 or min(a.rounds, a.pairs, a.timeout, a.profile_decode_steps) < 1:
        p.error("warmup/rounds/pairs/timeout must be positive")
    if not (0 < a.mem_util < 1) or not (math.isfinite(a.temperature) and a.temperature > 1e-10):
        p.error("invalid memory utilization or temperature")
    if "graph" in a.modes and any(b not in (1, 2, 4, 8) and (b % 16 or b > 512) for b in a.batches):
        p.error("exact graph batches: 1/2/4/8 or multiples of 16 through 512")
    return a


def measure(engine, a, torch, seed, profile=False):
    from nano_qwen.engine.sequence import Sequence
    from nano_qwen.sampling_params import SamplingParams
    from bench_fp8_pressure import memory_snapshot

    b, n, o = a.batches[0], a.inputs[0], a.outputs[0]
    if not engine.is_finished():
        raise RuntimeError("Previous run did not drain")
    rng = random.Random(seed)
    prompts = [[rng.randrange(engine.config.hf_config.vocab_size) for _ in range(n)] for _ in range(b)]
    seqs = [Sequence(prompt, SamplingParams(temperature=a.temperature, max_tokens=o, ignore_eos=True))
            for prompt in prompts]
    runner = engine.model_runner
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    runner.cuda_graphs.reset_dispatch_stats()
    pipeline = engine._dual_pipeline
    before = dict(pipeline.stats) if pipeline is not None else {}
    mark = torch.profiler.record_function if profile else lambda name: nullcontext()
    started = time.perf_counter()
    with mark("nq/phase/prefill"):
        for seq in seqs:
            engine.scheduler.add(seq)
        engine.step()  # Budget admits the entire prompt batch in one prefill.
        prefill_lengths = [seq.num_completion_tokens for seq in seqs]
        torch.cuda.synchronize()  # One common phase boundary, never one sync per tick.
    prefill_end = time.perf_counter()
    ticks = 0
    with mark("nq/phase/decode"):
        if profile:
            while not engine.is_finished():
                with mark(f"nq/tick/{ticks:04d}"):
                    engine.step()
                ticks += 1
        else:
            while not engine.is_finished():
                engine.step()  # Includes the dual pipeline's empty prime/drain ticks.
                ticks += 1
        torch.cuda.synchronize()
    ended = time.perf_counter()

    if prefill_lengths != [1] * b:
        raise RuntimeError(f"Not a single complete prefill: {prefill_lengths}")
    if any(not s.is_finished or s.num_completion_tokens != o for s in seqs):
        raise RuntimeError("Incorrect output count/lifecycle")
    if runner.input_batch.seq_id_to_slot or engine.scheduler.block_manager.used_block_ids:
        raise RuntimeError("Request slots/KV pages did not drain")
    if pipeline is not None and (pipeline.pending or pipeline.requests or any(s.busy for s in runner.decode_slots)):
        raise RuntimeError("Dual execution slots did not drain")
    paths = runner.cuda_graphs.dispatch_stats()
    decode = paths.get("decode", {})
    expected = o - 1
    kind = "graph_calls" if a.modes[0] == "graph" else "eager_calls"
    other = "eager_calls" if kind == "graph_calls" else "graph_calls"
    if (decode.get(kind) != expected or decode.get(other, 0) != 0
            or decode.get("batch_counts") != {b: expected}):
        raise RuntimeError(f"Decode path/batch changed: {decode}")
    dec, e2e = ended - prefill_end, ended - started
    return dict(seed=seed, prompt_sha256=digest(prompts),
                output_sha256=digest([s.completion_token_ids for s in seqs]),
                output_tokens=b * o, decode_tokens=b * (o - 1), host_decode_ticks=ticks,
                ttft_ms=(prefill_end - started) * 1000, decode_ms=dec * 1000,
                tpot_ms=dec * 1000 / (o - 1), decode_tok_s=b * (o - 1) / dec,
                e2e_ms=e2e * 1000, output_tok_s=b * o / e2e, dispatch=paths,
                pipeline={k: v - before[k] for k, v in pipeline.stats.items()} if pipeline else {},
                memory=memory_snapshot(torch))


@contextmanager
def profile_labels(engine, torch):
    """Temporary annotations only: no production source edits or extra CUDA waits."""
    from nano_qwen.engine.async_output import AsyncModelOutput
    r = engine.model_runner
    dual = engine._dual_pipeline
    with ExitStack() as undo:
        def patch(obj, attr, replacement):
            own = attr in vars(obj)
            old = getattr(obj, attr)
            setattr(obj, attr, replacement)
            undo.callback(setattr if own else delattr, obj, attr, *([old] if own else []))

        def wrap(obj, attr, label):
            original = getattr(obj, attr)
            @functools.wraps(original)
            def call(*args, **kwargs):
                name = label(*args, **kwargs) if callable(label) else label
                with torch.profiler.record_function(name):
                    return original(*args, **kwargs)
            patch(obj, attr, call)

        def suffix():
            return f"/slot{r._active_slot_id if dual is not None else 0}"

        # Bridge existing CPU trace scopes into Kineto's clock domain; do not
        # combine the old perf_counter Chrome trace with CUDA timestamps.
        for obj in (sys.modules[type(r).__module__], sys.modules[type(engine).__module__]):
            original_trace = obj.trace_event
            @contextmanager
            def bridge(name, cat="bench", args=None, original=original_trace):
                names = {"run_model": "forward", "d2h_copy": "d2h_enqueue"}
                label = "nq/cpu/" + names.get(name, name) + suffix()
                with torch.profiler.record_function(label), original(name, cat, args):
                    yield
            patch(obj, "trace_event", bridge)
        wrap(AsyncModelOutput, "get_output", "nq/cpu/get_output_wait")
        wrap(r.model, "compute_logits", lambda *x, **kw: "nq/cpu/lm_head" + suffix())
        wrap(r.cuda_graphs, "run_decode", lambda *x, **kw: "nq/cpu/graph_dispatch" + suffix())
        if dual is None:
            wrap(engine.scheduler, "postprocess", "nq/cpu/commit/slot0")
        else:
            wrap(dual, "_plan_decode", "nq/cpu/schedule/dual")
            wrap(dual, "_plan_prefill", "nq/cpu/schedule/prefill")
            wrap(dual, "_commit_oldest", lambda: f"nq/cpu/commit/slot{dual.pending[0].slot_id}")
            wrap(r, "launch_slot", lambda seqs, prefill, slot_id: f"nq/cpu/launch_slot/slot{slot_id}")
            wrap(r, "sample_slot", lambda slot_id, ids: f"nq/cpu/sample_slot/slot{slot_id}")
        yield


def gpu_kind(event):
    """Only physical GPU activities, never nested GPU annotations."""
    cat, name = event.get("cat"), event.get("name", "").lower()
    if event.get("ph") != "X":
        return None
    if cat == "kernel":
        return "kernel"
    if cat == "gpu_memset":
        return "memset"
    if cat == "gpu_memcpy":
        if any(k in name for k in ("dtoh", "d2h", "device -> pinned", "device -> pageable")):
            return "d2h"
        if any(k in name for k in ("htod", "h2d", "pinned -> device", "pageable -> device")):
            return "h2d"
        return "memcpy"
    return None


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def intersection(a, b):
    a, b = merge_intervals(a), merge_intervals(b)
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if end > start:
            result.append([start, end])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def duration(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def analyze_trace(events):
    result = {}
    for phase in ("prefill", "decode"):
        windows = [e for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation"
                   and e.get("name") == f"nq/phase/{phase}"]
        if len(windows) != 1:
            raise RuntimeError(f"Missing/ambiguous profiler phase: {phase}")
        start, end = windows[0]["ts"], windows[0]["ts"] + windows[0]["dur"]
        def clips(rows):
            return [(max(start, e["ts"]), min(end, e["ts"] + e["dur"])) for e in rows
                    if e["ts"] < end and e["ts"] + e["dur"] > start]
        scoped = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e
                  and e["ts"] < end and e["ts"] + e["dur"] > start]
        gpu = [e for e in scoped if gpu_kind(e)]
        compute = clips([e for e in gpu if gpu_kind(e) == "kernel"])
        d2h = clips([e for e in gpu if gpu_kind(e) == "d2h"])
        physical = clips(gpu)
        waits = clips([e for e in scoped if e.get("cat") == "user_annotation"
                       and e.get("name") == "nq/cpu/get_output_wait"])
        commits = clips([e for e in scoped if e.get("cat") == "user_annotation"
                         and e.get("name", "").startswith("nq/cpu/commit/")])
        sync = clips([e for e in scoped if e.get("cat") == "cuda_runtime"
                      and e.get("name") in ("cudaEventSynchronize", "cuEventSynchronize")])
        busy, merged = duration(physical), merge_intervals(physical)
        span = merged[-1][1] - merged[0][0] if merged else 0
        kinds, kernels, tracks, cpu = {}, {}, {}, {}
        timeline = []
        for e in scoped:
            kind, name = gpu_kind(e), e.get("name", "")
            s, t = max(start, e["ts"]), min(end, e["ts"] + e["dur"])
            if kind:
                kinds[kind] = kinds.get(kind, 0) + (t-s)/1000
                if kind == "kernel":
                    row = kernels.setdefault(name, dict(name=name, calls=0, sum_ms=0))
                    row["calls"] += 1
                    row["sum_ms"] += (t-s)/1000
                track = f"GPU {e.get('args', {}).get('device', 0)} · stream {e.get('args', {}).get('stream', e.get('tid'))}"
                tracks.setdefault(track, set()).add(kind)
            elif e.get("cat") == "user_annotation" and name.startswith("nq/cpu/"):
                stage = name.split("/")[2]
                cpu.setdefault(stage, []).append((s,t))
                if stage not in ("prepare_inputs", "forward", "sampler", "d2h_enqueue", "get_output_wait", "commit", "schedule"):
                    continue
                track = f"CPU {e.get('tid')} · {stage}"
                kind = "slot1" if "/slot1" in name else "cpu"
            else:
                continue
            timeline.append([track, (s-start)/1000, (t-s)/1000, kind, name])
        wait_compute = duration(intersection(waits, compute))
        commit_compute = duration(intersection(commits, compute)) - duration(intersection(intersection(commits, waits), compute))
        result[phase] = dict(profile_wall_ms=(end-start)/1000, device_event_count=len(gpu),
            device_event_sum_ms=sum(kinds.values()), kernel_sum_ms=kinds.get("kernel", 0),
            compute_busy_ms=duration(compute)/1000, device_busy_ms=busy/1000,
            no_device_event_ms=max(0, end-start-busy)/1000,
            device_span_ms=span/1000, internal_device_gap_ms=max(0,span-busy)/1000,
            get_output_cpu_ms=duration(waits)/1000, cuda_event_sync_cpu_ms=duration(sync)/1000,
            get_output_overlap_compute_ms=wait_compute/1000,
            get_output_overlap_compute_pct=100*wait_compute/duration(waits) if duration(waits) else 0,
            commit_excluding_get_output_overlap_compute_ms=max(0,commit_compute)/1000,
            d2h_sum_ms=kinds.get("d2h", 0), d2h_busy_ms=duration(d2h)/1000,
            d2h_overlap_compute_ms=duration(intersection(d2h, compute))/1000,
            gpu_tracks={k: sorted(v) for k,v in tracks.items()}, kinds_sum_ms=kinds,
            cpu_scope_union_ms={k: duration(v)/1000 for k,v in cpu.items()},
            top_kernels=sorted(kernels.values(), key=lambda x: -x["sum_ms"])[:30], timeline=timeline)
    return result


def name_trace_tracks(data, variant):
    """Preserve Kineto pids/tids/flows; give device tracks explicit names."""
    events = data["traceEvents"]
    host_pids = {e.get("pid") for e in events if e.get("ph") == "X"
                 and e.get("cat") == "user_annotation" and e.get("name") == "nq/phase/decode"}
    metadata = {(e.get("name"), e.get("pid"), e.get("tid", 0)): e for e in events if e.get("ph") == "M"}
    def meta(name, pid, tid, label):
        key = name, pid, tid
        if key in metadata:
            metadata[key].setdefault("args", {})["name"] = label
        else:
            e = dict(ph="M", name=name, pid=pid, tid=tid, args=dict(name=label))
            events.append(e)
            metadata[key] = e
    gpu_tracks = {}
    for e in list(events):
        kind = gpu_kind(e)
        if kind:
            key = (e["pid"], e["tid"], e.get("args", {}).get("device", 0),
                   e.get("args", {}).get("stream", e["tid"]))
            gpu_tracks.setdefault(key, set()).add(kind)
        elif (e.get("ph") == "X" and e.get("cat") == "user_annotation"
                and e.get("name") == "nq/phase/decode"):
            meta("process_name", e["pid"], 0, f"CPU · {variant}")
            meta("thread_name", e["pid"], e["tid"], f"CPU main · {variant}")
    for (pid, tid, device, stream), kinds in gpu_tracks.items():
        label = "CPU / CUDA" if pid in host_pids else f"CUDA device {device}"
        meta("process_name", pid, 0, f"{label} · {variant}")
        meta("thread_name", pid, tid, f"GPU stream {stream} · {','.join(sorted(kinds))}")


def write_trace_comparison(reports, out):
    profiles = [r for r in reports if r["status"] == "ok" and r["args"]["phase"] == "trace"]
    groups, rows = {}, []
    for r in profiles:
        a = r["args"]
        key = (a["modes"][0], a["batches"][0], a["inputs"][0], a["outputs"][0])
        groups.setdefault(key, {})[a["worker"]] = r
    lines = ["# 单slot与双slot的CPU / CUDA时间线", "",
        "正式加速比见summary.md；本报告所有时长来自带profiler的独立诊断进程。",
        "短诊断请求包含prefill、decode启动和排空；与正式长输出的耗时不可直接互换。",
        "CPU阶段行属于同一线程的嵌套范围；GPU行是CUDA stream。CPU提交范围不等于GPU执行时长。",
        "设备空隙只表示没有采集到物理GPU事件，不能直接当作CPU耗时。不同范围/stream可能重叠，禁止相加。", ""]
    for key, variants in groups.items():
        if set(variants) != {"baseline", "dual"}:
            lines.append(f"{key}：trace不完整，未作配对分析。")
            continue
        base, dual = variants["baseline"], variants["dual"]
        if (base["source"]["source_sha256"] != dual["source"]["source_sha256"]
                or base["script_sha256"] != dual["script_sha256"]
                or base["model_metadata"] != dual["model_metadata"]
                or base["engine"]["kv_blocks"] != dual["engine"]["kv_blocks"]
                or base["profile_output_len"] != dual["profile_output_len"]
                or base["profile_measurement"]["prompt_sha256"] != dual["profile_measurement"]["prompt_sha256"]):
            raise RuntimeError(f"Trace pair changed source/KV/output: {key}")
        title = f"{key[0]} · B{key[1]} · 输入{key[2]} · 诊断输出{base['profile_output_len']}"
        lines += [f"## {title}", "", "| Decode诊断指标 | 原路径 | 双slot |", "|---|---:|---:|"]
        fields = [("profile_wall_ms", "诊断窗口ms"), ("kernel_sum_ms", "Kernel累计ms（可重叠）"),
            ("internal_device_gap_ms", "首尾GPU事件之间的空隙ms"), ("get_output_cpu_ms", "get_output范围ms（含等待与读回）"),
            ("get_output_overlap_compute_ms", "get_output期间GPU计算ms"),
            ("commit_excluding_get_output_overlap_compute_ms", "提交范围扣除get_output后与GPU计算重叠ms"),
            ("d2h_sum_ms", "D2H累计ms"), ("d2h_overlap_compute_ms", "D2H与计算重叠ms")]
        a, b = base["trace_analysis"]["decode"], dual["trace_analysis"]["decode"]
        for field, label in fields:
            lines.append(f"| {label} | {a[field]:.4f} | {b[field]:.4f} |")
        lines += ["", "阅读重点：在Perfetto展开CPU main和GPU stream，搜索nq/phase/decode。",
            "双slot中应检查launch_slot/slot1是否先于slot0的commit，以及对应GPU kernel是否覆盖CPU的get_output/commit。",
            "d2h_enqueue只是CPU发起复制；真正传输耗时查看GPU上的Memcpy DtoH。", ""]
        gap = a["internal_device_gap_ms"] - b["internal_device_gap_ms"]
        overlap = b["get_output_overlap_compute_ms"] - a["get_output_overlap_compute_ms"]
        lines.append(f"观测差值：GPU内部空隙减少{gap:.4f}ms；get_output与GPU计算重叠增加{overlap:.4f}ms。")
        if gap > 0 and overlap > 0:
            lines.append("这与下一轮forward覆盖CPU等待/提交、减少GPU供给空隙的解释相符；最终收益需由正式计时确认。")
        else:
            lines.append("该诊断窗口未同时表现出空隙缩小和等待覆盖增加，需查看具体kernel、启动/排空及正式计时。")
        lines += ["提交范围的两种实现包含的CPU工作略有差别，重叠量用于定位，不能视为严格的逐项耗时归因。", ""]
        rows.append(dict(title=title, baseline=base["trace_analysis"], dual=dual["trace_analysis"]))
    save(out / "trace_compare.json", rows)
    (out / "trace_compare.md").write_text("\n".join(lines) + "\n")
    if rows:
        write_timeline_html(rows, out / "trace_compare.html")


def write_timeline_html(rows, path):
    # Offline, reduced overview. Full CPU operators, runtime APIs and correlation
    # arrows remain available in each unmodified-id Kineto trace.json.
    payload = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    template = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CPU / CUDA slot对照</title>
<style>body{margin:24px;font:14px system-ui;color:#192b42;background:#f3f6fa}h1{font-size:23px}header,p{max-width:1200px}section{background:white;padding:18px;margin:16px 0;border-radius:10px;border:1px solid #dbe3ee}label{margin-right:16px}input{width:82px}button,select,input{padding:5px}canvas{width:100%;display:block;background:#fff}#tip{position:fixed;display:none;background:#152840;color:white;padding:9px;border-radius:6px;max-width:650px;pointer-events:none;white-space:pre-wrap}small{color:#526278}.legend span{display:inline-block;padding:4px 10px;margin-right:8px;color:white;border-radius:4px}</style>
<h1>单slot / 双slot：CPU与CUDA时间线</h1>
<p>上下两图使用相同毫秒刻度，各自以所选阶段起点对齐。CPU阶段是同一线程的不同范围；GPU按stream分行。诊断带profiler，最终加速比请看summary.md。</p>
<header><label>配置 <select id="which"></select></label><label>阶段 <select id="phase"><option>decode</option><option>prefill</option></select></label><label>起点ms <input id="offset" type="number" min="0" step="0.1" value="0"></label><label>窗口ms <input id="window" type="number" min="0.01" step="0.1" value="20"></label><button id="zoom">放大2倍</button> <button id="pan">向后平移</button></header>
<p class="legend"><span style="background:#526c94">CPU / slot0</span><span style="background:#8254ba">CPU / slot1</span><span style="background:#246bce">GPU kernel</span><span style="background:#129b80">H2D</span><span style="background:#ed8b2d">D2H</span><span style="background:#89929c">其他复制</span></p>
<small>鼠标停留查看事件；缩小窗口观察kernel间隙。图中隐藏了底层CPU算子；完整相关箭头与算子请打开.trace.json。</small>
<section><h2>原路径（单执行slot）</h2><p id="baseStats"></p><canvas id="base"></canvas></section>
<section><h2>双slot</h2><p id="dualStats"></p><canvas id="dual"></canvas></section><div id="tip"></div>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id), colors={cpu:'#526c94',slot1:'#8254ba',kernel:'#246bce',h2d:'#129b80',d2h:'#ed8b2d',memcpy:'#89929c',memset:'#89929c'};
DATA.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.title;$('which').appendChild(o)});
function plot(id,stats,s){let c=$(id),ratio=window.devicePixelRatio||1,w=Math.max(700,c.parentElement.clientWidth-36),left=245,right=16,off=+$('offset').value||0,win=Math.max(.01,+$('window').value||20);
let order=['schedule','prepare_inputs','forward','sampler','d2h_enqueue','get_output_wait','commit'];
let lanes=[...new Set(s.timeline.map(e=>e[0]))].sort((a,b)=>{let ac=a.startsWith('CPU'),bc=b.startsWith('CPU');if(ac!==bc)return ac?-1:1;if(ac){let rank=v=>order.indexOf(v.split(' · ')[1]);return rank(a)-rank(b)}return a.localeCompare(b,'en',{numeric:true})}),h=40+lanes.length*25;
c.width=w*ratio;c.height=h*ratio;c.style.height=h+'px';let ctx=c.getContext('2d');ctx.scale(ratio,ratio);ctx.font='11px system-ui';ctx.clearRect(0,0,w,h);let hits=[];
for(let i=0;i<=10;i++){let x=left+(w-left-right)*i/10;ctx.strokeStyle='#e3e9f2';ctx.beginPath();ctx.moveTo(x,22);ctx.lineTo(x,h);ctx.stroke();ctx.fillStyle='#526278';ctx.fillText((off+win*i/10).toFixed(2),x-12,13)}
lanes.forEach((lane,i)=>{let y=28+i*25;ctx.fillStyle='#253f60';let label=lane;while(ctx.measureText(label).width>left-12)label=label.slice(0,-2)+'…';ctx.fillText(label,3,y+13);for(let e of s.timeline){if(e[0]!==lane||e[1]+e[2]<=off||e[1]>=off+win)continue;let x=left+Math.max(0,e[1]-off)/win*(w-left-right),end=left+Math.min(win,e[1]+e[2]-off)/win*(w-left-right),width=Math.max(.7,end-x);ctx.fillStyle=colors[e[3]]||colors.cpu;ctx.fillRect(x,y,width,18);hits.push({x,y,w:width,e})}});
c.onmousemove=ev=>{let r=c.getBoundingClientRect(),x=(ev.clientX-r.left)*w/r.width,y=ev.clientY-r.top,hit=hits.findLast(v=>x>=v.x&&x<=v.x+v.w&&y>=v.y&&y<=v.y+18);if(!hit){$('tip').style.display='none';return}let e=hit.e;$('tip').textContent=e[0]+'\\n'+e[4]+'\\nstart '+e[1].toFixed(4)+'ms · duration '+e[2].toFixed(4)+'ms';Object.assign($('tip').style,{display:'block',left:Math.min(window.innerWidth-400,ev.clientX+12)+'px',top:ev.clientY+15+'px'})};c.onmouseleave=()=>$('tip').style.display='none';
$(stats).textContent='诊断窗口 '+s.profile_wall_ms.toFixed(3)+'ms；GPU内部空隙 '+s.internal_device_gap_ms.toFixed(3)+'ms；get_output与计算重叠 '+s.get_output_overlap_compute_ms.toFixed(3)+'ms';}
function render(){let r=DATA[+$('which').value],p=$('phase').value;plot('base','baseStats',r.baseline[p]);plot('dual','dualStats',r.dual[p])}
['which','phase','offset','window'].forEach(id=>$(id).addEventListener('change',render));$('zoom').onclick=()=>{$('window').value=Math.max(.01,+$('window').value/2);render()};$('pan').onclick=()=>{$('offset').value=(+$('offset').value + +$('window').value/2).toFixed(3);render()};window.onresize=render;render();</script></html>'''
    path.write_text(template.replace("__DATA__", payload))


def worker(a):
    report = dict(status="failed", args=vars(a), samples=[])
    engine = None
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; activate nano-qwen-fp8-test")
        from bench_fp8_pressure import git_info, kv_allocation_policy, memory_snapshot, telemetry
        from nano_qwen.engine.llm_engine import LLMEngine
        from nano_qwen.config import Config
        if "dual_slot_decode" not in Config.__dataclass_fields__:
            raise RuntimeError("Run on feat/moondream-dual-slot with the dual-slot patch applied")
        b, n, o = a.batches[0], a.inputs[0], a.outputs[0]
        context = n + o
        report.update(source=git_info(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                      conda_env=os.environ.get("CONDA_DEFAULT_ENV"), telemetry_before=telemetry(),
                      model_metadata={name: hashlib.sha256((Path(a.model)/name).read_bytes()).hexdigest()
                          for name in ("config.json", "nano_qwen_quantization.json")
                          if (Path(a.model)/name).is_file()})
        torch.cuda.set_per_process_memory_fraction(a.mem_util)
        torch.manual_seed(a.seed)
        policy = dict(batch_size=b, max_model_len=context, options=dict(kv_cache_policy="workload"))
        with torch.inference_mode(), kv_allocation_policy(policy, torch):
            engine = LLMEngine(a.model, dual_slot_decode=a.worker == "dual",
                enforce_eager=a.modes[0] == "eager", use_prefill_cudagraph=False,
                tensor_parallel_size=1, max_num_seqs=b, max_num_batched_tokens=b*n,
                max_model_len=context, gpu_memory_utilization=a.mem_util, enable_prefix_cache=False)
        r = engine.model_runner
        engine.max_concurrent_batches = 2
        r.async_output = True
        if engine.config.max_model_len < context:
            raise RuntimeError("Workload exceeds model context")
        if a.modes[0] == "graph" and b not in r.cuda_graphs.decode_graphs:
            raise RuntimeError(f"Missing exact graph batch {b}")
        report["engine"] = dict(kv_blocks=engine.config.num_kvcache_blocks, max_model_len=context,
            kv_policy="workload", quantization=engine.config.quantization,
            decode_graph_sizes=r.cuda_graphs.decode_graph_sizes,
            graph_banks=len(r.cuda_graphs.decode_graph_banks), prefill_graph=False, queue_depth=2,
            quantization_methods=dict(Counter(type(m.quant_method).__name__ for m in r.model.modules()
                                              if hasattr(m, "quant_method"))))
        def reject_preempt(seq):
            raise RuntimeError(f"Unexpected preemption: {seq.seq_id}")
        engine.scheduler.preempt = reject_preempt
        with torch.inference_mode():
            for i in range(a.warmup):
                measure(engine, a, torch, a.seed + i)
            report["resident_after_warmup"] = memory_snapshot(torch)
            if a.phase == "timing":
                for i in range(a.rounds):
                    sample = measure(engine, a, torch, a.seed + a.warmup + i)
                    report["samples"].append(sample)
                    save(a.result, report)
                    print(f"{a.worker} r{i+1}: {sample['decode_tok_s']:.2f} decode tok/s", flush=True)
            else:
                if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
                    raise RuntimeError("CUDA profiling unavailable; check CUPTI")
                diagnostic = copy.copy(a)
                diagnostic.outputs = [min(o, a.profile_decode_steps + 1)]
                report["profile_output_len"] = diagnostic.outputs[0]
                # Keep model/KV/capture configuration identical to formal runs,
                # but shorten diagnostic generation to bound trace file size.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=a.record_shapes, with_stack=a.with_stack,
                                            profile_memory=False) as prof, profile_labels(engine, torch):
                    report["profile_measurement"] = measure(engine, diagnostic, torch,
                                                           a.seed + a.warmup, profile=True)
                trace = Path(a.result).with_suffix(".trace.json")
                prof.export_chrome_trace(str(trace))
                data = json.loads(trace.read_text())
                name_trace_tracks(data, a.worker)
                trace.write_text(json.dumps(data, ensure_ascii=False))
                count = sum(gpu_kind(e) is not None for e in data["traceEvents"])
                report.update(trace=str(trace), gpu_events=count, diagnostic_only=True)
                if not count:
                    raise RuntimeError("Trace contains no CUDA device events")
                report["trace_analysis"] = analyze_trace(data["traceEvents"])
                if not report["trace_analysis"]["decode"]["kinds_sum_ms"].get("kernel"):
                    raise RuntimeError("No physical decode kernels captured; inspect CUPTI/CUDA Graph support")
                # These are profiler aggregates, not unprofiled latency metrics.
                averages = prof.key_averages()
                device_sort = ("self_device_time_total" if averages and
                               hasattr(averages[0], "self_device_time_total") else "self_cuda_time_total")
                trace.with_suffix(".operators.txt").write_text(
                    averages.table(sort_by="self_cpu_time_total", row_limit=40) + "\n\n" +
                    averages.table(sort_by=device_sort, row_limit=40))
        report["telemetry_after"] = telemetry()
        report["status"] = "ok"
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if engine is not None:
            try:
                engine.exit()
            except Exception:
                report.update(status="failed", cleanup_error=traceback.format_exc())
        save(a.result, report)
    return 0 if report["status"] == "ok" else 1


def summarize(reports):
    rows, warnings = [], []
    grouped = {}
    for report in reports:
        a = report["args"]
        if a["phase"] != "timing":
            continue
        key = (a["modes"][0], a["batches"][0], a["inputs"][0], a["outputs"][0])
        grouped.setdefault(key, {"baseline": [], "dual": []})[a["worker"]].append(report)
    for key, variants in grouped.items():
        if any(not runs or any(r["status"] != "ok" for r in runs) for runs in variants.values()):
            warnings.append(f"{key}: failed/incomplete pair; no speedup reported")
            continue
        all_runs = variants["baseline"] + variants["dual"]
        capacity = {r["engine"]["kv_blocks"] for r in all_runs}
        identities = {(r["source"]["source_sha256"], r["script_sha256"], digest(r["model_metadata"]))
                      for r in all_runs}
        if len(capacity) != 1 or len(identities) != 1:
            raise RuntimeError(f"Source/script/KV capacity differs: {key}")
        samples = {v: [s for r in runs for s in r["samples"]] for v, runs in variants.items()}
        signature = lambda s: (s["seed"], s["prompt_sha256"])
        if Counter(map(signature, samples["baseline"])) != Counter(map(signature, samples["dual"])):
            raise RuntimeError(f"Input samples differ: {key}")
        metrics = {}
        for v, ss in samples.items():
            metrics[v] = {k: statistics.median(s[k] for s in ss) for k in
                          ("ttft_ms", "decode_ms", "tpot_ms", "decode_tok_s", "e2e_ms", "output_tok_s")}
            values = [s["decode_ms"] for s in ss]
            metrics[v]["decode_cv"] = statistics.stdev(values)/statistics.mean(values) if len(values) > 1 else 0
            metrics[v]["peak_allocated_mib"] = max(s["memory"]["peak_allocated_bytes"] for s in ss) / 2**20
            if metrics[v]["decode_cv"] > 0.05:
                warnings.append(f"{key}/{v}: decode CV > 5%; inspect load/clocks and rerun paired processes")
        base, dual = metrics["baseline"], metrics["dual"]
        rows.append(dict(mode=key[0], batch=key[1], input=key[2], output=key[3], metrics=metrics,
                         decode_speedup=base["decode_ms"]/dual["decode_ms"],
                         e2e_speedup=base["e2e_ms"]/dual["e2e_ms"],
                         extra_peak_mib=dual["peak_allocated_mib"]-base["peak_allocated_mib"]))
    return rows, warnings


def parent(a):
    if not (ROOT / "src/nano_qwen").is_dir() or not (ROOT / "benchmarks/bench_fp8_pressure.py").is_file():
        raise RuntimeError("Place this script under nano_qwen/benchmarks before running")
    a.model = str(Path(a.model).expanduser().resolve())
    out = Path(a.output_dir or f"reports/dual_slot_perf_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {out}")
    reports, failures = [], []
    cases = list(itertools.product(a.modes, a.batches, a.inputs, a.outputs))
    for case_index, (mode, b, n, o) in enumerate(cases):
        phases = ["trace"] if a.trace_only else ["timing", "trace"] if a.trace else ["timing"]
        for phase in phases:
            for pair in range(a.pairs if phase == "timing" else 1):
                order = ["baseline", "dual"] if (case_index+pair) % 2 == 0 else ["dual", "baseline"]
                for variant in order:
                    stem = out / f"{mode}_b{b}_i{n}_o{o}_p{pair}_{variant}_{phase}"
                    result = stem.with_suffix(".json")
                    cmd = [sys.executable, str(Path(__file__).resolve()), "--model", a.model,
                           "--batches", str(b), "--inputs", str(n), "--outputs", str(o), "--modes", mode,
                           "--warmup", str(a.warmup), "--rounds", str(a.rounds), "--mem-util", str(a.mem_util),
                           "--temperature", str(a.temperature), "--seed", str(a.seed), "--worker", variant,
                           "--phase", phase, "--result", str(result),
                           "--profile-decode-steps", str(a.profile_decode_steps)]
                    if a.record_shapes:
                        cmd.append("--record-shapes")
                    if a.with_stack:
                        cmd.append("--with-stack")
                    print(f"{phase}: {mode} B{b} input{n} output{o} {variant} pair{pair+1}", flush=True)
                    try:
                        with stem.with_suffix(".log").open("w") as log:
                            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                       cwd=ROOT, timeout=a.timeout, check=False)
                        report = json.loads(result.read_text()) if result.exists() else {"status": "failed"}
                        if completed.returncode or report["status"] != "ok":
                            raise RuntimeError(f"worker failed; see {stem.with_suffix('.log')}")
                    except Exception as exc:
                        report = dict(status="failed", error=str(exc), args=dict(vars(a), worker=variant,
                            phase=phase, modes=[mode], batches=[b], inputs=[n], outputs=[o]))
                        failures.append(str(exc))
                        print(str(exc), flush=True)
                    reports.append(report)
                    save(out / "raw.json", reports)
    rows, warnings = summarize(reports)
    if a.trace or a.trace_only:
        write_trace_comparison(reports, out)
    save(out / "summary.json", dict(status="failed" if failures else "ok", args=vars(a),
                                   comparisons=rows, warnings=warnings, failures=failures))
    lines = ["| Mode | B | In/Out | Base tok/s | Dual tok/s | Decode x | E2E x | Extra peak MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        base, dual = r["metrics"]["baseline"], r["metrics"]["dual"]
        lines.append(f"| {r['mode']} | {r['batch']} | {r['input']}/{r['output']} | "
                     f"{base['decode_tok_s']:.1f} | {dual['decode_tok_s']:.1f} | "
                     f"{r['decode_speedup']:.3f} | {r['e2e_speedup']:.3f} | {r['extra_peak_mib']:+.1f} |")
    text = "\n".join(lines + ["", *warnings, *failures, ""])
    (out / "summary.md").write_text(text)
    print(text, f"Results: {out}", sep="\n")
    return 1 if failures else 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(worker(args) if args.worker else parent(args))
