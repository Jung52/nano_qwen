"""Separate true-off/reuse prefill and steady decode traces for B2/B4/B8."""
import argparse
import ast
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'benchmarks')]
from bench_mtp_paired import make_engine,production_hashes,engine_info  # noqa: E402
from bench_mtp_batch2 import run_batch  # noqa: E402
from bench_qwen3_5_mtp_production import make_prompt,release  # noqa: E402
from nano_qwen.engine.sequence import Sequence  # noqa: E402
from nano_qwen.sampling_params import SamplingParams  # noqa: E402
from mtp_profile_instrumentation import PhaseRecorder  # noqa: E402
from mtp_profile_analysis import summarize_trace  # noqa: E402


def restore_configs(tuners, snapshot):
    """Install recorded candidates before constructing this diagnostic engine."""
    for name, values in snapshot.items():
        tuner = tuners[name]
        candidates = {repr(config.all_kwargs()):config for config in tuner.configs}
        for encoded, kwargs in values.items():
            key = ast.literal_eval(encoded)
            existing = tuner.cache.get(key)
            if existing is not None:
                if existing.all_kwargs() != kwargs:
                    raise ValueError("Existing tuning config differs from formal timing")
                continue
            config = next((v for v in candidates.values() if v.all_kwargs() == kwargs),None)
            if config is None:
                raise ValueError("Recorded tuning config is absent from this kernel's candidates")
            tuner.cache[key] = config


def telemetry():
    # Read outside diagnostic windows; useful context for noisy workstation runs.
    fields = 'timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used'
    result = subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv'],
                            text=True,capture_output=True,check=False)
    return dict(returncode=result.returncode,output=result.stdout.strip(),error=result.stderr.strip())


def progress(engine,seqs):
    live = [s for s in seqs if not s.is_finished]
    return dict(active=len(live),completed=[s.num_completion_tokens for s in seqs],
                slots=[engine.model_runner.input_batch.seq_id_to_slot.get(s.seq_id) for s in seqs])


def step(engine,seqs,recorder):
    before = progress(engine,seqs)
    recorder.begin()
    start = perf_counter_ns()
    engine.step()
    elapsed = (perf_counter_ns()-start)/1e6
    spans = recorder.finish()
    after = progress(engine,seqs)
    return dict(before=before,after=after,committed=sum(after['completed'])-sum(before['completed']),
                wall_ms=elapsed,**spans)


def add(engine,prompt,batch):
    if not engine.is_finished() or engine.model_runner.input_batch.seq_id_to_slot:
        raise ValueError('Profile needs an idle engine')
    seqs = [Sequence(prompt,SamplingParams(temperature=1e-6,max_tokens=512,ignore_eos=True)) for _ in range(batch)]
    for s in seqs:
        engine.scheduler.add(s)
    return seqs


def finish(engine,seqs,expected):
    while not all(s.is_finished for s in seqs):
        engine.step()
    completion = [s.completion_token_ids for s in seqs]
    if completion != expected:
        raise RuntimeError('Instrumentation changed strict completion')
    for s in seqs:
        release(engine,s)
    return dict(completion=completion,strict_pass=True)


def stage_pass(engine,prompt,batch,expected,kind):
    with PhaseRecorder(engine.model_runner,kind).install() as recorder:
        seqs = add(engine,prompt,batch)
        rows = []
        try:
            while not all(s.is_finished for s in seqs):
                rows.append(step(engine,seqs,recorder))
            gate = finish(engine,seqs,expected)
            field = 'gpu_ms' if kind=='events' else 'host_ms'
            prefill,decode = defaultdict(float),defaultdict(float)
            for i,row in enumerate(rows):
                dest = prefill if i==0 else decode
                for label,ms in row[field].items():
                    dest[label] += ms
            return dict(rows=rows,prefill=dict(prefill),decode=dict(decode),
                        actual_decode_commits=sum(r['committed'] for r in rows[1:]),**gate)
        finally:
            for s in seqs:
                release(engine,s)


def trace_pass(engine,prompt,batch,expected,path,start,end):
    with PhaseRecorder(engine.model_runner,'trace').install() as recorder:
        seqs = add(engine,prompt,batch)
        rows = []
        try:
            while min(s.num_completion_tokens for s in seqs)<start:
                engine.step()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function('mtp.window'):
                    while min(s.num_completion_tokens for s in seqs)<end:
                        rows.append(step(engine,seqs,recorder))
                    torch.cuda.synchronize()
            prof.export_chrome_trace(str(path))
            raw = path.read_bytes()
            summary = summarize_trace(json.loads(raw)['traceEvents'])
            if not summary['physical_device_events']:
                raise RuntimeError('Trace has no physical GPU evidence')
            target = path.with_suffix(path.suffix+'.gz')
            target.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
            path.unlink()
            summary.update(trace=str(target),raw_sha256=hashlib.sha256(raw).hexdigest(),
                gzip_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),raw_bytes=len(raw),gzip_bytes=target.stat().st_size,
                rows=rows,actual_commits=sum(r['committed'] for r in rows),**finish(engine,seqs,expected))
            return summary
        finally:
            for s in seqs:
                release(engine,s)


@torch.inference_mode()
def measure(args):
    timing = json.loads(args.timing.read_text())
    frozen = production_hashes()
    if frozen != timing['source']['sha256']:
        raise RuntimeError('Timing and profile production sources differ')
    prompt = make_prompt(args.model,64)
    report = dict(input=64,output=512,batches=args.batches,source_sha256=frozen,
                  diagnostic_only=True,tuning_protocol='Recorded candidates restored before engine construction; existing choices cannot change',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),profiles={})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    from nano_qwen.quantization.fp8 import _per_channel_mm_tuned
    from nano_qwen.quantization.fp8_block import _block_mm_tuned
    tuners = {'per_channel':_per_channel_mm_tuned,'block_wise':_block_mm_tuned}
    for batch in args.batches:
        restore_configs(tuners,timing['tuning'][str(batch)])
        expected = next(s['completion'] for s in timing['samples']
                        if s['batch']==batch and s['input']==64 and s['output']==512 and s['mode']=='off')
        for mode in ('off','reuse'):
            engine = make_engine(args.model,args.mtp_model,batch,mode)
            try:
                for _ in range(2):
                    run_batch(engine,[prompt]*batch,8)
                value = report['profiles'][f'B{batch}-{mode}'] = dict(engine=engine_info(engine),telemetry_before=telemetry())
                # Profiler first, before any event recapture; steady graph kernels remain untouched.
                for phase,start,end in (('prefill',0,1),('decode',112,144)):
                    path = args.output.with_suffix(f'.B{batch}.{mode}.{phase}.trace.json')
                    value[phase+'_trace'] = trace_pass(engine,prompt,batch,expected,path,start,end)
                    save()
                    print(f'B{batch} {mode} {phase} trace strict=True',flush=True)
                for kind in ('host','events'):
                    value[kind] = stage_pass(engine,prompt,batch,expected,kind)
                    save()
                    print(f'B{batch} {mode} {kind} diagnostic strict=True',flush=True)
                if production_hashes()!=frozen:
                    raise RuntimeError('Production changed during profile')
                value['telemetry_after'] = telemetry()
                save()
            finally:
                engine.exit()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--mtp-model',required=True)
    p.add_argument('--timing',required=True,type=Path)
    p.add_argument('--batches',nargs='+',type=int,default=[2,4,8],choices=[2,4,8])
    p.add_argument('--output',required=True,type=Path)
    measure(p.parse_args())


if __name__=='__main__':
    main()
