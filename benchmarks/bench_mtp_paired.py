"""Five counterbalanced true-off/reuse pairs; no profiling or production edits."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import math
from pathlib import Path
import statistics as stats
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'benchmarks')]

from bench_mtp_batch2 import run_batch  # noqa: E402
from bench_qwen3_5_mtp_production import make_prompt  # noqa: E402
from nano_qwen.engine.llm_engine import LLMEngine  # noqa: E402


def case_matrix(batch):
    if batch == 1:
        return [(64,128),(64,512),(1024,128),(1024,512)]
    if batch == 2:
        return [(64,512)]
    if batch in (4,8):
        return [(64,128),(64,512)]
    raise ValueError('Validated batch capacities:1/2/4/8')


def pair_modes(pair):
    return ('off','reuse') if pair%2 else ('reuse','off')


def execution_plan(batches):
    return [(b,n,o,p,m) for b in batches for n,o in case_matrix(b)
            for p in range(1,6) for m in pair_modes(p)]


def pending_plan(samples,batches):
    plan = execution_plan(batches)
    if len(samples)>len(plan):
        raise ValueError('Completed execution prefix exceeds plan')
    for i,(row,key) in enumerate(zip(samples,plan),1):
        if tuple(row[k] for k in ('batch','input','output','pair','mode')) != key or row['execution_order'] != i:
            raise ValueError('Resume must retain the exact completed execution prefix')
        if not row['strict_pass'] or not row['actual_commits_pass'] or not row.get('invariants_pass',False):
            raise ValueError('Cannot resume across a failed strict gate')
    return plan[len(samples):]


def check_environment(reference,current):
    if reference != current:
        raise ValueError('Resume environment differs: GPU/PyTorch/CUDA must match')


def acceptance_rate(row):
    trials = sum(int(b)*n for b,n in row['graphs'].get('verify',{}).get('batch_counts',{}).items())
    accepted = row['metrics']['accepted_tokens']
    if not 0 <= accepted <= trials:
        raise ValueError('Invalid accepted request count')
    return accepted/trials if trials else None


def distribution(values):
    return dict(values=values,mean=stats.mean(values),median=stats.median(values),
                stdev=stats.stdev(values) if len(values)>1 else 0.,min=min(values),max=max(values))


def speedup_ratios(values):
    # Five paired log ratios; a descriptive t interval, not a hardware-wide guarantee.
    if len(values) != 5:
        raise ValueError('Expected exactly five complete pairs')
    logs = [math.log(v) for v in values]
    center = stats.mean(logs)
    half = 2.7764451051977987*stats.stdev(logs)/math.sqrt(5)
    return dict(**distribution([100*(v-1) for v in values]),geometric_mean=100*math.expm1(center),
                ci95=[100*math.expm1(center-half),100*math.expm1(center+half)])


def summarize_pairs(rows):
    pairs = defaultdict(dict)
    for row in rows:
        if row['mode'] in pairs[row['pair']]:
            raise ValueError('Duplicate pair mode')
        pairs[row['pair']][row['mode']] = row
    if sorted(pairs) != list(range(1,6)) or any(set(p)!={'off','reuse'} for p in pairs.values()):
        raise ValueError('Missing complete pair')
    e2e,tpot,extra = [],[],[]
    for pair in sorted(pairs):
        off,reuse = (pairs[pair][m]['metrics'] for m in ('off','reuse'))
        e2e.append(off['e2e_s']/reuse['e2e_s'])
        tpot.append(stats.mean(off['request_tpot_ms'])/stats.mean(reuse['request_tpot_ms']))
        extra.append(stats.mean(reuse['request_ttft_ms'])-stats.mean(off['request_ttft_ms']))
    return dict(e2e_speedup_pct=speedup_ratios(e2e),tpot_speedup_pct=speedup_ratios(tpot),
                ttft_extra_ms=distribution(extra))


def tuning_snapshot():
    from nano_qwen.quantization.fp8 import _per_channel_mm_tuned
    from nano_qwen.quantization.fp8_block import _block_mm_tuned
    return {name:{repr(key):config.all_kwargs() for key,config in tuner.cache.items()}
            for name,tuner in (('per_channel',_per_channel_mm_tuned),('block_wise',_block_mm_tuned))}


def check_tuning(reference,current):
    for name,configs in reference.items():
        if any(current[name].get(key)!=value for key,value in configs.items()):
            raise RuntimeError('A previously frozen GEMM configuration changed')


def production_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'src/nano_qwen').rglob('*.py')}


def make_engine(model,mtp_model,batch,mode):
    # Construction is untimed. Dump its Python call site if a CUDA wait stalls;
    # cancel the timer before any warmup or formal measurement can begin.
    faulthandler.dump_traceback_later(120,repeat=True)
    try:
        return LLMEngine(model,max_num_seqs=batch,max_num_batched_tokens=2048,max_model_len=1538,
                     gpu_memory_utilization=.85,enforce_eager=False,use_prefill_cudagraph=True,
                     enable_mtp=mode=='reuse',mtp_model=mtp_model if mode=='reuse' else None,
                     mtp_verify_mode='reuse' if mode=='reuse' else 'serial')
    finally:
        faulthandler.cancel_dump_traceback_later()


def engine_info(engine):
    r = engine.model_runner
    def nbytes(t):
        return 0 if t is None else t.numel()*t.element_size()
    return dict(mtp_loaded=r.mtp is not None,mtp_weights_bytes=0 if r.mtp is None else sum(p.numel()*p.element_size() for p in r.mtp.parameters()),
                mtp_cache_bytes=nbytes(r.mtp_kv_cache),target_cache_bytes=nbytes(r.kv_cache),
                kv_blocks=engine.config.num_kvcache_blocks,prefill_graph=r.use_prefill_cudagraph,
                snapshot_bytes=sum(nbytes(m._speculative_conv_state)+nbytes(m._speculative_recurrent_state) for m in r.gdn_layers),
                allocated_bytes=torch.cuda.memory_allocated())


@torch.inference_mode()
def measure(args):
    frozen_source = production_hashes()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = dict(model=args.model,mtp_model=args.mtp_model,repeats=5,batches=args.batches,
        protocol='Independent engines rebuilt sequentially per pair; odd off/reuse, even reuse/off;2×O8 warmup per input/mode;no profiler in timing',
        environment=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda),
        source=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sha256=frozen_source,
                    script_sha256=script_sha),
        samples=[],engines=[],tuning={},summaries={})
    if getattr(args,'resume',False):
        current_environment = report['environment']
        report = json.loads(args.output.read_text())
        check_environment(report['environment'],current_environment)
        if (report['source']['sha256']!=frozen_source or report['model']!=args.model
                or report['mtp_model']!=args.mtp_model or report['batches']!=args.batches
                or len(report['engines'])!=len(report['samples'])):
            raise RuntimeError('Resume source/model/plan/engine alignment differs')
        pending_plan(report['samples'],args.batches)
        report.setdefault('continuations',[]).append(dict(after_samples=len(report['samples']),
            utc=datetime.now(timezone.utc).isoformat(),script_sha256=script_sha,
            environment=current_environment,
            note='Retain all completed samples, restore recorded tuning, continue original prefix; independent engines and fixed warmup unchanged'))
        if report['tuning']:
            from profile_mtp_paired import restore_configs
            from nano_qwen.quantization.fp8 import _per_channel_mm_tuned
            from nano_qwen.quantization.fp8_block import _block_mm_tuned
            restore_configs({'per_channel':_per_channel_mm_tuned,'block_wise':_block_mm_tuned},
                            list(report['tuning'].values())[-1])
    pending = set(pending_plan(report['samples'],args.batches))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    prompts = {n:make_prompt(args.model,n) for n in (64,1024)}
    expected = {s['case']:s['completion'] for s in report['samples']}
    def save():
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    save()
    for batch in args.batches:
        cases = case_matrix(batch)
        canonical = report['tuning'].get(str(batch),{})
        # Establish both engines' configs once; subsequent paired rebuilds reuse the persistent tuner cache.
        for mode in (() if str(batch) in report['tuning'] else ('off','reuse')):
            engine = make_engine(args.model,args.mtp_model,batch,mode)
            try:
                for warm_length in sorted({length for length,output in cases}):
                    for _ in range(2):
                        run_batch(engine,[prompts[warm_length]]*batch,8)
                current = tuning_snapshot()
                check_tuning(canonical,current)
                canonical = current
            finally:
                engine.exit()
        report['tuning'][str(batch)] = canonical
        for n,o in cases:
            for pair in range(1,6):
                for mode in pair_modes(pair):
                    if (batch,n,o,pair,mode) not in pending:
                        continue
                    print(f'begin B{batch}-I{n}-O{o} pair{pair} {mode}: construct',flush=True)
                    engine = make_engine(args.model,args.mtp_model,batch,mode)
                    try:
                        print('constructed; fixed warmup',flush=True)
                        info = engine_info(engine)
                        if mode=='off' and (info['mtp_loaded'] or info['mtp_cache_bytes'] or info['snapshot_bytes'] or not info['prefill_graph']):
                            raise RuntimeError('Independent off engine allocation is invalid')
                        report['engines'].append(dict(batch=batch,pair=pair,mode=mode,**info))
                        for warm_length in sorted({length for length,output in cases}):
                            for _ in range(2):
                                run_batch(engine,[prompts[warm_length]]*batch,8)
                        current = tuning_snapshot()
                        check_tuning(canonical,current)
                        if current != canonical:
                            raise RuntimeError('Shape preflight did not freeze every warmed GEMM configuration')
                        key = f'B{batch}-I{n}-O{o}'
                        print('warmup complete; formal timing',flush=True)
                        engine.model_runner.cuda_graphs.reset_dispatch_stats()
                        metrics,tokens = run_batch(engine,[prompts[n]]*batch,o)
                        row = dict(case=key,batch=batch,input=n,output=o,mode=mode,pair=pair,
                            order_in_pair=pair_modes(pair).index(mode),execution_order=len(report['samples'])+1,
                            metrics=metrics,graphs=engine.model_runner.cuda_graphs.dispatch_stats(),completion=tokens)
                        expected.setdefault(key,tokens)
                        row['strict_pass'] = tokens==expected[key] and len(tokens)==batch and all(len(t)==o for t in tokens)
                        row['acceptance_rate'] = acceptance_rate(row)
                        row['actual_commits_pass'] = metrics['tokens']==(o-1)*batch==sum(s['committed'] for s in metrics['steps'])
                        observed_tuning,observed_source = tuning_snapshot(),production_hashes()
                        row['invariants_pass'] = observed_tuning==canonical and observed_source==frozen_source
                        if not row['invariants_pass']:
                            row['invariant_failure'] = dict(observed_tuning=observed_tuning,observed_source=observed_source)
                        report['samples'].append(row)
                        save()
                        print(f"{key} pair{pair} {mode}: e2e={metrics['e2e_s']:.4f}s TPOT={stats.mean(metrics['request_tpot_ms']):.3f}ms strict={row['strict_pass']}",flush=True)
                        if not row['strict_pass'] or not row['actual_commits_pass']:
                            raise RuntimeError('Strict completion/actual commit gate failed; retain evidence')
                        if not row['invariants_pass']:
                            raise RuntimeError('Production source or frozen tuning changed during timing')
                    finally:
                        engine.exit()
        for n,o in cases:
            key = f'B{batch}-I{n}-O{o}'
            report['summaries'][key] = summarize_pairs([s for s in report['samples'] if s['case']==key])
        save()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--mtp-model',required=True)
    p.add_argument('--batches',nargs='+',type=int,default=[1,2,4,8],choices=[1,2,4,8])
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--resume',action='store_true',help='Continue only an unchanged completed execution prefix')
    measure(p.parse_args())


if __name__=='__main__':
    main()
