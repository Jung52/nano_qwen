"""Two-repeat 4B B1/B2 boundary validation and resident/independent-off timing."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'benchmarks')]

from bench_mtp_batch2 import run_batch  # noqa: E402
from bench_mtp_interleaved import execution_mode  # noqa: E402
from bench_mtp_prefill_tail import snapshot  # noqa: E402
from bench_qwen3_5_mtp_production import make_prompt  # noqa: E402
from nano_qwen.engine.llm_engine import LLMEngine  # noqa: E402
from nano_qwen.layers.sampler import Sampler  # noqa: E402
from profile_mtp_reuse import tuning_choices  # noqa: E402
from nano_qwen.utils.context import get_context  # noqa: E402


@contextmanager
def projection_probe(engine, all_rows):
    """Untimed numerical/state oracle; no instrumentation in formal samples."""
    runner = engine.model_runner
    project, postprocess = runner.project_target_hidden, engine.scheduler.postprocess
    record = dict(tails=[], hidden=[], drafts=[], final=[])

    def projected(hidden):
        plan = runner._sampling_plan
        if all_rows:
            runner._target_hidden = hidden
            full = runner.model.compute_logits(hidden)
            rows = torch.tensor(plan.row_indices, device=hidden.device, dtype=torch.int64)
            logits = full.index_select(0, rows)
        else:
            logits = project(hidden)
        if get_context().is_prefill:
            record['hidden'].append(hidden.cpu())
            record['tails'].append(logits.cpu())
        return logits

    def post(seqs, results, prefill):
        if prefill and not any(s.is_speculative for s in seqs):
            record['drafts'].append([r.next_draft for r in results])
        if all(s.num_completion_tokens+len(r.committed_tokens) == s.max_tokens
               for s, r in zip(seqs, results)):
            record['final'].append(snapshot(runner, seqs, results))
        return postprocess(seqs, results, prefill)

    with patch.object(runner, 'project_target_hidden', projected), \
         patch.object(engine.scheduler, 'postprocess', post):
        yield record


def compare_gates(reference, actual, old_tokens, new_tokens):
    if old_tokens != new_tokens or reference['drafts'] != actual['drafts']:
        raise RuntimeError('Completion or first draft changed after target row selection')
    if len(reference['tails']) != len(actual['tails']) or len(reference['final']) != 1 or len(actual['final']) != 1:
        raise RuntimeError('Projection/state gate coverage differs')
    deltas = []
    for a, b in zip(reference['tails'], actual['tails']):
        if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.equal(a.argmax(-1), b.argmax(-1)):
            raise RuntimeError('Nonfinite or changed target tail top1')
        deltas.append(float((a.float()-b.float()).abs().max()))
    pairs = list(zip(reference['hidden'], actual['hidden']))
    old, new = reference['final'][0], actual['final'][0]
    if old['metadata'] != new['metadata'] or old['tensors'].keys() != new['tensors'].keys():
        raise RuntimeError('Effective prefix metadata changed')
    pairs.extend((old['tensors'][k], v) for k, v in new['tensors'].items())
    checked = 0
    for a, b in pairs:
        if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            raise RuntimeError('Target hidden, GDN or effective target/MTP KV changed')
        checked += b.numel()*b.element_size()
    return dict(completion_exact=True, first_draft_exact=True, target_tail_top1_exact=True,
                tail_max_abs=deltas, hidden_gdn_effective_kv_bitwise=True, checked_bytes=checked,
                prefixes=new['metadata'])


@torch.inference_mode()
def measure(args):
    prompts = {n: make_prompt(args.model, n) for n in (64, 1024)}
    files = [*ROOT.joinpath('src/nano_qwen').rglob('*.py'),
             *ROOT.joinpath('benchmarks').glob('*.py')]
    report = dict(model=args.model, mtp_model=args.mtp_model, batches=[1, 2], inputs=[64, 1024],
        output=128, repeats=2, environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda),
        protocol='resident off1/reuse1/reuse2/off2 per case; independent off1/off2; diagnostics separate, profiler disabled',
        source=dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),
        samples=[], gates={}, engines={}, tuning={})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')

    def make(batch, enabled):
        return LLMEngine(args.model, max_num_seqs=batch, max_num_batched_tokens=2048,
            max_model_len=1154, gpu_memory_utilization=.85, enforce_eager=False,
            use_prefill_cudagraph=True, enable_mtp=enabled,
            mtp_model=args.mtp_model if enabled else None,
            mtp_verify_mode='reuse' if enabled else 'serial')

    expected = {}
    for batch in (1, 2):
        engine = make(batch, True)
        try:
            report['engines'][f'resident-B{batch}'] = dict(
                mtp_loaded=engine.model_runner.mtp is not None,
                mtp_weight_bytes=sum(p.numel()*p.element_size() for p in engine.model_runner.mtp.parameters()),
                target_cache_bytes=engine.model_runner.kv_cache.numel()*engine.model_runner.kv_cache.element_size(),
                kv_blocks=engine.config.num_kvcache_blocks,
                mtp_cache_bytes=engine.model_runner.mtp_kv_cache.numel()*engine.model_runner.mtp_kv_cache.element_size(),
                prefill_graph=engine.model_runner.use_prefill_cudagraph,
                allocated_bytes=torch.cuda.memory_allocated())
            # Untimed shape warmup; four outputs only, disjoint from formal runs.
            for n in (64, 1024):
                for enabled in (False, True):
                    with execution_mode(engine, enabled, Sampler()):
                        run_batch(engine, [prompts[n]]*batch, 4)
            frozen = tuning_choices()
            report['tuning'][f'resident-B{batch}'] = frozen
            for n in (64, 1024):
                key = f'B{batch}-I{n}-O128'
                for mode, repeat in (('resident_off', 1), ('reuse', 1), ('reuse', 2), ('resident_off', 2)):
                    with execution_mode(engine, mode == 'reuse', Sampler()):
                        engine.model_runner.cuda_graphs.reset_dispatch_stats()
                        metrics, tokens = run_batch(engine, [prompts[n]]*batch, 128)
                        graphs = engine.model_runner.cuda_graphs.dispatch_stats()
                    expected.setdefault(key, tokens)
                    strict = tokens == expected[key] and all(len(t) == 128 for t in tokens)
                    report['samples'].append(dict(case=key, batch=batch, input=n, mode=mode,
                        repeat=repeat, metrics=metrics, graphs=graphs, completion=tokens, strict_pass=strict))
                    save()
                    print(key, mode, repeat, round(metrics['tokens_per_s'], 2), 'strict', strict, flush=True)
                    if not strict or tuning_choices() != frozen:
                        raise RuntimeError('Strict completion or frozen tuner gate failed')
            # Reference full head versus selected rows: complete outputs and valid final states.
            for n in (64, 1024):
                records, tokens = [], []
                for full in (True, False):
                    with projection_probe(engine, full) as record:
                        _, completion = run_batch(engine, [prompts[n]]*batch, 128)
                        records.append(record)
                        tokens.append(completion)
                report['gates'][f'B{batch}-I{n}'] = compare_gates(*records, *tokens)
                save()
        finally:
            engine.exit()
        # Real off engine: no MTP weights/cache/snapshots, default prefill Graph retained.
        engine = make(batch, False)
        try:
            if (engine.model_runner.mtp is not None or engine.model_runner.mtp_kv_cache is not None
                    or any(layer._speculative_conv_state is not None
                           or layer._speculative_recurrent_state is not None
                           for layer in engine.model_runner.gdn_layers)):
                raise RuntimeError('Independent-off engine still owns MTP state')
            report['engines'][f'independent-off-B{batch}'] = dict(mtp_loaded=False, mtp_weight_bytes=0, mtp_cache_bytes=0,
                target_cache_bytes=engine.model_runner.kv_cache.numel()*engine.model_runner.kv_cache.element_size(),
                kv_blocks=engine.config.num_kvcache_blocks,
                prefill_graph=engine.model_runner.use_prefill_cudagraph,
                allocated_bytes=torch.cuda.memory_allocated())
            for n in (64, 1024):
                run_batch(engine, [prompts[n]]*batch, 4)
            report['tuning'][f'independent-off-B{batch}'] = tuning_choices()
            for n in (64, 1024):
                key = f'B{batch}-I{n}-O128'
                for repeat in (1, 2):
                    engine.model_runner.cuda_graphs.reset_dispatch_stats()
                    metrics, tokens = run_batch(engine, [prompts[n]]*batch, 128)
                    strict = tokens == expected[key]
                    report['samples'].append(dict(case=key, batch=batch, input=n, mode='independent_off',
                        repeat=repeat, metrics=metrics, graphs=engine.model_runner.cuda_graphs.dispatch_stats(),
                        completion=tokens, strict_pass=strict))
                    save()
                    print(key, 'independent_off', repeat, round(metrics['tokens_per_s'], 2), 'strict', strict, flush=True)
                    if not strict:
                        raise RuntimeError('Independent-off strict completion gate failed')
        finally:
            engine.exit()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--mtp-model', required=True)
    parser.add_argument('--output', required=True, type=Path)
    measure(parser.parse_args())


if __name__ == '__main__':
    main()
