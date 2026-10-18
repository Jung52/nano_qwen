"""Same-engine off/all-row/tail-only A/B; separate state and prefill-event gates."""

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]

from bench_mtp_batch2 import run_batch  # noqa: E402
from bench_mtp_interleaved import execution_mode  # noqa: E402
from bench_mtp_small_batch import make_cases  # noqa: E402
from mtp_prefill_reference import REFERENCE, all_row_head, prefill_reference  # noqa: E402
from nano_qwen.engine.llm_engine import LLMEngine  # noqa: E402
from nano_qwen.layers.attention import Attention  # noqa: E402
from nano_qwen.layers.sampler import Sampler  # noqa: E402
from profile_mtp_reuse import tuning_choices  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def snapshot(runner, seqs, outputs):
    tensors, metadata = {}, []
    for n, layer in enumerate(runner.gdn_layers):
        tensors[f"gdn{n}.conv"] = layer.conv_states.cpu()
        tensors[f"gdn{n}.recurrent"] = layer.recurrent_states.cpu()

    def valid(cache, blocks, count):
        indices = torch.tensor([blocks[p // runner.block_size]*runner.block_size + p % runner.block_size
                                for p in range(count)], device=cache.device, dtype=torch.int64)
        return cache.flatten(0, 1).index_select(0, indices).cpu()

    for i, (seq, output) in enumerate(zip(seqs, outputs)):
        slot = runner.input_batch.slots_for([seq])[0]
        prefix = seq.num_cached_tokens + output.cache_advance
        count = runner._mtp_cache_lens[slot]
        metadata.append(dict(slot=slot, target_prefix=prefix, mtp_prefix=count,
                             committed=seq.num_completion_tokens+len(output.committed_tokens)))
        for n, layer in enumerate(m for m in runner.model.modules() if isinstance(m, Attention)):
            for label, cache in (("k", layer.k_cache), ("v", layer.v_cache)):
                tensors[f"request{i}.target{n}.{label}"] = valid(cache, seq.block_table, prefix)
        for label, cache in zip(("k", "v"), runner.mtp_kv_cache):
            tensors[f"request{i}.mtp.{label}"] = valid(cache, seq.block_table, count)
    return dict(metadata=metadata, tensors=tensors)


@contextmanager
def prefill_probe(engine):
    """Diagnostic only; timing ends before GPU clones and CPU state copies."""
    runner = engine.model_runner
    rows, final_states = [], []
    active = None

    @contextmanager
    def timed(label):
        if active is None:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        active["events"][label] = (start, end)
        start.record()
        cpu = perf_counter_ns()
        try:
            yield
        finally:
            active["host_ms"][label] = (perf_counter_ns()-cpu)/1e6
            end.record()

    prefill, forward, compute = runner._run_mtp_prefill, runner.mtp.forward, runner.model.compute_logits
    postprocess = engine.scheduler.postprocess

    def body(*args):
        with timed("mtp_body"):
            output = forward(*args)
        if active is not None:
            active["hidden_ref"] = output.detach()
        return output

    def head(hidden):
        with timed("mtp_head"):
            output = compute(hidden)
        if active is not None:
            active["head_rows"] = hidden.shape[0]
            active["tail_ref"] = output[-1:].detach()
        return output

    def run(seq, slot, target_hidden, sampled_token, final_chunk):
        nonlocal active
        if active is not None:
            raise RuntimeError("Nested prefill probe")
        active = dict(slot=slot, final_chunk=final_chunk, events={}, host_ms={}, head_rows=0)
        try:
            with timed("mtp_prefill"):
                draft = prefill(seq, slot, target_hidden, sampled_token, final_chunk)
            # These copies are outside the measured stage spans and formal runs.
            for src, dst in (("hidden_ref", "hidden"), ("tail_ref", "tail")):
                if src in active:
                    active[dst] = active.pop(src).clone()
            active["draft"] = None if draft is None else draft.clone()
            rows.append(active)
            return draft
        finally:
            active = None

    def post(seqs, outputs, is_prefill):
        if seqs and all(s.num_completion_tokens+len(tokens.committed_tokens) >= s.max_tokens
                        for s, tokens in zip(seqs, outputs)):
            final_states.append(snapshot(runner, seqs, outputs))
        return postprocess(seqs, outputs, is_prefill)

    with ExitStack() as stack:
        stack.enter_context(patch.object(runner, "_run_mtp_prefill", run))
        stack.enter_context(patch.object(runner.mtp, "forward", body))
        stack.enter_context(patch.object(runner.model, "compute_logits", head))
        stack.enter_context(patch.object(engine.scheduler, "postprocess", post))
        yield rows, final_states


def bitwise(actual, expected, label):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(f"Shape/dtype changed: {label}")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError(f"Nonfinite values: {label}")
    if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
        raise RuntimeError(f"Bitwise state mismatch: {label}")


def finish_probe(rows, states):
    torch.cuda.synchronize()
    result = []
    for row in rows:
        result.append(dict(slot=row["slot"], final_chunk=row["final_chunk"], head_rows=row["head_rows"],
                           host_ms=row["host_ms"], gpu_ms={k:a.elapsed_time(b) for k,(a,b) in row["events"].items()},
                           hidden=row["hidden"].cpu(), tail=row["tail"].cpu(), draft=int(row["draft"])))
    if len(states) != 1:
        raise RuntimeError("Expected one final common-prefix snapshot")
    return result, states[0]


def compare_probe(old, new, batch):
    old_rows, old_state = old
    new_rows, new_state = new
    if len(old_rows) != batch or len(new_rows) != batch:
        raise RuntimeError("Prefill probe omitted a request")
    summaries = []
    for a, b in zip(old_rows, new_rows):
        bitwise(a["hidden"], b["hidden"], f"slot{a['slot']}.mtp_hidden")
        if a["slot"] != b["slot"] or a["draft"] != b["draft"] or b["head_rows"] != 1:
            raise RuntimeError("First draft/slot/row gate failed")
        if not torch.isfinite(a["tail"]).all() or not torch.isfinite(b["tail"]).all():
            raise RuntimeError("Nonfinite tail logits")
        if int(a["tail"].argmax()) != int(b["tail"].argmax()) or int(b["tail"].argmax()) != b["draft"]:
            raise RuntimeError("Tail top-1 changed")
        delta = (a["tail"].float()-b["tail"].float()).abs()
        summaries.append(dict(slot=a["slot"], first_draft=b["draft"], first_draft_equal=True, hidden_bitwise=True,
                              tail_bitwise=torch.equal(a["tail"].view(torch.uint8), b["tail"].view(torch.uint8)),
                              tail_max_abs=float(delta.max()), tail_mean_abs=float(delta.mean()), tail_top1_equal=True,
                              old={k:a[k] for k in ("head_rows", "host_ms", "gpu_ms")},
                              new={k:b[k] for k in ("head_rows", "host_ms", "gpu_ms")}))
    if old_state["metadata"] != new_state["metadata"] or old_state["tensors"].keys() != new_state["tensors"].keys():
        raise RuntimeError("Final consumed prefixes changed")
    hashes, total = {}, 0
    for name, tensor in new_state["tensors"].items():
        bitwise(tensor, old_state["tensors"][name], name)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        hashes[name], total = hashlib.sha256(raw).hexdigest(), total+len(raw)
    return dict(requests=summaries, final_state=dict(metadata=new_state["metadata"], sha256=hashes,
                                                    tensors=len(hashes), checked_bytes=total, bitwise_finite=True))


@torch.inference_mode()
def measure(args):
    reference = prefill_reference()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = {n:next(c["prompt"] for c in make_cases(tokenizer) if c["name"] == f"synthetic-{n}-128") for n in (64,256)}
    files = [*ROOT.joinpath("src/nano_qwen").rglob("*.py"), Path(__file__).resolve(), ROOT/"benchmarks/mtp_prefill_reference.py"]
    report = dict(model=args.model, mtp_model=args.mtp_model, batches=[2,4], inputs=[64,256], outputs=[128,512],
                  repeats=2, common_baseline="local batched rollback", reference_prefill=REFERENCE,
                  prefill_graph=False, decode_graph=True,
                  protocol="per case: off1/old1/new1/off2/old2/new2; independent diagnostic cohorts after each capacity's formal samples",
                  environment=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda),
                  source=dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                              sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),
                  timing=[],diagnostics={},tuning={})
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")

    for batch in (2,4):
        engine = LLMEngine(args.model, max_num_seqs=batch,max_num_batched_tokens=1024,max_model_len=1024,
                           gpu_memory_utilization=.85,enforce_eager=False,use_prefill_cudagraph=False,
                           enable_mtp=True,mtp_model=args.mtp_model,mtp_verify_mode="reuse")
        runner = engine.model_runner
        expected = {}
        try:
            for count in range(1,batch+1):
                slots = torch.arange(count,device="cuda",dtype=torch.int64)
                for layer in runner.gdn_layers:
                    layer.rollback_speculative_states(slots)
            for length in (64,256):
                for mode in ("off","old","new"):
                    with execution_mode(engine,mode!="off",Sampler()), \
                         all_row_head(runner,reference) if mode=="old" else nullcontext():
                        run_batch(engine,[prompts[length]]*batch,8)
            frozen = tuning_choices()
            report["tuning"][str(batch)] = dict(before=frozen)
            for length in (64,256):
                for output in (128,512):
                    key = f"B{batch}-I{length}-O{output}"
                    for repeat in (1,2):
                        for mode in ("off","old","new"):
                            with execution_mode(engine,mode!="off",Sampler()), \
                                 all_row_head(runner,reference) if mode=="old" else nullcontext():
                                metrics,tokens = run_batch(engine,[prompts[length]]*batch,output)
                            expected.setdefault(key,tokens)
                            strict = tokens==expected[key] and all(len(t)==output for t in tokens)
                            report["timing"].append(dict(case=key,batch=batch,input=length,output=output,repeat=repeat,
                                                         mode=mode,metrics=metrics,completion=tokens,strict_pass=strict))
                            save()
                            print(f"{key} {mode}{repeat}: TTFT={sum(metrics['request_ttft_ms'])/batch:.2f}ms "
                                  f"e2e={metrics['e2e_s']:.4f}s TPOT={sum(metrics['request_tpot_ms'])/batch:.3f}ms strict={strict}",flush=True)
                            if not strict or tuning_choices()!=frozen:
                                raise RuntimeError("Strict completion or fixed-tuning gate failed")
            # Independent diagnostics never enter formal timing averages.
            for length in (64,256):
                for output in (128,512):
                    key = f"B{batch}-I{length}-O{output}"
                    probes = []
                    for mode in ("old","new"):
                        with all_row_head(runner,reference) if mode=="old" else nullcontext(), \
                             prefill_probe(engine) as (rows,states):
                            _,tokens = run_batch(engine,[prompts[length]]*batch,output)
                        if tokens!=expected[key]:
                            raise RuntimeError("Diagnostic completion changed")
                        probes.append(finish_probe(rows,states))
                    report["diagnostics"][key] = compare_probe(*probes,batch)
                    del probes
                    save()
                    print(f"{key} diagnostics: tail top1/first draft/completion/hidden/final GDN/KV pass",flush=True)
            report["tuning"][str(batch)].update(after=tuning_choices(),unchanged=tuning_choices()==frozen)
            if tuning_choices()!=frozen:
                raise RuntimeError("Diagnostic changed tuning")
            save()
        finally:
            engine.exit()
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--mtp-model",required=True)
    parser.add_argument("--output",type=Path,required=True)
    measure(parser.parse_args())
