"""Reproduce the mixed-prefill divergence of continuous-batching replicas.

One 1024-token request, three chunk patterns on the same engine:
  * [512, 512]      budget 512 (the B1 reference path)
  * [511, 511, 2]   budget 511 alone (the chunk pattern the request gets when
                    its first prefill step shares the token budget with one
                    decode token of a co-resident request)
  * mixed           budget 512 with a 64-token partner admitted first (the
                    original replica scenario); the request's real chunks
                    are recorded
Plus an eager-mode control for budget 511 and an optional BF16 model to
attribute the drift to FP8 vs the chunk boundary itself.

Per path: tokens, per-position top-2/margin diagnostics, the prefill-end
GDN state (conv/recurrent per layer), and the actual prefill chunk sizes.
Pairwise: first divergence position with both sides' logits, and per-layer
relative RMSE of the prefill-end GDN state.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.engine.sequence import Sequence
from nano_qwen.sampling_params import SamplingParams
from engine_bench_utils import ArgmaxSampler, run_until_idle


def prompt_for(engine, length):
    seed = engine.tokenizer.encode("Explain how a computer stores information in memory. ",
                                   add_special_tokens=False)
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def gdn_state(engine, seq):
    """Pre-completion GDN state of the request's slot, on CPU."""
    slot = engine.model_runner.input_batch.seq_id_to_slot[seq.seq_id]
    out = {}
    for i, layer in enumerate(engine.model_runner.gdn_layers):
        out[f"gdn.{i}.conv"] = layer.conv_states[slot].detach().cpu().clone()
        out[f"gdn.{i}.recurrent"] = layer.recurrent_states[slot].detach().cpu().clone()
    return out


def rmse(a, b):
    a, b = a.float(), b.float()
    return ((a - b).square().mean().sqrt() / max(a.square().mean().sqrt(), 1e-12)).item()


def run_path(engine, args, budget, partner_len=None, eager=False):
    """Run the 1024-token request; record tokens, diagnostics, GDN, chunks."""
    engine.scheduler.max_num_batched_tokens = budget
    sampler = ArgmaxSampler()
    engine.model_runner.sampler = sampler
    sampler.enable_diagnostics(capture_logits=False)
    params = SamplingParams(temperature=1e-8, max_tokens=args.out_len, ignore_eos=True)
    partner = None
    if partner_len:
        partner = Sequence(prompt_for(engine, partner_len), params)
        partner._diagnostic_key = "partner"
        engine.scheduler.add(partner)
        while partner.num_completion_tokens < 1:
            engine.step()
    seq = Sequence(prompt_for(engine, args.prompt_len), params)
    key = "subject"
    seq._diagnostic_key = key
    chunks = []
    engine.scheduler.add(seq)
    while seq.num_cached_tokens < args.prompt_len:
        before = seq.num_cached_tokens
        engine.step()
        after = seq.num_cached_tokens
        if after > before:
            chunks.append(after - before)
    # one decode step to reach the committed boundary (token 1 sampled)
    while seq.num_completion_tokens < 1:
        engine.step()
    gdn = gdn_state(engine, seq)
    if partner is not None:
        run_until_idle(engine, [partner])  # drain the partner
    else:
        run_until_idle(engine, [seq])
    tokens = list(seq.completion_token_ids)
    diagnostics = dict(sampler.get_diagnostic_map(key))
    sampler.clear_diagnostics([key])
    engine.scheduler.max_num_batched_tokens = args.budget
    return dict(tokens=tokens, chunks=chunks, gdn=gdn, diagnostics=diagnostics,
                budget=budget, eager=eager, partner=partner_len)


def diag_summary(diag):
    if not diag:
        return None
    return {k: diag.get(k) for k in
            ("top_tokens", "top_logits", "margin", "effective_tolerance", "near_tie")}


def compare(reference, candidate, label):
    equal = reference["tokens"] == candidate["tokens"]
    result = dict(label=label, tokens_equal=equal,
                  chunks_reference=reference["chunks"], chunks_candidate=candidate["chunks"])
    if not equal:
        position = next(i for i, (a, b) in enumerate(zip(reference["tokens"],
                                                         candidate["tokens"])) if a != b)
        result.update(
            first_divergence=position,
            reference_token=reference["tokens"][position],
            candidate_token=candidate["tokens"][position],
            reference_diag=diag_summary(reference["diagnostics"].get(position)),
            candidate_diag=diag_summary(candidate["diagnostics"].get(position)))
    result["gdn_rmse"] = {name: rmse(reference["gdn"][name], candidate["gdn"][name])
                          for name in sorted(reference["gdn"])}
    return result


def run_one_path(path_args):
    """One path in its own engine (dist can only init once per process)."""
    engine = LLMEngine(path_args.model, enforce_eager=path_args.eager,
                       use_prefill_cudagraph=not path_args.eager,
                       max_num_seqs=4, max_num_batched_tokens=path_args.budget,
                       max_model_len=4096, gpu_memory_utilization=0.7,
                       tensor_parallel_size=1, async_scheduling=False,
                       enable_prefix_cache=False)
    try:
        result = run_path(engine, path_args, path_args.budget,
                          partner_len=path_args.partner, eager=path_args.eager)
    finally:
        engine.exit()
    result["model"] = path_args.model
    result["gdn"] = {k: v.tolist() for k, v in result["gdn"].items()}
    Path(path_args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(path_args.output).write_text(json.dumps(result, default=str))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--partner-len", type=int, default=64)
    parser.add_argument("--out-len", type=int, default=32)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--budgets", nargs="+", type=int, default=[512, 511, 448])
    parser.add_argument("--no-bf16", action="store_true",
                        help="skip the BF16 control (requires the unquantized model dir)")
    parser.add_argument("--bf16-model", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", choices=("full", "path"), default="full")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--partner", type=int, default=None)
    args = parser.parse_args()
    if args.role == "path":
        run_one_path(args)
        return

    import subprocess
    tmp = Path(args.output).parent
    specs = [dict(name="fp8_512", model=args.model, budget=args.budget, eager=False),
             dict(name="fp8_511_eager", model=args.model, budget=511, eager=True),
             dict(name="fp8_512_mixed", model=args.model, budget=args.budget,
                  eager=False, partner=args.partner_len)]
    specs += [dict(name=f"fp8_{b}", model=args.model, budget=b, eager=False)
              for b in sorted(set(args.budgets) - {args.budget})]
    if not args.no_bf16 and args.bf16_model:
        specs += [dict(name="bf16_512", model=args.bf16_model, budget=args.budget,
                       eager=False),
                  dict(name="bf16_511", model=args.bf16_model, budget=511, eager=False)]
    report = dict(args=vars(args), paths={}, comparisons=[])
    results = {}
    for spec in specs:
        out = tmp / f"mpx-{spec['name']}.json"
        argv = [sys.executable, str(Path(__file__).resolve()),
                "--role", "path", "--model", spec["model"],
                "--prompt-len", str(args.prompt_len),
                "--partner-len", str(args.partner_len),
                "--out-len", str(args.out_len),
                "--budget", str(spec["budget"]),
                "--output", str(out)]
        if spec.get("partner"):
            argv += ["--partner", str(spec["partner"])]
        if spec.get("eager"):
            argv += ["--eager"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
                   PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        subprocess.run(argv, env=env, check=True)
        results[spec["name"]] = json.loads(out.read_text())
        results[spec["name"]]["gdn"] = {k: torch.tensor(v, dtype=torch.bfloat16)
                                        for k, v in results[spec["name"]]["gdn"].items()}
        report["paths"][spec["name"]] = dict(tokens=results[spec["name"]]["tokens"],
                                             chunks=results[spec["name"]]["chunks"],
                                             budget=results[spec["name"]]["budget"],
                                             eager=results[spec["name"]].get("eager"),
                                             partner=results[spec["name"]].get("partner"))
    reference = dict(tokens=results["fp8_512"]["tokens"],
                     diagnostics=results["fp8_512"]["diagnostics"],
                     gdn=results["fp8_512"]["gdn"],
                     chunks=results["fp8_512"]["chunks"])
    for spec in specs:
        if spec["name"] == "fp8_512":
            continue
        candidate = results[spec["name"]]
        comparison = compare(reference, candidate, f"fp8_512_vs_{spec['name']}")
        report["comparisons"].append(comparison)
    Path(args.output).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(dict(
        paths=list(report["paths"]),
        comparisons=[dict(label=c["label"], tokens_equal=c["tokens_equal"],
                          first_divergence=c.get("first_divergence"),
                          gdn0_conv_rmse=c["gdn_rmse"].get("gdn.0.conv"))
                     for c in report["comparisons"]]), indent=2))


if __name__ == "__main__":
    main()
