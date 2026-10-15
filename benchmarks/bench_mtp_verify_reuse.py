"""Strict diagnostics and two-sample timing for production MTP verify modes."""

from contextlib import ExitStack

import torch
from nano_qwen.layers.attention import Attention


def cache_rows(runner, seq):
    count = seq.num_cached_tokens + 2
    slots = torch.tensor(
        [
            seq.block_table[p // runner.block_size] * runner.block_size
            + p % runner.block_size
            for p in range(count)
        ],
        device="cuda",
    )
    return [
        (
            m,
            slots,
            m.k_cache.flatten(0, 1).index_select(0, slots).clone(),
            m.v_cache.flatten(0, 1).index_select(0, slots).clone(),
        )
        for m in runner.model.modules()
        if isinstance(m, Attention)
    ]


def save_state(runner, seq):
    gdn = [
        (
            m,
            m.conv_states.clone(),
            m.recurrent_states.clone(),
            m._speculative_conv_state.clone(),
            m._speculative_recurrent_state.clone(),
        )
        for m in runner.gdn_layers
    ]
    return gdn, cache_rows(runner, seq)


def restore_state(state):
    gdn, kv = state
    for m, c, r, sc, sr in gdn:
        m.conv_states.copy_(c)
        m.recurrent_states.copy_(r)
        m._speculative_conv_state.copy_(sc)
        m._speculative_recurrent_state.copy_(sr)
    for m, slots, k, v in kv:
        m.k_cache.flatten(0, 1).index_copy_(0, slots, k)
        m.v_cache.flatten(0, 1).index_copy_(0, slots, v)


def difference(actual, expected):
    equal = torch.equal(actual, expected)
    delta = (actual.float() - expected.float()).abs()
    return dict(
        equal=equal,
        shape=list(actual.shape),
        mismatches=int((actual != expected).sum()),
        max_abs=float(delta.max()),
        relative_l2=float(delta.norm() / expected.float().norm().clamp_min(1e-30)),
        finite=bool(torch.isfinite(actual).all()),
    )


def first_failure(comparisons):
    return next((r for r in comparisons if not r["equal"] or not r["finite"]), None)


@torch.inference_mode()
def diagnose(runner, seq, graph=False, prepared=None):
    ids, positions = runner.prepare_prefill([seq]) if prepared is None else prepared
    original = save_state(runner, seq)
    references = {}

    def cloned(value):
        return (
            tuple(x.clone() for x in value)
            if isinstance(value, tuple)
            else value.clone()
        )

    def remember(index, name, value):
        references.setdefault((index, name), []).append(cloned(value))

    try:
        with ExitStack() as stack:
            for index, layer in enumerate(runner.model.model.layers):
                modules = [
                    ("input_norm", layer.input_layernorm),
                    ("post_norm", layer.post_attention_layernorm),
                    ("gate_up", layer.mlp.gate_up_proj),
                    ("down", layer.mlp.down_proj),
                    ("layer_output", layer),
                ]
                out_proj = (
                    layer.linear_attn.out_proj
                    if layer.block_type == "linear_attention"
                    else layer.self_attn.o_proj
                )
                modules.append(("attention_output", out_proj))
                for name, module in modules:

                    def hook(module, args, value, _index=index, _name=name):
                        remember(_index, _name, value)

                    stack.enter_context(module.register_forward_hook(hook))

                def pre_hook(module, args, _index=index):
                    remember(_index, "core_output", args[0])

                stack.enter_context(out_proj.register_forward_pre_hook(pre_hook))
            reference = runner.forward_verify_serial(ids, positions).clone()
        expected_logits = runner.compute_verify_logits(reference).clone()
        expected_state = save_state(runner, seq)
        # Compare dense kernels on actual B1 normalized inputs, independently
        # of any subsequent propagated error from the experimental backbone.
        for index, layer in enumerate(runner.model.model.layers):
            projection = (
                layer.linear_attn.forward_dense_pre
                if layer.block_type == "linear_attention"
                else layer.self_attn.forward_dense_pre
            )
            inputs = references[index, "input_norm"]
            inputs = [v[0] if isinstance(v, tuple) else v for v in inputs]
            references[index, "dense_pre"] = [projection(v) for v in inputs]
        restore_state(original)
        comparisons = []

        def compare(index, name, value):
            a = value if isinstance(value, tuple) else (value,)
            ref = references[index, name]
            for component, tensor in enumerate(a):
                parts = [v[component] if isinstance(v, tuple) else v for v in ref]
                expected = torch.cat([v.reshape(1, -1) for v in parts])
                comparisons.append(
                    dict(
                        layer=index,
                        stage=name,
                        component=component,
                        **difference(tensor.reshape(2, -1), expected),
                    )
                )

        actuals = {}

        def observe(index, name, value):
            actuals[index, name] = cloned(value)

        if graph:
            for _ in range(3):
                runner.forward_verify_reuse(ids, positions)
            torch.cuda.synchronize()
            restore_state(original)
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                actual = runner.forward_verify_reuse(ids, positions, observe)
            restore_state(original)
            capture.replay()
        else:
            actual = runner.forward_verify_reuse(ids, positions, observe)
        for (index, name), value in actuals.items():
            compare(index, name, value)
        logits = runner.compute_verify_logits(actual)
        comparisons.extend(
            [
                dict(layer="final", stage="hidden", **difference(actual, reference)),
                dict(
                    layer="final", stage="logits", **difference(logits, expected_logits)
                ),
            ]
        )
        for index, (m, c, r, sc, sr) in enumerate(expected_state[0]):
            for name, a, b in [
                ("conv", m.conv_states, c),
                ("recurrent", m.recurrent_states, r),
                ("snapshot_conv", m._speculative_conv_state, sc),
                ("snapshot_recurrent", m._speculative_recurrent_state, sr),
            ]:
                comparisons.append(
                    dict(layer=f"gdn{index}", stage=name, **difference(a, b))
                )
        for index, (m, slots, k, v) in enumerate(expected_state[1]):
            for name, a, b in [
                ("key", m.k_cache.flatten(0, 1).index_select(0, slots), k),
                ("value", m.v_cache.flatten(0, 1).index_select(0, slots), v),
            ]:
                comparisons.append(
                    dict(layer=f"attention{index}", stage=name, **difference(a, b))
                )
        failure = first_failure(comparisons)
        return dict(
            graph=graph,
            reference_policy="local_M1",
            strict_pass=failure is None,
            first_difference=failure,
            comparisons=comparisons,
        )
    finally:
        restore_state(original)


def measure(model, mtp_model, prompt_tokens, output):
    """Time the actual configured APIs, using no projection or method patches."""
    import json
    import hashlib
    from pathlib import Path
    import bench_qwen3_5_mtp_production as bench

    prompt = bench.make_prompt(model, prompt_tokens)
    _, expected = bench.run_mode(model, False, prompt, 128, 60.0, True)
    _, actual = bench.run_mode(
        model,
        True,
        prompt,
        128,
        60.0,
        True,
        mtp_model=mtp_model,
        mtp_verify_mode="reuse",
    )
    if actual != expected or len(actual) != 128:
        raise AssertionError("Production reuse changed eager committed tokens")
    data = dict(
        model=model,
        mtp_model=mtp_model,
        prompt_tokens=prompt_tokens,
        output_tokens=128,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        eager_exact=True,
        eager_output_tokens=128,
        production_api=True,
        profiler=False,
        samples=[],
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    references = []
    for enabled in (False, True):
        engine = bench.LLMEngine(
            model,
            max_num_seqs=1,
            max_num_batched_tokens=max(512, prompt_tokens + 1),
            max_model_len=prompt_tokens + 130,
            gpu_memory_utilization=0.85,
            use_prefill_cudagraph=False,
            enable_mtp=enabled,
            mtp_model=mtp_model if enabled else None,
            mtp_verify_mode="reuse" if enabled else "serial",
        )
        cases = (
            [(1, "off"), (2, "off")]
            if not enabled
            else [(1, "serial"), (1, "reuse"), (2, "reuse"), (2, "serial")]
        )
        try:
            for repeat, mode in cases:
                selected = mode if enabled else "serial"
                if engine.config.mtp_verify_mode != selected:
                    # Empty engine between requests; recapture before timing.
                    engine.config.mtp_verify_mode = selected
                    engine.model_runner.cuda_graphs._capture_verify()
                metrics, tokens = bench.run_mode(
                    model,
                    enabled,
                    prompt,
                    128,
                    60.0,
                    False,
                    mtp_model=mtp_model if enabled else None,
                    mtp_verify_mode=selected,
                    engine=engine,
                )
                if not enabled:
                    references.append(tokens)
                if tokens != references[repeat - 1] or len(tokens) != 128:
                    data["failure"] = dict(
                        mode=mode,
                        repeat=repeat,
                        expected=references[repeat - 1],
                        actual=tokens,
                    )
                    output.write_text(json.dumps(data, indent=2) + "\n")
                    raise AssertionError(
                        f"{mode} changed committed tokens on repeat {repeat}"
                    )
                if metrics["tokens"] != 127 or metrics["stop_reason"] != "max_tokens":
                    raise AssertionError(
                        "Timing did not commit the requested completion"
                    )
                sample = dict(
                    repeat=repeat, mode=mode, exact_tokens=True, metrics=metrics
                )
                data["samples"].append(sample)
                output.write_text(json.dumps(data, indent=2) + "\n")
                print(json.dumps(sample), flush=True)
        finally:
            engine.exit()
    return data


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from nano_qwen.engine.llm_engine import LLMEngine
    from nano_qwen.engine.sequence import Sequence
    from nano_qwen.sampling_params import SamplingParams
    from nano_qwen.utils.context import reset_context
    from bench_qwen3_5_mtp_production import make_prompt, release

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mtp-model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Two B1 decode samples per off/serial/reused mode",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.prompt_tokens < 2:
        parser.error("--prompt-tokens must be >=2")
    if args.timing:
        measure(args.model, args.mtp_model, args.prompt_tokens, args.output)
        return
    with torch.inference_mode():
        engine = LLMEngine(
            args.model,
            max_num_seqs=1,
            max_num_batched_tokens=max(512, args.prompt_tokens),
            max_model_len=args.prompt_tokens + 130,
            gpu_memory_utilization=0.85,
            enable_mtp=True,
            mtp_model=args.mtp_model,
            mtp_verify_mode="reuse",
            use_prefill_cudagraph=False,
        )
        try:
            params = SamplingParams(temperature=1e-6, max_tokens=128, ignore_eos=True)
            seq = Sequence(make_prompt(args.model, args.prompt_tokens), params)
            engine.scheduler.add(seq)
            engine.step()
            if not engine.scheduler.block_manager.ensure_capacity(seq, 2):
                raise RuntimeError("Insufficient KV capacity for two-token diagnostic")
            # Sequence's TP serialization deliberately omits lifecycle fields;
            # construct a diagnostic view explicitly rather than deepcopy it.
            probe = Sequence(list(seq.token_ids), params)
            probe.num_cached_tokens = seq.num_cached_tokens
            probe.block_table = list(seq.block_table)
            probe.append_token(seq.draft_token)
            probe.is_speculative = True
            probe.num_scheduled_tokens = 2
            reports = []
            for graph in [True] if args.graph_only else [False, True]:
                report = diagnose(engine.model_runner, probe, graph=graph)
                reports.append(report)
                print(
                    "STRICT",
                    report["strict_pass"],
                    "graph",
                    graph,
                    "first_difference",
                    report["first_difference"],
                    flush=True,
                )
            data = dict(
                model=args.model,
                mtp_model=args.mtp_model,
                mtp_verify_mode="reuse",
                prompt_tokens=args.prompt_tokens,
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                experiment_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                diagnostic_only=True,
                reports=reports,
            )
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(data, indent=2) + "\n")
            release(engine, seq)
            if any(not r["strict_pass"] for r in reports):
                raise AssertionError(
                    "Two-token reuse failed strict layer/state validation"
                )
        finally:
            reset_context()
            engine.exit()


if __name__ == "__main__":
    main()
