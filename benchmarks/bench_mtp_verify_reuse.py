"""Experimental two-token target, never enabled by production configuration.

Reuse actual dense/MLP modules; retain ordinary sequential GDN/attention cores.
Projection injection is local to this single-threaded diagnostic and restored.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import fields
from unittest.mock import patch

import torch
import nano_qwen.layers.gated_delta_net as gdn_module
from nano_qwen.layers.attention import Attention
from nano_qwen.utils.context import get_context, set_context


def row_context(context, row):
    set_context(
        False,
        slot_mapping=context.slot_mapping[row : row + 1],
        context_lens=context.context_lens - (1 - row),
        block_tables=context.block_tables,
        state_indices=context.state_indices,
    )


@contextmanager
def b1_tiles():
    from nano_qwen.quantization.fp8 import _per_channel_mm_tuned
    from nano_qwen.quantization.fp8_block import _block_mm_tuned

    tuners = (_per_channel_mm_tuned, _block_mm_tuned)
    saved = [dict(t.cache) for t in tuners]
    try:
        for t in tuners:
            for key, config in list(t.cache.items()):
                if key[0] == 1:
                    t.cache[(2, *key[1:])] = config
        yield
    finally:
        for t, cache in zip(tuners, saved):
            t.cache.clear()
            t.cache.update(cache)


def rowwise(function, *values):
    parts = [
        function(*(None if v is None else v[row : row + 1] for v in values))
        for row in range(2)
    ]
    if isinstance(parts[0], tuple):
        return tuple(torch.cat([v[i] for v in parts]) for i in range(len(parts[0])))
    return torch.cat(parts)


@torch.inference_mode()
def forward_reused(runner, ids, positions, observe=None):
    context = get_context()
    if runner.world_size != 1 or ids.numel() != 2 or context.state_indices.numel() != 1:
        raise ValueError("Two-token diagnostic requires B1 and two consecutive rows")
    hidden = runner.model.model.embed_tokens(ids)
    residual = None
    try:
        for index, layer in enumerate(runner.model.model.layers):
            # torch.mean may choose different reduction schedules for M=1/2.
            # Keep all normalization reductions on the reference's M=1 path.
            hidden, residual = rowwise(layer.forward_input, hidden, residual)
            if observe:
                observe(index, "input_norm", hidden)
            if layer.block_type == "linear_attention":
                gdn = layer.linear_attn
                dense = gdn.forward_dense_pre(hidden)
                out_proj = gdn.out_proj
            else:
                attention = layer.self_attn
                dense = attention.forward_dense_pre(hidden)
                out_proj = attention.o_proj
            if observe:
                observe(index, "dense_pre", dense)
            core = []
            for row in range(2):
                row_context(context, row)
                if layer.block_type == "linear_attention":

                    def projections(x, *modules):
                        assert modules == (gdn.in_proj_qkv, gdn.in_proj_z)
                        return tuple(v[row : row + 1].unsqueeze(1) for v in dense[:2])

                    def gate(x, projection):
                        assert projection in (gdn.in_proj_b, gdn.in_proj_a)
                        return dense[2 if projection is gdn.in_proj_b else 3][
                            row : row + 1
                        ].unsqueeze(1)

                    with (
                        patch.object(
                            gdn_module, "project_shared_fp8_input", projections
                        ),
                        patch.object(gdn, "_project_gate", gate),
                        patch.object(out_proj, "forward", lambda x: x),
                    ):
                        value = gdn._forward_decode(
                            hidden[row : row + 1].unsqueeze(1)
                        ).reshape(1, -1)
                    if row == 0:
                        gdn.save_speculative_state(context.state_indices)
                else:
                    with patch.object(out_proj, "forward", lambda x: x):
                        value = attention.forward_core_from_dense(
                            tuple(v[row : row + 1] for v in dense),
                            positions[row : row + 1],
                        )
                core.append(value)
            core = torch.cat(core)
            if observe:
                observe(index, "core_output", core)
            attention_output = out_proj(core)
            if observe:
                observe(index, "attention_output", attention_output)
            hidden, residual = rowwise(
                layer.post_attention_layernorm, attention_output, residual
            )
            if observe:
                observe(index, "post_norm", (hidden, residual))
            with ExitStack() as stack:
                if observe:
                    for name, module in [
                        ("gate_up", layer.mlp.gate_up_proj),
                        ("down", layer.mlp.down_proj),
                    ]:

                        def hook(module, args, value, _name=name):
                            observe(index, _name, value)

                        stack.enter_context(module.register_forward_hook(hook))
                hidden = layer.mlp(hidden)
            if observe:
                observe(index, "layer_output", (hidden, residual))
        hidden, _ = rowwise(runner.model.model.norm, hidden, residual)
        return hidden
    finally:
        set_context(**{f.name: getattr(context, f.name) for f in fields(context)})


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
def diagnose(runner, seq, pin_tiles, graph=False, prepared=None):
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
            reference = runner.forward_verify(ids, positions).clone()
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

        with b1_tiles() if pin_tiles else ExitStack():
            if graph:
                for _ in range(3):
                    forward_reused(runner, ids, positions)
                torch.cuda.synchronize()
                restore_state(original)
                capture = torch.cuda.CUDAGraph()
                with torch.cuda.graph(capture):
                    actual = forward_reused(runner, ids, positions, observe)
                restore_state(original)
                capture.replay()
            else:
                actual = forward_reused(runner, ids, positions, observe)
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
            pin_b1_tiles=pin_tiles,
            strict_pass=failure is None,
            first_difference=failure,
            comparisons=comparisons,
        )
    finally:
        restore_state(original)


def measure(model, mtp_model, prompt_tokens, output):
    """Two formal samples per mode; same-engine serial/reuse ABBA order."""
    import json
    import hashlib
    from pathlib import Path
    from types import MethodType
    import bench_qwen3_5_mtp_production as bench
    from nano_qwen.engine.model_runner import ModelRunner

    prompt = bench.make_prompt(model, prompt_tokens)
    original_engine = bench.LLMEngine

    def eager_reused(runner, ids, positions):
        with b1_tiles():
            return forward_reused(runner, ids, positions)

    # Correctness runs are separate from the two formal timing samples.
    _, expected = bench.run_mode(model, False, prompt, 128, 60.0, True)
    with patch.object(ModelRunner, "forward_verify", eager_reused):
        _, actual = bench.run_mode(
            model, True, prompt, 128, 60.0, True, mtp_model=mtp_model
        )
    if actual != expected or len(actual) != 128:
        raise AssertionError("Reused eager verification changed committed tokens")
    data = dict(
        model=model,
        mtp_model=mtp_model,
        prompt_tokens=prompt_tokens,
        output_tokens=128,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        eager_exact=True,
        eager_output_tokens=128,
        experimental=True,
        profiler=False,
        samples=[],
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    references = []
    for enabled in (False, True):
        held, exits = [], []
        selected = None
        serial = None
        cases = (
            [(1, "off"), (2, "off")]
            if not enabled
            else [(1, "serial"), (1, "reused"), (2, "reused"), (2, "serial")]
        )

        def reuse_engine(*args, **kwargs):
            nonlocal selected, serial
            if not held:
                engine = original_engine(*args, **kwargs)
                exits.append(engine.exit)
                engine.exit = lambda: None
                held.append(engine)
                serial = engine.model_runner.forward_verify
                selected = "serial"
            runner = held[0].model_runner
            if enabled and selected != mode:
                runner.forward_verify = (
                    MethodType(forward_reused, runner) if mode == "reused" else serial
                )
                with b1_tiles():
                    runner.cuda_graphs._capture_verify()
                selected = mode
            for key in runner.mtp_stats:
                runner.mtp_stats[key] = 0
            return held[0]

        try:
            with patch.object(bench, "LLMEngine", reuse_engine):
                for repeat, mode in cases:
                    metrics, tokens = bench.run_mode(
                        model,
                        enabled,
                        prompt,
                        128,
                        60.0,
                        False,
                        mtp_model=mtp_model if enabled else None,
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
                    if (
                        metrics["tokens"] != 127
                        or metrics["stop_reason"] != "max_tokens"
                    ):
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
            for close in exits:
                close()
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
            cases = (
                [(True, True)]
                if args.graph_only
                else [(False, False), (True, False), (True, True)]
            )
            reports = []
            for pinned, graph in cases:
                report = diagnose(engine.model_runner, probe, pinned, graph)
                reports.append(report)
                print(
                    "STRICT",
                    report["strict_pass"],
                    "graph",
                    graph,
                    "pinned",
                    pinned,
                    "first_difference",
                    report["first_difference"],
                    flush=True,
                )
            data = dict(
                model=args.model,
                mtp_model=args.mtp_model,
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
