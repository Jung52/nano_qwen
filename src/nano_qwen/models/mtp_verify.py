"""B1 FP8 verification: two-row matrices and sequential single-token cores."""

from contextlib import ExitStack
from dataclasses import fields

import torch

from nano_qwen.utils.context import get_context, set_context


def _rowwise(function, *values):
    parts = [
        function(*(None if v is None else v[row : row + 1] for v in values))
        for row in range(2)
    ]
    if isinstance(parts[0], tuple):
        return tuple(torch.cat([v[i] for v in parts]) for i in range(len(parts[0])))
    return torch.cat(parts)


def forward_verify_reuse(model, ids, positions, observe=None):
    """Return two hidden rows, with the first GDN update saved for rejection.

    The caller enforces FP8/TP1. Context routes both rows to one request;
    snapshots and ordinary cores retain the serial path's arithmetic.
    ``observe`` is optional stage instrumentation for strict diagnostics.
    """
    original = get_context()
    if (
        ids.ndim != 1
        or ids.numel() != 2
        or positions.shape != ids.shape
        or original.state_indices is None
        or original.state_indices.numel() != 1
    ):
        raise ValueError("MTP reuse requires B1 and two consecutive token rows")
    policy = {f.name: getattr(original, f.name) for f in fields(original)}
    policy["gemm_reference_m"] = 1
    hidden = model.model.embed_tokens(ids)
    residual = None
    try:
        for index, layer in enumerate(model.model.layers):
            set_context(**policy)
            # Preserve M1 reduction scheduling as well as M1 matrix tiles.
            hidden, residual = _rowwise(layer.forward_input, hidden, residual)
            if observe:
                observe(index, "input_norm", hidden)
            linear = layer.block_type == "linear_attention"
            attention = layer.linear_attn if linear else layer.self_attn
            dense = attention.forward_dense_pre(hidden)
            if observe:
                observe(index, "dense_pre", dense)
            cores = []
            for row in range(2):
                set_context(
                    False,
                    slot_mapping=original.slot_mapping[row : row + 1],
                    context_lens=original.context_lens - (1 - row),
                    block_tables=original.block_tables,
                    state_indices=original.state_indices,
                )
                row_dense = tuple(v[row : row + 1] for v in dense)
                if linear:
                    value = attention.forward_decode_core(
                        tuple(v.unsqueeze(1) for v in row_dense)
                    )
                    value = value.reshape(1, -1)
                    if row == 0:
                        attention.save_speculative_state(original.state_indices)
                else:
                    value = attention.forward_decode_core(
                        row_dense, positions[row : row + 1]
                    )
                cores.append(value)
            set_context(**policy)
            core = torch.cat(cores)
            if observe:
                observe(index, "core_output", core)
            output_projection = attention.out_proj if linear else attention.o_proj
            attention_output = output_projection(core)
            if observe:
                observe(index, "attention_output", attention_output)
            hidden, residual = _rowwise(
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
        hidden, _ = _rowwise(model.model.norm, hidden, residual)
        return hidden
    finally:
        set_context(**{f.name: getattr(original, f.name) for f in fields(original)})
