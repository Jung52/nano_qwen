"""FP8 verification: 2B dense rows and two ordered B-request state steps."""

from contextlib import ExitStack
from dataclasses import fields

import torch

from nano_qwen.utils.context import get_context, set_context


def _rowwise(function, batch, *values):
    parts = [
        function(*(None if v is None else v[start : start + batch] for v in values))
        for start in (0, batch)
    ]
    if isinstance(parts[0], tuple):
        return tuple(torch.cat([v[i] for v in parts]) for i in range(len(parts[0])))
    return torch.cat(parts)


def forward_verify_reuse(model, ids, positions, observe=None):
    """Return request-major hidden rows, saving each request's first state.

    Dense work is time-major [A0,B0,A1,B1], preserving contiguous B-row
    reference arithmetic for norms and excluded BF16 projections. External
    ids, slots and outputs stay request-major [A0,A1,B0,B1]. No padding.
    ``observe`` is optional stage instrumentation for strict diagnostics.
    """
    original = get_context()
    batch = 0 if original.state_indices is None else original.state_indices.numel()
    if (
        ids.ndim != 1
        or batch not in (1, 2)
        or ids.numel() != 2 * batch
        or positions.shape != ids.shape
    ):
        raise ValueError("MTP reuse requires B1/B2 with two token rows per request")
    policy = {f.name: getattr(original, f.name) for f in fields(original)}
    policy["gemm_reference_m"] = batch
    time_ids = ids.reshape(batch, 2).T.contiguous().flatten()
    time_positions = positions.reshape(batch, 2).T.contiguous()
    time_slots = original.slot_mapping.reshape(batch, 2).T.contiguous()
    hidden = model.model.embed_tokens(time_ids)
    residual = None
    try:
        for index, layer in enumerate(model.model.layers):
            set_context(**policy)
            # Preserve the ordinary B-row reductions as well as matrix tiles.
            hidden, residual = _rowwise(layer.forward_input, batch, hidden, residual)
            if observe:
                observe(index, "input_norm", hidden)
            linear = layer.block_type == "linear_attention"
            attention = layer.linear_attn if linear else layer.self_attn
            dense = attention.forward_dense_pre(hidden)
            if observe:
                observe(index, "dense_pre", dense)
            cores = []
            for row in range(2):
                start = row * batch
                set_context(
                    False,
                    slot_mapping=time_slots[row],
                    context_lens=original.context_lens - (1 - row),
                    block_tables=original.block_tables,
                    state_indices=original.state_indices,
                )
                row_dense = tuple(v[start : start + batch] for v in dense)
                if linear:
                    value = attention.forward_decode_core(
                        tuple(v.unsqueeze(1) for v in row_dense)
                    )
                    value = value.reshape(batch, -1)
                    if row == 0:
                        attention.save_speculative_state(original.state_indices)
                else:
                    value = attention.forward_decode_core(
                        row_dense, time_positions[row]
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
                layer.post_attention_layernorm, batch, attention_output, residual
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
        hidden, _ = _rowwise(model.model.norm, batch, hidden, residual)
        return hidden.reshape(2, batch, -1).transpose(0, 1).reshape(2 * batch, -1)
    finally:
        set_context(**{f.name: getattr(original, f.name) for f in fields(original)})
