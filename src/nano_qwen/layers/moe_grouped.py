"""Grouped BF16 projections for Qwen3.5 routed experts."""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_projection(
    Inputs, Weights, Assignments, ExpertStarts, TileEnds, Output,
    K: tl.constexpr, N: tl.constexpr, NUM_EXPERTS: tl.constexpr, TOP_K: tl.constexpr,
    BLOCK_E: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GATE_UP: tl.constexpr,
):
    tile = tl.program_id(0)
    experts = tl.arange(0, BLOCK_E)
    tile_ends = tl.load(TileEnds + experts, experts < NUM_EXPERTS, other=2147483647)
    expert = tl.sum((tile >= tile_ends).to(tl.int32), 0)
    if expert < NUM_EXPERTS:
        first_tile = tl.load(TileEnds + expert - 1, expert > 0, other=0)
        start = tl.load(ExpertStarts + expert)
        end = tl.load(ExpertStarts + expert + 1)
        rows = start + (tile - first_tile) * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        if GATE_UP:
            assignments = tl.load(Assignments + rows, rows < end, other=0).to(tl.int32)
            input_rows = assignments // TOP_K
        else:
            input_rows = rows

        weight_stride: tl.constexpr = 2 * N if GATE_UP else N
        weight_base = Weights + expert * weight_stride * K
        accum = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
        if GATE_UP:
            up_accum = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
        for k in range(tl.cdiv(K, BLOCK_K)):
            inner = k * BLOCK_K + tl.arange(0, BLOCK_K)
            values = tl.load(
                Inputs + input_rows[:, None] * K + inner[None, :],
                (rows[:, None] < end) & (inner[None, :] < K), other=0,
            )
            weight_offset = inner[:, None] + columns[None, :] * K
            weight_mask = (inner[:, None] < K) & (columns[None, :] < N)
            weight = tl.load(weight_base + weight_offset, weight_mask, other=0)
            accum = tl.dot(values, weight, accum)
            if GATE_UP:
                up = tl.load(weight_base + N * K + weight_offset, weight_mask, other=0)
                up_accum = tl.dot(values, up, up_accum)

        if GATE_UP:
            # Preserve the BF16 outputs of the linear, SiLU and multiply operations.
            gate = accum.to(tl.bfloat16).to(tl.float32)
            activated = (gate / (1 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
            up = up_accum.to(tl.bfloat16).to(tl.float32)
            result = activated * up
            output_rows = rows
        else:
            result = accum
            output_rows = tl.load(Assignments + rows, rows < end, other=0).to(tl.int32)
        tl.store(
            Output + output_rows[:, None] * N + columns[None, :], result,
            (rows[:, None] < end) & (columns[None, :] < N),
        )


@triton.jit
def _combine_experts(
    Projected, RoutingWeights, Output,
    HIDDEN: tl.constexpr, TOP_K: tl.constexpr, BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    output = tl.full((BLOCK,), 0, tl.float32)
    # Match the eager loop's ascending expert order and BF16 accumulation.
    for route in range(TOP_K):
        assignment = token * TOP_K + route
        value = tl.load(Projected + assignment * HIDDEN + columns, columns < HIDDEN, other=0)
        weight = tl.load(RoutingWeights + token * TOP_K + route)
        contribution = (value.to(tl.float32) * weight.to(tl.float32)).to(tl.bfloat16)
        output = (output + contribution.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    tl.store(Output + token * HIDDEN + columns, output, columns < HIDDEN)


def grouped_experts(hidden, expert_ids, expert_weights, gate_up, down):
    tokens, hidden_size = hidden.shape
    num_experts, _, intermediate_size = down.shape
    top_k = expert_ids.size(1)
    num_assignments = tokens * top_k
    block_m, block_n, block_k = 16, 64, 64

    # Order each token's routes for reduction, then group all assignments by
    # expert. The tensors and launch grid depend only on T, K, and model sizes.
    ordered_ids, route_order = expert_ids.sort(dim=1)
    ordered_weights = expert_weights.gather(1, route_order)
    sorted_ids, assignments = ordered_ids.flatten().sort()
    expert_starts = torch.searchsorted(
        sorted_ids, torch.arange(num_experts + 1, device=hidden.device), out_int32=True,
    )
    expert_counts = expert_starts[1:] - expert_starts[:-1]
    tile_ends = ((expert_counts + block_m - 1) // block_m).cumsum(0, dtype=torch.int32)
    # Each nonempty expert consumes one tile for its first row. Every additional
    # tile requires BLOCK_M more rows for that expert.
    max_groups = min(num_experts, num_assignments)
    max_tiles = max_groups + (num_assignments - max_groups) // block_m

    intermediate = hidden.new_empty((num_assignments, intermediate_size))
    projected = hidden.new_empty((num_assignments, hidden_size))
    output = torch.empty_like(hidden)
    block_e = triton.next_power_of_2(num_experts)
    _grouped_projection[(max_tiles, triton.cdiv(intermediate_size, block_n))](
        hidden, gate_up, assignments, expert_starts, tile_ends, intermediate,
        hidden_size, intermediate_size, num_experts, top_k,
        block_e, block_m, block_n, block_k, True,
    )
    _grouped_projection[(max_tiles, triton.cdiv(hidden_size, block_n))](
        intermediate, down, assignments, expert_starts, tile_ends, projected,
        intermediate_size, hidden_size, num_experts, top_k,
        block_e, block_m, block_n, block_k, False,
    )
    _combine_experts[(tokens, triton.cdiv(hidden_size, 256))](
        projected, ordered_weights, output, hidden_size, top_k, 256,
    )
    return output
