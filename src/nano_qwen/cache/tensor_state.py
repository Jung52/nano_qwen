"""Export straight into typed LMCache CPU buffers; no payload codec or clones."""
import torch

from .state import Snapshot, validate


def synchronize(runner):
    if runner.kv_cache.device.type == "cuda":
        torch.cuda.synchronize(runner.kv_cache.device)


def describe(runner, seq, identity):
    if runner.world_size != 1 or getattr(runner, "_pending", None) is not None:
        raise ValueError("export requires completed single-GPU sampling")
    if (seq.is_finished or seq.is_prefill or seq.num_scheduled_tokens != 0
            or seq.num_cached_tokens != len(seq.token_ids) - 1 or seq.num_cached_tokens < 1):
        raise ValueError("request is not at a committed decode boundary")
    n, block_size = seq.num_cached_tokens, runner.block_size
    count = (n + block_size - 1) // block_size
    if len(seq.block_table) != count or seq.seq_id not in runner.input_batch.seq_id_to_slot:
        raise ValueError("invalid source block/slot assignment")
    shape = list(runner.kv_cache.shape)
    shape[2] = count
    specs = {"kv": (shape, runner.kv_cache.dtype)}
    for i, layer in enumerate(runner.gdn_layers):
        specs[f"gdn.{i}.conv"] = (list(layer.conv_states.shape[1:]), layer.conv_states.dtype)
        specs[f"gdn.{i}.recurrent"] = (list(layer.recurrent_states.shape[1:]), layer.recurrent_states.dtype)
    layout = [dict(name=name, shape=shape, dtype=str(dtype).removeprefix("torch."))
              for name, (shape, dtype) in sorted(specs.items())]
    metadata = dict(schema=1, model_id=identity, block_size=block_size,
        tokens=list(seq.token_ids), prompt_len=seq.num_prompt_tokens, cached_tokens=n,
        temperature=seq.temperature, max_tokens=seq.max_tokens, ignore_eos=seq.ignore_eos,
        gdn_layers=len(runner.gdn_layers))
    return metadata, layout


def copy_into(runner, seq, identity, metadata, tensors):
    expected, _ = describe(runner, seq, identity)
    if metadata != expected:
        raise ValueError("source progress changed after reserving L1 buffers")
    validate(runner, Snapshot(metadata, tensors), identity)
    if any(not tensor.is_contiguous() for tensor in tensors.values()):
        raise ValueError("L1 tensors must be contiguous")
    synchronize(runner)
    slot = runner.input_batch.seq_id_to_slot[seq.seq_id]
    ids = torch.tensor(seq.block_table, dtype=torch.int64, device=runner.kv_cache.device)
    # index_select gathers GPU pages. The destination is the mapped LMCache
    # tensor itself, not an intermediate .cpu().clone() allocation.
    tensors["kv"].copy_(runner.kv_cache.index_select(2, ids), non_blocking=False)
    n = seq.num_cached_tokens
    if n % runner.block_size:
        tensors["kv"][:, :, -1, n % runner.block_size:].zero_()
    for i, layer in enumerate(runner.gdn_layers):
        tensors[f"gdn.{i}.conv"].copy_(layer.conv_states[slot], non_blocking=False)
        tensors[f"gdn.{i}.recurrent"].copy_(layer.recurrent_states[slot], non_blocking=False)
    synchronize(runner)  # Commit may only publish completed writes.
