"""Synchronous, one-request P/D handoff orchestration; no model imports."""
from copy import copy

from .state import pack, unpack


def require_idle(engine):
    c = engine.config
    if (c.tensor_parallel_size != 1 or c.async_scheduling or c.enable_prefix_cache):
        raise ValueError("PD prototype requires TP=1, async_scheduling=False, enable_prefix_cache=False")
    if (engine.batch_queue or not engine.scheduler.is_finished()
            or engine.scheduler.async_decode is not None):
        raise ValueError("PD prototype requires an idle engine with no in-flight work")


def prefill_to_cache(engine, prompt, sampling_params, connector, transfer_id, sequence_type):
    require_idle(engine)
    if sampling_params.max_tokens < 2 or not sampling_params.ignore_eos:
        raise ValueError("initial PD prototype requires max_tokens>=2 and ignore_eos=True")
    tokens = engine.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
    if not tokens or len(tokens) + sampling_params.max_tokens > engine.config.max_model_len:
        raise ValueError("empty prompt or prompt+output exceeds max_model_len")
    connector.health()
    identity = engine.model_runner.call("pd_model_identity")
    seq = sequence_type(tokens, sampling_params)
    engine.scheduler.add(seq)
    # Partial prefill chunks do not sample; finish exactly the prompt.
    limit = len(tokens) + 2
    for _ in range(limit):
        engine.step()
        if seq.num_cached_tokens == len(tokens) and seq.num_completion_tokens == 1:
            break
    else:
        raise RuntimeError("prefill did not reach a decode boundary")
    if (seq.num_cached_tokens != len(tokens) or seq.num_completion_tokens != 1
            or engine.batch_queue or engine.scheduler.in_flight):
        raise RuntimeError("unexpected prefill progress/in-flight work")
    # The ordinary scheduler leaves is_prefill describing the previous
    # dispatch until it schedules decode. Mark the completed boundary without
    # actually dispatching/consuming the first generated token a second time.
    seq.is_prefill = False
    snapshot = engine.model_runner.call("export_request_state", seq, identity)
    payload = pack(snapshot)
    # The producer retains live GPU state if put/ACK fails.
    receipt = connector.put(transfer_id, payload)
    source_slot = engine.model_runner.input_batch.seq_id_to_slot[seq.seq_id]
    source_blocks = list(seq.block_table)
    engine.scheduler.running.remove(seq)
    engine.model_runner.call("remove_request", seq.seq_id)
    engine.scheduler.block_manager.deallocate(seq)
    return dict(receipt, prompt_tokens=len(tokens), first_token=snapshot.metadata["tokens"][-1],
                model_id=identity, source_slot=source_slot, source_blocks=source_blocks)


def resume_from_cache(engine, connector, transfer_id, sequence_type, params_type, running_status,
                      delete_after_restore=True):
    require_idle(engine)
    connector.health()
    snapshot = unpack(connector.get(transfer_id))
    identity = engine.model_runner.call("pd_model_identity")
    # Validate every model/tensor/progress field before allocating GPU blocks.
    engine.model_runner.call("validate_request_state", snapshot, identity)
    m = snapshot.metadata
    params = params_type(temperature=m["temperature"], max_tokens=m["max_tokens"],
                         ignore_eos=m["ignore_eos"])
    seq = sequence_type(m["tokens"][:m["prompt_len"]], params)
    seq.token_ids = list(m["tokens"])
    seq.num_tokens = len(seq.token_ids)
    seq.last_token = seq.token_ids[-1]
    seq.num_cached_tokens = m["cached_tokens"]
    seq.num_scheduled_tokens = 0
    seq.is_prefill = False
    seq.status = running_status
    # Allocate only processed tokens. When L is exactly block-aligned, the
    # scheduler allocates the page for the first generated token on next decode.
    allocation = copy(seq)
    allocation.num_tokens = seq.num_cached_tokens
    allocation.block_table = []
    count = allocation.num_blocks
    manager = engine.scheduler.block_manager
    if len(manager.free_block_ids) < count:
        raise MemoryError(f"need {count} KV blocks; have {len(manager.free_block_ids)}")
    manager.allocate(allocation, 0)
    seq.block_table = allocation.block_table
    try:
        engine.model_runner.call("restore_request_state", seq, snapshot, identity)
    except BaseException:
        manager.deallocate(seq)
        engine.model_runner.call("remove_request", seq.seq_id)
        raise
    engine.scheduler.running.append(seq)
    seq.pd_handoff = dict(transfer_id=transfer_id, restored=True, acknowledged=False)
    if delete_after_restore:
        try:
            connector.delete(transfer_id)
            seq.pd_handoff["acknowledged"] = True
        except Exception as exc:
            # State is already restored; preserve the runnable request and
            # expose the failed cleanup so callers can retry ACK independently.
            seq.pd_handoff["ack_error"] = str(exc)
    return seq
