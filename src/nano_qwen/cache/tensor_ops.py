"""Single-request handoff through actual LMCache typed shared-memory tensors."""
from copy import copy
from time import perf_counter_ns

from . import tensor_state
from .state import Snapshot


def _timed(metrics, name, operation):
    start = perf_counter_ns()
    try:
        return operation()
    finally:
        metrics[name] = metrics.get(name, 0.0) + (perf_counter_ns() - start) / 1e6


def prefill_to_cache(engine, prompt, params, connector, key, sequence_type):
    from .engine_ops import require_idle
    require_idle(engine)
    if params.max_tokens < 2 or not params.ignore_eos:
        raise ValueError("tensor PD requires max_tokens>=2 and ignore_eos=True")
    tokens = engine.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
    if not tokens or len(tokens) + params.max_tokens > engine.config.max_model_len:
        raise ValueError("empty prompt or prompt+output exceeds max_model_len")
    metrics, started = {}, perf_counter_ns()
    _timed(metrics, "health_ms", connector.health)
    runner = engine.model_runner
    identity = _timed(metrics, "identity_ms", lambda: runner.call("pd_model_identity"))
    seq = sequence_type(tokens, params)
    engine.scheduler.add(seq)
    start = perf_counter_ns()
    for _ in range(len(tokens) + 2):
        engine.step()
        if seq.num_cached_tokens == len(tokens) and seq.num_completion_tokens == 1:
            break
    else:
        raise RuntimeError("prefill did not reach a committed decode boundary")
    metrics["prefill_ms"] = (perf_counter_ns() - start) / 1e6
    first_token_at = perf_counter_ns()
    if engine.batch_queue or engine.scheduler.in_flight:
        raise RuntimeError("in-flight work at handoff boundary")
    seq.is_prefill = False
    metadata, layout = _timed(metrics, "describe_ms", lambda: tensor_state.describe(runner, seq, identity))
    lease = _timed(metrics, "reserve_write_ms", lambda: connector.allocate(key, metadata, layout))
    try:
        _timed(metrics, "d2h_ms", lambda: tensor_state.copy_into(runner, seq, identity, metadata, lease.tensors))
        receipt = _timed(metrics, "commit_ms", lambda: connector.commit(lease))
    except BaseException:
        # Do not recycle a buffer if a failed CUDA operation could still use it.
        # On sync failure the pidfd lease survives until this process exits.
        try:
            tensor_state.synchronize(runner)
        except Exception:
            pass
        else:
            if not lease.closed:
                try:
                    connector.abort(lease)
                except Exception:
                    pass
        raise
    slot = runner.input_batch.seq_id_to_slot[seq.seq_id]
    blocks = list(seq.block_table)
    start = perf_counter_ns()
    engine.scheduler.running.remove(seq)
    runner.call("remove_request", seq.seq_id)
    engine.scheduler.block_manager.deallocate(seq)
    metrics["producer_release_ms"] = (perf_counter_ns() - start) / 1e6
    metrics["prefill_total_ms"] = (perf_counter_ns() - started) / 1e6
    return dict(receipt, model_id=identity, prompt_tokens=len(tokens),
                first_token=metadata["tokens"][-1], source_slot=slot, source_blocks=blocks,
                pd_metrics=metrics, first_token_at_ns=first_token_at,
                payload_pack_calls=0, payload_unpack_calls=0, payload_http_bytes=0)


def resume_from_cache(engine, connector, key, sequence_type, params_type, running_status,
                      delete_after_restore=True):
    from .engine_ops import require_idle
    require_idle(engine)
    metrics, started = {}, perf_counter_ns()
    _timed(metrics, "health_ms", connector.health)
    runner = engine.model_runner
    identity = _timed(metrics, "identity_ms", lambda: runner.call("pd_model_identity"))
    lease = _timed(metrics, "acquire_ms", lambda: connector.acquire(key))
    seq, allocated, restored = None, False, False
    try:
        snapshot = Snapshot(lease.metadata, lease.tensors)  # Tensor views; no data reconstruction.
        _timed(metrics, "validate_ms", lambda: runner.call("validate_request_state", snapshot, identity))
        m = lease.metadata
        start = perf_counter_ns()
        params = params_type(temperature=m["temperature"], max_tokens=m["max_tokens"], ignore_eos=m["ignore_eos"])
        seq = sequence_type(m["tokens"][:m["prompt_len"]], params)
        seq.token_ids = list(m["tokens"])
        seq.num_tokens, seq.last_token = len(seq.token_ids), seq.token_ids[-1]
        seq.num_cached_tokens, seq.num_scheduled_tokens = m["cached_tokens"], 0
        seq.is_prefill, seq.status = False, running_status
        allocation = copy(seq)
        allocation.num_tokens, allocation.block_table = seq.num_cached_tokens, []
        manager = engine.scheduler.block_manager
        if len(manager.free_block_ids) < allocation.num_blocks:
            raise MemoryError("insufficient destination KV blocks")
        manager.allocate(allocation, 0)
        seq.block_table, allocated = allocation.block_table, True
        metrics["allocate_gpu_ms"] = (perf_counter_ns() - start) / 1e6
        _timed(metrics, "h2d_ms", lambda: runner.call("restore_request_state", seq, snapshot, identity))
        restored = True  # Production restore synchronizes all copies before returning.
        engine.scheduler.running.append(seq)
        seq.pd_handoff = dict(transfer_id=key, restored=True, acknowledged=False,
                              transport="lmcache_l1_shm", metrics=metrics)
        try:
            _timed(metrics, "ack_ms", lambda: connector.release(lease, delete=delete_after_restore))
            seq.pd_handoff["acknowledged"] = delete_after_restore
            seq.pd_handoff["lease_released"] = True
        except Exception as exc:
            # The GPU request is usable. Keep enough information to retry ACK.
            seq.pd_handoff.update(ack_error=str(exc), lease=lease)
        metrics["restore_total_ms"] = (perf_counter_ns() - started) / 1e6
        return seq
    except BaseException:
        if allocated and not restored:
            engine.scheduler.block_manager.deallocate(seq)
            runner.call("remove_request", seq.seq_id)
        try:
            tensor_state.synchronize(runner)
        except Exception:
            pass
        else:
            try:
                connector.release(lease, delete=False)
            except Exception:
                pass
        raise
