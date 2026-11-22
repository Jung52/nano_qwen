"""Synchronous, one-request P/D handoff orchestration; no model imports.

Only the direct shared-tensor path exists: the producer copies GPU KV/GDN into
LMCache L1's shared CPU pool and the consumer maps it back. The former binary
snapshot + HTTP payload path was removed.
"""
from .tensor_ops import prefill_to_cache as _tensor_prefill, resume_from_cache as _tensor_resume


def require_idle(engine):
    c = engine.config
    if (c.tensor_parallel_size != 1 or c.async_scheduling or c.enable_prefix_cache):
        raise ValueError("PD prototype requires TP=1, async_scheduling=False, enable_prefix_cache=False")
    if (engine.batch_queue or not engine.scheduler.is_finished()
            or engine.scheduler.async_decode is not None):
        raise ValueError("PD prototype requires an idle engine with no in-flight work")


def require_resume_ready(engine):
    """D-side entry guard: allow restore between decode steps, never mid-batch.

    The producer side must still satisfy require_idle(); the consumer may have
    running decode requests, as long as no batch is in flight, nothing waits
    for prefill, and every running request sits at a committed decode boundary.
    """
    c = engine.config
    if (c.tensor_parallel_size != 1 or c.async_scheduling or c.enable_prefix_cache):
        raise ValueError("PD prototype requires TP=1, async_scheduling=False, enable_prefix_cache=False")
    scheduler = engine.scheduler
    if (engine.batch_queue or scheduler.in_flight or scheduler.waiting
            or scheduler.async_decode is not None):
        raise ValueError("PD resume requires no in-flight batch and no waiting prompts")
    for seq in scheduler.running:
        if seq.is_prefill or seq.num_scheduled_tokens:
            raise ValueError("running requests must be at committed decode boundaries")


def require_resume_capacity(engine, cached_tokens, max_tokens, completion_tokens):
    """Free GDN slot plus KV blocks for the remaining generation of every
    active request, so the restored request and all running requests can
    decode to completion without preemption."""
    if not any(item is None for item in engine.model_runner.input_batch.seqs):
        raise ValueError("no free GDN state slot for the restored request")
    manager = engine.scheduler.block_manager
    block_size = manager.block_size

    def future_blocks(cached, remaining, allocated):
        return max(0, (cached + remaining + block_size - 1) // block_size - allocated) + 1

    remaining = max_tokens - completion_tokens
    initial = (cached_tokens + block_size - 1) // block_size
    needed = initial + future_blocks(cached_tokens, remaining, initial)
    for seq in engine.scheduler.running:
        remaining = seq.max_tokens - seq.num_completion_tokens
        needed += future_blocks(seq.num_cached_tokens, remaining, len(seq.block_table))
    if len(manager.free_block_ids) < needed:
        raise MemoryError(f"insufficient destination KV blocks: need {needed} additional, "
                          f"have {len(manager.free_block_ids)} free")


def _require_direct(connector):
    if not getattr(connector, "direct_tensors", False):
        raise ValueError("PD handoff requires nano_qwen.cache.shm_connector.LMCacheShmConnector "
                         "(direct shared tensors); the HTTP payload path was removed")


def prefill_to_cache(engine, prompt, sampling_params, connector, transfer_id, sequence_type):
    _require_direct(connector)
    return _tensor_prefill(engine, prompt, sampling_params, connector, transfer_id, sequence_type)


def resume_from_cache(engine, connector, transfer_id, sequence_type, params_type, running_status,
                      delete_after_restore=True):
    _require_direct(connector)
    return _tensor_resume(engine, connector, transfer_id, sequence_type, params_type,
                          running_status, delete_after_restore)
