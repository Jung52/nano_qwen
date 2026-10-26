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
