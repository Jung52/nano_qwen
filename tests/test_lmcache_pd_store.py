"""Real LMCache L1 + HTTP tests. Run in the isolated cache-server environment."""
import os
import threading
import time

import pytest

from nano_qwen.cache.connector import LMCacheConnector
from nano_qwen.cache.lmcache_l1 import LMCacheL1Store
from nano_qwen.cache.service import make_server


@pytest.fixture
def store():
    # Missing LMCache is a skip in the ordinary inference environment; with
    # REQUIRE_LMCACHE=1 it fails, preventing a fake/skip-only acceptance run.
    if os.environ.get("REQUIRE_LMCACHE") != "1":
        pytest.importorskip("lmcache")
    result = LMCacheL1Store(1 << 20)
    yield result
    result.close()


def test_real_l1_allocation_commit_read_delete(store):
    initial = store.status()["used_bytes"]
    payload = bytes(range(256)) * 32
    receipt = store.put("one", payload)
    assert receipt["backend"] == "lmcache_l1" and receipt["committed"]
    assert store.status()["used_bytes"] > initial
    assert store.get("one") == payload
    assert store.put("one", payload) == receipt
    with pytest.raises(FileExistsError): store.put("one", b"different")
    store.delete("one")
    assert store.status()["used_bytes"] == initial
    with pytest.raises(KeyError): store.get("one")


def test_capacity_failure_is_not_a_partial_commit(store):
    with pytest.raises(MemoryError): store.put("too-big", b"x" * (2 << 20))
    assert store.status()["payload_objects"] == 0
    assert store.status()["used_bytes"] == 0


def test_expiry_reclaims_real_l1_memory(store):
    store.ttl = 0.01
    store.put("expiry", b"abc")
    time.sleep(0.02)
    assert store.status()["payload_objects"] == 0
    assert store.status()["used_bytes"] == 0


def test_http_two_independent_clients_use_one_l1(store):
    server = make_server("127.0.0.1", 0, store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        address = f"http://127.0.0.1:{server.server_port}"
        p, d = LMCacheConnector(address), LMCacheConnector(address)
        assert p.health()["backend"] == "lmcache_l1"
        p.put("transfer-1", b"real LMCache object")
        assert d.get("transfer-1") == b"real LMCache object"
        d.delete("transfer-1")
        with pytest.raises(KeyError): p.get("transfer-1")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_real_l1_handoff_preserves_kv_gdn_and_continuation(store):
    from test_lmcache_pd_state import engine, Sequence, SamplingParams, Status, finish
    from nano_qwen.cache import engine_ops
    p, d, reference = engine(), engine(), engine()
    params = SamplingParams(temperature=1e-8, max_tokens=6, ignore_eos=True)
    tokens = [3 + i % 17 for i in range(257)]
    seq = Sequence(tokens, params)
    reference.scheduler.add(seq)
    expected = finish(reference)[0][1]
    server = make_server("127.0.0.1", 0, store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = LMCacheConnector(f"http://127.0.0.1:{server.server_port}")
        engine_ops.prefill_to_cache(p, tokens, params, client, "full-state", Sequence)
        assert store.status()["used_bytes"] > 0
        restored = engine_ops.resume_from_cache(d, client, "full-state", Sequence,
                                                SamplingParams, Status.RUNNING)
        assert restored.num_cached_tokens == 257
        assert finish(d)[0][1] == expected
        assert d.model_runner.prefill_calls == 0
        assert store.status()["used_bytes"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
