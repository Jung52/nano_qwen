"""Real named LMCache L1; CPU model fixture; cross-process producer/consumer.

Production state, scheduler and block management are exercised. Only model
compute is deterministic CPU arithmetic. No serialized tensor payload fallback.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
import torch
import importlib.util

from nano_qwen.cache import engine_ops, state
from nano_qwen.cache.shm_connector import LMCacheShmConnector, TensorLease
from nano_qwen.cache.shm_service import make_server
from nano_qwen.cache.lmcache_shm import LMCacheShmStore
from pd_cpu_fixture import engine, Sequence, SamplingParams, Status, finish

ROOT = Path(__file__).resolve().parents[1]


def ready_engine(blocks=16):
    e = engine(blocks=blocks, budget=128)
    e.prefill_to_lmcache = lambda tokens, params, connector, key: engine_ops.prefill_to_cache(
        e, tokens, params, connector, key, Sequence)
    e.resume_from_lmcache = lambda connector, key, delete_after_restore=True: engine_ops.resume_from_cache(
        e, connector, key, Sequence, SamplingParams, Status.RUNNING, delete_after_restore)
    return e


@pytest.fixture
def endpoint():
    if os.environ.get("REQUIRE_LMCACHE") != "1":
        pytest.importorskip("lmcache")
    store = LMCacheShmStore(1 << 20)
    server = make_server("127.0.0.1", 0, store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = LMCacheShmConnector(f"http://127.0.0.1:{server.server_port}")
    try:
        yield store, client
    finally:
        # All normal tests explicitly finish their leases before teardown.
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        store.close()


def sample(client, key="sample"):
    writer = client.allocate(key, {"hello": "world"}, [dict(name="kv", shape=[64, 64], dtype="bfloat16")])
    writer.tensors["kv"].fill_(7)
    client.commit(writer)


def test_real_l1_typed_storage_has_no_binary_payload(endpoint):
    store, client = endpoint
    sample(client)
    assert store.status()["used_bytes"] > 0
    k = store._key("sample")
    error, obj = store.manager.reserve_read([k])[k]
    try:
        assert error == store.Error.SUCCESS
        tensor = obj.get_tensor(0)
        assert tensor.dtype == torch.bfloat16 and tensor.shape == (64, 64)
        assert torch.all(tensor == 7)
        assert tensor.data_ptr() - store.arena.ptr == store.entries["sample"]["tensors"][0]["offset"]
    finally:
        store.manager.finish_read([k])
    reader = client.acquire("sample")
    assert torch.all(reader.tensors["kv"] == 7)
    assert reader.metadata == {"hello": "world"}
    client.release(reader, delete=True)
    assert store.status()["used_bytes"] == 0


@pytest.mark.parametrize("length", [1, 127, 255, 256, 257, 511, 512, 513])
def test_native_handoff_full_state_relocation_and_output(endpoint, monkeypatch, length):
    store, client = endpoint
    def forbidden(*args, **kwargs):
        raise AssertionError("the native tensor path must not call pack/unpack/capture")
    for module in (engine_ops, state):
        for name in ("pack", "unpack"):
            assert not hasattr(module, name), f"payload codec {module}.{name} still present"
            monkeypatch.setattr(module, name, forbidden, raising=False)
    p, d, reference = ready_engine(), ready_engine(), ready_engine()
    params = SamplingParams(temperature=1e-8, max_tokens=8, ignore_eos=True)
    tokens = [3 + i % 17 for i in range(length)]
    ref = Sequence(tokens, params)
    reference.scheduler.add(ref)
    while not (ref.num_cached_tokens == length and ref.num_completion_tokens == 1):
        reference.step()
    ref.is_prefill = False
    expected_state = state.capture(reference.model_runner, ref, "test-model")
    monkeypatch.setattr(state, "capture", forbidden)
    receipt = p.prefill_to_lmcache(tokens, params, client, "native")
    assert receipt["committed"] and receipt["payload_pack_calls"] == 0
    assert receipt["payload_http_bytes"] == 0 and p.is_finished()
    reader = client.acquire("native")
    assert reader.metadata == expected_state.metadata
    for name, tensor in expected_state.tensors.items():
        assert torch.equal(tensor.reshape(-1).view(torch.uint8), reader.tensors[name].reshape(-1).view(torch.uint8))
    client.release(reader)
    guard = Sequence([9], params)
    d.scheduler.block_manager.allocate(guard, 0)
    d.model_runner.input_batch.update([guard])
    seq = d.resume_from_lmcache(client, "native")
    assert d.model_runner.input_batch.seq_id_to_slot[seq.seq_id] == 1
    assert seq.pd_handoff["acknowledged"]
    assert finish(d)[0][1] == finish(reference)[0][1]
    assert d.model_runner.prefill_calls == 0 and d.model_runner.decode_calls == 7
    assert store.status()["used_bytes"] == 0
    assert d.scheduler.block_manager.used_block_ids == set(guard.block_table)


def test_live_leases_are_not_reclaimed_by_ttl(endpoint):
    store, client = endpoint
    store.ttl = 0.001
    writer = client.allocate("live", {}, [dict(name="kv", shape=[128], dtype="float32")])
    time.sleep(0.01)
    assert store.status()["states"]["writing"] == 1
    with pytest.raises(FileExistsError):
        client.delete("live")
    store.ttl = 900
    writer.tensors["kv"].fill_(2)
    client.commit(writer)
    reader = client.acquire("live")
    store.ttl = 0.001
    time.sleep(0.01)
    assert store.status()["states"]["reading"] == 1
    assert torch.all(reader.tensors["kv"] == 2)
    with pytest.raises(FileExistsError):
        client.delete("live")
    client.release(reader, delete=True)


def test_uncommitted_state_hidden_and_active_reader_exclusive(endpoint):
    store, client = endpoint
    writer = client.allocate("hidden", {}, [dict(name="kv", shape=[64], dtype="float32")])
    with pytest.raises(FileExistsError):
        client.acquire("hidden")
    writer.tensors["kv"].zero_()
    client.commit(writer)
    reader = client.acquire("hidden")
    with pytest.raises(FileExistsError):
        client.acquire("hidden")
    first = client.release(reader, delete=True)
    assert client.release(reader, delete=True) == first  # ACK retry is idempotent.
    assert store.status()["used_bytes"] == 0


def test_capacity_failure_leaves_existing_writer_intact(endpoint):
    store, client = endpoint
    writer = client.allocate("large", {}, [dict(name="kv", shape=[900 << 10], dtype="uint8")])
    with pytest.raises(MemoryError):
        client.allocate("no-space", {}, [dict(name="kv", shape=[256 << 10], dtype="uint8")])
    assert store.status()["payload_objects"] == 1
    client.abort(writer)
    assert store.status()["used_bytes"] == 0


def test_commit_failure_keeps_producer_gpu_state(endpoint, monkeypatch):
    store, client = endpoint
    p = ready_engine()
    params = SamplingParams(temperature=1e-8, max_tokens=8, ignore_eos=True)
    def fail(lease):
        raise RuntimeError("commit failure")
    monkeypatch.setattr(client, "commit", fail)
    with pytest.raises(RuntimeError, match="commit failure"):
        p.prefill_to_lmcache([3] * 257, params, client, "failed")
    assert len(p.scheduler.running) == 1
    assert len(finish(p)[0][1]) == 8
    assert store.status()["used_bytes"] == 0


def test_validation_and_restore_failure_preserve_cache_and_rollback(endpoint):
    store, client = endpoint
    p, d = ready_engine(), ready_engine()
    params = SamplingParams(temperature=1e-8, max_tokens=8, ignore_eos=True)
    p.prefill_to_lmcache([3] * 257, params, client, "retry")
    original = d.model_runner.call
    def wrong(method, *args):
        return "wrong-model" if method == "pd_model_identity" else original(method, *args)
    d.model_runner.call = wrong
    with pytest.raises(ValueError, match="identity"):
        d.resume_from_lmcache(client, "retry")
    assert not d.scheduler.block_manager.used_block_ids
    assert store.status()["states"]["ready"] == 1
    def fail(method, *args):
        result = original(method, *args)
        if method == "restore_request_state":
            raise RuntimeError("restore dispatch failed")
        return result
    d.model_runner.call = fail
    with pytest.raises(RuntimeError, match="dispatch failed"):
        d.resume_from_lmcache(client, "retry")
    assert not d.scheduler.block_manager.used_block_ids and not d.model_runner.input_batch.seq_id_to_slot
    d.model_runner.call = original
    seq = d.resume_from_lmcache(client, "retry")
    assert seq.pd_handoff["acknowledged"] and len(finish(d)[0][1]) == 8


def test_ack_failure_preserves_runnable_request_and_can_retry(endpoint, monkeypatch):
    store, client = endpoint
    p, d = ready_engine(), ready_engine()
    params = SamplingParams(temperature=1e-8, max_tokens=8, ignore_eos=True)
    p.prefill_to_lmcache([3] * 257, params, client, "ack")
    original = client.release
    def fail(*args, **kwargs):
        raise RuntimeError("ACK unavailable")
    monkeypatch.setattr(client, "release", fail)
    seq = d.resume_from_lmcache(client, "ack")
    assert seq.pd_handoff["restored"] and not seq.pd_handoff["acknowledged"]
    assert len(finish(d)[0][1]) == 8
    assert store.status()["states"]["reading"] == 1
    monkeypatch.setattr(client, "release", original)
    client.release(seq.pd_handoff.pop("lease"), delete=True)
    assert store.status()["used_bytes"] == 0


def test_bad_descriptor_is_rejected_before_tensor_access(endpoint):
    store, client = endpoint
    writer = client.allocate("bad", {}, [dict(name="kv", shape=[64], dtype="float32")])
    desc = deepcopy(writer.descriptor)
    desc["tensors"][0]["offset"] = desc["pool"]["size"]
    with pytest.raises(ValueError, match="offset"):
        TensorLease(client, desc, writing=True)
    client.abort(writer)


def load_benchmark():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    spec = importlib.util.spec_from_file_location("pd_shm_bench", ROOT / "benchmarks/bench_lmcache_pd_shm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_uses_snapshot_context_instead_of_unused_default_prompt_lengths():
    bench = load_benchmark()
    args = bench.arguments(["--mode", "decode", "--transfer-id", "saved", "--max-model-len", "512"])
    assert args.max_model_len == 512
    with pytest.raises(SystemExit):
        bench.arguments(["--mode", "perf", "--max-model-len", "512"])


@pytest.mark.parametrize("verify", [False, True])
def test_benchmark_distinguishes_performance_and_state_verification(endpoint, verify):
    store, client = endpoint
    bench = load_benchmark()
    e = ready_engine()
    params = SamplingParams(temperature=1e-8, max_tokens=8, ignore_eos=True)
    row = bench.run_pair(e, [3] * 257, params, client, engine_ops, state, Sequence,
                         lambda: None, ("baseline", "handoff"), verify_state=verify)
    assert row["passed"] and row["tokens_equal"]
    h = row["handoff"]
    assert h["performance_valid"] is not verify
    assert h["payload_pack_calls"] == h["payload_unpack_calls"] == h["payload_http_bytes"] == 0
    if verify:
        assert h["state_bytes_equal"] is True and h["destination_slot"] == 1
        assert "e2e_ms" not in h["metrics"]
    else:
        assert h["state_bytes_equal"] is None
        m = h["metrics"]
        assert sum(m.get(k, 0) for k in bench.ATOMIC_STAGES) + m["other_ms"] == pytest.approx(m["e2e_ms"])
    assert store.status()["used_bytes"] == 0


def child(script, endpoint, *args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "tests"), str(ROOT / "src"), env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, "-c", script, endpoint, *map(str, args)],
                          env=env, capture_output=True, text=True, timeout=60, check=True)


def test_independent_prefill_exit_decode_process_restore(endpoint, tmp_path):
    store, client = endpoint
    output = tmp_path / "reference.json"
    child('''
import json,sys
from pathlib import Path
from test_lmcache_pd_shm import ready_engine,SamplingParams,Sequence,finish
from nano_qwen.cache.shm_connector import LMCacheShmConnector
c=LMCacheShmConnector(sys.argv[1]);p=ready_engine();ref=ready_engine()
params=SamplingParams(temperature=1e-8,max_tokens=8,ignore_eos=True)
tokens=[3+i%17 for i in range(257)]
ref.scheduler.add(Sequence(tokens,params));expected=finish(ref)[0][1]
p.prefill_to_lmcache(tokens,params,c,"separate")
Path(sys.argv[2]).write_text(json.dumps(expected));c.close()
''', client.url, output)
    assert (Path("/dev/shm") / store.shm_name).exists()
    assert store.status()["states"]["ready"] == 1
    child('''
import json,sys
from pathlib import Path
from test_lmcache_pd_shm import ready_engine,finish
from nano_qwen.cache.shm_connector import LMCacheShmConnector
c=LMCacheShmConnector(sys.argv[1]);d=ready_engine()
seq=d.resume_from_lmcache(c,"separate")
assert finish(d)[0][1]==json.loads(Path(sys.argv[2]).read_text())
assert d.model_runner.prefill_calls==0 and d.model_runner.decode_calls==7
assert seq.pd_handoff["acknowledged"];c.close()
''', client.url, output)
    assert store.status()["used_bytes"] == 0


def test_store_smoke_cli_sends_only_small_control_messages(endpoint, tmp_path):
    store, client = endpoint
    path = tmp_path / "smoke.json"
    child('''
import runpy,sys
from pathlib import Path
script=Path("benchmarks/bench_lmcache_pd_shm.py").resolve()
sys.path.insert(0,str(script.parent))
sys.argv=[str(script),"--mode","store-smoke","--cache-url",sys.argv[1],"--output",sys.argv[2]]
runpy.run_path(str(script),run_name="__main__")
''', client.url, path)
    value = json.loads(path.read_text())
    row = value["cases"][0]
    assert value["passed"] and row["real_l1_allocation"]
    assert row["nbytes"] == 1 << 20
    assert row["control_sent_bytes"] + row["control_received_bytes"] < 1 << 16
    assert row["payload_http_bytes"] == 0 and store.status()["used_bytes"] == 0


@pytest.mark.parametrize("mode", ["writing", "reading"])
def test_dead_process_releases_lease_without_unlinking_pool(endpoint, mode):
    store, client = endpoint
    if mode == "reading":
        sample(client, "dead")
    child('''
import sys
from nano_qwen.cache.shm_connector import LMCacheShmConnector
c=LMCacheShmConnector(sys.argv[1])
if sys.argv[2]=="writing":
    lease=c.allocate("dead",{},[dict(name="kv",shape=[64],dtype="float32")])
    lease.tensors["kv"].fill_(9)
else:
    lease=c.acquire("dead")
    assert float(lease.tensors["kv"][0,0])==7
# Exit intentionally without commit/release; no SharedMemory tracker is used.
''', client.url, mode)
    status = store.status()
    assert (Path("/dev/shm") / store.shm_name).exists()
    if mode == "writing":
        assert status["used_bytes"] == 0
    else:
        assert status["states"]["ready"] == 1
        reader = client.acquire("dead")
        client.release(reader, delete=True)
