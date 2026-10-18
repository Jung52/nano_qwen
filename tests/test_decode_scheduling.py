"""Ready decode retains a budget under long continuous prompt arrivals."""
from collections import deque
from types import SimpleNamespace

import pytest

from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.engine.sequence import Sequence
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.scheduler import Scheduler


def owner(blocks=128,slots=8,budget=128):
    return Scheduler(SimpleNamespace(max_num_seqs=slots,max_num_batched_tokens=budget,eos=1,
        kvcache_block_size=256,enable_prefix_cache=False,num_kvcache_blocks=blocks))


def ready_decode(scheduler):
    seq = Sequence([2],SamplingParams(temperature=1e-6,max_tokens=20))
    scheduler.add(seq)
    rows,prefill = scheduler.schedule()
    scheduler.postprocess(rows,[7],prefill)
    return seq


def test_ready_decode_gets_first_row_and_prefill_uses_remaining_budget():
    s = owner()
    a = ready_decode(s)
    for _ in range(2):
        s.add(Sequence([2]*1024))
    rows,prefill = s.schedule()
    assert rows[0] is a and a.num_scheduled_tokens == 1
    assert prefill and rows[1].num_scheduled_tokens == 127


def test_continuous_arrivals_through_real_engine_queue_bound_decode_wait():
    s = owner()
    a = ready_decode(s)
    e = LLMEngine.__new__(LLMEngine)
    e.scheduler,e.batch_queue = s,deque()
    e.max_concurrent_batches,e._coalesce_armed = 2,False
    e.coalesce_stats = dict(coalesce_attempts=0,coalesce_successes=0,merged_batches=0,
                           merged_requests=0,batch_size_before={},batch_size_after={})
    trace,pending = [],None
    def call(name,*args):
        nonlocal pending
        if name=='execute_model':
            rows,_ = args
            trace.append(tuple(x.seq_id for x in rows))
            output = [7]*len(rows)
            pending = SimpleNamespace(get_output=lambda:output)
        elif name=='sample_tokens':
            return pending
    e.model_runner = SimpleNamespace(call=call)
    e.config = SimpleNamespace(eos=1)
    for _ in range(8):
        s.add(Sequence([2]*1024))
        e.step()
    indices = [i for i,ids in enumerate(trace) if a.seq_id in ids]
    assert indices[0] == 0 and len(indices)>=4
    assert all(b-a<=2 for a,b in zip(indices,indices[1:]))


def test_partial_prefill_reserves_slots_against_new_admission():
    s = owner(slots=2,budget=64)
    a,b,c = [Sequence([2]*128,SamplingParams(max_tokens=1)) for _ in range(3)]
    for x in (a,b,c):
        s.add(x)
    rows,prefill = s.schedule()
    assert rows==[a]
    s.postprocess(rows,[-1],prefill)
    rows,prefill = s.schedule()
    assert rows==[b]
    s.postprocess(rows,[-1],prefill)
    assert s.schedule()[0] == [a]
    assert s.schedule()[0] == [b]
    assert s.schedule()[0] == []


def test_inflight_owner_blocks_preemption_and_deallocation():
    s = owner()
    a = ready_decode(s)
    s.schedule()
    before = list(a.block_table)
    with pytest.raises(ValueError,match='flight'):
        s.preempt(a)
    assert a.seq_id in s.in_flight and a.block_table == before


def test_replay_resets_its_existing_gdn_slot_without_touching_idle_request(monkeypatch):
    import torch
    from nano_qwen.engine.model_runner import InputBatch,ModelRunner
    a,b = Sequence([2]*3),Sequence([3]*3)
    b.num_cached_tokens = 2
    a.num_scheduled_tokens = 3
    r = ModelRunner.__new__(ModelRunner)
    r.input_batch = InputBatch(2)
    r.input_batch.update([a,b])
    r.rank = 0
    r.batch_slots_gpu = torch.zeros(2,dtype=torch.int64)
    r.sampled_token_ids_gpu = torch.zeros(2,dtype=torch.int64)
    state = torch.tensor([1.,2.])
    def reset(indices):
        state[indices] = 0
    r.gdn_layers = [SimpleNamespace(reset_state=reset)]
    r.prepare_inputs = lambda *args:(torch.ones(3,dtype=torch.int64),torch.arange(3),torch.ones(1))
    r.run_model = lambda *args:torch.zeros(1,3)
    tensor = torch.tensor
    def host(*args,**kwargs):
        kwargs.pop('pin_memory',None)
        return tensor(*args,**kwargs)
    monkeypatch.setattr(torch,'tensor',host)
    r.execute_model([a],True)
    assert state.tolist() == [0.,2.]
