"""Pair protocol and request-based acceptance units, without running a model."""
import math

import pytest

from bench_mtp_paired import case_matrix, pair_modes, acceptance_rate, summarize_pairs


def test_purposeful_matrix_and_counterbalanced_pair_order():
    assert case_matrix(1) == [(64,128),(64,512),(1024,128),(1024,512)]
    assert case_matrix(2) == [(64,512)]
    assert case_matrix(4) == case_matrix(8) == [(64,128),(64,512)]
    assert [pair_modes(i) for i in range(1,6)] == [('off','reuse'),('reuse','off'),('off','reuse'),('reuse','off'),('off','reuse')]


def test_acceptance_divides_by_verified_requests_not_batch_rounds():
    row = dict(metrics=dict(accepted_tokens=10,verify_steps=3),graphs=dict(verify=dict(batch_counts={'4':2,'2':1})))
    assert acceptance_rate(row) == 1
    row['metrics']['accepted_tokens'] = 11
    with pytest.raises(ValueError,match='accept'):
        acceptance_rate(row)


def test_paired_summary_requires_complete_pairs_and_uses_actual_ratios():
    rows = []
    for pair in range(1,6):
        for mode in pair_modes(pair):
            rows.append(dict(pair=pair,mode=mode,batch=1,input=64,output=128,
                metrics=dict(e2e_s=1.2 if mode=='off' else 1.,request_tpot_ms=[12. if mode=='off' else 10.],
                             request_ttft_ms=[30. if mode=='off' else 50.])))
    summary = summarize_pairs(rows)
    assert math.isclose(summary['e2e_speedup_pct']['geometric_mean'],20.)
    assert summary['e2e_speedup_pct']['ci95'][0] == pytest.approx(20.)
    assert summary['ttft_extra_ms']['mean'] == 20.
    with pytest.raises(ValueError,match='pair'):
        summarize_pairs(rows[:-1])


def test_driver_runs_each_case_with_adjacent_counterbalanced_pairs(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from collections import Counter
    import bench_mtp_paired as b
    monkeypatch.setattr(b.torch.cuda,'get_device_name',lambda:'CPU protocol test')
    monkeypatch.setattr(b,'production_hashes',lambda:{})
    monkeypatch.setattr(b,'tuning_snapshot',lambda:{})
    monkeypatch.setattr(b,'make_prompt',lambda model,n:[2]*n)
    class Engine:
        def __init__(self,batch,mode):
            self.batch,self.mode = batch,mode
            graphs = SimpleNamespace(reset_dispatch_stats=lambda:None,
                dispatch_stats=lambda:{'verify':{'batch_counts':{str(batch):1}}} if mode=='reuse' else {})
            self.model_runner = SimpleNamespace(cuda_graphs=graphs)
        def exit(self):
            pass
    monkeypatch.setattr(b,'make_engine',lambda model,mtp,batch,mode:Engine(batch,mode))
    monkeypatch.setattr(b,'engine_info',lambda e:dict(mtp_loaded=e.mode=='reuse',mtp_cache_bytes=0,
                                                   snapshot_bytes=0,prefill_graph=e.mode=='off'))
    formal_calls = []
    def run(engine,prompts,output):
        batch = len(prompts)
        if output!=8:
            formal_calls.append((len(prompts[0]),output,engine.mode))
        tokens = (output-1)*batch
        return dict(e2e_s=1.,tokens=tokens,request_tpot_ms=[1.]*batch,request_ttft_ms=[1.]*batch,
                    accepted_tokens=0,verify_steps=0,steps=[dict(committed=tokens)]),[[3]*output for _ in prompts]
    monkeypatch.setattr(b,'run_batch',run)
    report = b.measure(SimpleNamespace(model='fake',mtp_model='fake',batches=[1],output=tmp_path/'result.json'))
    assert Counter((s['input'],s['output']) for s in report['samples']) == {case:10 for case in case_matrix(1)}
    for i in range(0,40,2):
        a,c = report['samples'][i:i+2]
        assert a['case'] == c['case'] and a['pair'] == c['pair']
        assert (a['mode'],c['mode']) == pair_modes(a['pair'])
    original = report['samples'][:18]
    report['samples'] = original
    report['engines'] = report['engines'][:18]
    report['summaries'] = {}
    (tmp_path/'result.json').write_text(json.dumps(report))
    resumed = b.measure(SimpleNamespace(model='fake',mtp_model='fake',batches=[1],output=tmp_path/'result.json',resume=True))
    assert len(formal_calls) == 40+22
    assert len(resumed['samples']) == 40
    assert resumed['samples'][:18] == original
    assert resumed['continuations'][0]['after_samples'] == 18
    hash_calls = 0
    def changed_source():
        nonlocal hash_calls
        hash_calls += 1
        return {} if hash_calls==1 else {'changed':'hash'}
    monkeypatch.setattr(b,'production_hashes',changed_source)
    failed=tmp_path/'failed.json'
    with pytest.raises(RuntimeError,match='source|tuning'):
        b.measure(SimpleNamespace(model='fake',mtp_model='fake',batches=[1],output=failed))
    evidence=json.loads(failed.read_text())
    assert evidence['samples'][0]['invariants_pass'] is False
    monkeypatch.setattr(b,'production_hashes',lambda:{})
    with pytest.raises(ValueError,match='gate'):
        b.measure(SimpleNamespace(model='fake',mtp_model='fake',batches=[1],output=failed,resume=True))


def test_phase_recorder_supports_a_true_off_engine_without_mtp():
    from types import SimpleNamespace
    from mtp_profile_instrumentation import PhaseRecorder
    def noop(*args,**kwargs):
        return None
    graphs = SimpleNamespace(**{k:noop for k in ('_capture_decode','_capture_verify_batch','_capture_mtp_decode_batch',
        'run_decode','run_verify','run_mtp_decode','run_mtp_cache_only')})
    runner = SimpleNamespace(cuda_graphs=graphs,mtp=None,gdn_layers=[],enforce_eager=True,
        run_model=noop,forward_verify=noop,_run_mtp_decode=noop,_run_mtp_prefill=noop,sample_tokens=noop,
        model=SimpleNamespace(model=SimpleNamespace(forward=noop),compute_logits=noop))
    recorder = PhaseRecorder(runner,'host')
    with recorder.install():
        recorder.begin()
        runner.run_model(None,None,False)
        row = recorder.finish()
    assert row['host_calls']['target'] == 1
    assert runner.run_model is noop


def test_trace_reuses_recorded_tuning_without_replacing_existing_choices():
    from types import SimpleNamespace
    from profile_mtp_paired import restore_configs
    class Config:
        def __init__(self,meta):
            self.meta = meta
        def all_kwargs(self):
            return self.meta
    a,b = Config({'BM':16,'num_warps':4}),Config({'BM':32,'num_warps':4})
    tuner = SimpleNamespace(configs=[a,b],cache={})
    record = {'per_channel':{"(1, 128, 128, 'torch.bfloat16')":a.all_kwargs()}}
    restore_configs({'per_channel':tuner},record)
    key = (1,128,128,'torch.bfloat16')
    assert tuner.cache[key] is a
    tuner.cache[key] = b
    with pytest.raises(ValueError,match='tuning|config'):
        restore_configs({'per_channel':tuner},record)


def test_resume_only_accepts_a_strict_completed_execution_prefix():
    from bench_mtp_paired import execution_plan, pending_plan
    plan = execution_plan([1,2,4,8])
    assert len(plan) == 90
    rows = [dict(batch=b,input=n,output=o,pair=p,mode=m,execution_order=i,
                 strict_pass=True,actual_commits_pass=True,invariants_pass=True)
            for i,(b,n,o,p,m) in enumerate(plan[:18],1)]
    assert pending_plan(rows,[1,2,4,8]) == plan[18:]
    assert pending_plan(rows[:17],[1,2,4,8]) == plan[17:]
    rows[1]['mode'] = 'off'
    with pytest.raises(ValueError,match='prefix'):
        pending_plan(rows,[1,2,4,8])
    rows[1]['mode'] = 'reuse'
    rows[1]['strict_pass'] = False
    with pytest.raises(ValueError,match='gate'):
        pending_plan(rows,[1,2,4,8])


def test_resume_rejects_missing_or_failed_invariant_gate():
    from bench_mtp_paired import execution_plan,pending_plan
    b,n,o,p,m=execution_plan([1])[0]
    row=dict(batch=b,input=n,output=o,pair=p,mode=m,execution_order=1,
             strict_pass=True,actual_commits_pass=True)
    with pytest.raises(ValueError,match='gate'):
        pending_plan([row],[1])
    row['invariants_pass']=False
    with pytest.raises(ValueError,match='gate'):
        pending_plan([row],[1])


def test_resume_requires_identical_gpu_torch_and_cuda_environment():
    from bench_mtp_paired import check_environment
    env=dict(gpu='4070 Ti SUPER',torch='2.12.1+cu130',cuda='13.0')
    check_environment(env,env.copy())
    for field in env:
        changed=dict(env);changed[field]='different'
        with pytest.raises(ValueError,match='environment'):
            check_environment(env,changed)


def test_runtime_overlap_uses_unions_without_double_counting():
    from report_mtp_paired import linear_overlap
    assert linear_overlap([(0,10),(20,30)],[(5,25),(26,40)]) == 14
    assert linear_overlap([(0,10),(5,15)],[(7,11)]) == 4
    assert linear_overlap([],[(1,2)]) == 0
