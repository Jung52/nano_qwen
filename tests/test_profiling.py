"""Optional profiling changes host annotations only and isolates failures."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

# This module deliberately has no torch import. Load it without importing the
# package's unrelated utils.context module, so these tests also run on CPU-only CI.
path=Path(__file__).resolve().parents[1]/"src/nano_qwen/utils/profiling.py"
spec=importlib.util.spec_from_file_location("nq_profiling_unit",path)
profiling=importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiling)


class Output:
    def __init__(self, log): self.log=log
    def get_output(self):
        self.log.append("wait")
        return [7]


def environment(tmp_path, *, fail_start=False, fail_stop=False, fail_export=False):
    log, marks=[],[]
    class Graph:
        def replay(self): log.append("replay")
    graph=Graph()
    class Runner:
        def __init__(self):
            self.cuda_graphs=SimpleNamespace(decode_graphs={1:graph})
            self.model=SimpleNamespace(compute_logits=lambda x: (log.append("head"),x)[1])
        def execute_model(self, seqs, is_prefill):
            log.append("execute")
            graph.replay()
            return self.model.compute_logits([5])
        def sample_tokens(self):
            log.append("sample")
            return Output(log)
    class Scheduler:
        def schedule(self):
            log.append("schedule")
            return [SimpleNamespace(seq_id=3)],False
        def postprocess(self,*args): log.append("commit")
    engine=SimpleNamespace(model_runner=Runner(),scheduler=Scheduler(),
                           config=SimpleNamespace(async_scheduling=True))
    class Profiler:
        def __init__(self): self.stops=self.exports=0
        def start(self):
            if fail_start: raise RuntimeError("start failed")
        def stop(self):
            self.stops+=1
            if fail_stop: raise RuntimeError("stop failed")
        def export_chrome_trace(self,p):
            self.exports+=1
            if fail_export: raise OSError("disk full")
            Path(p).write_bytes(b"trace")
    prof=Profiler()
    @contextmanager
    def mark(name):
        marks.append(name)
        yield
    torch=SimpleNamespace(profiler=SimpleNamespace(
        record_function=mark, profile=lambda **kwargs:prof,
        ProfilerActivity=SimpleNamespace(CPU=1,CUDA=2)))
    return engine,torch,prof,log,marks,graph


def one_dispatch(engine):
    rows,prefill=engine.scheduler.schedule()
    engine.model_runner.execute_model(rows,prefill)
    output=engine.model_runner.sample_tokens()
    engine.scheduler.postprocess(rows,output.get_output(),prefill)


def test_labels_preserve_every_original_call_and_return(tmp_path):
    engine,torch,prof,log,marks,graph=environment(tmp_path)
    one_dispatch(engine)
    expected=log.copy(); log.clear()
    with profiling.Labels(engine,torch,[graph]) as labels:
        one_dispatch(engine)
    assert log==expected
    assert labels.batches==[dict(batch_id=0,batch_size=1,is_prefill=False,seq_ids=[3])]
    assert "nq/launch/b00000" in marks and "nq/commit/b00000" in marks
    assert "execute_model" not in vars(engine.model_runner)
    assert "replay" not in vars(graph)


def test_two_queued_dispatches_keep_independent_wait_and_commit_ids(tmp_path):
    engine,torch,prof,log,marks,graph=environment(tmp_path)
    with profiling.Labels(engine,torch,[graph]):
        outputs=[]
        for _ in range(2):
            rows,p=engine.scheduler.schedule()
            engine.model_runner.execute_model(rows,p)
            outputs.append((rows,engine.model_runner.sample_tokens()))
        for rows,out in outputs:
            engine.scheduler.postprocess(rows,out.get_output(),False)
    assert [x for x in marks if x.startswith("nq/commit")] == ["nq/commit/b00000","nq/commit/b00001"]


def test_partial_annotation_install_is_rolled_back(tmp_path):
    engine,torch,prof,log,marks,graph=environment(tmp_path)
    del engine.model_runner.model.compute_logits
    with pytest.raises(AttributeError):
        with profiling.Labels(engine,torch,[graph]): pass
    assert "execute_model" not in vars(engine.model_runner)
    assert "schedule" not in vars(engine.scheduler)
    assert "replay" not in vars(graph)


def test_decode_only_labels_reject_prefill_without_executing_it(tmp_path):
    engine,torch,prof,log,marks,graph=environment(tmp_path)
    with profiling.Labels(engine,torch,[graph]):
        with pytest.raises(RuntimeError,match="Prefill"):
            engine.model_runner.execute_model([],True)
    assert log==[]


def test_export_happens_after_stop_and_is_idempotent(tmp_path):
    engine,torch,prof,log,marks,graph=environment(tmp_path)
    session=profiling.ProfileSession(torch,engine,tmp_path,1).start()
    one_dispatch(engine)
    session.stop()
    assert prof.stops==1 and prof.exports==0
    first=session.finish()
    assert "exported" in first and prof.exports==1
    assert session.finish()==first and prof.exports==1
    assert len(list(tmp_path.glob('*.metadata.json')))==1


@pytest.mark.parametrize("failure",["fail_start","fail_stop","fail_export"])
def test_profiler_failure_never_breaks_chat_or_retries_stop(tmp_path,failure):
    engine,torch,prof,log,marks,graph=environment(tmp_path,**{failure:True})
    session=profiling.ProfileSession(torch,engine,tmp_path,1).start()
    if session.active:
        one_dispatch(engine)
    message=session.finish()
    assert "failed" in message or "disabled" in message
    assert session.finish()==message
    assert prof.stops==1
    assert "execute_model" not in vars(engine.model_runner)
    one_dispatch(engine)  # Model path remains callable after diagnostics fail.


def test_mkdir_failure_does_not_escape(tmp_path):
    engine,torch,prof,log,marks,graph=environment(tmp_path)
    target=tmp_path/'file'; target.write_text('occupied')
    session=profiling.ProfileSession(torch,engine,target/'child',1).start()
    assert "export failed" in session.finish()
