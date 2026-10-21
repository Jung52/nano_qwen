"""Small real-CUDA replay/profiler smoke test; no model checkpoint needed."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks"))
from bench_one_batch_slots import replay_labels
from slot_trace_full import export_full_trace, trace_records

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@torch.inference_mode()
def test_real_replays_keep_cpu_ops_gpu_kernels_and_slot_alternation(tmp_path):
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        pytest.fail("CUPTI CUDA profiler unavailable")
    banks=[]
    tensors=[]
    for slot in range(2):
        x=torch.randn(8,128,device="cuda")
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                x.sin().cos()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y=x.sin().cos()
        banks.append(({8:graph},{},graph.pool()))
        tensors.append((x,y))
    # Minimal runner interface: the replay instrumentation is the exact same
    # context manager used by the full-engine benchmark.
    class Node:
        def prepare_inputs(self,*a): pass
        def sample_tokens(self,*a): pass
        def compute_logits(self,*a): pass
        def launch_slot(self,*a): pass
        def sample_slot(self,*a): pass
        def _plan_decode(self,*a): pass
        def _commit_oldest(self,*a): pass
    manager=SimpleNamespace(decode_graph_banks=banks)
    runner=Node()
    runner.cuda_graphs,runner.model,runner._active_slot_id=manager,Node(),0
    pipeline=Node()
    pipeline.pending=[SimpleNamespace(slot_id=0)]
    engine=SimpleNamespace(model_runner=runner,_dual_pipeline=pipeline)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False,profile_memory=False) as prof:
        with replay_labels(engine,torch) as counts:
            for i in range(6):
                slot=i%2
                tensors[slot][0].add_(0.01)  # Real CPU aten + GPU kernel outside Graph.
                banks[slot][0][8].replay()
            torch.cuda.synchronize()
    assert sum(counts.values())==6
    source=tmp_path/"raw.json"
    dest=tmp_path/"dual.trace.json.gz"
    prof.export_chrome_trace(str(source))
    result=export_full_trace(source,dest,mode="dual")
    assert result["cpu_graph_launches"]==6
    assert result["slot_order"]==[0,1,0,1,0,1]
    assert result["graph_kernel_events"]>=12
    assert result["original_counts"]["cpu_op"]>0
    categories=[e.get("cat") for k,e in trace_records(dest) if k=="event"]
    assert "cpu_op" in categories and "kernel" in categories
    for x,y in tensors:
        torch.testing.assert_close(y,x.sin().cos())
