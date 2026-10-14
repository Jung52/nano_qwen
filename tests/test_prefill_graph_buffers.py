from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nano_qwen.engine.cuda_graph import CudaGraphManager


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_sequential_prefill_graphs_share_matching_projection_buffers_without_changing_results():
    manager = CudaGraphManager(SimpleNamespace())
    manager.prefill_graph_sizes = [128]
    x = torch.zeros((4,3), device='cuda', dtype=torch.float32)
    residual = torch.zeros_like(x)
    manager.piecewise_buffers = {'pre_out':{}, 'pre_residual_out':torch.empty((128,3),device='cuda')}
    first, _ = manager._copy_pre_outputs(0, (x*2,), residual)
    second, _ = manager._copy_pre_outputs(1, (x+3,), residual)
    assert first[0].data_ptr() == second[0].data_ptr(), 'Matching sequential layers must share storage'
    different = manager._allocate_pre_out(2, (torch.zeros((4,5),device='cuda'),))
    other_dtype = manager._allocate_pre_out(3, (x.half(),))
    assert different[0].data_ptr() != first[0].data_ptr()
    assert other_dtype[0].data_ptr() != first[0].data_ptr()
    intermediate = torch.empty_like(x)
    graph0, graph1 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph0):
        manager._copy_pre_outputs(0, (x*2,), residual)
    with torch.cuda.graph(graph1):
        manager._copy_pre_outputs(1, (intermediate+3,), residual)
    for value, expected in ((2,19), (-3,39), (5,103)):
        x.fill_(value)
        graph0.replay()
        intermediate.copy_(first[0][:4].square())
        graph1.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(second[0][:4], torch.full_like(x,expected), rtol=0, atol=0)
    manager.clear_prefill()
    assert manager.piecewise_buffers is None
