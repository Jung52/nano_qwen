from types import SimpleNamespace

import torch

from nano_qwen.engine.cuda_graph import CudaGraphManager
from nano_qwen.utils.context import reset_context, set_context


def test_dynamic_batch_records_exact_hits_without_padding_live_slots():
    replayed = []
    hidden = torch.zeros(64, 4)
    class Model:
        def __call__(self, ids, pos):
            return hidden[:len(ids)]
    runner = SimpleNamespace(config=SimpleNamespace(), model=Model(),
        project_target_hidden=lambda h: h)
    graphs = CudaGraphManager(runner)
    sizes = [1,2,4,8]+list(range(16,65,16))
    graphs.decode_graphs = {b: SimpleNamespace(replay=lambda b=b: replayed.append(b)) for b in sizes}
    graphs.decode_graph_vars = dict(
        input_ids=torch.zeros(64, dtype=torch.int64), positions=torch.zeros(64, dtype=torch.int64),
        slot_mapping=torch.zeros(64, dtype=torch.int32), context_lens=torch.zeros(64, dtype=torch.int32),
        block_tables=torch.zeros(64, 1, dtype=torch.int32),
        state_indices=torch.zeros(64, dtype=torch.int64), outputs=hidden)
    try:
        for b in (64, 63, 48, 47, 32, 31, 1):
            set_context(False, slot_mapping=torch.arange(b), context_lens=torch.ones(b),
                block_tables=torch.zeros(b, 1), state_indices=torch.arange(b))
            assert graphs.run_decode(torch.zeros(b), torch.zeros(b)).shape == (b, 4)
        assert replayed == [64, 48, 32, 1]
        stats = graphs.dispatch_stats()
        assert stats['decode']['graph_calls'] == 4
        assert stats['decode']['eager_calls'] == 3
        assert stats['decode']['batch_counts'] == {b: 1 for b in (64, 63, 48, 47, 32, 31, 1)}
        graphs.reset_dispatch_stats()
        assert graphs.dispatch_stats() == {}
    finally:
        reset_context()
