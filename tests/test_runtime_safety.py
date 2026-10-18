"""CPU guards for worker metadata and unsupported state restoration."""

import pickle
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nano_qwen.config import Config
from nano_qwen.engine.model_runner import InputBatch
from nano_qwen.engine.sequence import Sequence, SequenceStatus
from nano_qwen.sampling_params import SamplingParams


@pytest.mark.parametrize("prefill", [True, False])
def test_worker_roundtrip_preserves_all_execution_metadata(prefill):
    seq = Sequence([2, 3, 4], SamplingParams(temperature=.7, max_tokens=8, ignore_eos=True))
    seq.status = SequenceStatus.RUNNING
    seq.append_token(5)
    seq.num_cached_tokens = 2
    seq.num_scheduled_tokens = 1
    seq.block_table = [7]
    seq.is_prefill = prefill
    restored = pickle.loads(pickle.dumps(seq))
    assert vars(restored) == vars(seq)
    rows, entries = InputBatch(2).update([restored])
    assert rows == [restored] and entries[0][0].seq_id == seq.seq_id
    restored.token_ids.append(9)
    assert seq.token_ids == [2, 3, 4, 5]


def hf_config(hybrid):
    return SimpleNamespace(max_position_embeddings=4096,
                           layer_types=["linear_attention", "full_attention"] if hybrid else ["full_attention"])


def test_hybrid_prefix_cache_fails_before_runtime_allocation(tmp_path):
    with patch("nano_qwen.config.config.AutoConfig.from_pretrained", return_value=hf_config(True)):
        with pytest.raises(ValueError, match="GDN|hybrid"):
            Config(str(tmp_path), enable_prefix_cache=True)
        assert not Config(str(tmp_path)).enable_prefix_cache


def test_attention_only_prefix_cache_remains_available(tmp_path):
    with patch("nano_qwen.config.config.AutoConfig.from_pretrained", return_value=hf_config(False)):
        assert Config(str(tmp_path), enable_prefix_cache=True).enable_prefix_cache


def test_unvalidated_tensor_parallel_fails_explicitly(tmp_path):
    with patch("nano_qwen.config.config.AutoConfig.from_pretrained", return_value=hf_config(False)):
        with pytest.raises(ValueError, match="tensor_parallel_size=1"):
            Config(str(tmp_path), tensor_parallel_size=2)
