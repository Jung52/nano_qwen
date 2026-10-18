"""Actual main-only eager/Graph lifecycle gates for generic execution changes."""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoConfig, PreTrainedTokenizerFast

from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.engine.model_runner import ModelRunner
from nano_qwen.engine.sequence import Sequence
from nano_qwen.layers.attention import Attention
from nano_qwen.models.qwen3_5 import Qwen3_5ForCausalLM
from nano_qwen.quantization.convert import convert_checkpoint
from nano_qwen.sampling_params import SamplingParams
from nano_qwen.utils.context import get_context


@pytest.fixture(scope='module')
def source_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp('main-lifecycle-source')
    config = AutoConfig.for_model('qwen3_5_text')
    for name, value in dict(hidden_size=128, intermediate_size=256, vocab_size=256,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=128,
        linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=128,
        linear_value_head_dim=128, max_position_embeddings=1024,
        layer_types=['linear_attention','full_attention'], dtype=torch.bfloat16).items():
        setattr(config, name, value)
    config.save_pretrained(path)
    torch.manual_seed(614)
    prior = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with patch.object(dist,'get_rank',return_value=0), patch.object(dist,'get_world_size',return_value=1):
            model = Qwen3_5ForCausalLM(config)
        tensors = {}
        for name, p in model.named_parameters():
            p.data.normal_(std=.02)
            if '.gate_up_proj.' in name:
                a,b = p.detach().chunk(2,dim=0)
                tensors[name.replace('gate_up_proj','gate_proj')] = a.clone()
                tensors[name.replace('gate_up_proj','up_proj')] = b.clone()
            else:
                tensors[name] = p.detach().clone()
        save_file(tensors,str(path/'model.safetensors'))
    finally:
        torch.set_default_dtype(prior)
    vocab = {'<unk>':0,'<eos>':1,**{f't{i}':i for i in range(2,256)}}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab,unk_token='<unk>')),
                                        unk_token='<unk>',eos_token='<eos>')
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope='module',params=['bf16','per_channel','block_wise'])
def checkpoint(request,source_checkpoint,tmp_path_factory):
    if request.param == 'bf16':
        return source_checkpoint
    path = tmp_path_factory.mktemp('main-lifecycle-fp8')
    convert_checkpoint(source_checkpoint,path,scheme=request.param)
    return path


@pytest.fixture(scope='module',params=[True,False],ids=['eager','graph'])
def engine(request,checkpoint):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    def cache(runner):
        runner.config.num_kvcache_blocks = 16
        attentions = [m for m in runner.model.modules() if isinstance(m,Attention)]
        runner.kv_cache = torch.empty(2,len(attentions),16,256,1,128,device='cuda',dtype=torch.bfloat16)
        for i,m in enumerate(attentions):
            m.k_cache,m.v_cache = runner.kv_cache[0,i],runner.kv_cache[1,i]
    with patch.object(ModelRunner,'allocate_kv_cache',cache):
        value = LLMEngine(str(checkpoint),max_num_seqs=2,max_num_batched_tokens=128,
                          max_model_len=1024,enforce_eager=request.param,use_prefill_cudagraph=True)
    try:
        yield value
    finally:
        value.exit()


@torch.inference_mode()
def lifecycle(engine, full=False):
    runner = engine.model_runner
    seqs = [Sequence([2+i]*n,SamplingParams(temperature=1e-6,max_tokens=8,ignore_eos=True))
            for i,n in enumerate((63,257))]
    record = SimpleNamespace(hidden={},logits={},states={},batches=[],heads=[])
    seen_slots = {}
    forward,postprocess,compute = runner.run_model,engine.scheduler.postprocess,runner.model.compute_logits
    def all_rows(hidden):
        record.heads.append(len(hidden))
        runner._target_hidden = hidden
        logits = compute(hidden)
        rows = torch.tensor(runner._sampling_plan.row_indices,device=hidden.device,dtype=torch.int64)
        return logits.index_select(0,rows)
    def run(ids,positions,prefill):
        ctx = get_context()
        slots = ctx.state_indices.tolist()
        sizes = (ctx.cu_seqlens_q[1:]-ctx.cu_seqlens_q[:-1]).tolist() if prefill else [1]*len(slots)
        logits = forward(ids,positions,prefill)
        selected = {r:i for i,r in enumerate(runner._sampling_plan.row_indices)}
        result = torch.zeros_like(logits)
        offset = 0
        for slot,size in zip(slots,sizes):
            current = runner.input_batch.seqs[slot]
            seen_slots[current.seq_id] = slot
            i = seqs.index(current)
            for row in range(offset,offset+size):
                p = int(positions[row])
                record.hidden[i,p] = runner._target_hidden[row].clone()
                if row in selected:
                    record.logits[i,p] = logits[selected[row]].clone()
                    result[selected[row],2+i*20+(p+1)%10] = 10
            offset += size
        return result
    def post(rows,outputs,prefill):
        record.batches.append(tuple(s.is_prefill for s in rows))
        for seq in rows:
            i = seqs.index(seq)
            # main releases a finishing request before scheduler postprocess.
            slot = seen_slots[seq.seq_id]
            end = seq.num_cached_tokens+seq.num_scheduled_tokens
            state = []
            for m in runner.gdn_layers:
                state.extend([m.conv_states[slot].clone(),m.recurrent_states[slot].clone()])
            indices = torch.tensor([seq.block_table[p//256]*256+p%256 for p in range(end)],device='cuda')
            for m in runner.model.modules():
                if isinstance(m,Attention):
                    state.extend(t.flatten(0,1).index_select(0,indices).clone() for t in (m.k_cache,m.v_cache))
            record.states[i,end] = state
        return postprocess(rows,outputs,prefill)
    def head(hidden):
        record.heads.append(len(hidden))
        return compute(hidden)
    with ExitStack() as stack:
        if full:
            stack.enter_context(patch.object(runner,'project_target_hidden',all_rows))
        stack.enter_context(patch.object(runner.model,'compute_logits',head))
        stack.enter_context(patch.object(runner,'run_model',run))
        stack.enter_context(patch.object(engine.scheduler,'postprocess',post))
        for seq in seqs:
            engine.scheduler.add(seq)
        for _ in range(100):
            engine.step()
            if all(s.is_finished for s in seqs):
                break
    assert all(s.is_finished for s in seqs) and not runner.input_batch.seq_id_to_slot
    record.tokens = [s.completion_token_ids for s in seqs]
    return record


@torch.inference_mode()
def test_selected_rows_preserve_main_hidden_completion_and_effective_states(engine):
    old,new = lifecycle(engine,True),lifecycle(engine)
    assert old.tokens == new.tokens and old.batches == new.batches
    assert (False,True) in new.batches, "decode must precede the partial prompt"
    assert old.hidden.keys() == new.hidden.keys() and old.states.keys() == new.states.keys()
    assert sum(new.heads) < sum(old.heads)
    for key,a in old.hidden.items():
        assert torch.equal(a,new.hidden[key])
    for key,tensors in old.states.items():
        for a,b in zip(tensors,new.states[key]):
            assert torch.isfinite(b).all() and torch.equal(a.view(torch.uint8),b.view(torch.uint8)),key
    for key,logits in old.logits.items():
        b = new.logits[key]
        assert torch.isfinite(b).all() and logits.argmax() == b.argmax(),key
