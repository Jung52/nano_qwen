"""Full-engine dual-slot smoke: greedy parity, EOS zombies and slot reuse.

Run once with --eager, then without it to validate two independent graph banks.
This is a correctness check, not a throughput benchmark.
"""
import argparse
import json
from pathlib import Path

import torch
from nano_qwen.engine.llm_engine import LLMEngine
from nano_qwen.sampling_params import SamplingParams


class ForceEOS(torch.nn.Module):
    def __init__(self, sampler, target, eos, first_token):
        super().__init__()
        self.sampler, self.target, self.eos = sampler, target, eos
        self.first_token = first_token
        self.count = 0
        self.rows = []

    def set_batch_context(self, rows, prefill):
        self.rows = rows

    def forward(self, logits, temperatures, all_greedy=False):
        tokens = self.sampler(logits, temperatures, all_greedy=all_greedy)
        for i, seq in enumerate(self.rows):
            if seq.seq_id == self.target:
                self.count += 1
                if self.count == 1:
                    tokens[i] = self.first_token
                elif self.count == 2:
                    tokens[i] = self.eos
        return tokens


def assert_drained(engine):
    r, s = engine.model_runner, engine.scheduler
    assert engine.is_finished()
    assert not s.in_flight and not r.input_batch.seq_id_to_slot
    p = engine._dual_pipeline
    if p is not None:
        assert not p.pending and not p.requests
        assert all(not slot.busy and slot.pending is None and slot.output is None
                   for slot in r.decode_slots)


def run(args, dual):
    engine = LLMEngine(args.model, dual_slot_decode=dual,
        enforce_eager=args.eager, use_prefill_cudagraph=False,
        max_num_seqs=4, max_num_batched_tokens=2048, max_model_len=512,
        gpu_memory_utilization=args.mem_util, enable_prefix_cache=False)
    try:
        r = engine.model_runner
        # Correctness-only synchronization: prove reused persistent slots are
        # clean immediately after reset and before their first forward.
        reset_checks = [0]
        for layer in r.gdn_layers:
            original_reset = layer.reset_state
            def checked_reset(slots, fn=original_reset, module=layer):
                fn(slots)
                assert torch.count_nonzero(module.conv_states[slots]).item() == 0
                assert torch.count_nonzero(module.recurrent_states[slots]).item() == 0
                reset_checks[0] += 1
            layer.reset_state = checked_reset
        if dual and not args.eager:
            a, b = r.cuda_graphs.decode_graph_banks
            assert a[2] != b[2], "graph pools alias"
            for key in ("input_ids", "positions", "state_indices", "outputs"):
                assert a[1][key].data_ptr() != b[1][key].data_ptr(), key
        seed = engine.tokenizer.encode("验证双slot状态与缓存边界。", add_special_tokens=False)
        assert seed
        results = []
        for batch, prompt_len, output_len in [(1,255,1),(1,255,12),(2,256,12),(4,257,12)]:
            prompt = (seed*((prompt_len+len(seed)-1)//len(seed)))[:prompt_len]
            outputs = engine.generate([prompt]*batch,
                SamplingParams(temperature=1e-6, max_tokens=output_len, ignore_eos=True),
                use_tqdm=False)
            ids = [out["token_ids"] for out in outputs]
            assert all(len(row) == output_len for row in ids)
            assert_drained(engine)
            results.append(dict(batch=batch,prompt=prompt_len,output=output_len,tokens=ids))
        if dual:
            # Force EOS on the second sampled token while t+1 is already
            # launched. Check lifecycle/length, not BF16-vs-batch numerics.
            original = r.sampler
            for _ in range(2):
                engine.add_request(seed, SamplingParams(temperature=1e-6,max_tokens=8))
            target = engine.scheduler.waiting[0].seq_id
            assert seed[0] != engine.config.eos
            r.sampler = ForceEOS(original,target,engine.config.eos,seed[0])
            emitted = {}
            while not engine.is_finished():
                output,_ = engine.step()
                emitted.update(output)
            assert emitted[target][-1] == engine.config.eos
            assert len(emitted[target]) == 2
            assert engine._dual_pipeline.stats["zombie_rows"] >= 1
            r.sampler = original
            assert_drained(engine)
            # Same engine admits another request after the zombie retires.
            out = engine.generate([seed], SamplingParams(temperature=1e-6,max_tokens=2,
                                                        ignore_eos=True),use_tqdm=False)
            assert len(out[0]["token_ids"]) == 2
            assert_drained(engine)
        if r.gdn_layers:
            assert reset_checks[0] > 0
        return results
    finally:
        engine.exit()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--eager',action='store_true')
    ap.add_argument('--mem-util',type=float,default=0.65)
    ap.add_argument('--json-out',default='dual_slot_validation.json')
    args = ap.parse_args()
    report = dict(status='failed',eager=args.eager,model=args.model)
    try:
        report['baseline'] = run(args,False)
        report['dual_slot'] = run(args,True)
        assert report['baseline'] == report['dual_slot'], 'first greedy mismatch: inspect JSON'
        report['status'] = 'passed'
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('dual-slot validation passed')


if __name__ == '__main__':
    main()
