"""Streaming must expose committed owners, never decode lookahead views.

Uses the existing fake GPU boundary with the real engine/scheduler/cohort.
Run in the project's inference environment:
  PYTHONPATH=src python -m pytest -q tests/test_chat_streaming.py tests/test_chat_cli.py
"""
from contextlib import closing, redirect_stdout
from io import StringIO
import unittest

from nano_qwen.sampling_params import SamplingParams
from test_async_decode_pipeline import engine, add, finish


class Tokenizer:
    def encode(self, text):
        return [3] * len(text)


def stream_engine(**kwargs):
    value = engine(**kwargs)
    value.config.max_model_len = 4096
    value.tokenizer = Tokenizer()
    return value


def params(count=8, ignore_eos=False):
    return SamplingParams(temperature=1e-6, max_tokens=count, ignore_eos=ignore_eos)


class StreamingTests(unittest.TestCase):
    def assert_idle(self, e, blocks=32):
        self.assertTrue(e.is_finished())
        self.assertIsNone(e.scheduler.async_decode)
        self.assertFalse(e.batch_queue)
        self.assertEqual(len(e.scheduler.block_manager.free_block_ids), blocks)

    def test_streams_before_completion_and_keeps_two_dispatch_lookahead(self):
        e = stream_engine()
        with closing(e.generate_stream([3] * 8, params())) as chunks:
            first = next(chunks)
            self.assertFalse(first.finished)
            self.assertEqual(first.token_ids, [14])
            self.assertFalse(e.is_finished())
            e.model_runner.events.clear()
            second = next(chunks)
            self.assertEqual(second.token_ids, [15])
            self.assertEqual([event[0] for event in e.model_runner.events][:5],
                             ["execute", "sample", "execute", "sample", "wait"])
            # The GPU feedback has advanced further than the visible stream.
            self.assertEqual(e.model_runner.gpu_tokens[first.request_id], 16)
            rest = list(chunks)
        self.assertTrue(rest[-1].finished)
        self.assertEqual(rest[-1].finish_reason, "length")
        self.assertEqual(sum(len(c.token_ids) for c in [first, second] + rest), 8)
        self.assert_idle(e)

    def test_async_and_serial_streams_match_across_prompt_and_kv_boundaries(self):
        for length in (1, 127, 128, 129, 254, 255, 256, 257, 513):
            outputs = []
            for enabled in (False, True):
                e = stream_engine(enabled=enabled, budget=64)
                chunks = list(e.generate_stream([3] * length, params(9, True)))
                outputs.append([t for c in chunks for t in c.token_ids])
                self.assertEqual(len(chunks), 9)
                self.assertEqual(sum(c.finished for c in chunks), 1)
                self.assertEqual(chunks[-1].finish_reason, "length")
                self.assert_idle(e)
            self.assertEqual(*outputs)

    def test_eos_never_leaks_already_queued_token_and_drains_before_reuse(self):
        e = stream_engine(token_fn=lambda seq, pos: 1 if pos == 8 else 7)
        chunks = list(e.generate_stream([3] * 8, params()))
        self.assertEqual([t for c in chunks for t in c.token_ids], [7, 1])
        self.assertEqual(chunks[-1].finish_reason, "eos")
        self.assertEqual(len(e.model_runner.removed), 1)
        self.assert_idle(e)
        second = list(e.generate_stream([3] * 3, params(1)))
        self.assertNotEqual(second[0].request_id, chunks[0].request_id)
        self.assert_idle(e)

    def test_max_tokens_one_has_one_final_chunk(self):
        for enabled in (False, True):
            e = stream_engine(enabled=enabled)
            chunks = list(e.generate_stream([3] * 257, params(1)))
            self.assertEqual(len(chunks), 1)
            self.assertTrue(chunks[0].finished)
            self.assertEqual(chunks[0].finish_reason, "length")
            self.assertEqual(len(e.model_runner.removed), 1)
            self.assert_idle(e)

    def test_close_after_prefill_cancels_and_allows_next_turn(self):
        e = stream_engine()
        chunks = e.generate_stream([3] * 8, params())
        first = next(chunks)
        executions = len([v for v in e.model_runner.events if v[0] == "execute"])
        chunks.close()
        self.assertEqual(executions, len([v for v in e.model_runner.events if v[0] == "execute"]))
        self.assertEqual(e.model_runner.removed, [first.request_id])
        self.assert_idle(e)
        self.assertEqual(len(list(e.generate_stream([3], params(2)))), 2)
        self.assert_idle(e)

    def test_close_with_queued_lookahead_discards_result_before_releasing_slot(self):
        e = stream_engine()
        chunks = e.generate_stream([3] * 255, params())
        next(chunks)
        second = next(chunks)
        self.assertTrue(e.batch_queue)
        owner = e.scheduler.async_decode.live[second.request_id]
        committed = owner.completion_token_ids.copy()
        e.model_runner.events.clear()
        chunks.close()
        self.assertEqual(owner.completion_token_ids, committed)
        self.assertEqual([v[0] for v in e.model_runner.events], ["wait", "remove"])
        self.assert_idle(e)

    def test_busy_engine_is_rejected_without_stealing_existing_request(self):
        e = stream_engine()
        seq = add(e, output=2)
        with self.assertRaisesRegex(RuntimeError, "idle"):
            next(e.generate_stream([3], params()))
        self.assertEqual(list(e.scheduler.waiting), [seq])
        finish(e)
        self.assert_idle(e)

    def test_invalid_budget_or_prompt_does_not_admit_request(self):
        e = stream_engine()
        for prompt, budget in (([], 1), ([3], 0), ([3], -1), ([3] * 4096, 1)):
            with self.assertRaises(ValueError):
                list(e.generate_stream(prompt, params(budget)))
            self.assert_idle(e)

    def test_string_prompt_uses_engine_tokenizer(self):
        e = stream_engine()
        chunks = list(e.generate_stream("hello", params(1)))
        self.assertEqual(chunks[0].token_ids, [11])
        self.assert_idle(e)

    def test_yielded_id_lists_are_independent_of_engine_history(self):
        e = stream_engine()
        chunks = e.generate_stream([3] * 8, params(3))
        first = next(chunks)
        first.token_ids[0] = 999
        rest = list(chunks)
        self.assertEqual([t for c in rest for t in c.token_ids], [15, 16])
        self.assert_idle(e)

    def test_model_execution_failure_is_not_hidden_by_retirement(self):
        e = stream_engine()
        original = e.step
        def fail():
            raise RuntimeError("execution failed")
        e.step = fail
        with self.assertRaisesRegex(RuntimeError, "execution failed"):
            list(e.generate_stream([3] * 8, params()))
        e.step = original
        # An execution failure requires engine.exit() in a real caller. No
        # normal result consumption/release was attempted by generate_stream.
        self.assertFalse(e.model_runner.events)

    def test_real_transformers_streamer_handles_multibyte_token_fragments(self):
        try:
            import numpy as np
            from transformers import TextStreamer
        except ImportError:
            self.skipTest("transformers streamer not installed")
        class ByteTokenizer:
            def decode(self, ids, **kwargs):
                return bytes(ids).decode("utf-8", errors="replace")
        out = StringIO()
        with redirect_stdout(out):
            streamer = TextStreamer(ByteTokenizer(), skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
            for byte in "你好 world".encode("utf-8"):
                streamer.put(np.array([byte]))
            streamer.end()
        self.assertEqual(out.getvalue(), "你好 world\n")


if __name__ == "__main__":
    unittest.main()
