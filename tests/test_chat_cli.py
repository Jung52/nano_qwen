"""Terminal entry point behavior, using a fake model/runtime boundary."""
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nano_qwen
from nano_qwen import chat
from nano_qwen.engine.llm_engine import StreamChunk
from nano_qwen.sampling_params import SamplingParams


class Tokenizer:
    def __init__(self):
        self.calls = []
        self.raw_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), kwargs))
        return [3] * (3 * len(messages))

    def encode(self, text):
        self.raw_calls.append(text)
        return [3] * len(text)

    def decode(self, ids, **kwargs):
        return "".join({10: "你", 11: "好", 12: "！"}.get(t, "") for t in ids)


class Streamer:
    calls = []

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.calls.append(kwargs)

    def put(self, value):
        print(self.tokenizer.decode(value), end="", flush=True)

    def end(self):
        print(flush=True)


class Graphs:
    def __init__(self, eager):
        self.decode_graphs = {} if eager else {1: object()}
        self.graph = self.eager = 0

    def dispatch_stats(self):
        return {"decode": {"graph_calls": self.graph, "eager_calls": self.eager}}


class Engine:
    instances = []
    interrupt = False
    disable_copy = False

    def __init__(self, model, **kwargs):
        self.instances.append(self)
        self.model = model
        self.kwargs = kwargs
        self.config = SimpleNamespace(max_model_len=kwargs["max_model_len"],
                                      async_scheduling=kwargs["async_scheduling"])
        self.max_concurrent_batches = 2
        self.model_runner = SimpleNamespace(async_output=not self.disable_copy,
                                            cuda_graphs=Graphs(kwargs["enforce_eager"]))
        self.tokenizer = Tokenizer()
        self.requests = []
        self.closed_streams = 0
        self.exited = False

    def generate_stream(self, ids, params):
        self.requests.append((list(ids), params))
        try:
            if self.interrupt:
                raise KeyboardInterrupt()
            for index, token in enumerate((10, 11, 12)[:params.max_tokens]):
                if index:
                    graphs = self.model_runner.cuda_graphs
                    if self.kwargs["enforce_eager"]:
                        graphs.eager += 1
                    else:
                        graphs.graph += 1
                final = index == min(3, params.max_tokens) - 1
                yield StreamChunk(5, [token], final, "length" if final else None)
        finally:
            self.closed_streams += 1

    def exit(self):
        self.exited = True


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        Engine.instances.clear()
        Engine.interrupt = Engine.disable_copy = False
        Streamer.calls.clear()
        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True), int64="int64",
            inference_mode=nullcontext,
            tensor=lambda values, **kwargs: values,
        )

    def run_cli(self, extra=(), inputs=()):
        stdout, stderr = StringIO(), StringIO()
        with patch.object(chat, "_load_runtime", return_value=(self.torch, Streamer, Engine, SamplingParams)), \
                patch("builtins.input", side_effect=list(inputs)), redirect_stdout(stdout), redirect_stderr(stderr):
            result = chat.main(["--model", self.directory.name, *extra])
        return result, stdout.getvalue(), stderr.getvalue()

    def test_defaults_enable_single_graph_async_and_safe_chat_rendering(self):
        code, output, log = self.run_cli(["--prompt", "解释CUDA Graph"])
        self.assertEqual(code, 0)
        self.assertEqual(output, "你好！\n")
        e = Engine.instances[-1]
        self.assertEqual((e.kwargs["max_num_seqs"], e.kwargs["tensor_parallel_size"]), (1, 1))
        self.assertTrue(e.kwargs["async_scheduling"])
        self.assertFalse(e.kwargs["enforce_eager"])
        self.assertFalse(e.kwargs["use_prefill_cudagraph"])
        self.assertEqual(e.requests[0][1].temperature, 1e-6)
        self.assertFalse(e.requests[0][1].ignore_eos)
        self.assertTrue(e.exited)
        self.assertEqual(e.closed_streams, 1)
        self.assertIn("single CUDA Graph", log)
        self.assertIn("decode graph/eager=2/0", log)
        self.assertEqual(Streamer.calls[0], dict(skip_prompt=False, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False))

    def test_two_turn_history_is_formatted_through_checkpoint_template(self):
        code, _, _ = self.run_cli(["--no-stats", "--system-prompt", "请用中文回答"],
                                  ["第一个问题", "第二个问题", "/exit"])
        self.assertEqual(code, 0)
        calls = Engine.instances[-1].tokenizer.calls
        self.assertEqual([m["role"] for m in calls[1][0]],
                         ["system", "user", "assistant", "user"])
        self.assertEqual(calls[1][0][2]["content"], "你好！")
        self.assertEqual(calls[1][1], dict(tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False))

    def test_reset_removes_both_user_and_assistant_history(self):
        code, output, _ = self.run_cli(["--no-stats"], ["第一轮", "/reset", "第二轮", "/exit"])
        self.assertEqual(code, 0)
        self.assertIn("对话已清空", output)
        second = Engine.instances[-1].tokenizer.calls[1][0]
        self.assertEqual(second, [{"role": "user", "content": "第二轮"}])

    def test_reference_flags_are_explicit_and_effective(self):
        code, _, log = self.run_cli(["--prompt", "hello", "--eager", "--no-async-scheduling",
                                    "--thinking", "--no-stats"])
        self.assertEqual(code, 0)
        e = Engine.instances[-1]
        self.assertFalse(e.kwargs["async_scheduling"])
        self.assertTrue(e.kwargs["enforce_eager"])
        self.assertTrue(e.tokenizer.calls[0][1]["enable_thinking"])
        self.assertIn("async=off", log)

    def test_raw_one_shot_skips_chat_template(self):
        code, _, _ = self.run_cli(["--raw", "--prompt", "raw input", "--no-stats"])
        self.assertEqual(code, 0)
        tokenizer = Engine.instances[-1].tokenizer
        self.assertEqual(tokenizer.raw_calls, ["raw input"])
        self.assertFalse(tokenizer.calls)

    def test_context_error_keeps_conversation_and_engine_usable(self):
        code, _, log = self.run_cli(["--max-model-len", "4", "--max-tokens", "4"],
                                   ["too long", "/reset", "/exit"])
        self.assertEqual(code, 0)
        e = Engine.instances[-1]
        self.assertFalse(e.requests)
        self.assertTrue(e.exited)
        self.assertIn("Context budget exceeded", log)

    def test_keyboard_interrupt_closes_stream_and_engine(self):
        Engine.interrupt = True
        code, _, _ = self.run_cli(["--prompt", "interrupt"])
        self.assertEqual(code, 130)
        e = Engine.instances[-1]
        self.assertTrue(e.exited)
        self.assertEqual(e.closed_streams, 1)

    def test_async_request_cannot_silently_fall_back_to_sync_output(self):
        Engine.disable_copy = True
        with self.assertRaisesRegex(RuntimeError, "async decode"):
            self.run_cli(["--prompt", "hello"])
        self.assertTrue(Engine.instances[-1].exited)

    def test_no_gpu_fails_before_constructing_engine(self):
        self.torch.cuda.is_available = lambda: False
        code, _, log = self.run_cli(["--prompt", "hello"])
        self.assertEqual(code, 1)
        self.assertFalse(Engine.instances)
        self.assertIn("CUDA GPU", log)

    def test_invalid_cli_options_fail_before_runtime_load(self):
        for extra in (["--temperature", "-1"], ["--temperature", "nan"],
                      ["--max-tokens", "0"], ["--raw"],
                      ["--gpu-memory-utilization", "1.1"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as exc:
                with redirect_stderr(StringIO()):
                    chat.main(["--model", self.directory.name, *extra])
            self.assertEqual(exc.exception.code, 2)
        self.assertFalse(Engine.instances)

    def test_existing_console_script_dispatches_to_chat(self):
        with patch.object(chat, "main", return_value=0) as call:
            self.assertEqual(nano_qwen.main(), 0)
            call.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
