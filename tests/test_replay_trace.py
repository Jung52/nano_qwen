"""CUDA-free trace integrity and temporary replay instrumentation tests."""
from collections import Counter
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"benchmarks"))
from slot_trace_full import export_full_trace, inspect_trace, trace_records, PHYSICAL
from bench_one_batch_slots import replay_labels, parse_args


def ev(name, cat, ts, dur, corr=None, gpu=False, tid=None):
    return dict(ph="X", name=name, cat=cat, pid=0 if gpu else 100,
        tid=tid if tid is not None else 17 if gpu else 101, ts=ts, dur=dur,
        args={} if corr is None else dict(correlation=corr))


def data(slots=(0, 1)):
    events = []
    for i,slot in enumerate(slots):
        t, corr = i*1000, i+10
        events += [ev(f"CUDAGraph.replay/slot{slot}/bs8", "user_annotation", t+10, 20),
            ev("cudaGraphLaunch", "cuda_runtime", t+12, 2, corr),
            ev("aten::copy_", "cpu_op", t+1, 2),
            ev("kernel_a", "kernel", t+100, 3, corr, True),
            ev("kernel_b", "kernel", t+105, 4, corr, True),
            ev("Memcpy DtoH", "gpu_memcpy", t+120, 1, corr+100, True, 13),
            ev("native GPU scope", "gpu_user_annotation", t+100, 9, gpu=True),
            dict(ph="s", cat="ac2g", name="flow", pid=100, tid=101, ts=t+12, id=corr),
            dict(ph="f", cat="ac2g", name="flow", pid=0, tid=17, ts=t+100, id=corr)]
    return dict(baseTimeNanoseconds=1000000, traceEvents=events)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, value):
        p = self.root/"in.json"
        p.write_text(json.dumps(value))
        return p

    def test_cpu_ops_apis_kernels_copies_flows_are_preserved_exactly(self):
        original = data()
        out = self.root/"out.trace.json.gz"
        report = export_full_trace(self.write(original), out, mode="dual")
        with gzip.open(out,"rt") as f:
            events = json.load(f)["traceEvents"]
        cats = PHYSICAL | {"cpu_op", "cuda_runtime", "cuda_driver", "ac2g", "user_annotation"}
        self.assertEqual([e for e in original["traceEvents"] if e.get("cat") in cats],
                         [e for e in events if e.get("cat") in cats])
        self.assertFalse(report["operators_filtered"])
        self.assertEqual(report["slot_counts"], {0:1,1:1})
        self.assertEqual(report["gpu_graph_ranges"], 2)
        self.assertEqual(report["graph_kernel_events"], 4)

    def test_gpu_parent_uses_gpu_time_cpu_lane_uses_actual_api_time(self):
        out = self.root/"out.trace.json.gz"
        export_full_trace(self.write(data()), out, mode="dual")
        events = [e for k,e in trace_records(out) if k == "event"]
        graphs = [e for e in events if e.get("name", "").startswith("cuda_graph_slot")]
        self.assertEqual([(e["name"],e["ts"],e["dur"],e["pid"],e["tid"]) for e in graphs],
            [("cuda_graph_slot0",100,9,0,17),("cuda_graph_slot1",1100,9,0,17)])
        launches = [e for e in events if e.get("cat") == "nq_display_cuda_api"]
        self.assertEqual([(e["ts"],e["dur"]) for e in launches], [(12,2),(1012,2)])
        self.assertTrue(all(e["args"]["display_only"] for e in launches))
        # At equal GPU start timestamps, parents must reach the importer
        # before child kernels so the expanded view has the right nesting.
        for parent in graphs:
            child=next(e for e in events if e.get("cat")=="kernel" and e["ts"]==parent["ts"])
            self.assertLess(events.index(parent),events.index(child))

    def test_native_gpu_scopes_move_but_are_not_removed(self):
        out = self.root/"out.trace.json.gz"
        r = export_full_trace(self.write(data()), out, mode="dual")
        scopes = [e for k,e in trace_records(out) if k=="event" and e.get("name")=="native GPU scope"]
        self.assertEqual(len(scopes),2)
        self.assertEqual(r["moved_gpu_annotations"],2)
        self.assertTrue(all(e["pid"]==0 and e["tid"]!=17 and e["args"]["original_tid"]==17 for e in scopes))

    def test_single_and_multidigit_slot_inference(self):
        source = self.write(data((0,0)))
        self.assertEqual(inspect_trace(source)["slot_order"],[0,0])
        export_full_trace(source,self.root/"single.gz",mode="single")
        self.assertEqual(inspect_trace(self.write(data((10,11))))["slot_order"],[10,11])

    def test_missing_correlation_fails_instead_of_fabricating_ranges(self):
        d = data()
        for e in d["traceEvents"]:
            if e.get("cat")=="kernel":
                e["args"].pop("correlation")
        with self.assertRaisesRegex(ValueError,"Cannot associate"):
            export_full_trace(self.write(d), self.root/"out.gz",mode="dual")

    def test_non_alternating_dual_slots_fail(self):
        with self.assertRaisesRegex(ValueError,"alternating"):
            export_full_trace(self.write(data((0,0,1))),self.root/"out.gz",mode="dual")

    def test_legacy_trace_can_be_grouped_without_replay_wrapper(self):
        d=data()
        for e in d["traceEvents"]:
            if e.get("name","").startswith("CUDAGraph.replay"):
                slot=e["name"].split("/")[1]
                e["name"]=f"nq/cpu/forward/{slot}"
        result=export_full_trace(self.write(d),self.root/"out.gz",mode="dual")
        self.assertEqual(result["cpu_graph_launches"],2)

    def test_truncated_source_leaves_no_output(self):
        p=self.root/"broken.json"
        p.write_text('{"traceEvents":[{"ph":"X"}')
        dest=self.root/"out.gz"
        with self.assertRaises(ValueError):
            export_full_trace(p,dest,mode="dual")
        self.assertFalse(dest.exists())
        self.assertFalse(list(self.root.glob("*.tmp")))


class ReplayInstrumentationTests(unittest.TestCase):
    def test_wrappers_call_real_replay_once_and_restore_on_failure(self):
        entered, called = [], []
        @contextmanager
        def record(name):
            entered.append(name)
            yield
        class Graph:
            def __init__(self, slot):
                self.slot=slot
            def replay(self):
                called.append(self.slot)
                return 123
        class Node:
            def prepare_inputs(self,*a): pass
            def sample_tokens(self,*a): pass
            def compute_logits(self,*a): pass
            def launch_slot(self,*a): pass
            def sample_slot(self,*a): pass
            def _plan_decode(self,*a): pass
            def _commit_oldest(self,*a): pass
            def get_output(self,*a): pass
        gs=[Graph(0),Graph(1)]
        manager=SimpleNamespace(decode_graph_banks=[({8:g},{},None) for g in gs])
        runner=Node()
        runner.cuda_graphs,runner.model,runner._active_slot_id=manager,Node(),0
        pipeline=Node()
        pipeline.pending=[SimpleNamespace(slot_id=0)]
        engine=SimpleNamespace(model_runner=runner,_dual_pipeline=pipeline)
        torch=SimpleNamespace(profiler=SimpleNamespace(record_function=record))
        with patch.dict(sys.modules,{"nano_qwen.engine.async_output":SimpleNamespace(AsyncModelOutput=Node)}):
            with self.assertRaisesRegex(RuntimeError,"test failure"):
                with replay_labels(engine,torch) as counts:
                    self.assertEqual(gs[0].replay(),123)
                    gs[1].replay()
                    self.assertEqual(counts,Counter({(0,8):1,(1,8):1}))
                    raise RuntimeError("test failure")
        self.assertEqual(called,[0,1])
        self.assertEqual(entered,["CUDAGraph.replay/slot0/bs8","CUDAGraph.replay/slot1/bs8"])
        self.assertTrue(all("replay" not in vars(g) for g in gs))
        self.assertNotIn("prepare_inputs",vars(runner))

    def test_cli_defaults_and_window_validation(self):
        args=parse_args([])
        self.assertEqual((args.batch_size,args.input_len,args.output_len,args.profile_rounds),(8,128,256,3))
        self.assertEqual(args.timing_rounds,0)
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parse_args(["--profile-decode-steps","240"])


if __name__=="__main__":
    unittest.main()
