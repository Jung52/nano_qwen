"""CUDA-free regressions for export correlation, clocks and acceptance gates."""
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from profile_slots_b8 import CompactTrace, trace_records
from slot_trace import scan_slot_groups, PHYSICAL
from bench_fused_slots import compare_tokens, summarize


def event(name, cat, ts, dur, corr=None, gpu=False, tid=None):
    return dict(name=name, cat=cat, ph="X", pid=0 if gpu else 100,
        tid=tid if tid is not None else 17 if gpu else 101, ts=ts, dur=dur,
        args={} if corr is None else dict(correlation=corr))


def fixture(slot=0, single=False):
    prefix = "forward" if single else "launch_forward"
    return dict(baseTimeNanoseconds=1000000, traceEvents=[
        event(f"nq/cpu/{prefix}/slot{slot}", "user_annotation", 10, 40),
        event(f"nq/cpu/forward/slot{slot}", "user_annotation", 11, 38),
        event(f"nq/cpu/lm_head/slot{slot}", "user_annotation", 30, 10),
        event(f"nq/cpu/{'sampler' if single else 'sample_slot'}/slot{slot}", "user_annotation", 50, 10),
        event(f"nq/cpu/d2h_enqueue/slot{slot}", "user_annotation", 55, 4),
        event("cudaGraphLaunch", "cuda_runtime", 20, 1, 7),
        event("cudaLaunchKernel", "cuda_runtime", 32, 1, 8),
        event("cudaLaunchKernel", "cuda_runtime", 51, 1, 9),
        event("cudaMemcpyAsync", "cuda_runtime", 56, 1, 10),
        event("kernel_a", "kernel", 100, 2, 7, True),
        event("kernel_b", "kernel", 103, 2, 7, True),
        event("lm_head", "kernel", 106, 4, 8, True),
        event("_norm_kernel", "kernel", 110, 1, 9, True),
        event("Memcpy DtoH (Device -> Pinned)", "gpu_memcpy", 112, 1, 10, True, 13),
        dict(name="process_name", ph="M", pid=0, tid=0, args=dict(name="GPU")),
        dict(name="process_name", ph="M", pid=9, tid=0, args=dict(name="unused GPU")),
        dict(name="flow", cat="ac2g", ph="s", id=1, pid=100, tid=101, ts=20),
        dict(name="flow", cat="ac2g", ph="f", id=1, pid=0, tid=17, ts=100)])


class SlotTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, data):
        path = self.root/name
        path.write_text(json.dumps(data))
        return path

    def test_gpu_ranges_use_gpu_clock_and_multidigit_slot(self):
        source = self.write("in.json", fixture(10))
        groups, _, stats = scan_slot_groups(source, trace_records)
        graph = next(g for g in groups if g["name"] == "cuda_graph_slot_10")
        self.assertEqual((graph["ts"], graph["dur"], graph["pid"], graph["tid"]), (100, 5, 0, 17))
        self.assertEqual(graph["args"]["physical_events"], 2)
        self.assertEqual(stats["graph_kernel_events"], 2)
        self.assertEqual(stats["graph_groups"], 1)  # Nested scopes must not double count.
        self.assertEqual(stats["fused_norm_kernels"], 1)
        copy_range = next(g for g in groups if g["name"] == "D2H_slot_10")
        self.assertEqual((copy_range["ts"], copy_range["tid"]), (112, 13))

    def test_single_slot_fallback_scopes(self):
        groups, _, _ = scan_slot_groups(self.write("single.json", fixture(single=True)), trace_records)
        self.assertTrue(any(g["name"] == "cuda_graph_slot_0" for g in groups))
        self.assertTrue(any(g["name"] == "sample_slot_0" for g in groups))

    def test_detail_preserves_physical_events_and_overview_hides_them(self):
        data = fixture()
        source, dest = self.write("in.json", data), self.root/"test.trace.json.gz"
        with CompactTrace(dest) as sink:
            sink.add(source, 1)
        with gzip.open(dest, "rt") as f:
            detail = json.load(f)["traceEvents"]
        with gzip.open(sink.overview_path, "rt") as f:
            overview = json.load(f)["traceEvents"]
        self.assertEqual([e for e in detail if e.get("cat") in PHYSICAL],
                         [e for e in data["traceEvents"] if e.get("cat") in PHYSICAL])
        self.assertFalse(any(e.get("cat") in PHYSICAL for e in overview))
        self.assertFalse(any(e.get("cat") == "ac2g" for e in detail))
        self.assertFalse(any(e.get("pid") == 9 for e in detail))
        self.assertTrue(any(e.get("name") == "cuda_graph_slot_0" for e in overview))

    def test_merged_windows_isolate_correlations_and_normalize_time(self):
        a, b = fixture(0), fixture(1)
        b["baseTimeNanoseconds"] += 1000000
        # Same CUPTI correlations may restart in each independent profiler.
        paths = [self.write("a.json", a), self.write("b.json", b)]
        dest = self.root/"merged.trace.json.gz"
        with CompactTrace(dest, keep_flows=True) as sink:
            for i, path in enumerate(paths):
                sink.add(path, i)
        with gzip.open(dest, "rt") as f:
            events = json.load(f)["traceEvents"]
        groups = [e for e in events if e.get("name", "").startswith("cuda_graph_slot_")]
        self.assertEqual([(g["name"], g["ts"], g["dur"]) for g in groups],
                         [("cuda_graph_slot_0", 100, 5), ("cuda_graph_slot_1", 1100, 5)])
        flows = [e["id"] for e in events if e.get("cat") == "ac2g"]
        self.assertEqual(flows[0], flows[1])
        self.assertEqual(flows[2], flows[3])
        self.assertNotEqual(flows[0], flows[2])

    def test_missing_gpu_correlation_is_not_fabricated(self):
        data = fixture()
        for e in data["traceEvents"]:
            if e.get("cat") == "kernel" and e["args"]["correlation"] == 7:
                del e["args"]["correlation"]
        groups, _, stats = scan_slot_groups(self.write("in.json", data), trace_records)
        self.assertEqual(stats["graph_launches"], 1)
        self.assertEqual(stats["graph_groups"], 0)
        self.assertFalse(any(g["name"].startswith("cuda_graph_slot_") for g in groups))

    def test_truncation_aborts_both_outputs(self):
        source = self.root/"bad.json"
        source.write_text('{"traceEvents":[{"ph":"X"}')
        dest = self.root/"broken.trace.json.gz"
        with self.assertRaises(ValueError), CompactTrace(dest):
            pass  # An empty export must fail too.
        with self.assertRaises(ValueError):
            with CompactTrace(dest) as sink:
                sink.add(source, 1)
        self.assertFalse(dest.exists())
        self.assertFalse(sink.overview_path.exists())
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_reexport_does_not_duplicate_derived_groups(self):
        raw = self.write("raw.json", fixture())
        a, b = self.root/"a.trace.json.gz", self.root/"b.trace.json.gz"
        with CompactTrace(a) as sink:
            sink.add(raw, 1)
        with CompactTrace(b) as sink:
            sink.add(a, 1)
        self.assertEqual(sink.kept["gpu_user_annotation"], 5)


class AcceptanceTests(unittest.TestCase):
    @staticmethod
    def rounds():
        return [dict(seed=43, prompt_sha256="same", token_ids=[[1, 2, 3], [4, 5, 6]],
            ttft_ms=10, decode_ms=20, tpot_ms=1, decode_tok_s=100, e2e_ms=30, output_tok_s=90)]

    def test_first_token_mismatch_and_length_are_failures(self):
        a, b = self.rounds(), self.rounds()
        b[0]["token_ids"][1][1] = 99
        diff = compare_tokens(a, b)
        self.assertEqual((diff[0]["request"], diff[0]["first_token"]), (1, 1))
        b = self.rounds()
        b[0]["token_ids"][0].append(7)
        self.assertTrue(compare_tokens(a, b))

    def test_performance_does_not_override_failed_correctness(self):
        reports = {name: dict(status="ok", measurements=self.rounds(), kv_blocks=24,
            model="same", source=dict(source_sha256="same"), trace_stats=dict(gpu_grouping={
            "kernel": 4, "graph_launches": 1, "graph_groups": 1, "graph_kernel_events": 2}))
            for name in ("reference_single", "fused_single", "fused_dual", "reference_dual")}
        self.assertEqual(summarize(reports)["status"], "ok")
        changed = copy.deepcopy(reports)
        changed["fused_dual"]["measurements"][0]["token_ids"][0][0] = 50
        summary = summarize(changed)
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(all(not row["correctness_accepted"] for row in summary["comparisons"]))


if __name__ == "__main__":
    unittest.main()
