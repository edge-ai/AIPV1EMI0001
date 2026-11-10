import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import benchmark  # noqa: E402
import check_package  # noqa: E402
from public_reference import assert_no_private_references  # noqa: E402


def package_fixture(directory):
    """Source tree with every expected file, and a dist/ copy as eais-tools would make it."""
    root = Path(directory)
    (root / ".include").write_text((ROOT / ".include").read_text())
    names = check_package.EAIS_FILES | check_package.include_entries(ROOT)
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture: {name}\n")
    for name in names:
        target = root / "dist" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    return root, names


def run_check(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = check_package.main(list(args))
    return code, out.getvalue() + err.getvalue()


class CheckPackageTests(unittest.TestCase):
    def test_repository_include_is_an_explicit_safe_allowlist(self):
        entries = check_package.include_entries(ROOT)
        self.assertIn("logic/flows.json", entries)
        self.assertFalse({"tools/benchmark.py", "presentation.ipynb", "localdev.ipynb"} & entries)
        self.assertFalse(any(name.startswith("doc/benchmarks/") for name in entries))

    def test_include_rejects_unsafe_entries(self):
        for entry, message in (("logic/*.json", "explicit files"), ("../secret", "unsafe"),
                               ("logic/flows_cred.json", "credential"), ("missing.txt", "missing"),
                               ("dashboard/node_modules/x.js", "dependency")):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ("logic/flows_cred.json", "dashboard/node_modules/x.js"):
                    (root / name).parent.mkdir(parents=True, exist_ok=True)
                    (root / name).write_text("x")
                (root / ".include").write_text(f"{entry}\n")
                with self.assertRaisesRegex(check_package.PackageError, message):
                    check_package.include_entries(root)

    def test_exact_dist_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, names = package_fixture(directory)
            code, output = run_check("--root", str(root))
            self.assertEqual(code, 0, output)
            self.assertIn(f"OK: {len(names)} files", output)

    def test_dist_rejects_unexpected_missing_changed_or_symlinked_files(self):
        mutations = {
            "source": lambda dist: (dist / "models/source.onnx").write_text("x"),
            "credential": lambda dist: (dist / "logic/flows_cred.json").write_text("{}"),
            "dependency": lambda dist: ((dist / "dashboard/node_modules").mkdir(),
                                        (dist / "dashboard/node_modules/x.js").write_text("x")),
            "missing": lambda dist: (dist / "callback.py").unlink(),
            "changed": lambda dist: (dist / "callback.py").write_text("different\n"),
            "symlink": lambda dist: ((dist / "callback.py").unlink(),
                                     (dist / "callback.py").symlink_to(dist.parent / "callback.py")),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name), tempfile.TemporaryDirectory() as directory:
                root, _ = package_fixture(directory)
                mutate(root / "dist")
                code, output = run_check("--root", str(root))
                self.assertEqual(code, 1)
                self.assertIn("FAIL:", output)

    def archive(self, root, extra=None):
        path = root / "AIPV1EMI0001_dist.tgz"
        with tarfile.open(path, "w:gz") as tar:
            tar.add(root / "dist", arcname="dist")
            if extra:
                extra(tar)
        return path

    def test_archive_checks(self):
        def add(name, data=b"x"):
            def extra(tar):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            return extra

        cases = {
            "valid": (None, 0),
            "duplicate": (add("dist/callback.py", b"pass\n"), 1),
            "traversal": (add("../escape"), 1),
            "outside dist": (add("other/file"), 1),
            "unexpected": (add("dist/models/source.onnx"), 1),
        }
        for name, (extra, expected) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                root, _ = package_fixture(directory)
                archive = self.archive(root, extra)
                code, output = run_check("--root", str(root), "--archive", str(archive))
                self.assertEqual(code, expected, output)


class BenchmarkTests(unittest.TestCase):
    def test_all_steps_build_matching_engines_before_measuring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, deployment = root / "deployment.onnx", root / "deployment.engine"
            source.write_text("original model")
            deployment.write_text("original engine")
            output = root / "benchmark"
            events, commands = [], []

            def raw_graph(source_path, destination):
                events.append("graph")
                destination.write_text("raw graph")
                return destination

            def build(command, **kwargs):
                events.append("build")
                commands.append(command)
                self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
                self.assertTrue(kwargs["check"])
                engine = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--saveEngine="))
                Path(engine).write_text("benchmark engine")

            def measure(*args, **kwargs):
                events.append("measure")
                return {"complete": True}

            with patch.object(benchmark, "make_raw_onnx", side_effect=raw_graph), \
                    patch.object(benchmark.subprocess, "run", side_effect=build), \
                    patch.object(benchmark, "run_benchmark", side_effect=measure) as measured:
                report = benchmark.prepare_and_run(source, output, deployment_engine=deployment,
                                                   trtexec="trtexec-test", workspace_mib=768)
            self.assertEqual(report, {"complete": True})
            self.assertEqual(events, ["graph", "build", "build", "measure"])
            fp32, fp16 = commands
            self.assertEqual(fp32[0], "trtexec-test")
            for flag in (f"--onnx={output / 'raw.onnx'}", "--noTF32", "--workspace=768",
                         "--minShapes=images:1x3x640x640", "--optShapes=images:8x3x640x640",
                         "--maxShapes=images:8x3x640x640", "--skipInference"):
                self.assertIn(flag, fp32)
            self.assertNotIn("--fp16", fp32)
            self.assertIn("--fp16", fp16)
            self.assertEqual([arg.replace("raw-fp16", "raw-fp32") for arg in fp16 if arg != "--fp16"],
                             fp32)
            measured.assert_called_once_with(
                output / "raw.onnx",
                {"TensorRT FP32": output / "raw-fp32.engine",
                 "TensorRT FP16": output / "raw-fp16.engine"},
                output, source_onnx=source.resolve(), deployment_engine=deployment.resolve(),
                trtexec="trtexec-test", workspace_mib=768)
            self.assertEqual(source.read_text(), "original model")
            self.assertEqual(deployment.read_text(), "original engine")

    def test_all_steps_reject_existing_results_and_invalid_inputs_before_building(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "deployment.onnx"
            source.write_text("original")
            output = root / "new-results"
            with patch.object(benchmark, "make_raw_onnx") as graph, \
                    patch.object(benchmark.subprocess, "run") as build:
                with self.assertRaises(FileExistsError):
                    benchmark.prepare_and_run(source, root)
                with self.assertRaises(FileNotFoundError):
                    benchmark.prepare_and_run(root / "missing.onnx", output)
                with self.assertRaises(FileNotFoundError):
                    benchmark.prepare_and_run(source, output, deployment_engine=root / "missing.engine")
                with self.assertRaises(ValueError):
                    benchmark.prepare_and_run(source, output, workspace_mib=0)
                graph.assert_not_called()
                build.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(source.read_text(), "original")

    def test_all_steps_abort_on_failed_build_and_preserve_its_log(self):
        for fail_at in (1, 2):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "deployment.onnx"
                source.write_text("original")
                output, error = root / "benchmark", io.StringIO()
                calls = []

                def build(command, **kwargs):
                    calls.append(command)
                    kwargs["stdout"].write("builder diagnostic\n")
                    if len(calls) == fail_at:
                        raise subprocess.CalledProcessError(9, command)

                with patch.object(benchmark, "make_raw_onnx", return_value=output / "raw.onnx"), \
                        patch.object(benchmark.subprocess, "run", side_effect=build), \
                        patch.object(benchmark, "run_benchmark") as measure, \
                        contextlib.redirect_stderr(error), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(subprocess.CalledProcessError):
                        benchmark.prepare_and_run(source, output)
                self.assertEqual(len(calls), fail_at)
                measure.assert_not_called()
                self.assertIn("builder diagnostic", error.getvalue())
                failed_log = output / ("raw-fp32.log" if fail_at == 1 else "raw-fp16.log")
                self.assertEqual(failed_log.read_text(), "builder diagnostic\n")

    def test_graph_preparation_failure_prevents_builds_and_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            source.write_text("original")
            with patch.object(benchmark, "make_raw_onnx", side_effect=ValueError("invalid graph")), \
                    patch.object(benchmark.subprocess, "run") as build, \
                    patch.object(benchmark, "run_benchmark") as measure:
                with self.assertRaisesRegex(ValueError, "invalid graph"):
                    benchmark.prepare_and_run(source, root / "benchmark")
                build.assert_not_called()
                measure.assert_not_called()

    def test_all_command_dispatches_to_the_complete_workflow(self):
        report = json.loads((ROOT / "doc/benchmarks/orin-nx-16gb-jp6.0.json").read_text())
        with patch.object(benchmark, "prepare_and_run", return_value=report) as complete, \
                contextlib.redirect_stdout(io.StringIO()):
            code = benchmark.main(["all", "--onnx", "source.onnx", "--output", "results",
                                   "--deployment-engine", "deployment.engine",
                                   "--trtexec", "trtexec-test", "--workspace-mib", "512"])
        self.assertEqual(code, 0)
        complete.assert_called_once_with("source.onnx", "results", trtexec="trtexec-test",
                                         workspace_mib=512, deployment_engine="deployment.engine")

    def test_run_command_still_accepts_prebuilt_benchmark_artifacts(self):
        report = json.loads((ROOT / "doc/benchmarks/orin-nx-16gb-jp6.0.json").read_text())
        with patch.object(benchmark, "run_benchmark", return_value=report) as measure, \
                contextlib.redirect_stdout(io.StringIO()):
            code = benchmark.main(["run", "--onnx", "raw.onnx",
                                   "--engine", "TensorRT FP16=raw.engine", "--output", "results"])
        self.assertEqual(code, 0)
        measure.assert_called_once_with("raw.onnx", {"TensorRT FP16": "raw.engine"}, "results",
                                        source_onnx=None, deployment_engine=None,
                                        trtexec=benchmark.DEFAULT_TRTEXEC, workspace_mib=None)

    def test_ort_profile_summarizes_unique_executed_nodes(self):
        def node(index, provider, operation):
            return {"cat": "Node", "name": f"{operation}_{index}_kernel_time",
                    "args": {"node_index": str(index), "provider": provider, "op_name": operation}}

        gpu = node(1, "CUDAExecutionProvider", "Conv")
        cpu = node(2, "CPUExecutionProvider", "Shape")
        events = [
            {"cat": "Node", "name": "Conv_1_fence_before", "args": {"op_name": "Conv"}},
            gpu, cpu, gpu.copy(), cpu.copy(),
        ]
        self.assertEqual(benchmark.summarize_ort_profile(events), {
            "CPUExecutionProvider": {"node_count": 1, "operator_counts": {"Shape": 1}},
            "CUDAExecutionProvider": {"node_count": 1, "operator_counts": {"Conv": 1}},
        })

        with self.assertRaisesRegex(ValueError, "No CUDA"):
            benchmark.summarize_ort_profile([cpu])
        with self.assertRaisesRegex(ValueError, "missing"):
            benchmark.summarize_ort_profile([
                {"cat": "Node", "name": "bad_kernel_time", "args": {"op_name": "Conv"}},
            ])
        with self.assertRaisesRegex(ValueError, "changed"):
            benchmark.summarize_ort_profile([gpu, node(1, "CPUExecutionProvider", "Conv")])

    def test_tensorrt_binding_dependencies_must_match_exactly(self):
        installed = {"libnvinfer8": "8.6.2.3-1+cuda12.2", "libnvonnxparsers8": "8.6.2.3-1+cuda12.2"}.get
        depends = ("python3 (>= 3.10), python3 (<< 3.11), libnvinfer8 (= 8.6.2.3-1+cuda12.2), "
                   "libnvonnxparsers8 (= 8.6.2.3-1+cuda12.2)")
        unmet = benchmark.unmet_exact_dependencies
        self.assertEqual(unmet(depends, installed), [])
        self.assertEqual(unmet(depends.replace("8.6.2.3-1", "10.3.0.26-1"), installed),
                         ["libnvinfer8 (= 10.3.0.26-1+cuda12.2)",
                          "libnvonnxparsers8 (= 10.3.0.26-1+cuda12.2)"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
    def test_latency_summary(self):
        summarize = benchmark.summarize
        reference = summarize("ORT", 4, [[10, 10, 10], [10, 10, 10]], expected_rounds=2, expected_requests=3)
        self.assertEqual(reference["speedup_vs_onnxruntime"], 1.0)
        row = summarize("TRT", 4, [[4, 5, 6], [5, 5, 5]], reference["median_ms"],
                        expected_rounds=2, expected_requests=3)
        self.assertEqual((row["median_ms"], row["ms_per_image"], row["serial_images_per_second"],
                          row["speedup_vs_onnxruntime"], row["round_medians_ms"]),
                         (5.0, 1.25, 800.0, 2.0, [5.0, 5.0]))
        for bad in ([[1, 2, 3]], [[1, 2], [1, 2]], [[1, 2, 0], [1, 2, 3]], [[1, 2, float("nan")], [1, 2, 3]]):
            with self.assertRaises(AssertionError):
                summarize("bad", 1, bad, expected_rounds=2, expected_requests=3)

    def test_one_timing_boundary_and_no_hidden_side_effects(self):
        source = (ROOT / "tools/benchmark.py").read_text()
        timed = ("device_sync()\n                    start = time.perf_counter_ns()\n"
                 "                    runners[label]()\n                    device_sync()")
        self.assertEqual(source.count(timed), 1, "every backend must share one timing block")
        for required in ('("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0})',
                         "session.disable_fallback()", "binding.bind_ortvalue_input",
                         "execute_async_v3(stream.value)", "order_rng.shuffle(order)",
                         '"apt-get", "download"'):
            self.assertIn(required, source)
        self.assertLess(source.index("session.end_profiling()"),
                        source.index("start = time.perf_counter_ns()"))
        for forbidden in ("urllib", "socket", "pip install", "sudo", "apt-get install",
                          "--loadEngine", "systemctl", "nvpmodel", "jetson_clocks"):
            self.assertNotIn(forbidden, source)

    def test_engine_arguments(self):
        self.assertEqual(benchmark.parse_engine("TensorRT FP16=a/b.engine"), ("TensorRT FP16", "a/b.engine"))
        for bad in ("no-separator", "=path", "label="):
            with self.assertRaises(Exception):
                benchmark.parse_engine(bad)


class PublishedResultsTests(unittest.TestCase):
    report = json.loads((ROOT / "doc/benchmarks/orin-nx-16gb-jp6.0.json").read_text())

    def test_results_are_complete_and_consistent(self):
        rows = {(r["variant"], r["batch"]): r for r in self.report["results"]}
        variants = (benchmark.ORT_LABEL, "TensorRT FP32", "TensorRT FP16")
        self.assertEqual(set(rows), {(v, b) for v in variants for b in benchmark.BATCHES})
        for (variant, batch), row in rows.items():
            reference = rows[(benchmark.ORT_LABEL, batch)]["median_ms"]
            self.assertAlmostEqual(row["speedup_vs_onnxruntime"], reference / row["median_ms"], places=6)
            self.assertAlmostEqual(row["ms_per_image"], row["median_ms"] / batch, places=6)

    def test_readme_quotes_the_published_medians(self):
        readme = (ROOT / "README.md").read_text()
        for row in self.report["results"]:
            self.assertIn(f"{row['median_ms']:.1f} ms", readme, row["variant"])

    def test_published_conditions_and_provider_placement_are_consistent(self):
        placement = self.report["onnxruntime_node_placement"]
        self.assertEqual(placement["profiled_batches"], list(benchmark.BATCHES))
        for provider in placement["providers"].values():
            self.assertEqual(provider["node_count"], sum(provider["operator_counts"].values()))
        self.assertGreater(placement["providers"]["CUDAExecutionProvider"]["node_count"], 0)
        self.assertGreater(placement["providers"]["CPUExecutionProvider"]["node_count"], 0)
        states = self.report["gpu_state_during_rounds"]
        self.assertEqual({(s["variant"], s["batch"], s["round"]) for s in states},
                         {(v, b, r) for v in (benchmark.ORT_LABEL, "TensorRT FP32", "TensorRT FP16")
                          for b in benchmark.BATCHES for r in range(1, benchmark.ROUNDS + 1)})
        conditions = self.report["run_conditions"]
        self.assertGreater(conditions["pipeline_checks_with_zero_active"], 0)
        self.assertLessEqual(max(conditions["idle_gpu_load_percent"]), 10)
        self.assertLessEqual(max(conditions["idle_gpu_temperature_c"]), 80)

    def test_report_contains_no_station_identity(self):
        assert_no_private_references(self, json.dumps(self.report), "benchmark report")


if __name__ == "__main__":
    unittest.main()
