import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CELLS = {
    cell["id"]: "".join(cell["source"])
    for cell in json.loads((ROOT / "localdev.ipynb").read_text())["cells"]
    if cell["cell_type"] == "code"
}


class NotebookTests(unittest.TestCase):
    def test_code_cells_compile(self):
        for name, source in CELLS.items():
            with self.subTest(cell=name):
                compile(source, name, "exec")

    def test_onnx_filename_is_consistent(self):
        source = CELLS["step-04"]
        namespace = {"AIP_ROOT": ROOT}
        start = source.index('HF_BASE =')
        end = source.index('assert ENGINE_SOURCE')
        exec(source[start:end], namespace)
        name, (destination, _) = next(iter(namespace["ONNX"].items()))
        self.assertEqual(name, "yolov7tiny-640.onnx")
        self.assertEqual(destination.name, name)
        self.assertIn(f'OnnxModel="{name}"', (ROOT / "bin/build.sh").read_text())
        self.assertIn(name, CELLS["step-23"])
        self.assertNotIn("parking-rev3", "\n".join(CELLS.values()))

    def test_commands_reject_unexpected_exit_codes(self):
        namespace = {"subprocess": subprocess}
        source = CELLS["step-02"]
        exec(source[source.index("def run("):source.index('AIP_ID =')], namespace)
        run = namespace["run"]
        run(["sh", "-c", "exit 0"])
        run(["sh", "-c", "exit 124"], expected=(0, 124))
        with self.assertRaises(subprocess.CalledProcessError):
            run(["sh", "-c", "exit 1"])
        with self.assertRaises(subprocess.CalledProcessError):
            run(["sh", "-c", "exit 137"], expected=(0, 124))

    def test_benchmarks_can_be_skipped_without_optional_dependencies(self):
        namespace = {"RUN_BENCHMARKS": False}
        for name in ("bench-graph", "bench-build", "bench-trt-python", "bench-run"):
            exec(CELLS[name], namespace)
        self.assertNotIn("BENCH_ROOT", namespace)
        self.assertNotIn("trt", namespace)

    def test_benchmark_uses_one_harness_without_triton(self):
        source = "\n".join(CELLS[name] for name in
                           ("bench-graph", "bench-build", "bench-trt-python", "bench-run"))
        for forbidden in ("urlopen", "urllib", "socket", "http://", "--http-port",
                          "--load-model", "TRITON_ROOT", "BENCH_DRIVER", "-ltritonserver",
                          "pip install", "sudo", "apt-get install", "--loadEngine",
                          "TensorrtExecutionProvider\"]"):
            self.assertNotIn(forbidden, source)
        run_cell = CELLS["bench-run"]
        for required in ('("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0})',
                         "session.disable_fallback()", "binding.bind_ortvalue_input",
                         "binding.bind_ortvalue_output", "execute_async_v3(stream.value)",
                         "set_tensor_address(\"images\", gpu_input.data_ptr())",
                         "order_rng.shuffle(order)"):
            self.assertIn(required, run_cell)
        timed = ("device_sync()\n                    start = time.perf_counter_ns()\n"
                 "                    runners[label]()\n                    device_sync()")
        self.assertEqual(run_cell.count(timed), 1, "every backend must share one timing block")
        self.assertIn('apt-get", "download", package', CELLS["bench-trt-python"])

    def test_tensorrt_binding_dependencies_must_match_exactly(self):
        source = CELLS["bench-trt-python"]
        start = source.index("    def unmet_exact_dependencies(")
        end = source.index("    LIBNVINFER_VERSION =")
        namespace = {"re": __import__("re"), "dpkg_version": None}
        exec(__import__("textwrap").dedent(source[start:end]), namespace)
        unmet = namespace["unmet_exact_dependencies"]
        installed = {"libnvinfer8": "8.6.2.3-1+cuda12.2", "libnvonnxparsers8": "8.6.2.3-1+cuda12.2"}.get
        depends = ("python3 (>= 3.10), python3 (<< 3.11), libnvinfer8 (= 8.6.2.3-1+cuda12.2), "
                   "libnvonnxparsers8 (= 8.6.2.3-1+cuda12.2)")
        self.assertEqual(unmet(depends, installed), [])
        self.assertEqual(unmet(depends.replace("8.6.2.3-1", "10.3.0.26-1"), installed),
                         ["libnvinfer8 (= 10.3.0.26-1+cuda12.2)",
                          "libnvonnxparsers8 (= 10.3.0.26-1+cuda12.2)"])
        self.assertEqual(unmet(depends + ", libnvinfer-plugin8 (= 8.6.2.3-1+cuda12.2)", installed),
                         ["libnvinfer-plugin8 (= 8.6.2.3-1+cuda12.2)"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
    def test_latency_summary_uses_shared_statistics(self):
        import numpy as np

        source = CELLS["bench-run"]
        start = source.index("    def summarize(")
        end = source.index("    state_before =")
        namespace = {"np": np, "ROUNDS": 2, "MEASURED_REQUESTS": 3}
        exec(__import__("textwrap").dedent(source[start:end]), namespace)
        summarize = namespace["summarize"]
        reference = summarize("ORT", 4, [[10, 10, 10], [10, 10, 10]])
        self.assertEqual(reference["speedup_vs_onnxruntime"], 1.0)
        row = summarize("TRT", 4, [[4, 5, 6], [5, 5, 5]], reference["median_ms"])
        self.assertEqual(row["median_ms"], 5.0)
        self.assertEqual(row["ms_per_image"], 1.25)
        self.assertEqual(row["serial_images_per_second"], 800.0)
        self.assertEqual(row["speedup_vs_onnxruntime"], 2.0)
        self.assertEqual(row["round_medians_ms"], [5.0, 5.0])
        for bad in ([[1, 2, 3]], [[1, 2], [1, 2]], [[1, 2, 0], [1, 2, 3]],
                    [[1, 2, float("nan")], [1, 2, 3]]):
            with self.assertRaises(AssertionError):
                summarize("bad", 1, bad)

    def test_prebuilt_path_downloads_onnx_only_when_benchmarking(self):
        source = CELLS["step-04"].replace('ENGINE_SOURCE = "build"', 'ENGINE_SOURCE = "prebuilt"')
        source = source[:source.index("for name, (dest, expected) in downloads.items():")]
        for enabled in (True, False):
            with self.subTest(benchmarks=enabled):
                namespace = {"AIP_ROOT": ROOT, "TRT_VERSION": "8602"}
                candidate = source.replace("RUN_BENCHMARKS = True", f"RUN_BENCHMARKS = {enabled}")
                exec(candidate, namespace)
                self.assertEqual(len(namespace["downloads"]), 2 if enabled else 1)
                self.assertEqual("yolov7tiny-640.onnx" in namespace["downloads"], enabled)

    def test_benchmark_engines_share_graph_profile_and_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.onnx"
            raw.write_bytes(b"benchmark-only graph")
            commands = []

            def build(args, log_path):
                commands.append(args)
                engine = next(a.split("=", 1)[1] for a in args if a.startswith("--saveEngine="))
                Path(engine).write_bytes(b"engine")

            namespace = {
                "RUN_BENCHMARKS": True, "BENCH_ROOT": root,
                "RAW_ONNX": raw, "TRTEXEC": Path("/sdk/trtexec"),
                "WORKSPACE_MIB": 1024, "shutil": shutil, "run_logged": build,
            }
            exec(CELLS["bench-build"], namespace)
            self.assertEqual(len(commands), 2)
            for args in commands:
                self.assertIn(f"--onnx={raw}", args)
                self.assertIn("--noTF32", args)
                self.assertIn("--workspace=1024", args)
                self.assertIn("--skipInference", args)
                self.assertIn("--minShapes=images:1x3x640x640", args)
                self.assertIn("--optShapes=images:8x3x640x640", args)
                self.assertIn("--maxShapes=images:8x3x640x640", args)
                engine = Path(next(a.split("=", 1)[1] for a in args if a.startswith("--saveEngine=")))
                self.assertEqual(engine.parent, root)
            self.assertNotIn("--fp16", commands[0])
            self.assertIn("--fp16", commands[1])
            for engine in namespace["ENGINES"].values():
                self.assertTrue(engine.is_file())
            self.assertEqual(raw.read_bytes(), b"benchmark-only graph")
            self.assertEqual(list(root.rglob("config.pbtxt")), [])

    def test_include_is_an_explicit_existing_file_allowlist(self):
        names = [line.split("#", 1)[0].strip()
                 for line in (ROOT / ".include").read_text().splitlines()]
        names = [name for name in names if name]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            with self.subTest(path=name):
                path = Path(name)
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                self.assertFalse(any(char in name for char in "!*?[]"))
                self.assertTrue((ROOT / path).is_file())
                self.assertFalse((ROOT / path).is_symlink())
                self.assertNotIn("node_modules", path.parts)
                self.assertNotEqual(path.name, "flows_cred.json")

    def package_fixture(self, directory):
        root = Path(directory)
        entries = [line.split("#", 1)[0].strip()
                   for line in (ROOT / ".include").read_text().splitlines()]
        included = {name for name in entries if name}
        assignment = next(node for node in ast.parse(CELLS["step-10"]).body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "expected_files"
                                  for target in node.targets))
        names = ast.literal_eval(assignment.value.right) | included
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture: {name}\n")
        (root / ".include").write_text((ROOT / ".include").read_text())

        def build(args):
            self.assertEqual(args, ["eais-tools", "aip", "build", "--uuid",
                                    "AIPV1EMI0001", "--include-file", ".include"])
            for name in names:
                target = root / "dist" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / name, target)

        namespace = {"AIP_ROOT": root, "AIP_ID": "AIPV1EMI0001",
                     "Path": Path, "run": Mock(side_effect=build),
                     "sha256": lambda p: hashlib.sha256(p.read_bytes()).hexdigest()}
        resolver = Mock(resolve_include_file=Mock(return_value=list(included)))
        return root, names, namespace, resolver

    def test_eais_build_and_exact_dist_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root, names, namespace, resolver = self.package_fixture(directory)
            with patch.dict("sys.modules", {"eais_tools.include_parser": resolver}):
                exec(CELLS["step-10"], namespace)
            resolver.resolve_include_file.assert_called_once_with(str(root), include_filename=".include")
            exec(CELLS["step-11"], namespace)
            self.assertEqual(set(namespace["expected_hashes"]), names)
            self.assertNotIn("rmtree", CELLS["step-10"])

    def test_distribution_rejects_unexpected_missing_or_changed_files(self):
        for mutation in ("source", "credential", "dependency", "missing", "changed", "symlink"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root, _, namespace, resolver = self.package_fixture(directory)
                with patch.dict("sys.modules", {"eais_tools.include_parser": resolver}):
                    exec(CELLS["step-10"], namespace)
                dist = root / "dist"
                if mutation in ("source", "credential", "dependency"):
                    name = {
                        "source": "models/AIPV1EMI0001_Yolo_Detector/1/source.onnx",
                        "credential": "configs/.env",
                        "dependency": "dashboard/node_modules/dependency.js",
                    }[mutation]
                    target = dist / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("must not ship\n")
                elif mutation == "missing":
                    (dist / "callback.py").unlink()
                elif mutation == "changed":
                    (dist / "callback.py").write_text("different bytes\n")
                else:
                    (dist / "callback.py").unlink()
                    (dist / "callback.py").symlink_to(root / "callback.py")
                with self.assertRaises(AssertionError):
                    exec(CELLS["step-11"], namespace)

    def test_missing_include_entry_fails_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, namespace, resolver = self.package_fixture(directory)
            (root / "dashboard/package.json").unlink()
            with patch.dict("sys.modules", {"eais_tools.include_parser": resolver}):
                with self.assertRaisesRegex(AssertionError, "missing included file"):
                    exec(CELLS["step-10"], namespace)
            namespace["run"].assert_not_called()

    def test_runtime_acceptance_uses_only_eais_pipeline_test(self):
        source = "\n".join(CELLS.values())
        self.assertNotIn("config-live", CELLS)
        self.assertNotIn("config_diagnostic", source)
        self.assertNotIn("gst-launch-1.0", source)
        self.assertIn('"eais-tools", "pipeline", "test"', CELLS["step-16"])
        self.assertIn('expected=(0, 124)', CELLS["step-16"])
        self.assertIn('assert records', CELLS["step-17"])
        self.assertIn('assert any(peak.values())', CELLS["step-17"])

    def test_pipeline_preflight_refuses_existing_test_resources(self):
        for endpoint, item in (
            ("pipelines", {"pk": "another-pipeline", "name": "eais_pipeline"}),
            ("pipelines", {"pk": "another-pipeline", "name": "eais_reference"}),
            ("medias", {"pk": "another-media", "name": "eais_media_1"}),
            ("medias", {"pk": "another-media", "name": "eais_reference"}),
        ):
            with self.subTest(endpoint=endpoint):
                responses = {
                    path: Mock(json=Mock(return_value=[item] if path == endpoint else []))
                    for path in ("pipelines", "medias")
                }
                requests = Mock(get=lambda url, **kw: responses[url.rsplit("/", 1)[1]])
                namespace = {"run": Mock()}
                with patch.dict("sys.modules", {"requests": requests}):
                    with self.assertRaisesRegex(AssertionError, "replace existing resources"):
                        exec(CELLS["step-16"], namespace)
                namespace["run"].assert_not_called()

    def test_cleanup_refuses_other_runs_resources(self):
        for endpoint, item in (
            ("pipelines", {"pk": "another-pipeline", "name": "eais_other"}),
            ("medias", {"pk": "another-media", "name": "eais_media_2"}),
            ("medias", {"pk": "another-media", "name": "eais_reference"}),
        ):
            with self.subTest(endpoint=endpoint):
                namespace = {
                    "station_list": lambda path: [item] if path == endpoint else [],
                    "PIPELINE_ID": "owned-pipeline", "TEST_MEDIA_IDS": {"owned-media"},
                    "run": Mock(),
                }
                with self.assertRaisesRegex(AssertionError, "another run"):
                    exec(CELLS["step-23"], namespace)
                namespace["run"].assert_not_called()

    def test_pipeline_identity_rejects_preexisting_or_ambiguous_test(self):
        for pipelines in (
            [{"pk": "preexisting", "name": "eais_pipeline"}],
            [{"pk": "new-1", "name": "eais_pipeline"}, {"pk": "new-2", "name": "eais_pipeline"}],
            [],
        ):
            with self.subTest(pipelines=pipelines):
                namespace = {"station_list": lambda path: pipelines, "PIPELINES_BEFORE": {"preexisting"}}
                with self.assertRaises(AssertionError):
                    exec(CELLS["step-18"], namespace)

    def test_callback_acceptance_rejects_empty_or_zero_detection_data(self):
        empty_counts = {
            "obj_counter": {name: {"max_val": 0} for name in ("car", "person", "truck")},
            "analytics_obj_cnt": {"objInROIcnt": {}, "objLCCurrCnt": {}},
        }
        cases = ("", "[]", json.dumps([{"since": 1, "until": 2, "data": {"0": empty_counts}}]))
        for data in cases:
            with self.subTest(data=data), tempfile.TemporaryDirectory() as directory:
                dump = Path(directory) / "dump.json"
                dump.write_text(data)
                with self.assertRaises(AssertionError):
                    exec(CELLS["step-17"], {"DUMP": dump})

    def test_notebook_parser_failure_does_not_copy_stale_library(self):
        import shlex

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parser = root / "src/nvdsinfer_customparser"
            parser.mkdir(parents=True)
            (parser / "libnvds_infercustomparser.so").write_bytes(b"stale library")
            model = root / "models"
            model.mkdir()
            processes = Mock()
            processes.run.side_effect = [
                subprocess.CompletedProcess([], 0, stdout="release 12.2"),
                subprocess.CompletedProcess([], 37),
            ]
            processes.CalledProcessError = subprocess.CalledProcessError
            namespace = {
                "Path": Path, "AIP_ROOT": root, "MODEL_DIR": model,
                "subprocess": processes, "re": __import__("re"),
                "shutil": shutil, "shlex": shlex,
            }
            with self.assertRaises(subprocess.CalledProcessError):
                exec(CELLS["step-06"], namespace)
            self.assertFalse((model / "libnvds_infercustomparser.so").exists())

    def check_archive(self, mutation=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist = root / "dist"
            dist.mkdir()
            manifest = {"unique_id": "AIPV1EMI0001", "version": "6.0.0"}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (dist / "manifest.json").write_text(json.dumps(manifest))
            (dist / "callback.py").write_text("pass\n")

            def export(args):
                self.assertEqual(args, ["eais-tools", "aip", "export", "--uuid", "AIPV1EMI0001"])
                with tarfile.open(root / "AIPV1EMI0001_dist.tgz", "w:gz") as archive:
                    archive.add(dist, arcname="dist")
                    if mutation:
                        mutation(archive)

            namespace = {
                "AIP_ROOT": root,
                "AIP_ID": "AIPV1EMI0001",
                "DIST": dist,
                "Path": Path,
                "run": export,
                "hashlib": hashlib,
                "sha256": lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
                "expected_hashes": {p.relative_to(dist).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in dist.rglob("*") if p.is_file()},
            }
            exec(CELLS["step-12"], namespace)
            self.assertTrue(namespace["ARCHIVE"].is_file())

    def test_accepts_eais_tools_dist_prefix(self):
        self.check_archive()

    def test_rejects_duplicate_entries(self):
        def duplicate(archive):
            info = tarfile.TarInfo("dist/callback.py")
            info.size = 5
            archive.addfile(info, io.BytesIO(b"pass\n"))

        with self.assertRaisesRegex(AssertionError, "duplicate archive entry"):
            self.check_archive(duplicate)

    def test_rejects_path_traversal(self):
        def traversal(archive):
            archive.addfile(tarfile.TarInfo("../unexpected"))

        with self.assertRaises(AssertionError):
            self.check_archive(traversal)

    def test_export_rejects_dist_modified_since_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            dist = Path(directory) / "dist"
            dist.mkdir()
            (dist / "callback.py").write_text("modified\n")
            namespace = {
                "DIST": dist, "run": Mock(),
                "sha256": lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
                "expected_hashes": {"callback.py": hashlib.sha256(b"original\n").hexdigest()},
            }
            with self.assertRaisesRegex(AssertionError, "changed since validation"):
                exec(CELLS["step-12"], namespace)
            namespace["run"].assert_not_called()

    def test_cleanup_preserves_package(self):
        self.assertNotIn("rmtree", CELLS["step-23"])
        self.assertNotIn("ARCHIVE.unlink", CELLS["step-23"])
        self.assertIn("assert ARCHIVE.is_file()", CELLS["step-23"])

    def test_parser_build_failure_does_not_reuse_stale_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            (root / "mock").mkdir()
            source = root / "src/nvdsinfer_customparser"
            source.mkdir(parents=True)
            (root / "models/AIPV1EMI0001_Yolo_Detector").mkdir(parents=True)
            shutil.copy2(ROOT / "bin/build.sh", root / "bin/build.sh")
            (source / "libnvds_infercustomparser.so").write_text("stale library")
            make = root / "mock/make"
            make.write_text("#!/bin/sh\nexit 37\n")
            make.chmod(0o755)
            result = subprocess.run(
                ["bash", "./build.sh"], cwd=root / "bin",
                env={**os.environ, "SKIP_ENGINE": "1",
                     "PATH": f"{root / 'mock'}:{os.environ['PATH']}"},
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("success", result.stdout)


if __name__ == "__main__":
    unittest.main()
