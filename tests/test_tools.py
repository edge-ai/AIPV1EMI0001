import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import benchmark  # noqa: E402
import check_package  # noqa: E402


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

    def test_report_contains_no_station_identity(self):
        text = json.dumps(self.report)
        for forbidden in ("192.168.", "DEX5W", "nvidia@", "/home/"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
