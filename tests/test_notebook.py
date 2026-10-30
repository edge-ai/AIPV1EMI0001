import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_package  # noqa: E402


def load(name):
    cells = json.loads((ROOT / name).read_text())["cells"]
    return {cell["id"]: (cell["cell_type"], "".join(cell["source"])) for cell in cells}


NOTEBOOK = load("localdev.ipynb")
PRESENTATION = load("presentation.ipynb")
CODE = {name: source for name, (kind, source) in NOTEBOOK.items() if kind == "code"}
BASH = {name: source.split("\n", 1)[1] for name, source in CODE.items() if source.startswith("%%bash\n")}
MARKDOWN = "\n".join(source for kind, source in NOTEBOOK.values() if kind == "markdown")

# Station commands replaced by stubs. Each call is appended to $CALLS.
STUBS = {
    "curl": """
        echo "curl $*" >> "$CALLS"
        url=${@: -1}; name=${url##*/}
        [ -f "$FIXTURES/$name.json" ] || exit 22
        cat "$FIXTURES/$name.json"
    """,
    "timeout": """
        while [[ $1 == --* ]]; do shift; done; shift
        exec "$@"
    """,
    "trtexec": """
        echo "trtexec $*" >> "$CALLS"
        echo "&&&& RUNNING TensorRT.trtexec [TensorRT v${TRT_VERSION:-8602}]"
        for arg; do [[ $arg == --saveEngine=* ]] && echo engine > "${arg#--saveEngine=}"; done
        exit 0
    """,
    "eais-tools": """
        echo "eais-tools $*" >> "$CALLS"
        case "$1 $2" in
            "aip build") python3 "$STUB_DIR/eais_build.py" ;;
            "aip export") tar -czf "${AIP_ID}_dist.tgz" dist ;;
            "pipeline test") exit "${PIPELINE_TEST_STATUS:-0}" ;;
            "pipeline clean") echo "[]" > "$FIXTURES/pipelines.json" ;;
        esac
    """,
    "make": 'echo "make $*" >> "$CALLS"; exit "${MAKE_STATUS:-0}"',
    "nvcc": 'echo "Cuda compilation tools, release 12.2, V12.2.140"',
    "npm": """
        echo "npm $*" >> "$CALLS"
        for package in vue hls.js; do
            mkdir -p "node_modules/$package" && echo "{}" > "node_modules/$package/package.json"
        done
    """,
}

EAIS_BUILD = f"""
import os, shutil, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / "tools")!r})
from check_package import EAIS_FILES, include_entries
shutil.rmtree("dist", ignore_errors=True)
for name in EAIS_FILES | include_entries("."):
    Path("dist", name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(name, Path("dist", name))
extra = os.environ.get("EXTRA_DIST_FILE")
if extra:
    Path("dist", extra).parent.mkdir(parents=True, exist_ok=True)
    Path("dist", extra).write_text("must not ship")
"""


class Station:
    """Temporary AIP folder plus stub commands for running notebook bash cells."""

    def __init__(self, directory, fixtures=None):
        self.root = Path(directory) / "aip"
        self.root.mkdir()
        self.calls = Path(directory) / "calls.log"
        self.calls.touch()
        self.bin = Path(directory) / "bin"
        self.bin.mkdir()
        self.fixtures = Path(directory) / "fixtures"
        self.fixtures.mkdir()
        for name, body in STUBS.items():
            stub = self.bin / name
            stub.write_text("#!/bin/bash\n" + textwrap.dedent(body))
            stub.chmod(0o755)
        (self.bin / "eais_build.py").write_text(EAIS_BUILD)
        for name, data in (fixtures or {}).items():
            (self.fixtures / f"{name}.json").write_text(json.dumps(data))
        shutil.copytree(ROOT / "tools", self.root / "tools")

    def run(self, cell, **env):
        source = BASH[cell].replace("/usr/local/cuda/bin/nvcc", "nvcc").replace(
            "/data/projects/", f"{self.root.parent}/projects/")
        environment = {
            **os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "CALLS": str(self.calls),
            "FIXTURES": str(self.fixtures), "STUB_DIR": str(self.bin), "AIP_ID": "AIPV1EMI0001",
            "MODEL_DIR": "models/AIPV1EMI0001_Yolo_Detector", "TRTEXEC": "trtexec",
            "ONNX": "models/AIPV1EMI0001_Yolo_Detector/yolov7tiny-640.onnx",
            "ENGINE": "models/AIPV1EMI0001_Yolo_Detector/1/yolov7-fp16-1x8x8.engine",
            "HF_BASE": "https://hf.invalid", "MS3": "http://ms3", "BENCH": "tmp/bench",
            "ENGINE_SOURCE": "build", "RUN_BENCHMARKS": "0", "WORKSPACE_MIB": "1024",
            "PIPELINE_ID": "owned-pipeline", "COPYFILE_DISABLE": "1", **env,
        }
        return subprocess.run(["bash", "-c", source], cwd=self.root, env=environment,
                              capture_output=True, text=True)

    def log(self):
        return self.calls.read_text().splitlines()

    def add_package_sources(self):
        shutil.copy2(ROOT / ".include", self.root / ".include")
        for name in check_package.EAIS_FILES | check_package.include_entries(ROOT):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture: {name}\n")


class NotebookStructureTests(unittest.TestCase):
    def test_cell_ids_are_unique_and_code_is_valid(self):
        ids = [cell["id"] for cell in json.loads((ROOT / "localdev.ipynb").read_text())["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for name, source in CODE.items():
            with self.subTest(cell=name):
                if name in BASH:
                    self.assertTrue(BASH[name].startswith("set -euo pipefail\n"))
                    result = subprocess.run(["bash", "-n"], input=BASH[name], text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    compile(source, name, "exec")

    def test_cells_stay_short_and_use_station_commands(self):
        for name, source in CODE.items():
            with self.subTest(cell=name):
                self.assertLessEqual(len(source.splitlines()), 30, "move long logic into tools/")
        source = "\n".join(CODE.values())
        for command in ('eais-tools aip check --uuid "$AIP_ID"',
                        'eais-tools aip build --uuid "$AIP_ID" --include-file .include',
                        'eais-tools aip export --uuid "$AIP_ID"', "eais-tools aip scan",
                        "eais-tools pipeline test", "eais-tools pipeline clean",
                        "python3 tools/check_package.py", "python3 tools/benchmark.py run",
                        "validate_specified_triton_config", "validate_ds_config"):
            self.assertIn(command, source)
        for removed in ("config_diagnostic", "gst-launch", "tritonserver", "launch_triton_server",
                        "sudo", "pip install", "docker ", "systemctl"):
            self.assertNotIn(removed, source)

    def test_markdown_explains_the_pipeline_with_diagrams(self):
        self.assertGreaterEqual(MARKDOWN.count("```mermaid"), 3)
        for topic in ("nvinferserver", "NvDCF", "nvtracker", "callback.py", ".include"):
            self.assertIn(topic, MARKDOWN)

    def test_artifact_names_match_build_script_and_package(self):
        setup = CODE["setup"]
        onnx = re.search(r'ONNX=f"\{MODEL_DIR\}/([^"]+)"', setup)[1]
        engine = re.search(r'ENGINE=f"\{MODEL_DIR\}/([^"]+)"', setup)[1]
        self.assertIn(f'OnnxModel="{onnx}"', (ROOT / "bin/build.sh").read_text())
        self.assertIn(f"models/AIPV1EMI0001_Yolo_Detector/{engine}", check_package.EAIS_FILES)
        self.assertNotIn("parking-rev3", "\n".join(CODE.values()))


class NotebookBehaviourTests(unittest.TestCase):
    def test_prebuilt_engine_requires_matching_tensorrt(self):
        with tempfile.TemporaryDirectory() as directory:
            station = Station(directory)
            result = station.run("download", ENGINE_SOURCE="prebuilt", TRT_VERSION="100300")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("needs TensorRT 8.6.2", result.stdout)
            self.assertFalse([call for call in station.log() if call.startswith("curl")])

    def test_downloads_only_what_the_path_needs(self):
        for source, benchmarks, first in (("build", "0", "yolov7tiny-640.onnx"),
                                          ("prebuilt", "0", "yolov7-fp16-1x8x8-jp6-l4t36.3-arm64.engine"),
                                          ("prebuilt", "1", "yolov7tiny-640.onnx")):
            with self.subTest(source=source, benchmarks=benchmarks), tempfile.TemporaryDirectory() as d:
                station = Station(d)
                station.run("download", ENGINE_SOURCE=source, RUN_BENCHMARKS=benchmarks)
                downloads = [call for call in station.log() if call.startswith("curl")]
                self.assertEqual(len(downloads), 1)  # the stub fails, so the cell stops there
                self.assertTrue(downloads[0].endswith(f"https://hf.invalid/{first}"), downloads)

    def test_parser_failure_does_not_copy_a_stale_library(self):
        with tempfile.TemporaryDirectory() as directory:
            station = Station(directory)
            parser = station.root / "src/nvdsinfer_customparser"
            parser.mkdir(parents=True)
            (parser / "libnvds_infercustomparser.so").write_text("stale")
            (station.root / "models/AIPV1EMI0001_Yolo_Detector").mkdir(parents=True)
            result = station.run("parser", MAKE_STATUS="2")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((station.root / "models/AIPV1EMI0001_Yolo_Detector/libnvds_infercustomparser.so").exists())
            self.assertIn("make -C src/nvdsinfer_customparser CUDA_VER=12.2", station.log())

    def test_engine_build_and_smoke_test(self):
        with tempfile.TemporaryDirectory() as directory:
            station = Station(directory)
            self.assertEqual(station.run("engine").returncode, 0)
            (build,) = station.log()
            for flag in ("--fp16", "--workspace=1024", "--optShapes=images:8x3x640x640",
                         "--maxShapes=images:8x3x640x640", "--saveEngine=models/AIPV1EMI0001_Yolo_Detector/1/"):
                self.assertIn(flag, build)
            (station.bin / "trtexec").write_text("#!/bin/bash\necho '&&&& FAILED'\n")
            self.assertNotEqual(station.run("smoke-test").returncode, 0)

    def test_benchmark_is_optional_and_engines_differ_only_by_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            station = Station(directory)
            for cell in ("bench-graph", "bench-build", "bench-run"):
                result = station.run(cell)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Skipped", result.stdout)
            self.assertEqual(station.log(), [])
            (station.root / "tmp/bench").mkdir(parents=True)
            result = station.run("bench-build", RUN_BENCHMARKS="1")
            self.assertEqual(result.returncode, 0, result.stderr)
            fp32, fp16 = station.log()
            self.assertIn("--onnx=tmp/bench/raw.onnx", fp32)
            self.assertIn("--noTF32", fp32)
            normalize = lambda call: call.replace(" --fp16", "").replace("raw-fp16", "raw-fp32")
            self.assertNotIn("--fp16", fp32)
            self.assertEqual(normalize(fp16), fp32)

    def test_package_build_and_export_verify_exact_contents(self):
        for extra, ok in ((None, True), ("models/AIPV1EMI0001_Yolo_Detector/source.onnx", False),
                          ("logic/flows_cred.json", False), ("dashboard/node_modules/vue/index.js", False)):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                station = Station(directory)
                station.add_package_sources()
                env = {"EXTRA_DIST_FILE": extra} if extra else {}
                build = station.run("package-build", **env)
                self.assertEqual(build.returncode == 0, ok, build.stdout + build.stderr)
                if ok:
                    (station.root / "AIPV1EMI0001_dist.tgz").write_text("stale archive")
                    export = station.run("package-export")
                    self.assertEqual(export.returncode, 0, export.stdout + export.stderr)
                    self.assertIn("archive AIPV1EMI0001_dist.tgz matches", export.stdout)

    def test_pipeline_test_refuses_existing_test_resources(self):
        for endpoint in ("pipelines", "medias"):
            with self.subTest(endpoint=endpoint), tempfile.TemporaryDirectory() as directory:
                other = [{"pk": "other", "name": "eais_other"}]
                station = Station(directory, {"pipelines": [], "medias": [], endpoint: other})
                result = station.run("pipeline-test")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("another test owns them", result.stdout)
                self.assertFalse([call for call in station.log() if call.startswith("eais-tools")])

    def test_pipeline_test_accepts_only_success_or_timeout(self):
        for status, ok in (("0", True), ("124", True), ("1", False), ("137", False)):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                station = Station(directory, {"pipelines": [], "medias": []})
                result = station.run("pipeline-test", PIPELINE_TEST_STATUS=status)
                self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
                self.assertIn("--retain_resources", station.log()[-1])

    def test_cleanup_only_removes_this_runs_resources_and_keeps_the_package(self):
        owned = {"pk": "owned-pipeline", "name": "eais_pipeline"}
        cases = (([owned], [{"pk": "m", "name": "eais_media_1"}], False),
                 ([owned, {"pk": "other", "name": "eais_other"}], [], True),
                 ([owned], [{"pk": "1", "name": "eais_a"}, {"pk": "2", "name": "eais_b"}], True))
        for pipelines, medias, refused in cases:
            with self.subTest(pipelines=pipelines, medias=medias), tempfile.TemporaryDirectory() as d:
                station = Station(d, {"pipelines": pipelines, "medias": medias})
                (station.root / "AIPV1EMI0001_dist.tgz").write_text("package")
                result = station.run("cleanup")
                cleaned = "eais-tools pipeline clean" in station.log()
                self.assertEqual(result.returncode == 0, not refused, result.stdout + result.stderr)
                self.assertEqual(cleaned, not refused)
                self.assertTrue((station.root / "AIPV1EMI0001_dist.tgz").is_file())
        self.assertNotIn("rm -rf", BASH["cleanup"])
        self.assertNotIn("dist", BASH["cleanup"].replace("_dist.tgz", ""))

    def test_callback_acceptance(self):
        counts = lambda value: {
            "obj_counter": {name: {"max_val": value} for name in ("car", "person", "truck")},
            "analytics_obj_cnt": {"objInROIcnt": {}, "objLCCurrCnt": {}},
        }
        record = lambda value: {"since": 1, "until": 2, "data": {"0": counts(value)}}
        cases = {"": False, "[]": False, json.dumps([record(0)]): False,
                 json.dumps([record(0)]) + "\n" + json.dumps([record(2)]): True}
        manifest = json.loads((ROOT / "manifest.json").read_text())
        for dump, ok in cases.items():
            with self.subTest(dump=dump[:40]), tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "tmp").mkdir()
                (Path(directory) / "tmp/callback_dump.json").write_text(dump)
                requests = Mock()
                requests.get.return_value.json.return_value = [{"pk": "P1", "name": "eais_pipeline"},
                                                               {"pk": "P0", "name": "demo"}]
                namespace = {"Path": Path, "os": os, "manifest": manifest}
                cwd = os.getcwd()
                os.chdir(directory)
                try:
                    with patch.dict("sys.modules", {"requests": requests}), \
                            patch.dict(os.environ, {"MS3": "http://ms3"}):
                        if ok:
                            exec(CODE["callback-check"], namespace)
                            self.assertEqual(namespace["PIPELINE_ID"], "P1")
                        else:
                            with self.assertRaises(AssertionError):
                                exec(CODE["callback-check"], namespace)
                finally:
                    os.chdir(cwd)

    def test_nodered_project_is_wired_without_credentials_or_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            station = Station(directory)
            ignore = shutil.ignore_patterns("node_modules", "flows_cred.json")
            for folder in ("logic", "dashboard"):
                shutil.copytree(ROOT / folder, station.root / folder, ignore=ignore)
            (station.root / "logic/flows_cred.json").write_text('{"secret": "x"}')
            (station.root / "dashboard/node_modules/local").mkdir(parents=True)
            result = station.run("nodered", PIPELINE_ID="P1")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            project = Path(directory) / "projects/nodered-aipv1emi0001-dashboard"
            flows = json.loads((project / "flows.json").read_text())
            (tab,) = [n for n in flows if n.get("type") == "tab" and n.get("label") == "AIPV1EMI0001 Logic"]
            self.assertEqual(tab["env"], [{"name": "AIP_PIPELINE_ID", "value": "P1", "type": "str"}])
            self.assertFalse((project / "flows_cred.json").exists())
            self.assertFalse((project / "uibuilder/node_modules/local").exists())
            tracked = subprocess.run(["git", "-C", str(project), "ls-files"], capture_output=True,
                                     text=True, check=True).stdout.split()
            self.assertIn("uibuilder/aipv1emi0001_dash/src/index.js", tracked)
            self.assertFalse([name for name in tracked if "node_modules" in name or "cred" in name])
            self.assertNotEqual(station.run("nodered", PIPELINE_ID="P1").returncode, 0, "must not overwrite")


class PresentationTests(unittest.TestCase):
    def test_code_compiles_and_live_demo_is_opt_in(self):
        for name, (kind, source) in PRESENTATION.items():
            if kind == "code":
                with self.subTest(cell=name):
                    compile(source, name, "exec")
        self.assertIn("LIVE = False", PRESENTATION["live"][1])

    def test_uses_published_results_and_diagrams(self):
        text = "\n".join(source for _, source in PRESENTATION.values())
        self.assertIn("doc/benchmarks/orin-nx-16gb-jp6.0.json", text)
        self.assertGreaterEqual(text.count("```mermaid"), 4)

    def test_public_content_has_no_station_identity(self):
        for name in ("presentation.ipynb", "localdev.ipynb"):
            text = (ROOT / name).read_text()
            for forbidden in ("192.168.", "DEX5W", "nvidia@", "agent-workspace", "antoine"):
                self.assertNotIn(forbidden, text, name)


class BuildScriptTests(unittest.TestCase):
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
                env={**os.environ, "SKIP_ENGINE": "1", "PATH": f"{root / 'mock'}:{os.environ['PATH']}"},
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("success", result.stdout)


if __name__ == "__main__":
    unittest.main()
