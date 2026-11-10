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
AIP_VERSION = json.loads((ROOT / "manifest.json").read_text())["version"]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import check_package  # noqa: E402
from public_reference import assert_no_private_references  # noqa: E402


def load(name):
    cells = json.loads((ROOT / name).read_text())["cells"]
    return {cell["id"]: (cell["cell_type"], "".join(cell["source"])) for cell in cells}


NOTEBOOK = load("localdev.ipynb")
PRESENTATION = load("presentation.ipynb")
ACCURACY = load("accuracy.ipynb")
CODE = {name: source for name, (kind, source) in NOTEBOOK.items() if kind == "code"}
BASH = {name: source.split("\n", 1)[1] for name, source in CODE.items() if source.startswith("%%bash\n")}
MARKDOWN = "\n".join(source for kind, source in NOTEBOOK.values() if kind == "markdown")
ACCURACY_CODE = {name: source for name, (kind, source) in ACCURACY.items() if kind == "code"}
ACCURACY_BASH = {name: source.split("\n", 1)[1] for name, source in ACCURACY_CODE.items()
                 if source.startswith("%%bash\n")}

# Station commands replaced by stubs. Each call is appended to $CALLS.
STUBS = {
    "curl": """
        echo "curl $*" >> "$CALLS"
        url=${@: -1}; name=${url##*/}; name=${name%%[?]*}
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
            "aip build")
                [ "${EAIS_BUILD_STATUS:-0}" = 0 ] || exit "$EAIS_BUILD_STATUS"
                python3 "$STUB_DIR/eais_build.py" ;;
            "aip export")
                [ "${EAIS_EXPORT_STATUS:-0}" = 0 ] || exit "$EAIS_EXPORT_STATUS"
                tar -czf "${AIP_ID}_dist.tgz" dist ;;
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
            "ONNX_PREEXISTED": "0", "TIMING_CACHE_PREEXISTED": "0",
            "CLEANUP_EAIS_RESOURCES": "0",
            "HF_BASE": "https://hf.invalid", "MS3": "http://ms3", "BENCH": "tmp/bench",
            "ENGINE_SOURCE": "build", "RUN_BENCHMARKS": "0", "WORKSPACE_MIB": "1024",
            "TRT_VERSION": "8602",
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
                        'eais-tools aip export --uuid "$AIP_ID"',
                        'curl -fsS -X POST "$MS3/aiprocessor/scan/$AIP_ID?force=false"',
                        "eais-tools pipeline test", "eais-tools pipeline clean",
                        "python3 tools/benchmark.py all",
                        "validate_specified_triton_config", "validate_ds_config"):
            self.assertIn(command, source)
        self.assertNotIn("eais-tools aip scan", source)
        for removed in ("config_diagnostic", "gst-launch", "tritonserver", "launch_triton_server",
                        "sudo", "pip install", "docker ", "systemctl", "check_package"):
            self.assertNotIn(removed, source)

    def test_node_red_deployment_is_outside_the_build_tutorial(self):
        self.assertFalse({"nodered", "nodered-md", "open-dashboard"} & NOTEBOOK.keys())
        source = "\n".join(CODE.values())
        for removed in ("install-nodered-project", "npm", "/data/projects"):
            self.assertNotIn(removed, source)
        self.assertIn("[`logic/README.md`](logic/README.md)", MARKDOWN)

    def test_markdown_explains_the_pipeline_with_diagrams(self):
        self.assertGreaterEqual(MARKDOWN.count("```mermaid"), 1)
        for topic in ("nvinferserver", "NvDCF", "nvtracker", "callback.py", ".include"):
            self.assertIn(topic, MARKDOWN)

    def test_documented_node_red_version_matches_project(self):
        package = json.loads((ROOT / "logic/package.json").read_text())
        version = package["dependencies"]["@edgematrix/node-red-contrib-eais"]
        for name in ("README.md", "logic/README.md", "doc/USER_MANUAL.md"):
            with self.subTest(document=name):
                self.assertIn(version, (ROOT / name).read_text())
        self.assertNotIn("nodered-md", NOTEBOOK)

    def test_changelog_covers_manifest_version(self):
        changelog = (ROOT / "CHANGELOG.md").read_text()
        self.assertRegex(changelog, rf"(?m)^## v{re.escape(AIP_VERSION)}(?: - .+)?$")

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

    def test_prebuilt_download_does_not_inherit_the_host_tensorrt_version(self):
        with patch.dict(os.environ, {"TRT_VERSION": "8.6.2.3"}), tempfile.TemporaryDirectory() as directory:
            station = Station(directory)
            station.run("download", ENGINE_SOURCE="prebuilt")
            self.assertEqual(len([call for call in station.log() if call.startswith("curl")]), 1)

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

    def test_benchmark_is_optional_and_delegates_to_one_tool_command(self):
        with tempfile.TemporaryDirectory() as directory:
            station = Station(directory)
            result = station.run("bench-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Skipped", result.stdout)
            self.assertEqual(station.log(), [])
            self.assertFalse({"bench-graph", "bench-build"} & CODE.keys())
            tool = station.root / "tools/benchmark.py"
            tool.write_text(
                "import os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['CALLS']).write_text(' '.join(sys.argv[1:]))\n"
                "sys.exit(int(os.environ.get('BENCHMARK_STATUS', '0')))\n")
            result = station.run("bench-run", RUN_BENCHMARKS="1")
            self.assertEqual(result.returncode, 0, result.stderr)
            (command,) = station.log()
            self.assertTrue(command.startswith("all --onnx models/"))
            for argument in ("--output tmp/bench", "--deployment-engine models/",
                             "--trtexec trtexec", "--workspace-mib 1024"):
                self.assertIn(argument, command)
            result = station.run("bench-run", RUN_BENCHMARKS="1", BENCHMARK_STATUS="7")
            self.assertEqual(result.returncode, 7)

    def test_packaging_uses_native_eais_tools_without_a_custom_checker(self):
        with tempfile.TemporaryDirectory() as directory:
            station = Station(directory)
            station.add_package_sources()
            (station.root / "tools/check_package.py").unlink()
            build = station.run("package-build")
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            archive = station.root / "AIPV1EMI0001_dist.tgz"
            archive.write_text("stale archive")
            export = station.run("package-export")
            self.assertEqual(export.returncode, 0, export.stdout + export.stderr)
            self.assertEqual(station.log(), [
                "eais-tools aip build --uuid AIPV1EMI0001 --include-file .include",
                "eais-tools aip export --uuid AIPV1EMI0001"])
            check_package.check_archive(archive, check_package.check_dist(station.root))

    def test_packaging_propagates_native_eais_failures(self):
        for cell, environment in (("package-build", {"EAIS_BUILD_STATUS": "9"}),
                                  ("package-export", {"EAIS_EXPORT_STATUS": "9"})):
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory:
                station = Station(directory)
                result = station.run(cell, **environment)
                self.assertEqual(result.returncode, 9)
                self.assertFalse((station.root / "AIPV1EMI0001_dist.tgz").exists())

    def test_registration_scans_only_the_target_without_forcing(self):
        aip = {"pk": "aip-pk", "unique_id": "AIPV1EMI0001", "version": AIP_VERSION}
        with tempfile.TemporaryDirectory() as directory:
            station = Station(directory, {"AIPV1EMI0001": aip, "aiprocessors": [aip]})
            (station.root / "manifest.json").write_text(json.dumps({"version": AIP_VERSION}))
            result = station.run("register")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(station.log(), [
                "curl -fsS -X POST http://ms3/aiprocessor/scan/AIPV1EMI0001?force=false",
                "curl -fsS http://ms3/aiprocessors",
            ])

    def test_registration_stops_when_the_target_scan_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            station = Station(directory, {"aiprocessors": []})
            (station.root / "manifest.json").write_text(json.dumps({"version": AIP_VERSION}))
            result = station.run("register")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Targeted scan failed", result.stdout + result.stderr)
            self.assertEqual(station.log(), [
                "curl -fsS -X POST http://ms3/aiprocessor/scan/AIPV1EMI0001?force=false",
            ])

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
                result = station.run("cleanup", CLEANUP_EAIS_RESOURCES="1")
                cleaned = "eais-tools pipeline clean" in station.log()
                self.assertEqual(result.returncode == 0, not refused, result.stdout + result.stderr)
                self.assertEqual(cleaned, not refused)
                self.assertTrue((station.root / "AIPV1EMI0001_dist.tgz").is_file())
        self.assertNotIn("rm -rf", BASH["cleanup"])
        self.assertNotIn("dist", BASH["cleanup"].replace("_dist.tgz", ""))

    def test_cleanup_preserves_model_artifacts_that_preexisted_the_run(self):
        for existed, removed in ((True, False), (False, True)):
            with self.subTest(preexisted=existed), tempfile.TemporaryDirectory() as directory:
                station = Station(directory, {"pipelines": [{"pk": "owned-pipeline", "name": "eais_pipeline"}],
                                               "medias": []})
                (station.root / "AIPV1EMI0001_dist.tgz").write_text("package")
                onnx = station.root / "models/AIPV1EMI0001_Yolo_Detector/yolov7tiny-640.onnx"
                timing_cache = station.root / "bin/timing.cache"
                onnx.parent.mkdir(parents=True)
                timing_cache.parent.mkdir(parents=True)
                onnx.write_text("model")
                timing_cache.write_text("cache")
                result = station.run(
                    "cleanup",
                    CLEANUP_EAIS_RESOURCES="1",
                    ONNX_PREEXISTED=str(int(existed)),
                    TIMING_CACHE_PREEXISTED=str(int(existed)),
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(onnx.exists(), not removed)
                self.assertEqual(timing_cache.exists(), not removed)

    def test_cleanup_requires_explicit_opt_in_for_station_wide_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            station = Station(directory, {"pipelines": [{"pk": "owned-pipeline", "name": "eais_pipeline"}],
                                          "medias": []})
            (station.root / "AIPV1EMI0001_dist.tgz").write_text("package")
            result = station.run("cleanup")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Skipped station-wide pipeline cleanup", result.stdout)
            self.assertNotIn("eais-tools pipeline clean", station.log())

    def test_cleanup_preserves_artifacts_when_ownership_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            station = Station(directory, {"pipelines": [{"pk": "owned-pipeline", "name": "eais_pipeline"}],
                                          "medias": []})
            (station.root / "AIPV1EMI0001_dist.tgz").write_text("package")
            onnx = station.root / "models/AIPV1EMI0001_Yolo_Detector/yolov7tiny-640.onnx"
            timing_cache = station.root / "bin/timing.cache"
            onnx.parent.mkdir(parents=True)
            timing_cache.parent.mkdir(parents=True)
            onnx.write_text("model")
            timing_cache.write_text("cache")
            result = station.run("cleanup", CLEANUP_EAIS_RESOURCES="1",
                                 ONNX_PREEXISTED="", TIMING_CACHE_PREEXISTED="")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(onnx.exists())
            self.assertTrue(timing_cache.exists())

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


class InstallNodeRedProjectTests(unittest.TestCase):
    SCRIPT = "bin/install-nodered-project.sh"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.station = Station(self.directory.name)
        self.aip = self.station.root
        for folder in ("logic", "dashboard"):
            shutil.copytree(ROOT / folder, self.aip / folder,
                            ignore=shutil.ignore_patterns("node_modules", "flows_cred.json"))
        (self.aip / "bin").mkdir()
        shutil.copy2(ROOT / self.SCRIPT, self.aip / self.SCRIPT)
        self.projects = Path(self.directory.name) / "projects"
        self.project = self.projects / "nodered-aipv1emi0001-dashboard"

    def install(self, *args, cwd=None, **env):
        environment = {**os.environ, "PATH": f"{self.station.bin}:{os.environ['PATH']}",
                       "CALLS": str(self.station.calls), "NODE_RED_PROJECTS_DIR": str(self.projects),
                       "COPYFILE_DISABLE": "1", **env}
        return subprocess.run([str(self.aip / self.SCRIPT), *args], cwd=cwd or self.directory.name,
                              env=environment, capture_output=True, text=True)

    def test_installs_a_wired_project_without_credentials_or_dependencies(self):
        (self.aip / "logic/flows_cred.json").write_text('{"secret": "x"}')
        (self.aip / "dashboard/node_modules/local").mkdir(parents=True)
        result = self.install("--pipeline", "P1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Projects > Open > nodered-aipv1emi0001-dashboard", result.stdout)
        flows = json.loads((self.project / "flows.json").read_text())
        (callback,) = [n for n in flows if n.get("id") == "aipv1emi0001callback"]
        self.assertEqual(callback["pipelineId"], "P1")
        package = json.loads((self.project / "package.json").read_text())
        self.assertEqual(package["dependencies"]["@edgematrix/node-red-contrib-eais"], "0.7.10")
        self.assertNotIn("@edge-ai/node-red-contrib-eais", package["dependencies"])
        self.assertFalse((self.project / "node_modules").exists())
        self.assertFalse((self.project / "flows_cred.json").exists())
        self.assertFalse((self.project / "uibuilder/node_modules/local").exists())
        self.assertFalse((self.project / "uibuilder/README.md").exists())
        self.assertIn("npm ci --omit=dev --ignore-scripts --no-audit --no-fund", self.station.log())
        tracked = subprocess.run(["git", "-C", str(self.project), "ls-files"], capture_output=True,
                                 text=True, check=True).stdout.split()
        self.assertIn("uibuilder/aipv1emi0001_dash/src/index.js", tracked)
        self.assertIn("uibuilder/aipv1emi0001_dash/src/bootstrap.js", tracked)
        self.assertFalse([name for name in tracked if "node_modules" in name or "cred" in name])
        self.assertEqual(sorted(path.name for path in self.projects.iterdir()), [self.project.name])

    def test_refuses_to_overwrite_unless_forced(self):
        self.assertEqual(self.install("--pipeline", "P1").returncode, 0)
        result = self.install("--pipeline", "P2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pass --force", result.stderr)
        result = self.install("--pipeline", "P2", "--force")
        self.assertEqual(result.returncode, 0, result.stderr)
        flows = json.loads((self.project / "flows.json").read_text())
        self.assertEqual([n["pipelineId"] for n in flows if n.get("id") == "aipv1emi0001callback"], ["P2"])

    def test_validates_arguments_before_writing(self):
        for args, message in ((("--project", "x"), "--pipeline is required"),
                              (("--pipeline", "P1", "--project", "../escape"), "invalid project name"),
                              (("--pipeline",), "--pipeline needs a value"),
                              (("--bogus",), "unknown option")):
            with self.subTest(args=args):
                result = self.install(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
        self.assertFalse(self.projects.exists())
        self.assertEqual(self.install("--help").returncode, 0)

    def test_failed_install_leaves_no_partial_project(self):
        (self.station.bin / "npm").write_text("#!/bin/bash\nexit 9\n")
        result = self.install("--pipeline", "P1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(self.projects.iterdir()), [])

    def test_failed_replacement_preserves_the_existing_project(self):
        self.assertEqual(self.install("--pipeline", "P1").returncode, 0)
        original = (self.project / "flows.json").read_bytes()
        (self.station.bin / "npm").write_text("#!/bin/bash\nexit 9\n")
        result = self.install("--pipeline", "P2", "--force")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.project / "flows.json").read_bytes(), original)
        self.assertEqual(list(self.projects.iterdir()), [self.project])

    def test_runs_from_any_directory_and_custom_project_name(self):
        result = self.install("--pipeline", "P1", "--project", "my-dashboard", cwd="/")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.projects / "my-dashboard/flows.json").is_file())

    def test_script_ships_in_the_package(self):
        self.assertIn(self.SCRIPT, check_package.include_entries(ROOT))
        self.assertTrue(os.access(ROOT / self.SCRIPT, os.X_OK))


class AccuracyNotebookTests(unittest.TestCase):
    def test_cells_are_short_valid_and_do_not_manage_services(self):
        cells = json.loads((ROOT / "accuracy.ipynb").read_text())["cells"]
        ids = [cell["id"] for cell in cells]
        self.assertEqual(len(ids), len(set(ids)))
        for name, source in ACCURACY_CODE.items():
            with self.subTest(cell=name):
                self.assertLessEqual(len(source.splitlines()), 30)
                if name in ACCURACY_BASH:
                    self.assertTrue(ACCURACY_BASH[name].startswith("set -euo pipefail\n"))
                    result = subprocess.run(["bash", "-n"], input=ACCURACY_BASH[name],
                                            text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    compile(source, name, "exec")
        code = "\n".join(ACCURACY_CODE.values())
        for forbidden in ("pipeline stop", "/stop", "systemctl", "docker ", "sudo "):
            self.assertNotIn(forbidden, code)

    def test_labelled_workflow_is_command_first_and_explains_limits(self):
        code = "\n".join(ACCURACY_CODE.values())
        text = " ".join(" ".join(source.split()) for _, source in ACCURACY.values())
        for command in ("python3 tools/benchmark.py raw-onnx", "python3 tools/accuracy.py run",
                        "python3 tools/accuracy.py examples", "python3 tools/accuracy.py score",
                        "--fp32-engine", "--fp16-engine", "--deployment-engine",
                        "--source-onnx", "--batches 1 4 8", "--score-floor 0.35"):
            self.assertIn(command, code)
        for topic in ("COCO", "per-class", "mAP@0.50:0.95", "precision", "recall",
                      "0.001", "0.35", "not proven identical", "training dataset",
                      "does **not** run the DeepStream parser", "aggregated counts"):
            self.assertIn(topic, text)
        self.assertIn("cmp -s", ACCURACY_BASH["source-onnx"])
        self.assertIn("--noTF32", ACCURACY_BASH["build-fp32"])
        self.assertIn("--noTF32 --fp16", ACCURACY_BASH["build-fp16"])
        self.assertIn("pycocotools==2.0.10", (ROOT / "tools/requirements-accuracy.txt").read_text())

    def test_existing_dataset_does_not_download_and_pipeline_scoring_is_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "labels.json"
            annotations.write_text("{}")
            images = root / "images"
            images.mkdir()
            binary = root / "bin"
            binary.mkdir()
            curl = binary / "curl"
            curl.write_text("#!/bin/sh\nexit 83\n")
            curl.chmod(0o755)
            env = {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}", "DOWNLOAD_COCO": "0",
                   "DATA_ROOT": str(root), "ANNOTATIONS": str(annotations), "IMAGES": str(images),
                   "RUN_DIR": str(root / "run"), "EVAL_LIMIT": "0", "SCORE_PIPELINE": "0"}
            for name in ("annotations", "images", "pipeline-score"):
                with self.subTest(cell=name):
                    result = subprocess.run(["bash", "-c", ACCURACY_BASH[name]], cwd=ROOT, env=env,
                                            text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "run").exists())

    def test_public_tutorial_is_linked_and_contains_no_station_identity(self):
        for path in ("README.md", "localdev.ipynb", "presentation.ipynb"):
            self.assertIn("accuracy.ipynb", (ROOT / path).read_text(), path)
        text = (ROOT / "accuracy.ipynb").read_text()
        assert_no_private_references(self, text, "accuracy.ipynb")
        notebook = json.loads(text)
        self.assertFalse(any(cell.get("outputs") for cell in notebook["cells"]),
                         "the public tutorial must not embed dataset images or private paths")

    def test_published_accuracy_reference_matches_tutorials_and_latency_artifacts(self):
        path = ROOT / "doc/benchmarks/orin-nx-16gb-jp6.0-accuracy.json"
        report = json.loads(path.read_text())
        latency = json.loads((ROOT / "doc/benchmarks/orin-nx-16gb-jp6.0.json").read_text())
        readme = (ROOT / "README.md").read_text()
        slide = " ".join(PRESENTATION["accuracy-results"][1].split())
        self.assertEqual(report["dataset"]["image_count"], 5000)
        self.assertEqual(report["dataset"]["selection"], "all image IDs")
        self.assertNotIn("image_ids", report["dataset"])
        self.assertEqual(report["artifact_sha256"]["source_onnx"], latency["source_onnx_sha256"])
        self.assertEqual(report["artifact_sha256"]["raw_onnx"], latency["raw_onnx_sha256"])
        self.assertEqual(report["artifact_sha256"]["Deployment engine"], latency["deployment_engine_sha256"])
        for variant in ("TensorRT FP32", "TensorRT FP16"):
            self.assertEqual(report["artifact_sha256"][variant], latency["engine_sha256"][variant])
        rows = {(row["variant"], row["batch"]): row for row in report["results"]}
        self.assertEqual(len(rows), 12)
        self.assertTrue(all("predictions" not in row for row in rows.values()))
        for variant in ("ONNX Runtime FP32", "TensorRT FP32", "TensorRT FP16"):
            with self.subTest(variant=variant):
                maps = [rows[variant, batch]["metrics"]["map_50_95"] for batch in (1, 4, 8)]
                ap50 = rows[variant, 1]["metrics"]["ap_50"]
                self.assertIn(f"| {variant} | " + " | ".join(f"{value * 100:.2f}%" for value in maps)
                              + f" | {ap50 * 100:.2f}% |", readme)
                self.assertIn(f"| {variant} | {maps[0] * 100:.2f}% | {ap50 * 100:.2f}% |", slide)
        for batch in (1, 4, 8):
            self.assertIsNone(rows["Deployment engine", batch]["metrics"]["map_50_95"])
            self.assertEqual(report["fp16_parity_vs_trt_fp32"][str(batch)]["status"], "pass")
        for name, metrics in rows["Deployment engine", 1]["metrics"]["classes"].items():
            precision = metrics["precision_at_035_iou_050"] * 100
            recall = metrics["recall_at_035_iou_050"] * 100
            self.assertIn(f'| {name} | {metrics["ground_truth"]:,} | {precision:.2f}% | {recall:.2f}% |', readme)
            self.assertIn(f"{recall:.2f}% {name}", slide)
        assert_no_private_references(self, path.read_text(), "accuracy reference report")
        self.assertNotIn("/tmp/", path.read_text())

    def test_release_ci_installs_accuracy_test_dependencies(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text()
        self.assertIn("python3 -m pip install -r tests/requirements-accuracy.txt", workflow)
        self.assertIn("import cv2, numpy, onnxruntime, pycocotools._mask", workflow)
        self.assertIn("python3 -m unittest discover -s tests", workflow)


class PresentationTests(unittest.TestCase):
    def test_presentation_has_no_code(self):
        self.assertEqual([name for name, (kind, _) in PRESENTATION.items() if kind != "markdown"], [])
        self.assertIn("There is no code to run here", PRESENTATION["title"][1])

    def test_manifest_summary_matches_current_manifest(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        detector, tracker = manifest["primary_processor"], manifest["tracker_engine"]
        summary = PRESENTATION["manifest"][1]
        self.assertIn(f"`{manifest['unique_id']}` version **{AIP_VERSION}**", summary)
        self.assertIn(f"| Hardware targets | {', '.join(manifest['hardware_target'])} |", summary)
        self.assertIn(f"up to {manifest['maximum_input_streams']}", summary)
        classes = ", ".join(f"`{name}`" for name in detector["classes"])
        self.assertIn(f"{detector['model']}, classes {classes}, batch {detector['batch_size']}", summary)
        self.assertIn(f"{tracker['name']}, {tracker['tracker_width']}×{tracker['tracker_height']}", summary)

    def test_uses_published_results_and_diagrams(self):
        text = "\n".join(source for _, source in PRESENTATION.values())
        self.assertIn("doc/benchmarks/orin-nx-16gb-jp6.0.json", text)
        self.assertGreaterEqual(text.count("```mermaid"), 5)

    def test_results_table_and_chart_match_published_benchmark(self):
        report = json.loads((ROOT / "doc/benchmarks/orin-nx-16gb-jp6.0.json").read_text())
        table = PRESENTATION["results-table"][1]
        rows = {(row["variant"], row["batch"]): row for row in report["results"]}
        for variant in dict.fromkeys(row["variant"] for row in report["results"]):
            cells = " | ".join(f"{rows[variant, b]['median_ms']:.1f} ms "
                               f"({rows[variant, b]['speedup_vs_onnxruntime']:.2f}×)" for b in (1, 4, 8))
            self.assertIn(f"| {variant} | {cells} |", table)
        providers = report["onnxruntime_node_placement"]["providers"]
        self.assertIn(f"{providers['CUDAExecutionProvider']['node_count']} CUDA, "
                      f"{providers['CPUExecutionProvider']['node_count']} CPU", table)
        (image,) = re.findall(r"!\[[^\]]+\]\(([^)]+)\)", table)
        self.assertEqual(image, "doc/benchmarks/orin-nx-16gb-jp6.0.png")
        self.assertTrue((ROOT / image).is_file())

    def test_teaching_material_qualifies_portability_and_performance_claims(self):
        presentation = " ".join(" ".join(source.split()) for _, source in PRESENTATION.values())
        self.assertIn("includes `EfficientNMS_TRT`", presentation)
        self.assertIn("backend/runtime difference", presentation)
        self.assertIn("TensorRT FP32 in the reference run", MARKDOWN)
        self.assertIn("neither notebook trains a model from scratch", presentation.lower())
        self.assertIn("Neither notebook trains a model from scratch", presentation)
        self.assertIn("configuration files", presentation)
        self.assertIn("encoded HLS or WebRTC video follows a separate path", presentation)
        self.assertIn("not a model-training walkthrough or blank project",
                      " ".join(MARKDOWN.split()))
        self.assertIn("Learning goals", MARKDOWN)
        self.assertIn("station is reserved for this exercise", MARKDOWN)
        self.assertIn("cannot prove media ownership", MARKDOWN)
        report = json.loads((ROOT / "doc/benchmarks/orin-nx-16gb-jp6.0.json").read_text())
        medians = {(row["variant"], row["batch"]): row["median_ms"] for row in report["results"]}
        ratios = [medians[("TensorRT FP32", batch)] / medians[("TensorRT FP16", batch)]
                  for batch in (1, 4, 8)]
        ratio_range = f"{min(ratios):.1f}–{max(ratios):.1f}×"
        self.assertIn(f"{ratio_range} faster than TensorRT FP32", presentation)
        self.assertIn(f"{ratio_range} over TensorRT FP32 in the reference run", MARKDOWN)

    def test_teaching_material_explains_optimize_measure_validate(self):
        guide = " ".join(MARKDOWN.split())
        presentation = " ".join(" ".join(source.split()) for _, source in PRESENTATION.values())
        self.assertIn("What this notebook proves", guide)
        self.assertIn("does **not** prove detection accuracy", guide)
        self.assertIn("Every optimization is a claim", presentation)
        self.assertIn("**Raw model latency**", presentation)
        self.assertIn("**Pipeline throughput**", presentation)
        self.assertIn("| Check | Section | Proves | Does not prove |", guide)
        self.assertIn("results.json", guide)
        self.assertIn("ONNX Runtime may use CPU nodes even when CUDA is requested", guide)
        self.assertIn("this notebook never stops them automatically", guide)
        self.assertIn("| Optimization | Measured by | Validated by |", presentation)
        self.assertIn("Know what is not proven yet", presentation)
        report = json.loads((ROOT / "doc/benchmarks/orin-nx-16gb-jp6.0.json").read_text())
        medians = {(row["variant"], row["batch"]): row["median_ms"] for row in report["results"]}
        runtime_ratios = [medians[("ONNX Runtime CUDA FP32", batch)] / medians[("TensorRT FP32", batch)]
                          for batch in (1, 4, 8)]
        self.assertIn(f"{min(runtime_ratios):.1f}–{max(runtime_ratios):.1f}×", presentation)
        providers = report["onnxruntime_node_placement"]["providers"]
        cuda = providers["CUDAExecutionProvider"]["node_count"]
        cpu = providers["CPUExecutionProvider"]["node_count"]
        self.assertIn(f"{cuda} CUDA nodes and {cpu} CPU nodes", presentation)

    def test_public_content_has_no_station_identity(self):
        for name in ("README.md", "CHANGELOG.md", "presentation.ipynb", "localdev.ipynb",
                     "accuracy.ipynb", "tools/accuracy.py", "tests/test_accuracy.py",
                     "tests/test_notebook.py", "tests/test_tools.py", "tests/public_reference.py"):
            assert_no_private_references(self, (ROOT / name).read_text(), name)

    def test_generic_privacy_checks_detect_representative_identifiers(self):
        assert_no_private_references(self, "TensorRT 10.3.0.26-1+cuda12.2", "version fixture")
        examples = (
            ".".join(("10", "0", "0", "1")),
            "".join(("ABC", "1", "D", "234567")),
            "/".join(("", "home", "example", "project")),
            "-".join(("session", "workspace")),
            " ".join(("ssh", "example" + "@" + "station")),
        )
        for example in examples:
            with self.subTest(example=example), self.assertRaises(AssertionError):
                assert_no_private_references(self, example, "fixture")


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
