import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
AIP_VERSION = json.loads((ROOT / "manifest.json").read_text())["version"]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import check_package  # noqa: E402
import flavors  # noqa: E402
from public_reference import assert_no_private_references  # noqa: E402
from station import Station  # noqa: E402

NOTEBOOKS = ROOT / "notebooks"
PRESENTATION = "00-presentation.ipynb"
COMPARISON = "04-compare-models.ipynb"
BUILD = {"yolov7": "01-build-yolov7.ipynb", "yolov10": "02-build-yolov10.ipynb", "yolo26": "03-build-yolo26.ipynb"}
ALL = [PRESENTATION, *BUILD.values(), COMPARISON]
AUTHOR = "**Author:** Antoine ([@antowan](https://github.com/antowan))"
MODEL_DIR = "models/AIPV1EMI0001_Yolo_Detector"
DOCUMENTS = ("README.md", "CHANGELOG.md", "PRODUCT.md", "doc/README.md", "doc/USER_MANUAL.md", "logic/README.md")


class Notebook:
    def __init__(self, name):
        self.name = name
        self.raw = json.loads((NOTEBOOKS / name).read_text())
        self.cells = self.raw["cells"]
        self.ids = [cell["id"] for cell in self.cells]
        self.sources = {cell["id"]: "".join(cell["source"]) for cell in self.cells}
        self.code = {cell["id"]: self.sources[cell["id"]] for cell in self.cells if cell["cell_type"] == "code"}
        self.bash = {name: source.split("\n", 1)[1] for name, source in self.code.items()
                     if source.startswith("%%bash\n")}
        self.markdown = "\n".join(self.sources[cell["id"]] for cell in self.cells if cell["cell_type"] == "markdown")


def load_all():
    return {name: Notebook(name) for name in ALL if (NOTEBOOKS / name).is_file()}


NB = load_all()


def github_slug(heading):
    slug = re.sub(r"[^\w\- ]", "", heading.strip().lower().replace("`", ""))
    return slug.replace(" ", "-")


def anchors(path):
    return {github_slug(line.lstrip("#")) for line in path.read_text().splitlines() if line.startswith("#")}


class NotebookStructureTests(unittest.TestCase):
    def test_the_tutorial_is_split_into_one_notebook_per_purpose(self):
        self.assertEqual(sorted(path.name for path in NOTEBOOKS.glob("*.ipynb")), sorted(ALL))
        for old in ("localdev.ipynb", "presentation.ipynb", "accuracy.ipynb"):
            self.assertFalse((ROOT / old).exists(), f"{old} moved to notebooks/")

    def test_every_notebook_starts_with_a_title_and_the_author(self):
        for name, notebook in NB.items():
            with self.subTest(notebook=name):
                first = notebook.cells[0]
                self.assertEqual(first["cell_type"], "markdown")
                lines = [line for line in notebook.sources[first["id"]].splitlines() if line.strip()]
                self.assertTrue(lines[0].startswith("# "), lines[0])
                self.assertEqual(lines[1], AUTHOR)
                self.assertEqual(notebook.markdown.count(AUTHOR), 1)

    def test_notebooks_are_clean_and_their_code_is_valid(self):
        for name, notebook in NB.items():
            with self.subTest(notebook=name):
                self.assertEqual((notebook.raw["nbformat"], notebook.raw["nbformat_minor"]), (4, 5))
                self.assertEqual(len(notebook.ids), len(set(notebook.ids)))
                self.assertNotIn("widgets", notebook.raw["metadata"])
                for cell in notebook.cells:
                    if cell["cell_type"] == "code":
                        self.assertEqual(cell["outputs"], [], cell["id"])
                        self.assertIsNone(cell["execution_count"], cell["id"])
                for cell_id, source in notebook.code.items():
                    self.assertLessEqual(len(source.splitlines()), 30, f"{cell_id}: move long logic into tools/")
                    if cell_id in notebook.bash:
                        script = notebook.bash[cell_id]
                        self.assertTrue(script.startswith("set -euo pipefail\n"), cell_id)
                        result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
                        self.assertEqual(result.returncode, 0, f"{cell_id}: {result.stderr}")
                    else:
                        compile(source, cell_id, "exec")

    def test_notebooks_are_public_material(self):
        for name, notebook in NB.items():
            text = (NOTEBOOKS / name).read_text()
            with self.subTest(notebook=name):
                assert_no_private_references(self, text, name)
                self.assertNotIn("/resolve/main", text)
                self.assertNotRegex(text.lower(), r"tokyo\s*ai")

    def test_no_document_mentions_tokyo_ai(self):
        for name in DOCUMENTS:
            with self.subTest(document=name):
                self.assertNotRegex((ROOT / name).read_text().lower(), r"tokyo\s*ai")

    def test_links_and_referenced_files_exist(self):
        prefixes = "tools|bin|src|doc|logic|dashboard|flavors|licenses|tests|configs|resources"
        for name, notebook in NB.items():
            with self.subTest(notebook=name):
                for target in re.findall(r"\]\((?!https?:|#|mailto:)([^)\s]+)\)", notebook.markdown):
                    path, _, fragment = target.partition("#")
                    resolved = (NOTEBOOKS / path).resolve()
                    self.assertTrue(resolved.exists(), target)
                    if fragment and resolved.suffix == ".md":
                        self.assertIn(fragment, anchors(resolved), target)
                text = notebook.markdown + "\n".join(notebook.code.values())
                for reference in re.findall(rf"`((?:{prefixes})/[\w./-]+)`", text):
                    if reference.endswith("/") or "<" in reference:
                        continue
                    self.assertTrue((ROOT / reference).exists(), reference)

    def test_markdown_explains_the_pipeline_with_diagrams(self):
        # GitHub's notebook viewer shows Mermaid source as plain text.
        for name, notebook in NB.items():
            with self.subTest(notebook=name):
                self.assertNotIn("```mermaid", notebook.markdown)
        self.assertIn("](../doc/assets/aip-pipeline.svg)", NB[PRESENTATION].markdown)
        for name in BUILD.values():
            with self.subTest(notebook=name):
                self.assertIn("Weights → ONNX graph → TensorRT engine → AIP archive → Video test", NB[name].markdown)
        build = "\n".join(NB[name].markdown for name in BUILD.values())
        for topic in ("nvinferserver", "NvDCF", "callback.py", ".include", "EAIS Tools"):
            self.assertIn(topic, build)
        presentation = NB[PRESENTATION].markdown
        for topic in ("nvinferserver", "nvtracker", "callback.py", "manifest.json"):
            self.assertIn(topic, presentation)

    def test_documented_node_red_version_matches_project(self):
        package = json.loads((ROOT / "logic/package.json").read_text())
        version = package["dependencies"]["@edgematrix/node-red-contrib-eais"]
        for name in ("logic/README.md", "doc/USER_MANUAL.md"):
            with self.subTest(document=name):
                self.assertIn(version, (ROOT / name).read_text())

    def test_changelog_covers_manifest_version(self):
        changelog = (ROOT / "CHANGELOG.md").read_text()
        self.assertRegex(changelog, rf"(?m)^## v{re.escape(AIP_VERSION)}(?: - .+)?$")


class BuildNotebookStructureTests(unittest.TestCase):
    def test_the_three_notebooks_follow_the_same_steps(self):
        sequences = {}
        for flavor_id, name in BUILD.items():
            sequences[flavor_id] = [cell for cell in NB[name].ids if cell != "install-exporter"]
        self.assertEqual(sequences["yolov10"], sequences["yolov7"])
        self.assertEqual(sequences["yolo26"], sequences["yolov7"])

    def test_the_exporter_is_installed_only_for_the_ultralytics_flavors(self):
        for flavor_id, name in BUILD.items():
            flavor = flavors.load(ROOT, flavor_id)
            with self.subTest(flavor=flavor_id):
                installs = "install-exporter" in NB[name].ids
                self.assertEqual(installs, flavor["exporter"]["kind"] == "ultralytics")
                if installs:
                    self.assertTrue(NB[name].bash["install-exporter"].splitlines()[1].startswith(
                        "python3 -m pip install --user --no-deps -r tools/requirements-export.txt"))
                    self.assertIn(flavor["exporter"]["package"],
                                  (ROOT / "tools/requirements-export.txt").read_text())

    def test_each_notebook_builds_only_its_own_flavor(self):
        for flavor_id, name in BUILD.items():
            code = "\n".join(NB[name].code.values())
            with self.subTest(flavor=flavor_id):
                self.assertIn(f'FLAVOR = "{flavor_id}"', code)
                for other in set(BUILD) - {flavor_id}:
                    self.assertNotIn(other, code)
                    self.assertNotIn(flavors.load(ROOT, other)["engine"], code)

    def test_cells_use_station_commands_and_tools_from_this_repository(self):
        for flavor_id, name in BUILD.items():
            code = "\n".join(NB[name].code.values())
            with self.subTest(flavor=flavor_id):
                for command in ('eais-tools aip check --uuid "$AIP_ID"',
                                'eais-tools aip build --uuid "$AIP_ID" --include-file .include',
                                'eais-tools aip export --uuid "$AIP_ID"',
                                'curl -fsS -X POST "$MS3/aiprocessor/scan/$AIP_ID?force=false"',
                                "eais-tools pipeline test", "eais-tools pipeline clean",
                                'python3 tools/flavors.py use "$FLAVOR"', 'python3 tools/export_model.py "$FLAVOR"',
                                'bin/build.sh "$FLAVOR"', "validate_specified_triton_config", "validate_ds_config"):
                    self.assertIn(command, code)
                self.assertNotIn("eais-tools aip scan", code)
                for removed in ("config_diagnostic", "gst-launch", "tritonserver", "launch_triton_server",
                                "sudo", "docker ", "systemctl", "check_package", "install-nodered-project",
                                "npm", "/data/projects", "hf.co", "huggingface"):
                    self.assertNotIn(removed, code)
                self.assertEqual(code.count("pip install"), 1 if "install-exporter" in NB[name].ids else 0)

    def test_node_red_deployment_is_outside_the_build_tutorial(self):
        for name in BUILD.values():
            self.assertIn("(../logic/README.md)", NB[name].markdown)


def build_tree(directory):
    """A stand-in for /data/aip/AIPV1EMI0001 with the files the setup cell reads."""
    base = Path(directory).resolve() / "data/aip"
    root = base / "AIPV1EMI0001"
    (root / "notebooks").mkdir(parents=True)
    shutil.copytree(ROOT / "tools", root / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "flavors", root / "flavors")
    return root


def run_cells(notebook, cell_ids, root, cwd=None, **env):
    """Run notebook code cells in one Python process, with /data/aip redirected to a temporary tree."""
    program = "\n".join(notebook.code[cell_id] for cell_id in cell_ids).replace(
        'Path("/data/aip")', f"Path({str(root.parent)!r})")
    return subprocess.run([sys.executable, "-c", program], cwd=cwd or root / "notebooks",
                          env={**os.environ, **env}, capture_output=True, text=True)


class SetupCellTests(unittest.TestCase):
    def test_setup_exposes_the_selected_flavor_to_the_bash_cells(self):
        for flavor_id, name in BUILD.items():
            with self.subTest(flavor=flavor_id), tempfile.TemporaryDirectory() as directory:
                root = build_tree(directory)
                flavor = flavors.load(ROOT, flavor_id)
                probe = "import json\nprint(json.dumps({k: os.environ[k] for k in " \
                        "('AIP_ID', 'FLAVOR', 'MODEL_DIR', 'ENGINE', 'WORKSPACE_MIB', 'CLEANUP_EAIS_RESOURCES', 'MS3')}))"
                notebook = Notebook(name)
                notebook.code["probe"] = probe
                result = run_cells(notebook, ["setup", "probe"], root)
                self.assertEqual(result.returncode, 0, result.stderr)
                settings = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(settings, {
                    "AIP_ID": "AIPV1EMI0001", "FLAVOR": flavor_id, "MODEL_DIR": MODEL_DIR,
                    "ENGINE": f"{MODEL_DIR}/1/{flavor['engine']}", "WORKSPACE_MIB": "1024",
                    "CLEANUP_EAIS_RESOURCES": "0", "MS3": "http://ms3_videostreaming:8000"})
                self.assertIn(flavor["display_name"], result.stdout)

    def test_setup_refuses_to_run_outside_the_station_aip_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_tree(directory)
            result = run_cells(NB[BUILD["yolo26"]], ["setup"], root, cwd=root / "tools")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Open this notebook from /data/aip/AIPV1EMI0001/notebooks", result.stderr)

    @unittest.skipUnless(shutil.which("python3") and __import__("importlib").util.find_spec("onnx"), "needs onnx")
    def test_inspect_reads_inputs_and_outputs_back_from_the_exported_graph(self):
        import onnx
        from onnx import TensorProto, helper
        for flavor_id, name in BUILD.items():
            with self.subTest(flavor=flavor_id), tempfile.TemporaryDirectory() as directory:
                root = build_tree(directory)
                flavor = flavors.load(ROOT, flavor_id)
                graph = helper.make_graph(
                    [helper.make_node("Identity", ["images"], ["output0"])], "g",
                    [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 640, 640])],
                    [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", 3, 640, 640])])
                path = flavors.onnx_path(root, flavor)
                path.parent.mkdir(parents=True)
                onnx.save(helper.make_model(graph), path)
                result = run_cells(NB[name], ["setup", "inspect"], root)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("images: FLOAT ['batch', 3, 640, 640]", result.stdout)
                self.assertIn("output0", result.stdout)


class BuildNotebookBehaviourTests(unittest.TestCase):
    """The %%bash cells of all three notebooks, run against stub station commands."""

    def run_cell(self, station, flavor_id, cell, **env):
        flavor = flavors.load(ROOT, flavor_id)
        trtexec = str(station.bin / "trtexec")
        return station.run(NB[BUILD[flavor_id]].bash[cell], FLAVOR=flavor_id,
                           ENGINE=f"{MODEL_DIR}/1/{flavor['engine']}", TRTEXEC=trtexec, TRTEXEC_BIN=trtexec, **env)

    def each_flavor(self, fixtures=None):
        for flavor_id in BUILD:
            with self.subTest(flavor=flavor_id), tempfile.TemporaryDirectory() as directory:
                yield flavor_id, Station(directory, fixtures)

    def test_select_switches_the_repository_to_the_notebook_flavor(self):
        other = {"yolov7": "yolo26", "yolov10": "yolov7", "yolo26": "yolov10"}
        for flavor_id, station in self.each_flavor():
            station.add_flavor_sources(other[flavor_id])
            self.assertEqual(flavors.check(station.root), other[flavor_id])
            result = self.run_cell(station, flavor_id, "select")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(flavors.check(station.root), flavor_id)
            self.assertIn(flavor_id, result.stdout)

    def test_parser_failure_does_not_copy_a_stale_library(self):
        for flavor_id, station in self.each_flavor():
            station.add_flavor_sources()
            station.add_build_script()
            (station.root / "src/nvdsinfer_customparser/libnvds_infercustomparser.so").write_text("stale")
            result = self.run_cell(station, flavor_id, "parser", MAKE_STATUS="2")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((station.root / MODEL_DIR / "libnvds_infercustomparser.so").exists())
            result = self.run_cell(station, flavor_id, "parser")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(flavors.check(station.root), flavor_id)
            self.assertTrue((station.root / MODEL_DIR / "libnvds_infercustomparser.so").is_file())

    def test_engine_build_installs_one_engine_and_the_smoke_test_checks_it(self):
        for flavor_id, station in self.each_flavor():
            station.add_flavor_sources()
            station.add_build_script()
            flavor = station.add_onnx(flavor_id)
            stale = station.root / MODEL_DIR / "1/other-fp16-1x8x8.engine"
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_text("other flavor")
            self.assertEqual(self.run_cell(station, flavor_id, "engine").returncode, 0)
            self.assertEqual([p.name for p in (station.root / MODEL_DIR / "1").glob("*.engine")], [flavor["engine"]])
            (build,) = [call for call in station.log() if call.startswith("trtexec")]
            for flag in ("--fp16", "--workspace=1024", "--optShapes=images:8x3x640x640",
                         "--maxShapes=images:8x3x640x640", f"--onnx=../tmp/export/{flavor_id}/{flavor['onnx']}"):
                self.assertIn(flag, build)
            trtexec = station.bin / "trtexec"
            trtexec.write_text("#!/bin/bash\necho '&&&& PASSED TensorRT.trtexec'\n")
            self.assertEqual(self.run_cell(station, flavor_id, "smoke").returncode, 0)
            trtexec.write_text("#!/bin/bash\necho '&&&& FAILED'\n")
            self.assertNotEqual(self.run_cell(station, flavor_id, "smoke").returncode, 0)

    def test_packaging_uses_native_eais_tools_and_ships_one_flavor(self):
        for flavor_id, station in self.each_flavor():
            station.add_package_sources(flavor_id)
            (station.root / "tools/check_package.py").unlink()
            build = self.run_cell(station, flavor_id, "build-package")
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            archive = station.root / "AIPV1EMI0001_dist.tgz"
            archive.write_text("stale archive")
            export = self.run_cell(station, flavor_id, "export-package")
            self.assertEqual(export.returncode, 0, export.stdout + export.stderr)
            self.assertEqual(station.log(), [
                "eais-tools aip build --uuid AIPV1EMI0001 --include-file .include",
                "eais-tools aip export --uuid AIPV1EMI0001"])
            check_package.check_archive(archive, check_package.check_dist(station.root))
            engines = [p.name for p in (station.root / "dist" / MODEL_DIR / "1").glob("*.engine")]
            self.assertEqual(engines, [flavors.load(ROOT, flavor_id)["engine"]])
            self.assertFalse((station.root / "dist/flavors").exists())

    def test_packaging_propagates_native_eais_failures(self):
        for cell, environment in (("build-package", {"EAIS_BUILD_STATUS": "9"}),
                                  ("export-package", {"EAIS_EXPORT_STATUS": "9"})):
            for flavor_id, station in self.each_flavor():
                result = self.run_cell(station, flavor_id, cell, **environment)
                self.assertEqual(result.returncode, 9)
                self.assertFalse((station.root / "AIPV1EMI0001_dist.tgz").exists())

    def test_scan_registers_only_the_target_without_forcing(self):
        for flavor_id, station in self.each_flavor({"AIPV1EMI0001": {}}):
            primary = {"model": flavors.load(ROOT, flavor_id)["display_name"]}
            aip = {"pk": "aip-pk", "unique_id": "AIPV1EMI0001", "version": AIP_VERSION,
                   "primary_processor": primary}
            (station.root / "manifest.json").write_text(json.dumps(aip))
            (station.fixtures / "aiprocessors.json").write_text(json.dumps([aip]))
            result = self.run_cell(station, flavor_id, "scan")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(station.log(), [
                "curl -fsS -X POST http://ms3/aiprocessor/scan/AIPV1EMI0001?force=false",
                "curl -fsS http://ms3/aiprocessors"])

            other = {**aip, "primary_processor": {"model": "other detector"}}
            (station.fixtures / "aiprocessors.json").write_text(json.dumps([other]))
            result = self.run_cell(station, flavor_id, "scan")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Registered detector does not match", result.stderr)

    def test_scan_stops_when_the_target_scan_fails(self):
        for flavor_id, station in self.each_flavor({"aiprocessors": []}):
            (station.root / "manifest.json").write_text(json.dumps({"version": AIP_VERSION}))
            result = self.run_cell(station, flavor_id, "scan")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Targeted scan failed", result.stdout + result.stderr)
            self.assertEqual(station.log(), ["curl -fsS -X POST http://ms3/aiprocessor/scan/AIPV1EMI0001?force=false"])

    def test_pipeline_test_refuses_existing_test_resources(self):
        for endpoint in ("pipelines", "medias"):
            other = [{"pk": "other", "name": "eais_other"}]
            for flavor_id, station in self.each_flavor({"pipelines": [], "medias": [], endpoint: other}):
                result = self.run_cell(station, flavor_id, "pipeline-test")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("another test owns them", result.stdout)
                self.assertFalse([call for call in station.log() if call.startswith("eais-tools")])

    def test_pipeline_test_accepts_only_success_or_timeout(self):
        for status, ok in (("0", True), ("124", True), ("1", False), ("137", False)):
            for flavor_id, station in self.each_flavor({"pipelines": [], "medias": []}):
                result = self.run_cell(station, flavor_id, "pipeline-test", PIPELINE_TEST_STATUS=status)
                self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
                self.assertIn("--retain_resources", station.log()[-1])

    def test_cleanup_only_removes_this_runs_resources_and_keeps_the_package(self):
        owned = {"pk": "owned-pipeline", "name": "eais_pipeline"}
        cases = (([owned], [{"pk": "m", "name": "eais_media_1"}], False),
                 ([owned, {"pk": "other", "name": "eais_other"}], [], True),
                 ([owned], [{"pk": "1", "name": "eais_a"}, {"pk": "2", "name": "eais_b"}], True))
        for pipelines, medias, refused in cases:
            for flavor_id, station in self.each_flavor({"pipelines": pipelines, "medias": medias}):
                (station.root / "AIPV1EMI0001_dist.tgz").write_text("package")
                result = self.run_cell(station, flavor_id, "cleanup", CLEANUP_EAIS_RESOURCES="1")
                self.assertEqual(result.returncode == 0, not refused, result.stdout + result.stderr)
                self.assertEqual("eais-tools pipeline clean" in station.log(), not refused)
                self.assertTrue((station.root / "AIPV1EMI0001_dist.tgz").is_file())
        for notebook in (NB[name] for name in BUILD.values()):
            self.assertNotIn("rm -rf", notebook.bash["cleanup"])
            self.assertNotIn("dist", notebook.bash["cleanup"].replace("_dist.tgz", ""))

    def test_cleanup_requires_explicit_opt_in_for_station_wide_removal(self):
        fixtures = {"pipelines": [{"pk": "owned-pipeline", "name": "eais_pipeline"}], "medias": []}
        for flavor_id, station in self.each_flavor(fixtures):
            (station.root / "AIPV1EMI0001_dist.tgz").write_text("package")
            result = self.run_cell(station, flavor_id, "cleanup")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Skipped station-wide pipeline cleanup", result.stdout)
            self.assertNotIn("eais-tools pipeline clean", station.log())

    def test_callback_acceptance(self):
        counts = lambda value: {
            "available": value is not None,
            "obj_counter": {name: {"max_val": value} for name in ("car", "person", "truck")},
            "analytics_obj_cnt": {"objInROIcnt": {}, "objLCCurrCnt": {}, "objLCCumCnt": {}},
        }
        record = lambda value: {"since": 1, "until": 2, "data": {"0": counts(value)}}
        cases = {"": False, "[]": False, json.dumps([record(0)]): False, json.dumps([record(None)]): False,
                 json.dumps([record(None), record(2)]): True,
                 json.dumps([record(0)]) + "\n" + json.dumps([record(2)]): True}
        manifest = json.loads((ROOT / "manifest.json").read_text())
        for name in BUILD.values():
            for dump, ok in cases.items():
                with self.subTest(notebook=name, dump=dump[:40]), tempfile.TemporaryDirectory() as directory:
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
                                exec(NB[name].code["callback-check"], namespace)
                                self.assertEqual(namespace["PIPELINE_ID"], "P1")
                            else:
                                with self.assertRaises(AssertionError):
                                    exec(NB[name].code["callback-check"], namespace)
                    finally:
                        os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
