import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import check_package  # noqa: E402
import flavors  # noqa: E402
from station import Station  # noqa: E402

MODEL_DIR = "models/AIPV1EMI0001_Yolo_Detector"


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

    def fail_renames_from(self, source_glob):
        """Make the installer's rename(2) helper fail for sources matching a bash glob."""
        stub = self.station.bin / "python3"
        stub.write_text(textwrap.dedent(f"""\
            #!/bin/bash
            if [[ $# -eq 4 && $1 == -c && $3 == {source_glob} ]]; then
                echo "ERROR: injected rename failure" >&2
                exit 9
            fi
            exec "{sys.executable}" "$@"
        """))
        stub.chmod(0o755)

    def test_never_replaces_a_project_created_during_installation(self):
        npm = self.station.bin / "npm"
        stub = npm.read_text()
        for kind, create in (("directory", 'mkdir -p "$TARGET" && echo theirs > "$TARGET/flows.json"'),
                             ("file", 'echo theirs > "$TARGET"')):
            with self.subTest(kind=kind):
                npm.write_text(f"{stub}\nTARGET={self.project}\n{create}\n")
                result = self.install("--pipeline", "P1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("created during installation: pass --force", result.stderr)
                theirs = self.project / "flows.json" if kind == "directory" else self.project
                self.assertEqual(theirs.read_text(), "theirs\n")
                self.assertEqual([path.name for path in self.projects.iterdir()], [self.project.name])
                if kind == "directory":
                    shutil.rmtree(self.project)
                else:
                    self.project.unlink()

    def test_failed_forced_move_restores_the_previous_project(self):
        self.assertEqual(self.install("--pipeline", "P1").returncode, 0)
        original = (self.project / "flows.json").read_bytes()
        self.fail_renames_from("*/.nodered-aipv1emi0001-dashboard.??????")
        result = self.install("--pipeline", "P2", "--force")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Restored the previous project", result.stderr)
        self.assertEqual((self.project / "flows.json").read_bytes(), original)
        self.assertEqual(list(self.projects.iterdir()), [self.project])

    def test_keeps_the_previous_project_when_it_cannot_be_restored(self):
        self.assertEqual(self.install("--pipeline", "P1").returncode, 0)
        original = (self.project / "flows.json").read_bytes()
        self.fail_renames_from("*/.nodered-aipv1emi0001-dashboard.*")
        result = self.install("--pipeline", "P2", "--force")
        self.assertNotEqual(result.returncode, 0)
        (kept,) = self.projects.glob(".nodered-aipv1emi0001-dashboard.previous.*/" + self.project.name)
        self.assertIn(f"the previous project is kept in {kept}", result.stderr)
        self.assertEqual((kept / "flows.json").read_bytes(), original)
        self.assertEqual([path.name for path in self.projects.iterdir()], [kept.parent.name])

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


class BuildScriptTests(unittest.TestCase):
    """bin/build.sh against stub make, nvcc and trtexec commands."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.station = Station(self.directory.name)
        self.aip = self.station.root
        self.station.add_flavor_sources()
        self.station.add_build_script()

    def build(self, *args, **env):
        env.setdefault("TRTEXEC_BIN", str(self.station.bin / "trtexec"))
        return subprocess.run(["bash", "bin/build.sh", *args], cwd=self.aip,
                              env=self.station.environment(**env), capture_output=True, text=True)

    def add_onnx(self, flavor_id):
        return self.station.add_onnx(flavor_id)

    def test_parser_build_failure_does_not_reuse_stale_library(self):
        source = self.aip / "src/nvdsinfer_customparser/libnvds_infercustomparser.so"
        source.write_text("stale library")
        result = self.build("yolo26", SKIP_ENGINE="1", MAKE_STATUS="37")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("success", result.stdout)
        self.assertFalse((self.aip / MODEL_DIR / "libnvds_infercustomparser.so").exists())

    def test_each_flavor_builds_its_own_engine_and_replaces_the_previous_one(self):
        for flavor_id in ("yolov7", "yolov10", "yolo26"):
            with self.subTest(flavor=flavor_id):
                flavor = self.add_onnx(flavor_id)
                result = self.build(flavor_id)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                engines = sorted(p.name for p in (self.aip / MODEL_DIR / "1").glob("*.engine"))
                self.assertEqual(engines, [flavor["engine"]])
                self.assertEqual(flavors.check(self.aip), flavor_id)
                self.assertTrue((self.aip / MODEL_DIR / "libnvds_infercustomparser.so").is_file())
        build_command = next(call for call in self.station.log() if call.startswith("trtexec"))
        for flag in ("--fp16", "--workspace=1024", "--minShapes=images:1x3x640x640",
                     "--optShapes=images:8x3x640x640", "--maxShapes=images:8x3x640x640",
                     "--saveEngine=../tmp/engines/yolov7/yolov7-fp16-1x8x8.engine"):
            self.assertIn(flag, build_command)
        self.assertIn("--onnx=../tmp/export/yolov7/yolov7-tiny.onnx", build_command)

    def test_missing_onnx_points_to_the_exporter(self):
        result = self.build("yolov10")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("python3 tools/export_model.py yolov10", result.stderr)
        self.assertFalse(list((self.aip / MODEL_DIR).glob("1/*.engine")))

    def test_failed_engine_build_installs_nothing_and_names_the_log(self):
        self.add_onnx("yolo26")
        result = self.build("yolo26", TRTEXEC_STATUS="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tmp/engines/yolo26/build.log", result.stderr)
        self.assertFalse(list((self.aip / MODEL_DIR).glob("1/*.engine")))

    def test_unknown_flavor_is_rejected_before_anything_is_built(self):
        result = self.build("yolov99")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse([call for call in self.station.log() if call.startswith(("make", "trtexec"))])


class ParserHarnessTests(unittest.TestCase):
    """Compile the real parser source against stub DeepStream headers and feed it synthetic outputs."""

    @unittest.skipUnless(shutil.which("c++"), "no C++ compiler")
    def test_both_output_layouts_produce_the_same_three_aip_detections(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = Path(directory) / "harness"
            stub = ROOT / "tests/parser_stub"
            subprocess.run(["c++", "-std=c++11", f"-I{stub}", "-w", "-o", str(harness),
                            str(stub / "parser_harness.cpp")], check=True)
            lines = subprocess.run([str(harness)], check=True, capture_output=True, text=True).stdout.splitlines()
        self.assertIn("nms ok=1 count=3", lines)
        self.assertIn("e2e ok=1 count=3", lines)
        self.assertIn("nms_bad ok=0 count=0", lines)
        self.assertIn("e2e_bad ok=0 count=0", lines)
        nms = [line.split(" ", 1)[1] for line in lines if line.startswith("nms ") and "ok=" not in line]
        e2e = [line.split(" ", 1)[1] for line in lines if line.startswith("e2e ") and "ok=" not in line]
        self.assertEqual(nms, e2e)
        self.assertEqual([line.split()[1] for line in lines if line.startswith("nms ") and "ok=" not in line],
                         ["0", "1", "2"])


if __name__ == "__main__":
    unittest.main()
