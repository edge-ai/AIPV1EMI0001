import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


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
