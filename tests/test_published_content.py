import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import compare_models  # noqa: E402
import flavors  # noqa: E402
from public_reference import assert_no_private_references  # noqa: E402

RESULTS = ROOT / "doc/benchmarks/orin-nx-16gb-jp6.0"
NOTEBOOKS = ROOT / "notebooks"
BUILD = {"yolov7": "01-build-yolov7.ipynb", "yolov10": "02-build-yolov10.ipynb", "yolo26": "03-build-yolo26.ipynb"}


def notebook(name):
    return json.loads((NOTEBOOKS / name).read_text())


def source(cell):
    return "".join(cell["source"])


def markdown(name):
    return "\n".join(source(cell) for cell in notebook(name)["cells"] if cell["cell_type"] == "markdown")


class PublishedContentTests(unittest.TestCase):
    """The README and the notebooks quote the committed station measurements, and nothing else."""

    @classmethod
    def setUpClass(cls):
        cls.entries = compare_models.collect(RESULTS, ROOT)
        cls.by_id = {entry["flavor"]["id"]: entry for entry in cls.entries}

    def test_every_flavor_has_recorded_measurements(self):
        self.assertEqual({entry["flavor"]["id"] for entry in self.entries}, set(flavors.flavor_ids(ROOT)))

    def test_the_readme_shows_the_recorded_summary_table(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn(compare_models.summary_table(self.entries), readme)
        for marker in ("TBD", "DRAFT", "TODO"):
            self.assertNotIn(marker, readme)

    def test_the_presentation_reads_only_and_quotes_the_recorded_table(self):
        presentation = notebook("00-presentation.ipynb")
        self.assertFalse([cell for cell in presentation["cells"] if cell["cell_type"] == "code"])
        self.assertIn(compare_models.summary_table(self.entries), markdown("00-presentation.ipynb"))

    def test_no_notebook_keeps_placeholder_text(self):
        for name in (*BUILD.values(), "00-presentation.ipynb", "04-compare-models.ipynb"):
            with self.subTest(notebook=name):
                text = markdown(name)
                for marker in ("DRAFT", "TBD", "TODO"):
                    self.assertNotIn(marker, text)

    def test_each_build_notebook_quotes_the_numbers_of_its_own_flavor(self):
        for flavor_id, name in BUILD.items():
            entry = self.by_id[flavor_id]
            metrics = compare_models.accuracy_row(entry)["metrics"]
            text = markdown(name)
            with self.subTest(flavor=flavor_id):
                self.assertIn(f"**{100 * metrics['map_50_95']:.2f}** mAP 50:95", text)
                self.assertIn(f"**{100 * metrics['aip_map_50_95']:.2f}** over the three classes", text)
                batch1 = compare_models.row(entry["latency"], compare_models.TENSORRT_FP16, 1)["ms_per_image"]
                batch8 = compare_models.row(entry["latency"], compare_models.TENSORRT_FP16, 8)["ms_per_image"]
                self.assertIn(f"{batch1:.1f} ms per image at batch 1 and {batch8:.1f} ms at batch 8", text)
                for other_id, other in self.by_id.items():
                    if other_id != flavor_id:
                        other_metrics = compare_models.accuracy_row(other)["metrics"]
                        self.assertNotIn(f"**{100 * other_metrics['map_50_95']:.2f}** mAP 50:95", text)

    def test_the_metric_claim_holds_for_the_flavors_with_a_published_figure(self):
        # The README and notebooks claim agreement within 0.2 mAP points.
        checked = 0
        for entry in self.entries:
            published = entry["flavor"]["reference"]["coco_map50_95"]
            if published is None:
                continue
            checked += 1
            measured = 100 * compare_models.accuracy_row(entry)["metrics"]["map_50_95"]
            self.assertLess(abs(measured - published), 0.2, entry["flavor"]["id"])
        self.assertGreaterEqual(checked, 2)

    def test_the_recommended_starting_flavor_is_the_most_accurate_and_the_fastest_recorded_one(self):
        default = "yolo26"
        accuracy = {i: compare_models.accuracy_row(e)["metrics"]["map_50_95"] for i, e in self.by_id.items()}
        latency = {i: compare_models.row(e["latency"], compare_models.TENSORRT_FP16, 1)["ms_per_image"]
                   for i, e in self.by_id.items()}
        self.assertEqual(max(accuracy, key=accuracy.get), default)
        self.assertEqual(min(latency, key=latency.get), default)

    def test_recorded_files_are_public_material(self):
        for path in sorted(RESULTS.rglob("*")):
            if path.suffix in {".json", ".md"}:
                text = path.read_text()
                assert_no_private_references(self, text, path.name)
                self.assertNotIn("/tmp/", text, path.name)

    def test_the_chart_and_the_conditions_are_recorded(self):
        for name in ("comparison.png", "comparison.md", "conditions.md"):
            self.assertTrue((RESULTS / name).is_file(), name)
        self.assertEqual(compare_models.render_markdown(self.entries, compare_models.conditions(RESULTS)),
                         (RESULTS / "comparison.md").read_text())


class ComparisonNotebookTests(unittest.TestCase):
    """The read-only cells of the comparison notebook run anywhere and show the recorded tables."""

    def run_cells(self, cell_ids):
        shown = []
        stub = types.ModuleType("IPython.display")
        stub.Markdown = lambda text: ("markdown", text)
        stub.Image = lambda **kwargs: ("image", kwargs["filename"])
        stub.display = shown.append
        package = types.ModuleType("IPython")
        package.display = stub
        cells = {cell["id"]: source(cell) for cell in notebook("04-compare-models.ipynb")["cells"]
                 if cell["cell_type"] == "code"}
        previous_directory, previous_path = os.getcwd(), list(sys.path)
        os.chdir(NOTEBOOKS)
        try:
            with patch.dict(sys.modules, {"IPython": package, "IPython.display": stub}):
                namespace = {}
                for cell_id in cell_ids:
                    exec(compile(cells[cell_id], cell_id, "exec"), namespace)
        finally:
            os.chdir(previous_directory)
            sys.path[:] = previous_path
        return shown, namespace

    def test_the_tables_are_the_ones_rendered_from_the_recorded_results(self):
        entries = compare_models.collect(RESULTS, ROOT)
        shown, namespace = self.run_cells(["load"])
        self.assertEqual(shown, [])
        self.assertEqual([entry["flavor"]["id"] for entry in namespace["entries"]], [e["flavor"]["id"] for e in entries])

    def test_each_table_cell_displays_the_matching_table(self):
        entries = compare_models.collect(RESULTS, ROOT)
        notebook_cells = notebook("04-compare-models.ipynb")["cells"]
        expected = {"accuracy_table": compare_models.accuracy_table(entries),
                    "latency_table": compare_models.latency_table(entries),
                    "parity_table": compare_models.parity_table(entries)}
        for table, text in expected.items():
            cell = next(cell for cell in notebook_cells if cell["cell_type"] == "code" and f"compare_models.{table}" in source(cell))
            with self.subTest(table=table):
                shown, _ = self.run_cells(["load", cell["id"]])
                self.assertIn(("markdown", text), shown)

    def test_the_chart_cell_points_to_the_recorded_image(self):
        notebook_cells = notebook("04-compare-models.ipynb")["cells"]
        cell = next(cell for cell in notebook_cells if cell["cell_type"] == "code" and "comparison.png" in source(cell))
        shown, _ = self.run_cells(["load", cell["id"]])
        self.assertEqual([kind for kind, _ in shown], ["image"])
        self.assertTrue(Path(shown[0][1]).is_file())

    def test_the_measure_cells_do_nothing_unless_asked(self):
        for cell in notebook("04-compare-models.ipynb")["cells"]:
            text = source(cell)
            if text.startswith("%%bash"):
                with self.subTest(cell=cell["id"]):
                    self.assertIn('[ "$MEASURE" = 1 ]', text)
        self.assertIn("MEASURE = False", "\n".join(source(c) for c in notebook("04-compare-models.ipynb")["cells"]))

    def test_coco_download_is_skipped_when_the_data_is_already_present(self):
        coco_cell = next(source(cell) for cell in notebook("04-compare-models.ipynb")["cells"]
                         if cell.get("id") == "coco")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coco = root / "tmp/coco"
            (coco / "val2017").mkdir(parents=True)
            (coco / "annotations").mkdir()
            (coco / "annotations/instances_val2017.json").write_text("{}")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            marker = root / "curl-called"
            curl = bin_dir / "curl"
            curl.write_text(f"#!/bin/sh\ntouch {str(marker)!r}\n")
            curl.chmod(0o755)
            python3 = bin_dir / "python3"
            python3.write_text("#!/bin/sh\nexit 0\n")
            python3.chmod(0o755)
            result = subprocess.run(
                ["bash", "-c", coco_cell.removeprefix("%%bash\n")],
                cwd=root,
                env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "MEASURE": "1"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(marker.exists(), "curl should not run when both COCO directories are present")


if __name__ == "__main__":
    unittest.main()
