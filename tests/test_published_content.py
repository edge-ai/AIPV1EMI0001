import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import capacity  # noqa: E402
import compare_models  # noqa: E402
import flavors  # noqa: E402
from public_reference import assert_no_private_references  # noqa: E402

RESULTS = ROOT / "doc/benchmarks/orin-nx-16gb-jp6.0"
EX3_RESULTS = ROOT / "doc/benchmarks/orin-nano-8gb-jp6.0"
NOTEBOOKS = ROOT / "notebooks"
BUILD = {"yolov7": "01-build-yolov7.ipynb", "yolov10": "02-build-yolov10.ipynb", "yolo26": "03-build-yolo26.ipynb"}
BASELINE_COLUMNS = [("720p", "base"), ("1080p", "base")]
WEBRTC_SCREENING = [name for name in capacity.SCREENING if not capacity.parse_profile(name).eventrec]
PATCHED_EX3_FILES = ("capacity-fixed-ms3.jsonl", "capacity-fixed-ms3-recheck.jsonl",
                     "capacity-fixed-ms3-boundary.jsonl")


def notebook(name):
    return json.loads((NOTEBOOKS / name).read_text())


def source(cell):
    return "".join(cell["source"])


def markdown(name):
    return "\n".join(source(cell) for cell in notebook(name)["cells"] if cell["cell_type"] == "markdown")


def table_rows(table):
    return [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in table.splitlines()[2:] if line.startswith("| ")]


def day(stamp):
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc).date()


def span(days):
    first, last = min(days), max(days)
    if first == last:
        return f"{first.day} {first:%B %Y}"
    assert (first.year, first.month) == (last.year, last.month), "spans across months are not worded yet"
    return f"{first.day} to {last.day} {last:%B %Y}"


class PublishedContentTests(unittest.TestCase):
    """The overview points to the lessons; the comparison quotes the committed measurements."""

    @classmethod
    def setUpClass(cls):
        cls.entries = compare_models.collect(RESULTS, ROOT)
        cls.by_id = {entry["flavor"]["id"]: entry for entry in cls.entries}

    def test_every_flavor_has_recorded_measurements(self):
        self.assertEqual({entry["flavor"]["id"] for entry in self.entries}, set(flavors.flavor_ids(ROOT)))

    def test_the_readme_links_each_lesson_and_the_camera_report(self):
        readme = (ROOT / "README.md").read_text()
        for name in (*BUILD.values(), "00-presentation.ipynb", "04-compare-models.ipynb"):
            self.assertIn(f"(notebooks/{name})", readme)
        self.assertIn(f"({(RESULTS / 'capacity.md').relative_to(ROOT)})", readme)
        for marker in ("TBD", "DRAFT", "TODO"):
            self.assertNotIn(marker, readme)

    def test_the_readme_quotes_the_recorded_results(self):
        readme = (ROOT / "README.md").read_text()
        records = capacity.load(RESULTS / "capacity.jsonl")
        models = [entry["flavor"]["id"] for entry in self.entries]
        summary = table_rows(compare_models.summary_table(self.entries))
        cameras = table_rows(capacity.camera_table(records, models, BASELINE_COLUMNS))
        for model, accuracy, counts in zip(models, summary, cameras):
            with self.subTest(model=model):
                row = next(line for line in readme.splitlines()
                           if line.startswith("| [") and f"(notebooks/{BUILD[model]})" in line)
                counts = [count.rstrip("†") for count in counts[1:]]
                # The caption compares input resolutions, not unsupported output combinations.
                self.assertEqual(len(set(counts)), 1, counts)
                self.assertEqual(table_rows(f"|\n|\n{row}")[0][1:], [*accuracy[1:4], counts[0]])
                self.assertIn(f"| [{capacity.MODEL_NAMES[model]}]", row)

    def test_the_presentation_is_read_only(self):
        presentation = notebook("00-presentation.ipynb")
        self.assertFalse([cell for cell in presentation["cells"] if cell["cell_type"] == "code"])

    def test_supported_matrix_does_not_relabel_historical_output_results(self):
        readme = (ROOT / "README.md").read_text()
        plan = (ROOT / "doc/benchmarks/capacity/README.md").read_text()
        comparison = markdown("04-compare-models.ipynb")
        table = plan.split("| Device |", 1)[1].split("\n\n", 1)[0]
        self.assertIn("| Device |" + table, comparison)
        baseline = table_rows(capacity.camera_table(capacity.load(RESULTS / "capacity.jsonl"),
                                                    columns=[("720p", "base")]))
        counts = {row[0]: row[1].rstrip("†") for row in baseline}
        patched = [row for name in PATCHED_EX3_FILES for row in capacity.load(EX3_RESULTS / name)]
        ex3_counts = {row[0]: row[1:] for row in table_rows(capacity.camera_table(
            patched, columns=[("720p", profile) for profile in ("base", *capacity.CAPACITY_PROFILES)]))}
        self.assertEqual(len(table_rows("| Device |" + table)), 2 * len(self.by_id))
        for line in table_rows("| Device |" + table):
            device, model, *cells = line
            expected = ([f"{counts[model]} (baseline)", "Not measured", "Not measured", "Not measured"]
                        if device.startswith("EX5") else ex3_counts[model])
            self.assertEqual(cells, expected)
        for text in (readme, comparison):
            headers = [line.lower() for line in text.splitlines() if line.startswith("|")]
            self.assertFalse(any("continuous recording" in line or "hls" in line or "all outputs" in line
                                 for line in headers))
        self.assertFalse((RESULTS / "capacity-historical.md").exists())

    def test_retained_benchmark_evidence_contains_only_supported_profiles(self):
        def check_profiles(value):
            if isinstance(value, dict):
                if "profile" in value:
                    capacity.validate_live_profile(capacity.parse_profile(value["profile"]))
                for item in value.values():
                    check_profiles(item)
            elif isinstance(value, list):
                for item in value:
                    check_profiles(item)

        for directory in (RESULTS, EX3_RESULTS):
            for path in sorted(directory.glob("capacity*.json*")):
                with self.subTest(file=path):
                    values = capacity.load(path) if path.suffix == ".jsonl" else json.loads(path.read_text())
                    check_profiles(values)

    def test_recording_deficit_table_uses_saved_one_camera_clips(self):
        records = [row for name in PATCHED_EX3_FILES[:2] for row in capacity.load(EX3_RESULTS / name)]
        report = (EX3_RESULTS / "capacity.md").read_text()
        all_clips = []
        for model in self.by_id:
            rows = [row for row in records if row["kind"] == "trial" and row["model"] == model
                    and row["profile"] == "evt-osd" and row["n"] == 1]
            self.assertEqual(len(rows), 2)
            clips = [clip for row in rows for clip in row["smart_video"]["clips"]]
            all_clips.extend(clips)
            inference = [row["mean_fps"] for row in rows]
            decoded = [clip["decoded_fps"] for clip in clips]
            self.assertIn(f"| {capacity.MODEL_NAMES[model]} | {min(inference):.2f}-{max(inference):.2f} | "
                          f"{min(decoded):.2f}-{max(decoded):.2f} | "
                          f"{sum(fps < capacity.PASS_FPS for fps in decoded)}/{len(clips)} |", report)
            for clip in clips:
                self.assertEqual(clip["packets"], clip["decoded_frames"])
        failed = sum(clip["decoded_fps"] < capacity.PASS_FPS for clip in all_clips)
        claim = f"{failed} of {len(all_clips)}"
        for path in (EX3_RESULTS / "capacity.md", ROOT / "doc/benchmarks/capacity/README.md"):
            self.assertIn(claim, path.read_text())
        self.assertIn(claim, markdown("04-compare-models.ipynb"))

    def test_ex3_results_and_evidence_are_published_consistently(self):
        report = (EX3_RESULTS / "capacity.md").read_text()
        matrix = (ROOT / "doc/benchmarks/capacity/README.md").read_text()
        comparison = markdown("04-compare-models.ipynb")
        evidence = (EX3_RESULTS / "capacity.jsonl").read_text()
        assert_no_private_references(self, evidence, "EX3 capacity evidence")
        records = capacity.load(EX3_RESULTS / "capacity.jsonl")
        table = capacity.camera_table(
            records, self.by_id.keys(),
            [("720p", profile) for profile in ("base", "webrtc-osd", "evt-osd", "webrtc+evt-osd")])
        self.assertIn(f"{len(records)} pilot, search,", report)
        self.assertIn("](capacity.jsonl)", report)
        self.assertIn("| YOLO26n | 5 | Not qualified at 1 | 3† | Not qualified at 1 |", table)
        self.assertIn("WEBRTC + SMART VIDEO", report.upper())
        for text in (matrix, comparison):
            self.assertIn("capacity.md#released-stack-results-24-november-2026)", text)
        self.assertEqual({(row["device"], row["protocol"]) for row in records},
                         {("ex3-orin-nano-8gb-15w", "webrtc-smart-video-v1")})
        self.assertEqual({row["model"] for row in records if row["kind"] in {"trial", "fixed"}},
                         set(self.by_id))

    def test_ex3_six_camera_report_counts_attempts_not_search_summaries(self):
        report = (EX3_RESULTS / "capacity.md").read_text()
        all_attempts = []
        for name in ("capacity-fixed-ms3.jsonl", "capacity-fixed-ms3-recheck.jsonl",
                     "capacity-fixed-ms3-boundary.jsonl"):
            attempts = []
            for row in capacity.load(EX3_RESULTS / name):
                if row["kind"] not in ("trial", "fixed") or (row["model"], row["profile"]) != ("yolo26", "base"):
                    continue
                trial = row.get("trial", row)
                if trial["n"] != 6:
                    continue
                capacity.require_valid_trial(trial)
                attempts.extend(trial.get("attempts", [trial]))
            self.assertTrue(attempts, name)
            for attempt in attempts:
                self.assertEqual(len(attempt["per_input_fps"]), 6)
                self.assertEqual(min(attempt["per_input_fps"].values()), attempt["min_fps"])
                self.assertEqual(attempt["pass"], attempt["min_fps"] >= capacity.PASS_FPS)
            results = ", ".join(f"{a['min_fps']:.2f} {'pass' if a['pass'] else 'fail'}" for a in attempts)
            self.assertIn(f"]({name}) | {results} | {sum(a['pass'] for a in attempts)}/{len(attempts)} |", report)
            all_attempts.extend(attempts)
        self.assertEqual(len({a["started"] for a in all_attempts}), len(all_attempts))
        self.assertIn(f"{sum(a['pass'] for a in all_attempts)} of {len(all_attempts)} attempts passed", report)

    def test_the_recorded_summary_is_visible_without_running_the_comparison(self):
        self.assertIn(compare_models.summary_table(self.entries), markdown("04-compare-models.ipynb"))

    def test_no_notebook_keeps_placeholder_text(self):
        for name in (*BUILD.values(), "00-presentation.ipynb", "04-compare-models.ipynb"):
            with self.subTest(notebook=name):
                text = markdown(name)
                for marker in ("DRAFT", "TBD", "TODO"):
                    self.assertNotIn(marker, text)

    def test_each_build_explains_the_outputs_its_parser_expects(self):
        for flavor_id, name in BUILD.items():
            flavor = self.by_id[flavor_id]["flavor"]
            with self.subTest(flavor=flavor_id):
                text = markdown(name)
                self.assertIn(f"`{flavor['parser_function']}`", text)
                for output in flavor["outputs"]:
                    self.assertIn(f"`{output['name']}`", text)

    def test_the_metric_claim_holds_for_the_flavors_with_a_published_figure(self):
        # The comparison notebook claims agreement within 0.2 mAP points.
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
        for path in sorted((*RESULTS.rglob("*"), *EX3_RESULTS.rglob("*"))):
            if path.suffix in {".json", ".jsonl", ".md"}:
                text = path.read_text()
                assert_no_private_references(self, text, path.name)
                self.assertNotIn("/tmp/", text, path.name)

    def test_the_chart_and_the_conditions_are_recorded(self):
        for name in ("comparison.png", "comparison.md", "conditions.md"):
            self.assertTrue((RESULTS / name).is_file(), name)
        self.assertEqual(compare_models.render_markdown(self.entries, compare_models.conditions(RESULTS)),
                         (RESULTS / "comparison.md").read_text())


class ComparisonNotebookTests(unittest.TestCase):
    """The comparison notebook shows the recorded results as text; only the optional re-measure cells run code."""

    @classmethod
    def setUpClass(cls):
        cls.entries = compare_models.collect(RESULTS, ROOT)
        cls.records = capacity.load(RESULTS / "capacity.jsonl")
        cls.models = [entry["flavor"]["id"] for entry in cls.entries]
        cls.text = markdown("04-compare-models.ipynb")

    def run_cells(self, cell_ids, directory):
        shown = []
        stub = types.ModuleType("IPython.display")
        stub.Markdown = lambda text: ("markdown", text)
        stub.Image = lambda **kwargs: ("image", kwargs["filename"])
        stub.display = shown.append
        package = types.ModuleType("IPython")
        package.display = stub
        cells = {cell["id"]: source(cell) for cell in notebook("04-compare-models.ipynb")["cells"]
                 if cell["cell_type"] == "code"}
        previous_directory = os.getcwd()
        os.chdir(directory)
        try:
            with patch.dict(sys.modules, {"IPython": package, "IPython.display": stub}), \
                    patch.dict(os.environ), redirect_stdout(io.StringIO()) as printed:
                namespace = {}
                for cell_id in cell_ids:
                    exec(compile(cells[cell_id], cell_id, "exec"), namespace)
                environment, working_directory = dict(os.environ), Path.cwd()
        finally:
            os.chdir(previous_directory)
        return shown, printed.getvalue(), environment, working_directory

    def test_every_recorded_table_is_quoted(self):
        expected = {"accuracy": compare_models.accuracy_table(self.entries),
                    "latency": compare_models.latency_table(self.entries),
                    "precision": compare_models.precision_table(self.entries),
                    "parity": compare_models.parity_table(self.entries),
                    "cameras": capacity.camera_table(self.records, self.models, BASELINE_COLUMNS),
                    "headroom": capacity.headroom_table(self.records, self.models, profiles=("base",)),
                    "screening": capacity.screening_table(self.records, profiles=WEBRTC_SCREENING)}
        for name, table in expected.items():
            with self.subTest(table=name):
                self.assertIn(table, self.text)
        self.assertIn(f"]({os.path.relpath(RESULTS / 'comparison.png', NOTEBOOKS)})", self.text)

    def test_the_measurement_dates_come_from_the_evidence(self):
        accuracy = {day(json.loads(path.read_text())["utc"]) for path in RESULTS.glob("*-accuracy.json")}
        latency = {day(json.loads(path.read_text())["utc"]) for path in RESULTS.glob("*-latency.json")}
        trials = [row["trial"] if row["kind"] == "fixed" else row for row in self.records
                  if row["kind"] in ("trial", "fixed")]
        cameras = {day(attempt["started"]) for trial in trials for attempt in (trial.get("attempts") or [trial])}
        ex3_records = capacity.load(EX3_RESULTS / "capacity.jsonl")
        ex3_trials = [row["trial"] if row["kind"] == "fixed" else row for row in ex3_records
                      if row["kind"] in ("trial", "fixed")]
        ex3_cameras = {day(attempt["started"]) for trial in ex3_trials
                       for attempt in (trial.get("attempts") or [trial]) if attempt.get("started")}
        checks = json.loads((RESULTS / "capacity-outputs.json").read_text())["checks"]
        outputs = {day(check["probe"]["t"]) for check in checks}
        for text in (f"| Accuracy | {span(accuracy)} |", f"| Inference time | {span(latency)} |",
                     f"| Camera capacity (EX5) | {span(cameras)} | {len(trials)} live RTSP trials",
                     f"| Camera capacity (EX3) | {span(ex3_cameras)} | 720p, WebRTC and Smart Video tests",
                     f"| Output checks | {span(outputs)} | {len(checks)} YOLO26n WebRTC-only output diagnostics |"):
            with self.subTest(text=text):
                self.assertIn(text, self.text)

    def test_only_the_optional_cells_run_code_and_they_find_the_repository(self):
        cells = notebook("04-compare-models.ipynb")["cells"]
        ids = [cell["id"] for cell in cells]
        first_code = next(index for index, cell in enumerate(cells) if cell["cell_type"] == "code")
        self.assertGreater(first_code, ids.index("remeasure-intro"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "tools").mkdir()
            (root / "tools/compare_models.py").write_text("")
            (root / "notebooks").mkdir()
            shown, printed, environment, working_directory = self.run_cells(
                ["measure-switch", "show-own"], root / "notebooks")
            self.assertEqual(working_directory, root)
            self.assertEqual(environment["MEASURE"], "0")
            self.assertEqual(shown, [])
            self.assertIn("No report in tmp/benchmarks", printed)
            own = root / "tmp/benchmarks"
            own.mkdir(parents=True)
            (own / "comparison.md").write_text("| own table |")
            (own / "comparison.png").write_bytes(b"png")
            shown, _, _, _ = self.run_cells(["measure-switch", "show-own"], root / "notebooks")
            self.assertEqual(shown, [("markdown", "| own table |"), ("image", "tmp/benchmarks/comparison.png")])

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
