import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
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
V2_EX3_FILES = ("capacity-protocol-v2.jsonl", "capacity-protocol-v2-recheck.jsonl",
                "capacity-boundary-v2.jsonl")
MATRIX = ROOT / "doc/benchmarks/capacity/README.md"
# Each station's no-output evidence and the one-camera output runs it publishes.
STATIONS = {
    "EX5": {"results": RESULTS, "no_output": ("capacity.jsonl",), "tags": ("720p", "1080p"),
            "outputs": ("capacity-output-v2.jsonl",), "output_stage": "single-camera-screening"},
    "EX3": {"results": EX3_RESULTS, "no_output": V2_EX3_FILES, "tags": ("720p",),
            "outputs": V2_EX3_FILES[:2], "output_stage": "capacity"},
}
COUNT_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
PROFILE_LABELS = {"webrtc-osd": "WebRTC", "evt-osd": "Smart Video", "webrtc+evt-osd": "WebRTC + Smart Video"}
OUTPUT_HEADER = ("| Model | Profile | Runs | Processed FPS (minimum) | WebRTC publisher | Smart Video clips "
                 "| Result |")


def notebook(name):
    return json.loads((NOTEBOOKS / name).read_text())


def source(cell):
    return "".join(cell["source"])


def markdown(name):
    return "\n".join(source(cell) for cell in notebook(name)["cells"] if cell["cell_type"] == "markdown")


def normalized(text):
    return " ".join(text.split())


def qualification_rule():
    return (f"Every camera must average at least {capacity.PASS_FPS:g} processed FPS over "
            f"{capacity.WINDOW_SECONDS} seconds after a {capacity.WARMUP_SECONDS}-second warm-up, with no "
            "reconnects or inference stalls. A count is reported when it passed at least two attempts and failed "
            "none, every lower count tested also passed, and at least one attempt at the next count failed. "
            "† means a higher count passed in some attempts and failed in others; the table keeps the lower count.")


def output_criteria():
    clip_seconds = (capacity.SMART_PRE_MS + capacity.SMART_POST_MS) / 1000
    return (f"WebRTC must deliver at least {capacity.PASS_FPS:g} FPS, measured at the publisher for the first "
            f"{capacity.WEBRTC_PROBE_SECONDS} seconds of sampling. Every Smart Video clip must be H.264 at the "
            f"mosaic size, last {clip_seconds:g} seconds (±2 seconds), decode at {capacity.PASS_FPS:g} FPS or "
            "more and start on a keyframe. Neither output may have a packet gap over "
            f"{capacity.MAX_FRAME_GAP_SECONDS * 1000:.0f} ms (plus {capacity.GAP_TOLERANCE_SECONDS * 1000:g} ms "
            "timing tolerance).")


def station_records(station, names):
    return [row for name in names for row in capacity.load(STATIONS[station]["results"] / name)]


def base_attempts(records, model, tag):
    """Full-length no-output attempts; shorter pilot attempts do not qualify a count."""
    attempts = []
    for row in records:
        if row["kind"] in ("trial", "fixed") and (row["model"], row["profile"], row.get("camera_tag")) == (
                model, "base", tag):
            trial = row.get("trial", row)
            attempts += [attempt for attempt in trial.get("attempts", [trial])
                         if (attempt["warmup_seconds"], attempt["window_seconds"]) == (
                             capacity.WARMUP_SECONDS, capacity.WINDOW_SECONDS)]
    return attempts


def output_runs(station, model, profile):
    spec = STATIONS[station]
    return [row.get("trial", row) for row in station_records(station, spec["outputs"])
            if row["kind"] in ("trial", "fixed") and row["stage"] == spec["output_stage"]
            and (row["model"], row["profile"]) == (model, profile) and row.get("trial", row)["n"] == 1]


def meets_output_checks(run):
    checks = [run["min_fps"] >= capacity.PASS_FPS]
    if run.get("webrtc_delivery"):
        delivery = run["webrtc_delivery"]
        checks += [delivery["delivered_fps"] >= capacity.PASS_FPS, capacity.gap_ok(delivery["max_gap_seconds"])]
    for clip in (run.get("smart_video") or {}).get("clips", []):
        checks += [clip["codec"] == "h264", clip["decoded_fps"] >= capacity.PASS_FPS,
                   clip["first_packet_keyframe"], capacity.gap_ok(clip["max_gap_seconds"])]
    return all(checks)


def output_row(model, profile, runs):
    def fps_range(values):
        low, high = f"{min(values):.2f}", f"{max(values):.2f}"
        return low if low == high else f"{low}–{high}"

    def largest_gap(values):
        return f"{round(max(values) * 1000)} ms maximum gap"

    deliveries = [run["webrtc_delivery"] for run in runs if run.get("webrtc_delivery")]
    clips = [clip for run in runs for clip in (run.get("smart_video") or {}).get("clips", [])]
    webrtc = (f"{fps_range([d['delivered_fps'] for d in deliveries])} FPS; "
              f"{largest_gap([d['max_gap_seconds'] for d in deliveries])}") if deliveries else "—"
    smart_video = (f"{sum(clip['decoded_fps'] >= capacity.PASS_FPS for clip in clips)}/{len(clips)} at "
                   f"{capacity.PASS_FPS:g}+ FPS ({fps_range([clip['decoded_fps'] for clip in clips])}); "
                   f"{sum(clip['first_packet_keyframe'] for clip in clips)}/{len(clips)} keyframe starts; "
                   f"{largest_gap([clip['max_gap_seconds'] for clip in clips])}") if clips else "—"
    result = "Passed" if all(run["pass"] for run in runs) else "Not qualified"
    return (f"| {capacity.MODEL_NAMES[model]} | {PROFILE_LABELS[profile]} | {len(runs)} | "
            f"{min(run['min_fps'] for run in runs):.2f} | {webrtc} | {smart_video} | {result} |")


def table_rows(table):
    return [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in table.splitlines()[2:] if line.startswith("| ")]


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
        ex3_records = [row for name in V2_EX3_FILES for row in capacity.load(EX3_RESULTS / name)]
        ex3_cameras = table_rows(capacity.camera_table(ex3_records, models, [("720p", "base")]))
        for model, accuracy, counts, ex3 in zip(models, summary, cameras, ex3_cameras):
            with self.subTest(model=model):
                row = next(line for line in readme.splitlines()
                           if line.startswith("| [") and f"(notebooks/{BUILD[model]})" in line)
                counts = counts[1:]
                # The caption compares input resolutions, not unsupported output combinations.
                self.assertEqual(len({count.rstrip("†") for count in counts}), 1, counts)
                self.assertEqual(table_rows(f"|\n|\n{row}")[0][1:], [*accuracy[1:4], counts[0], ex3[1]])
                self.assertIn(f"| [{capacity.MODEL_NAMES[model]}]", row)

    def test_the_presentation_is_read_only(self):
        presentation = notebook("00-presentation.ipynb")
        self.assertFalse([cell for cell in presentation["cells"] if cell["cell_type"] == "code"])

    def test_batch_reading_example_matches_recorded_request_times(self):
        lesson = markdown("04-compare-models.ipynb")
        request_times = {}
        for batch in (1, 8):
            result = compare_models.row(self.by_id["yolov10"]["latency"], compare_models.TENSORRT_FP16, batch)
            median = result["median_ms"]
            with self.subTest(batch=batch):
                request_times[batch] = f"{median:.2f}"
                self.assertIn(f"| YOLOv10n | {batch} |", lesson)
        self.assertIn(
            f"YOLOv10n FP16 takes **{request_times[1]} ms** for one image, "
            f"or **{request_times[8]} ms** for eight.",
            lesson)
        per_image = compare_models.row(self.by_id["yolov10"]["latency"], compare_models.TENSORRT_FP16, 8)["ms_per_image"]
        self.assertIn(f"**{per_image:.2f} ms/image**", lesson)
        readme = (ROOT / "README.md").read_text()
        self.assertIn("(notebooks/04-compare-models.ipynb)", readme)
        for expected in ("The profile allows 1 through 8 images per request",
                         "For a **one-camera, one-frame request**, start with batch 1",
                         "an engine built to accept only one image per request",
                         "Triton's dynamic batching"):
            self.assertIn(expected, lesson)

    def test_supported_matrix_does_not_relabel_historical_output_results(self):
        readme = (ROOT / "README.md").read_text()
        plan = (ROOT / "doc/benchmarks/capacity/README.md").read_text()
        comparison = markdown("04-compare-models.ipynb")
        table = plan.split("| Device |", 1)[1].split("\n\n", 1)[0]
        self.assertIn("| Device |" + table, comparison)
        baseline = table_rows(capacity.camera_table(capacity.load(RESULTS / "capacity.jsonl"),
                                                    columns=[("720p", "base")]))
        counts = {row[0]: row[1] for row in baseline}
        v2_records = [row for name in V2_EX3_FILES for row in capacity.load(EX3_RESULTS / name)]
        ex3_counts = {row[0]: row[1:] for row in table_rows(capacity.camera_table(
            v2_records, columns=[("720p", profile) for profile in ("base", *capacity.CAPACITY_PROFILES)]))}
        self.assertEqual(len(table_rows("| Device |" + table)), 2 * len(self.by_id))
        for line in table_rows("| Device |" + table):
            device, model, *cells = line
            expected = ([counts[model], "At least 1 (screened)",
                         "At least 1 (screened)", "At least 1 (screened)"]
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
                self.assertNotIn("ms3_build", value)
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
        for name in V2_EX3_FILES:
            evidence = (EX3_RESULTS / name).read_text().lower()
            with self.subTest(file=name):
                self.assertNotIn('"hls"', evidence)
                self.assertNotIn("ms3_build", evidence)

    def test_ex5_protocol_v2_output_screening_matches_published_results(self):
        records = capacity.load(RESULTS / "capacity-output-v2.jsonl")
        expected = {(model, profile) for model in self.by_id
                    for profile in capacity.CAPACITY_PROFILES}
        self.assertEqual({(row["model"], row["profile"]) for row in records}, expected)
        self.assertEqual(len(records), 9)

        for row in records:
            with self.subTest(model=row["model"], profile=row["profile"]):
                self.assertEqual(row["kind"], "fixed")
                self.assertEqual(row["n"], 1)
                self.assertTrue(row["pass"])
                self.assertGreaterEqual(row["min_fps"], capacity.PASS_FPS)
                self.assertEqual(row["protocol"], capacity.PROTOCOL)
                self.assertEqual(row["stage"], "single-camera-screening")

                if "webrtc" in row["profile"]:
                    delivery = row["webrtc_delivery"]
                    self.assertTrue(delivery["pass"])
                    self.assertGreaterEqual(delivery["delivered_fps"], capacity.PASS_FPS)
                    self.assertLessEqual(delivery["max_gap_seconds"], capacity.MAX_FRAME_GAP_SECONDS)
                if "evt" in row["profile"]:
                    clips = row["smart_video"]["clips"]
                    self.assertEqual(len(clips), 4)
                    for clip in clips:
                        self.assertEqual(clip["codec"], "h264")
                        self.assertTrue(clip["fps_pass"])
                        self.assertTrue(clip["continuity_pass"])
                        self.assertTrue(clip["first_packet_keyframe"])
                        self.assertGreaterEqual(clip["decoded_fps"], capacity.PASS_FPS)
                        self.assertLessEqual(clip["max_gap_seconds"], capacity.MAX_FRAME_GAP_SECONDS)


        evidence = (RESULTS / "capacity-output-v2.jsonl").read_text().lower()
        for field in ("hls", "ms3_build", "pipeline", "run", "aip", "record_uuid", "camera_tag"):
            with self.subTest(field=field):
                self.assertNotIn(f'"{field}"', evidence)

    def test_one_qualification_rule_produces_every_published_count(self):
        rule = normalized(qualification_rule())
        matrix = MATRIX.read_text()
        readme = (ROOT / "README.md").read_text()
        cameras_cell = next(source(cell) for cell in notebook("04-compare-models.ipynb")["cells"]
                            if cell.get("id") == "cameras")
        self.assertIn(rule, normalized(matrix))
        models = list(self.by_id)
        for station, spec in STATIONS.items():
            report = (spec["results"] / "capacity.md").read_text()
            limits = report.split("## Measured no-output limits", 1)[1].split("\n## ", 1)[0]
            self.assertIn(rule, normalized(limits))
            records = station_records(station, spec["no_output"])
            published = table_rows(capacity.camera_table(records, models, [(tag, "base") for tag in spec["tags"]]))
            for model, (name, *cells) in zip(models, published):
                report_cells = [f"**{cell}**" for cell in cells] + ["Not tested"] * (2 - len(cells))
                self.assertIn(f"| {name} | {' | '.join(report_cells)} |", limits)
                for tag, cell in zip(spec["tags"], cells):
                    with self.subTest(station=station, model=model, tag=tag):
                        results = defaultdict(list)
                        for attempt in base_attempts(records, model, tag):
                            self.assertEqual(attempt["pass"], attempt["min_fps"] >= capacity.PASS_FPS)
                            results[attempt["n"]].append(attempt["pass"])
                        count = int(cell.rstrip("†"))
                        self.assertGreaterEqual(len(results[count]), 2)
                        self.assertTrue(all(passed for n, runs in results.items() if n <= count for passed in runs))
                        self.assertIn(False, results[count + 1])
                        contested = {n: runs for n, runs in results.items() if n > count and any(runs)}
                        self.assertEqual(cell.endswith("†"), bool(contested))
                        for n, runs in contested.items():
                            note = f"{name}: {COUNT_WORDS[n]} {tag} cameras passed {sum(runs)} of {len(runs)} attempts"
                            self.assertIn(f"† {note}.", limits)
                            for text in (matrix, readme, cameras_cell):
                                self.assertIn(f"{station} {note}", normalized(text))

    def test_one_camera_output_tables_apply_the_same_checks(self):
        criteria = normalized(output_criteria())
        self.assertIn(criteria, normalized(MATRIX.read_text()))
        for station, spec in STATIONS.items():
            report = (spec["results"] / "capacity.md").read_text()
            section = report.split("## Protocol v2: one-camera output checks", 1)[1].split("\n## ", 1)[0]
            self.assertIn(criteria, normalized(section))
            table = OUTPUT_HEADER + section.split(OUTPUT_HEADER, 1)[1].split("\n\n", 1)[0]
            self.assertEqual(len(table_rows(table)), len(self.by_id) * len(PROFILE_LABELS))
            for model in self.by_id:
                for profile in PROFILE_LABELS:
                    runs = output_runs(station, model, profile)
                    with self.subTest(station=station, model=model, profile=profile):
                        self.assertTrue(runs)
                        for run in runs:
                            self.assertEqual((run["warmup_seconds"], run["window_seconds"]),
                                             (capacity.WARMUP_SECONDS, capacity.WINDOW_SECONDS))
                            self.assertEqual(bool(run.get("webrtc_delivery")), "webrtc" in profile)
                            self.assertEqual(bool((run.get("smart_video") or {}).get("clips")), "evt" in profile)
                            config = capacity.parse_profile(profile)
                            settings = run["output_settings"]
                            self.assertEqual((settings["webrtc"], settings["eventrec"], settings["osd"],
                                              settings["bitrate"], tuple(settings["tiler"])),
                                             (config.webrtc, config.eventrec, config.osd, config.bitrate, config.tiler))
                            if config.eventrec:
                                timings = run["smart_video"]
                                self.assertEqual((timings["pre_duration_ms"], timings["post_duration_ms"],
                                                  timings["max_duration_ms"], timings["trigger_interval_seconds"]),
                                                 (capacity.SMART_PRE_MS, capacity.SMART_POST_MS,
                                                  capacity.SMART_MAX_MS, capacity.SMART_TRIGGER_SECONDS))
                            self.assertEqual(run["pass"], meets_output_checks(run))
                        self.assertIn(output_row(model, profile, runs), table)

    def test_matched_table_uses_measured_pairs_and_all_enabled_output_checks(self):
        folder = MATRIX.parent
        report = (folder / "matched.md").read_text()
        paths = [folder / f"matched-ex{n}.jsonl" for n in (3, 5)]
        stations = [capacity.load(path) for path in paths]
        for path in paths:
            self.assertNotRegex(path.read_text(), r'://|"(?:pk|record_uuid|pipeline_uuid|unique_id|run)"\s*:')
        self.assertIn(capacity.matched_comparison(*stations), report)
        expected = {(m, p) for m in self.by_id for p in capacity.MATCHED_PROFILES}
        for rows in stations:
            self.assertEqual({(r["model"], r["profile"]) for r in rows}, expected)
            for row in rows:
                with self.subTest(station=row["device"], model=row["model"], profile=row["profile"]):
                    trial = row["trial"]
                    self.assertIn(trial["matched_setup"]["source"]["file_sha256"], report)
                    self.assertEqual(trial["sample_count"], capacity.WINDOW_SECONDS // capacity.SAMPLE_SECONDS)
                    self.assertAlmostEqual(trial["sampling_duration_seconds"], capacity.WINDOW_SECONDS, delta=1)
                    passed = (trial["min_fps"] >= capacity.PASS_FPS and not trial["reconnects"]
                              and not trial["inference_stall_samples"]
                              and trial["running_samples"] == trial["streaming_samples"] == trial["sample_count"])
                    self.assertEqual(trial["processing_pass"], passed)
                    if "webrtc_delivery" in trial:
                        w = trial["webrtc_delivery"]
                        fps = (w["packets"] - 1) / w["span_seconds"]
                        self.assertEqual(w["delivered_fps"], round(fps, 2))
                        self.assertEqual(w["pass"], fps >= capacity.PASS_FPS and capacity.gap_ok(w["max_gap_seconds"]))
                        passed &= w["pass"]
                    if "smart_video" in trial:
                        smart = trial["smart_video"]
                        self.assertEqual(smart["trigger_count"], capacity.WINDOW_SECONDS // capacity.SMART_TRIGGER_SECONDS)
                        self.assertEqual(len(smart["clips"]), smart["trigger_count"])
                        for clip in smart["clips"]:
                            duration = clip["duration_seconds"]
                            self.assertAlmostEqual(duration, (capacity.SMART_PRE_MS + capacity.SMART_POST_MS) / 1000,
                                                   delta=2)
                            self.assertAlmostEqual(clip["decoded_fps"], clip["decoded_frames"] / duration)
                            self.assertEqual(clip["fps_pass"], clip["decoded_fps"] >= capacity.PASS_FPS)
                            self.assertEqual(clip["continuity_pass"], clip["first_packet_keyframe"]
                                             and capacity.gap_ok(clip["max_gap_seconds"]))
                            passed &= clip["fps_pass"] and clip["continuity_pass"]
                    self.assertEqual(trial["pass"], passed)
        for label, outcomes in (
                ("Input processing (all runs)", lambda rows: [r["trial"]["processing_pass"] for r in rows]),
                ("WebRTC publisher (runs with WebRTC enabled)", lambda rows: [
                    r["trial"]["webrtc_delivery"]["pass"] for r in rows if "webrtc_delivery" in r["trial"]]),
                ("Smart Video (individual clips)", lambda rows: [c["fps_pass"] and c["continuity_pass"]
                    for r in rows for c in r["trial"].get("smart_video", {}).get("clips", [])]),
                ("Processing and every requested output (runs with video output)", lambda rows: [
                    r["trial"]["pass"] for r in rows if r["profile"] != "base"])):
            counts = [outcomes(rows) for rows in stations]
            self.assertIn(f"| {label} | " + " | ".join(f"{sum(c)}/{len(c)}" for c in counts) + " |", report)

    def test_recording_deficit_table_uses_saved_one_camera_clips(self):
        records = [row for name in V2_EX3_FILES[:2] for row in capacity.load(EX3_RESULTS / name)]
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
            self.assertIn("capacity.md#earlier-measurements-protocol-v1)", text)
        self.assertEqual({(row["device"], row["protocol"]) for row in records},
                         {("ex3-orin-nano-8gb-15w", "webrtc-smart-video-v1")})
        self.assertEqual({row["model"] for row in records if row["kind"] in {"trial", "fixed"}},
                         set(self.by_id))

    def test_ex3_six_camera_report_counts_attempts_not_search_summaries(self):
        report = (EX3_RESULTS / "capacity.md").read_text()
        all_attempts = []
        for name in V2_EX3_FILES:
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

    def test_training_overview_explains_aip_layers_and_model_choice(self):
        readme = (ROOT / "README.md").read_text()
        overview = readme.split("## Choose a model", 1)[0]
        for concept in ("primary", "secondary", "Business logic", "Node-RED", "UIBuilder"):
            self.assertIn(concept, overview)
        for link in ("(logic/)", "(dashboard/)",
                     "(https://www.npmjs.com/package/@edgematrix/node-red-contrib-eais)",
                     "(https://github.com/TotallyInformation/node-red-contrib-uibuilder)"):
            self.assertIn(link, overview)
        for text in (readme, markdown("00-presentation.ipynb"), markdown("04-compare-models.ipynb")):
            self.assertIn("license", text.lower())
            self.assertNotIn("(recommended)", text)
            self.assertNotIn("a good first build", text)
        self.assertIn("all nine one-camera model/profile checks passed", readme)
        self.assertIn("multi-camera output limits have not been established", readme)

    def test_recorded_files_are_public_material(self):
        for path in sorted((*RESULTS.rglob("*"), *EX3_RESULTS.rglob("*"), *MATRIX.parent.iterdir())):
            if path.suffix in {".json", ".jsonl", ".md"}:
                text = path.read_text()
                assert_no_private_references(self, text, path.name)
                self.assertNotIn("/tmp/", text, path.name)

    def test_training_docs_omit_measurement_dates(self):
        paths = [ROOT / "README.md", *sorted((ROOT / "doc").rglob("*.md")),
                 *sorted(NOTEBOOKS.glob("*.ipynb"))]
        months = "January|February|March|April|May|June|July|August|September|October|November|December"
        date_pattern = (r"(?i:\b(?:20\d{2}-\d{2}-\d{2}|\d{1,2}\s+[a-z]+\s+20\d{2}|exact dates)\b)"
                        rf"|\b\d{{1,2}}(?:[–-]\d{{1,2}})?\s+(?:{months})\b|\b(?:{months})\s+\d{{1,2}}\b|\bJST\b")
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                text = markdown(path.name) if path.suffix == ".ipynb" else path.read_text()
                self.assertNotRegex(text, date_pattern)

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
                    "parity": compare_models.parity_table(self.entries)}
        for name, table in expected.items():
            with self.subTest(table=name):
                self.assertIn(table, self.text)
        ex3 = table_rows(capacity.camera_table(station_records("EX3", V2_EX3_FILES), self.models,
                                               [("720p", "base")]))
        ex5 = table_rows(capacity.camera_table(self.records, self.models, BASELINE_COLUMNS))
        table = self.text.split("| Model | EX3 720p | EX5 720p | EX5 1080p |", 1)[1].split("\n\n", 1)[0]
        self.assertEqual(table_rows("header" + table), [a + b[1:] for a, b in zip(ex3, ex5)])
        self.assertIn(f"]({os.path.relpath(RESULTS / 'comparison.png', NOTEBOOKS)})", self.text)

    def test_the_comparison_lesson_links_detailed_measurement_evidence(self):
        for expected in ("## 4. Where do these results come from?",
                         "../doc/benchmarks/orin-nx-16gb-jp6.0/capacity.md",
                         "../doc/benchmarks/orin-nano-8gb-jp6.0/capacity.md",
                         "../doc/benchmarks/orin-nx-16gb-jp6.0/conditions.md",
                         "not a controlled, scene-identical comparison"):
            self.assertIn(expected, self.text)

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
