import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import capacity  # noqa: E402
import flavors  # noqa: E402
from public_reference import assert_no_private_references  # noqa: E402


class CapacityProfileTests(unittest.TestCase):
    def test_profile_names_round_trip_and_irrelevant_settings_normalize(self):
        names = {
            "base", "rec", "webrtc", "webrtc-osd", "hls-osd", "evt-osd",
            "rec+webrtc-osd", "webrtc-osd-1M", "webrtc-osd-4M",
            "webrtc-osd-640x360", "webrtc-osd-1920x1080",
            "rec+webrtc+hls+evt-osd",
        }
        for name in names:
            with self.subTest(profile=name):
                config = capacity.parse_profile(name)
                self.assertEqual(config.name, name)
                self.assertEqual(capacity.parse_profile(config.name), config)

        collapsed = capacity.normalize(capacity.Config(recording=True, osd=True, bitrate=4_000_000))
        self.assertEqual(collapsed, capacity.Config(recording=True))
        with self.assertRaises(capacity.CapacityError):
            capacity.parse_profile("rec-osd")

    def test_pipeline_setup_enables_only_requested_output_branches(self):
        config = capacity.parse_profile("webrtc+evt-osd")
        body = capacity.pipeline_setup("test", config, {"pk": "aip-pk"}, ["m1", "m2", "m3"])
        pipeline = body["pipeline"]
        self.assertEqual(pipeline["ai_processor_setup"]["uuid"], "aip-pk")
        self.assertEqual(pipeline["ai_processor_setup"]["batch_size"], 3)
        self.assertFalse(pipeline["hls_output_sink_setup"]["active"])
        self.assertFalse(pipeline["mtx_output_sink_setup"]["recording_enabled"])
        self.assertTrue(pipeline["mtx_output_sink_setup"]["webrtc_enabled"])
        self.assertEqual(pipeline["mtx_output_sink_setup"]["num_slots"], 2)
        self.assertEqual(pipeline["eventrec_sink_setup"], {
            "active": True, "osd": True, "bitrate": capacity.EVENTREC_BITRATE,
            "pre_duration": 3000, "post_duration": 6000, "max_duration": 10000,
        })
        self.assertFalse(pipeline["auto_reload"])

        base = capacity.pipeline_setup("test", capacity.Config(), {"pk": "aip-pk"}, ["m1"])
        self.assertFalse(base["pipeline"]["hls_output_sink_setup"]["active"])
        self.assertFalse(base["pipeline"]["mtx_output_sink_setup"]["active"])

    def test_excluded_profiles_can_be_recovered_but_not_run_or_published(self):
        for name in ("rec", "hls", "rec+webrtc-osd", "rec+webrtc+hls+evt-osd"):
            with self.subTest(profile=name):
                config = capacity.parse_profile(name)
                self.assertEqual(config.name, name)
                with self.assertRaisesRegex(capacity.CapacityError, "excluded from benchmarks"):
                    capacity.pipeline_setup("test", config, {"pk": "aip"}, ["camera"])
                record = {"kind": "trial", "profile": name, "model": "yolo26",
                          "camera_tag": "720p", "n": 1, "min_fps": 30, "pass": True}
                for render in (capacity.summarize, capacity.camera_table):
                    with self.assertRaisesRegex(capacity.CapacityError, "excluded from benchmarks"):
                        render([record])
                with self.assertRaisesRegex(capacity.CapacityError, "excluded from benchmarks"):
                    capacity.describe(name)

    def test_batch_size_caps_at_engine_maximum(self):
        body = capacity.pipeline_setup("test", capacity.Config(), {"pk": "aip-pk"},
                                       [f"m{index}" for index in range(12)])
        self.assertEqual(body["pipeline"]["ai_processor_setup"]["batch_size"], capacity.ENGINE_MAX_BATCH)


class CapacityEvaluationTests(unittest.TestCase):
    def test_processed_metrics_are_scoped_to_the_pipeline_and_must_be_finite(self):
        metrics = (
            'pipeline_perf_fps_average_10sec{input_id="0",name="camera, \\"one\\" }",uuid="p1"} 2.95e1\n'
            'pipeline_perf_fps_average_10sec{uuid="another",input_id="0",name="other"} 999\n'
            'pipeline_input_fps_average_10sec{uuid="p1",input_id="0",name="input"} 30\n'
        )
        self.assertEqual(capacity.processed_fps(metrics, "p1"), {"0": 29.5})
        with self.assertRaisesRegex(capacity.CapacityError, "processed-FPS"):
            capacity.processed_fps(metrics.replace("2.95e1", "NaN"), "p1")

    @staticmethod
    def sample(fps=30, state="streaming", attempt=0, flags=()):
        return {
            "inputs": {"0": {"fps": fps, "state": state, "attempt": attempt}},
            "pipeline_fps": fps,
            "flags": list(flags),
        }

    def test_evaluate_requires_each_camera_to_meet_fps_and_remain_connected(self):
        passing = [
            {"inputs": {str(index): {"fps": 30, "state": "streaming", "attempt": 0} for index in range(2)},
             "pipeline_fps": 60, "flags": []}
            for _ in range(3)
        ]
        result = capacity.evaluate(passing, 2)
        self.assertTrue(result["pass"], result)
        self.assertEqual(result["min_fps"], 30)

        cases = (
            (self.sample(fps=28), "averaged 28 FPS"),
            (self.sample(state="reconnecting"), "left the streaming state"),
            (self.sample(attempt=1), "reconnected 1 time"),
            (self.sample(flags=("inference_stall",)), "inference stalled"),
        )
        for sample, reason in cases:
            with self.subTest(reason=reason):
                result = capacity.evaluate([sample], 1)
                self.assertFalse(result["pass"])
                self.assertTrue(any(reason in text for text in result["reasons"]))

    def test_threshold_is_checked_before_rounding_and_nonfinite_fps_cannot_pass(self):
        for fps in (28.999, float("nan"), float("inf")):
            with self.subTest(fps=fps):
                result = capacity.evaluate([self.sample(fps=fps)], 1)
                self.assertFalse(result["pass"], result)
                self.assertTrue(result["reasons"])
        self.assertTrue(capacity.evaluate([self.sample(fps=29)], 1)["pass"])

    def test_evaluate_fails_when_a_camera_disappears_during_the_window(self):
        samples = [
            {"inputs": {"0": {"fps": 30, "state": "streaming", "attempt": 0},
                        "1": {"fps": 30, "state": "streaming", "attempt": 0}}, "flags": []},
            {"inputs": {"0": {"fps": 30, "state": "streaming", "attempt": 0}}, "flags": []},
        ]
        result = capacity.evaluate(samples, 2)
        self.assertFalse(result["pass"])
        self.assertIn("did not report the same 2 cameras", result["reasons"][0])

    def test_search_capacity_reports_lower_bound_when_maximum_passes(self):
        capacity_found, bounded, results = capacity.search_capacity(
            lambda count: {"n": count, "pass": count <= 4}, 8
        )
        self.assertEqual(capacity_found, 4)
        self.assertTrue(bounded)
        self.assertTrue(all(result["n"] <= 8 for result in results))

        capacity_found, bounded, _ = capacity.search_capacity(
            lambda count: {"n": count, "pass": True}, 8
        )
        self.assertEqual(capacity_found, 8)
        self.assertFalse(bounded)

    def test_settled_repeats_borderline_trials_and_uses_both_results(self):
        results = iter([
            {"pass": True, "min_fps": 29.1, "reasons": [], "device": "ex3", "protocol": capacity.PROTOCOL},
            {"pass": False, "min_fps": 28.9, "reasons": ["camera 0 averaged 28.90 FPS"]},
        ])
        calls = []

        def trial(count):
            calls.append(count)
            return next(results)

        result = capacity.settled(trial, 3)
        self.assertEqual(calls, [3, 3])
        self.assertFalse(result["pass"])
        self.assertEqual(result["min_fps"], 28.9)
        self.assertEqual(result["reasons"], ["repeat: camera 0 averaged 28.90 FPS"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual((result["device"], result["protocol"]), ("ex3", capacity.PROTOCOL))
        self.assertEqual(capacity.worst_attempt(result)["min_fps"], 28.9)

    def test_invalid_trial_is_not_repeated_or_treated_as_a_capacity_limit(self):
        for error in ({"error": "API unavailable"}, {"cleanup_errors": ["pipeline still exists"]}):
            with self.subTest(error=error):
                record = {"pass": False, "min_fps": 29.1, "reasons": [], **error}
                run = mock.Mock(return_value=record)
                result = capacity.settled(run, 3)
                run.assert_called_once_with(3)
                with self.assertRaisesRegex(capacity.CapacityError, "trial invalid"):
                    capacity.require_valid_trial(result)

    def test_tegrastats_parser_and_summary_capture_gpu_and_encoder(self):
        line = ("RAM 6144/15600MB CPU [10%@102MHz,20%@102MHz] GR3D_FREQ 55% "
                "NVENC_FREQ 600 VIC_FREQ 300 tj@45.5C VDD_IN 12000mW")
        reading = capacity.parse_tegrastats(line)
        self.assertEqual(reading["gpu_percent"], 55)
        self.assertEqual(reading["nvenc_mhz"], 600)
        self.assertEqual(reading["ram_used_mb"], 6144)
        summary = capacity.summarize_readings([reading, {**reading, "gpu_percent": 65}])
        self.assertEqual(summary["gpu_percent"], {"mean": 60, "max": 65})


class FakeStation:
    """ms3 and the signaling service in memory: the calls capacity.py makes and the state they change."""

    AIP = {"pk": "aip-pk", "unique_id": "AIPV1EMI9026", "version": "1.0.0", "name": "YOLO26n",
           "maximum_input_streams": 8, "primary_processor": {"model": "YOLO26n", "batch_size": 8}}

    def __init__(self):
        self.media = {}
        self.pipelines = {}
        self.recording = set()
        self.files = {}
        self.calls = []
        self.broken = set()
        self.sticky = {}
        self.created = 0
        self.perf = 30.0
        self.records = {}
        self.paths = {
            "/pipeline/{uuid}/inputs/fps": {"get": {}}, "/pipeline/{uuid}/status": {"get": {}},
            "/metrics": {"get": {}}, "/pipeline/{uuid}/mtx/start": {"post": {}},
            "/pipeline/{uuid}/eventrec/trigger": {"post": {}}, "/pipeline/records": {"get": {}},
            "/pipeline/record/{uuid}": {"get": {}}, "/pipeline/records/deletion": {"post": {}},
        }

    def call(self, method, path, body=None, query=None, form=None, ok=(200, 201, 202)):
        self.calls.append((method, path, dict(query or {})))
        if (method, path) in self.broken:
            raise capacity.ApiError(method, path, 500, "broken")
        status, payload = self.respond(method, path, body, form)
        if status not in ok:
            raise capacity.ApiError(method, path, status, payload)
        return payload

    def add(self, kind, name):
        self.created += 1
        pk = f"{kind}-{self.created}"
        getattr(self, kind)[pk] = {"name": name}
        return pk

    def respond(self, method, path, body, form):
        query = self.calls[-1][2]
        if (method, path) == ("GET", "/openapi.json"):
            return 200, {"paths": self.paths}
        if (method, path) == ("GET", "/aiprocessors"):
            return 200, [self.AIP]
        if (method, path) == ("GET", "/medias"):
            return 200, [{"pk": pk, "name": item["name"]} for pk, item in self.media.items()]
        if (method, path) == ("GET", "/pipelines"):
            return 200, [{"pk": pk, "name": item["name"]} for pk, item in self.pipelines.items()]
        if (method, path) == ("GET", "/metrics"):
            return 200, "\n".join(
                f'{capacity.PERF_METRIC}{{uuid="{pk}",name="{item["name"]}",input_id="{index}"}} {self.perf}'
                for pk, item in self.pipelines.items() for index in range(item.get("inputs", 0)))
        if (method, path) == ("POST", "/media"):
            if json.loads(form.get("proxy_setup", "{}")).get("uri", "").startswith("rtsp://") \
                    and form.get("external_setup") is None:
                return 422, {"detail": "RTSP media must have external settings."}
            return 201, {"pk": self.add("media", form["name"])}
        if (method, path) == ("POST", "/pipeline"):
            pipeline = body["pipeline"]
            pk = self.add("pipelines", pipeline["name"])
            self.pipelines[pk]["inputs"] = len(pipeline["inputs"])
            return 201, {"pk": pk}
        if (method, path) == ("POST", "/pipeline/records/deletion"):
            self.records.pop(query["pipeline_uuid"], None)
            return 200, {"message": "deletion accepted"}
        if (method, path) == ("GET", "/pipeline/records"):
            return 200, {"records": self.records.get(query["pipeline_uuid"], []), "pagination": {"max_page": 1}}
        if path in ("/recordings/start", "/recordings/stop"):
            pipeline = body["pipeline_id"]
            if path.endswith("start"):
                self.recording.add(pipeline)
                self.files.setdefault(pipeline, set()).add("2026-01-01_00-00-00.mp4")
            else:
                self.recording.discard(pipeline)
            return 200, {"pipeline_id": pipeline}
        match = re.fullmatch(r"/recordings/([^/]+)/files", path)
        if match:
            stored = self.files.setdefault(match.group(1), set())
            if method == "GET":
                return 200, {"files": sorted(stored), "total": len(stored), "has_next": False}
            stored.difference_update(body["files"])
            return 200, {"deleted": body["files"], "not_found": [], "skipped_writing": []}
        match = re.fullmatch(r"/media/([^/]+)", path)
        if match and method == "DELETE":
            if self.sticky.get(match.group(1)):
                self.sticky[match.group(1)] -= 1
                return 200, None
            return (200, None) if self.media.pop(match.group(1), None) else (404, "gone")
        match = re.fullmatch(r"/pipeline/([^/]+)(/.*)?", path)
        pipeline = self.pipelines.get(match.group(1)) if match else None
        if pipeline is None:
            return 404, "no such pipeline"
        action = match.group(2) or ""
        inputs = pipeline["inputs"]
        if method == "DELETE" and not action:
            del self.pipelines[match.group(1)]
            return 200, None
        if action == "/status":
            return 200, {"overall": "Running", "inputs:live": inputs, "health": "good"}
        if action == "/inputs/fps":
            return 200, {str(index): {"arrival": {"average_10sec": 30.0, "raw": 30.0},
                                      "link": {"state": "streaming", "attempt": 0}} for index in range(inputs)}
        if action == "/fps":
            return 200, {"average_10sec": self.perf}
        if action == "/eventrec/trigger":
            pk = match.group(1)
            records = self.records.setdefault(pk, [])
            records.append({"record_uuid": f"clip-{len(records)}", "record_type": "eventrec", "pipeline_uuid": pk})
            return 200, {"status": 200}
        if action in ("/start", "/stop", "/mtx/start"):
            return 200, None
        return 404, f"unexpected {method} {path}"


class CapacitySessionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.ledger = capacity.Ledger(Path(directory.name) / "capacity.jsonl.owned.json")
        self.station = FakeStation()
        self.log = []
        settle = mock.patch.object(capacity, "MEDIA_SETTLE_SECONDS", 0)
        settle.start()
        self.addCleanup(settle.stop)
        probe = mock.patch.object(capacity, "probe_smart_clip", return_value={
            "duration_seconds": 9, "decoded_frames": 270, "decoded_fps": 30, "fps_pass": True, "codec": "h264"})
        probe.start()
        self.addCleanup(probe.stop)
        for name, value in (("start_webrtc_probe", mock.Mock()),
                            ("finish_webrtc_probe", mock.Mock(return_value={
                                "delivered_fps": 30.0, "max_gap_seconds": 0.033, "pass": True}))):
            patcher = mock.patch.object(capacity, name, value)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def session(self, **options):
        return capacity.Session(self.station, FakeStation.AIP, "yolo26", self.ledger, signaling=self.station,
                                run="run1", warmup=0, window=0.03, interval=0.01, containers=(),
                                log=self.log.append, **options)

    def test_session_creates_run_scoped_media_refuses_existing_ones_and_deletes_its_own(self):
        foreign = self.station.add("media", "cap-run1-1080p-02")
        with self.assertRaisesRegex(capacity.CapacityError, "already exist"):
            self.session().create_media("rtsp://VCAM:PORT/cam{index:02d}", 2, "1080p")
        self.assertEqual(list(self.station.media), [foreign])

        self.station.media.clear()
        session = self.session()
        session.create_media("rtsp://VCAM:PORT/cam{index:02d}", 2, "1080p")
        owned = self.ledger.read()["media"]
        self.assertEqual(sorted(info["name"] for info in owned.values()), ["cap-run1-1080p-01", "cap-run1-1080p-02"])
        self.assertEqual(session.close(), [])
        self.assertEqual(self.station.media, {})
        self.assertFalse(self.ledger.path.exists())

    def test_camera_still_listed_after_a_200_deletion_is_retried_and_otherwise_kept_in_the_ledger(self):
        session = self.session()
        session.create_media("rtsp://VCAM:PORT/cam{index:02d}", 2, "1080p")
        first, second = session.media
        self.station.sticky = {first: 1, second: 5}
        errors = session.close()
        self.assertEqual(list(self.station.media), [second])
        self.assertEqual(list(self.ledger.read()["media"]), [second])
        self.assertIn("still listed", errors[0])

        self.station.sticky.clear()
        self.assertEqual(capacity.cleanup(self.station, None, self.ledger, log=self.log.append), [])
        self.assertEqual(self.station.media, {})
        self.assertFalse(self.ledger.path.exists())

    def test_smart_video_trial_triggers_probes_clips_and_deletes_what_it_wrote(self):
        session = self.session()
        session.create_media("rtsp://VCAM:PORT/cam{index:02d}", 2, "1080p")
        record = session.trial(2, capacity.parse_profile("evt"))

        self.assertTrue(record["pass"], record["reasons"])
        self.assertEqual(len(record["smart_video"]["clips"]), 1)
        self.assertEqual(len(record["smart_video"]["triggers"]), 1)
        self.assertEqual(record["smart_video"]["pre_duration_ms"], 3000)
        self.assertEqual(record["smart_video"]["post_duration_ms"], 6000)
        self.assertEqual(record["aip"]["unique_id"], "AIPV1EMI9026")
        self.assertTrue(record["model_verified"])
        self.assertTrue(record["samples"] and all("t" in sample for sample in record["samples"]))
        self.assertFalse(any(path.endswith("/mtx/start") for _, path, _ in self.station.calls))
        self.assertTrue(any(path.endswith("/eventrec/trigger") for _, path, _ in self.station.calls))
        self.assertNotIn("/recordings/start", [path for _, path, _ in self.station.calls])
        self.assertEqual(self.station.pipelines, {})
        self.assertEqual(self.station.recording, set())
        self.assertFalse(any(self.station.files.values()))
        self.assertFalse(self.station.records)
        self.assertEqual(self.ledger.read()["pipelines"], {})

    def test_smart_video_low_clip_fps_is_a_measured_failure(self):
        with mock.patch.object(capacity, "probe_smart_clip", return_value={
                "duration_seconds": 9, "decoded_frames": 245, "decoded_fps": 27.2,
                "fps_pass": False, "codec": "h264"}):
            session = self.session()
            session.create_media("rtsp://VCAM:PORT/cam{index:02d}", 1, "720p")
            record = session.trial(1, capacity.parse_profile("evt"))

        self.assertFalse(record["pass"])
        self.assertNotIn("error", record)
        self.assertIn("got h264 at 27.2 FPS", record["reasons"][0])
        self.assertEqual(record["smart_video"]["clips"][0]["decoded_frames"], 245)
        self.assertEqual(self.station.pipelines, {})
        self.assertFalse(self.station.records)

    def test_smart_video_clip_with_a_skipped_gop_is_a_measured_failure(self):
        with mock.patch.object(capacity, "probe_smart_clip", return_value={
                "duration_seconds": 9, "decoded_frames": 271, "decoded_fps": 30.1, "fps_pass": True,
                "codec": "h264", "max_gap_seconds": 0.966667, "first_packet_keyframe": True,
                "continuity_pass": False}):
            session = self.session()
            session.create_media("rtsp://VCAM:PORT/cam{index:02d}", 1, "720p")
            record = session.trial(1, capacity.parse_profile("evt"))
        self.assertFalse(record["pass"])
        self.assertNotIn("error", record)
        self.assertIn("largest gap 0.966667s", record["reasons"][-1])

    def test_webrtc_trial_fails_when_delivery_drops_frames_even_if_inference_keeps_up(self):
        self.finish_webrtc_probe.return_value = {"delivered_fps": 17.7, "max_gap_seconds": 0.2, "pass": False}
        session = self.session()
        session.create_media("rtsp://VCAM:PORT/cam{index:02d}", 1, "720p")
        record = session.trial(1, capacity.parse_profile("webrtc"))
        self.assertGreaterEqual(record["min_fps"], capacity.PASS_FPS)
        self.assertFalse(record["pass"])
        self.assertEqual(record["webrtc_delivery"]["delivered_fps"], 17.7)
        self.assertIn("got 17.7 FPS", record["reasons"][-1])
        self.start_webrtc_probe.assert_called_once()
        self.assertEqual(self.station.pipelines, {})

    def test_fast_camera_arrival_cannot_hide_slow_inference(self):
        self.station.perf = 10.0
        session = self.session()
        session.create_media("rtsp://VCAM:PORT/{index}", 2, "1080p")
        record = session.trial(2, capacity.Config())
        self.assertFalse(record["pass"])
        self.assertNotIn("error", record)
        self.assertEqual(record["min_fps"], 10)
        self.assertEqual(record["samples"][0]["inputs"]["0"]["arrival_fps"], 30)
        self.assertEqual(record["fps_metric"], capacity.PERF_METRIC)

    def test_missing_processed_metrics_invalidate_trial_instead_of_falling_back_to_arrival(self):
        session = self.session()
        session.create_media("rtsp://VCAM:PORT/{index}", 1, "1080p")
        original = self.station.respond

        def no_metrics(method, path, body, form):
            return (200, "") if path == "/metrics" else original(method, path, body, form)

        self.station.respond = no_metrics
        record = session.trial(1, capacity.Config())
        self.assertFalse(record["pass"])
        self.assertIn("cannot measure inference capacity", record["error"])
        self.assertEqual(self.station.pipelines, {})

    def test_failed_teardown_leaves_the_pipeline_in_the_ledger(self):
        session = self.session()
        session.create_media("rtsp://VCAM:PORT/cam{index:02d}", 1, "1080p")
        original = self.station.respond

        def refuse_delete(method, path, body, form):
            if method == "DELETE" and path.startswith("/pipeline/"):
                return 500, "busy"
            return original(method, path, body, form)

        self.station.respond = refuse_delete
        record = session.trial(1, capacity.parse_profile("webrtc"))
        self.assertIn("deleting the pipeline", record["cleanup_errors"][0])
        self.assertEqual(len(self.ledger.read()["pipelines"]), 1)
        self.assertTrue(session.close())
        self.assertEqual(len(self.station.media), 1)
        self.assertEqual(len(self.ledger.read()["media"]), 1)

    def test_partial_samples_survive_a_polling_error(self):
        session = self.session()
        session.create_media("rtsp://VCAM:PORT/{index}", 1, "1080p")
        original = self.station.respond
        polls = 0

        def fail_second_poll(method, path, body, form):
            nonlocal polls
            if path.endswith("/inputs/fps"):
                polls += 1
                if polls == 2:
                    return 503, "temporarily unavailable"
            return original(method, path, body, form)

        self.station.respond = fail_second_poll
        record = session.trial(1, capacity.Config())
        self.assertFalse(record["pass"])
        self.assertIn("503", record["error"])
        self.assertEqual(len(record["samples"]), 1)
        self.assertEqual(record["samples"][0]["inputs"]["0"]["fps"], 30)
        self.assertEqual(record["window_seconds"], session.window)
        self.assertEqual(self.station.pipelines, {})

    def test_failed_recording_stop_does_not_delete_files_still_being_written(self):
        self.station.broken.add(("POST", "/recordings/stop"))
        pk = self.station.add("pipelines", "cap-old-rec")
        self.station.pipelines[pk]["inputs"] = 1
        self.station.recording.add(pk)
        self.station.files[pk] = {"old.mp4"}
        self.ledger.add("pipelines", pk, name="cap-old-rec", profile="rec", run="old")
        self.assertTrue(capacity.cleanup(self.station, self.station, self.ledger, log=self.log.append))
        self.assertTrue(any(self.station.files.values()))
        self.assertFalse(any(method == "DELETE" and path.endswith("/files")
                             for method, path, _ in self.station.calls))
        self.assertEqual(len(self.ledger.read()["pipelines"]), 1)
        self.station.broken.clear()
        self.assertEqual(capacity.cleanup(self.station, self.station, self.ledger, log=self.log.append), [])
        self.assertFalse(any(self.station.files.values()))
        self.assertFalse(self.ledger.path.exists())
        self.assertEqual(capacity.cleanup(self.station, self.station, self.ledger, log=self.log.append), [])

    def test_pipeline_ms3_already_stopped_is_still_deleted_but_other_stop_errors_are_kept(self):
        session = self.session()
        session.create_media("rtsp://VCAM:PORT/{index}", 2, "1080p")
        original = self.station.respond
        answers = {"error_code": "E3307", "message": "Pipeline is not running"}

        def stop_refused(method, path, body, form):
            if path.endswith("/stop"):
                return 422, answers
            return original(method, path, body, form)

        self.station.respond = stop_refused
        record = session.trial(1, capacity.parse_profile("base"))
        self.assertNotIn("cleanup_errors", record)
        self.assertEqual((self.station.pipelines, self.ledger.read()["pipelines"]), ({}, {}))

        answers = {"error_code": "E3399", "message": "something else"}
        record = session.trial(1, capacity.parse_profile("base"))
        self.assertTrue(record["cleanup_errors"])
        self.assertEqual(len(self.ledger.read()["pipelines"]), 1)

    def test_event_deletion_conflict_keeps_recovery_entry(self):
        session = self.session()
        session.create_media("rtsp://VCAM:PORT/{index}", 1, "1080p")
        original = self.station.respond

        def busy_deletion(method, path, body, form):
            if path == "/pipeline/records/deletion":
                return 409, "another deletion job is running"
            return original(method, path, body, form)

        self.station.respond = busy_deletion
        record = session.trial(1, capacity.parse_profile("evt"))
        self.assertTrue(record["cleanup_errors"])
        self.assertEqual(len(self.ledger.read()["pipelines"]), 1)

    def test_cleanup_deletes_only_ledger_resources_and_keeps_failures_listed(self):
        mine = self.station.add("pipelines", "cap-run1-yolo26-base-n1-t01")
        self.station.pipelines[mine]["inputs"] = 1
        kept = self.station.add("media", "cap-run1-1080p-01")
        foreign_pipeline = self.station.add("pipelines", "cap-other-yolo26-base-n1-t01")
        renamed = self.station.add("media", "someone-else")
        self.ledger.add("pipelines", mine, name="cap-run1-yolo26-base-n1-t01", profile="base", run="run1")
        self.ledger.add("media", kept, name="cap-run1-1080p-01", run="run1")
        self.ledger.add("media", renamed, name="cap-run1-1080p-02", run="run1")
        self.station.broken.add(("DELETE", f"/media/{kept}"))

        errors = capacity.cleanup(self.station, None, self.ledger, log=self.log.append)

        self.assertNotIn(mine, self.station.pipelines)
        self.assertIn(foreign_pipeline, self.station.pipelines)
        self.assertIn(renamed, self.station.media)
        self.assertEqual(len(errors), 2)
        self.assertEqual(sorted(self.ledger.read()["media"]), sorted([kept, renamed]))
        self.assertTrue(any("cap-other-yolo26-base-n1-t01" in line for line in self.log))

    def test_ledger_survives_on_disk_until_empty(self):
        self.ledger.add("media", "m1", name="cap-run1-1080p-01", run="run1")
        self.assertEqual(json.loads(self.ledger.path.read_text())["media"]["m1"]["run"], "run1")
        self.ledger.remove("media", "m1")
        self.assertFalse(self.ledger.path.exists())

    def test_command_line_checks_the_aip_and_records_provenance(self):
        out = self.ledger.path.parent / "capacity.jsonl"
        common = ["--ms3", "http://MS3:8000", "--device", "ex3", "--aip", "AIPV1EMI9026",
                  "--rtsp-template", "rtsp://VCAM:PORT/{index}",
                  "--warmup", "0", "--window", "0.03", "--interval", "0.01", "--containers", "", "--out", str(out)]

        def run(*argv):
            with mock.patch.object(capacity, "Api", lambda base: self.station), \
                    redirect_stdout(io.StringIO()) as output, redirect_stderr(output):
                return capacity.main(list(argv)), output.getvalue()

        code, output = run("fixed", "--model", "yolov7", "--n", "1", *common)
        self.assertEqual(code, 1)
        self.assertIn("not YOLOv7-tiny", output)
        code, output = run("fixed", "--model", "yolo26", "--n", "1", "--profile", "rec", *common)
        self.assertEqual(code, 1)
        self.assertIn("excluded from benchmarks", output)
        self.assertFalse(out.exists())

        code, output = run("fixed", "--model", "yolo26", "--n", "2", *common)
        self.assertEqual(code, 0, output)
        record = capacity.load(out)[-1]
        self.assertEqual((record["kind"], record["aip"]["version"], record["model_verified"]), ("fixed", "1.0.0", True))
        self.assertTrue(record["trial"]["pass"] and record["trial"]["samples"])
        self.assertEqual(record["device"], "ex3")
        self.assertEqual(record["protocol"], capacity.PROTOCOL)
        self.assertEqual((self.station.media, self.station.pipelines), ({}, {}))
        self.assertFalse(self.ledger.path.exists())
        self.assertIn("AIPs: yolo26 = AIPV1EMI9026 1.0.0", capacity.summarize(capacity.load(out)))

    def test_invalid_cli_timings_and_camera_count_fail_before_station_access(self):
        common = ["fixed", "--model", "yolo26", "--ms3", "http://MS3:8000", "--device", "ex3", "--aip", "AIPV1EMI9026",
                  "--rtsp-template", "rtsp://VCAM:PORT/{index}", "--out", str(self.ledger.path.parent / "out")]
        for options in (["--n", "0"], ["--n", "1", "--window", "0"],
                        ["--n", "1", "--interval", "-1"], ["--n", "1", "--window", "nan"],
                        ["--n", "1", "--warmup", "-1"]):
            with self.subTest(options=options), mock.patch.object(capacity, "Api") as api, \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(capacity.main(common + options), 1)
                api.assert_not_called()

    def test_search_persists_api_failure_but_does_not_publish_capacity(self):
        output = self.ledger.path.parent / "capacity.jsonl"
        original = self.station.respond

        def failed_start(method, path, body, form):
            if path.endswith("/start"):
                return 503, "service unavailable"
            return original(method, path, body, form)

        self.station.respond = failed_start
        with mock.patch.object(capacity, "Api", return_value=self.station), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = capacity.main(["search", "--model", "yolo26", "--ms3", "http://MS3:8000", "--device", "ex3",
                                  "--aip", "AIPV1EMI9026", "--rtsp-template", "rtsp://VCAM:PORT/{index}",
                                  "--max-n", "2", "--containers", "", "--out", str(output)])
        self.assertEqual(code, 1)
        records = capacity.load(output)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["kind"], "trial")
        self.assertTrue(records[0]["error"])
        self.assertEqual((self.station.media, self.station.pipelines), ({}, {}))

    def test_preflight_rejects_old_api_and_occupied_station_without_writes(self):
        del self.station.paths["/pipeline/{uuid}/inputs/fps"]
        with self.assertRaisesRegex(capacity.CapacityError, "inputs/fps"):
            capacity.preflight(self.station, capacity.Config())
        self.station.paths["/pipeline/{uuid}/inputs/fps"] = {"get": {}}
        pk = self.station.add("pipelines", "existing-demo")
        self.station.pipelines[pk]["inputs"] = 1
        with self.assertRaisesRegex(capacity.CapacityError, "exclusive station"):
            capacity.preflight(self.station, capacity.Config())
        self.assertTrue(all(method == "GET" for method, _, _ in self.station.calls))

    def test_smart_video_no_clip_is_an_error_not_a_recording_pass(self):
        with self.assertRaisesRegex(capacity.CapacityError, "produced 0 clips"):
            capacity.verify_smart_recordings(self.station, "p1", capacity.Config(eventrec=True),
                                             [{"t": 0}], "ffprobe", deadline=0)
        with self.assertRaisesRegex(capacity.CapacityError, "never triggered"):
            capacity.verify_smart_recordings(self.station, "p1", capacity.Config(eventrec=True), [], "ffprobe")

    def test_smart_video_cleanup_waits_for_deletion_and_keeps_failed_cleanup_owned(self):
        pk = self.station.add("pipelines", "cap-smart")
        self.station.pipelines[pk]["inputs"] = 1
        self.station.records[pk] = [{"record_uuid": "clip", "pipeline_uuid": pk, "record_type": "eventrec"}]
        self.ledger.add("pipelines", pk, name="cap-smart", profile="evt", run="mine")
        original = self.station.respond

        def asynchronous_delete(method, path, body, form):
            if (method, path) == ("POST", "/pipeline/records/deletion"):
                return 200, {"message": "accepted, not completed"}
            return original(method, path, body, form)

        self.station.respond = asynchronous_delete
        with self.assertRaisesRegex(capacity.CapacityError, "remain"):
            capacity.delete_smart_recordings(self.station, pk, deadline=0)
        self.assertTrue(self.ledger.read()["pipelines"])
        self.station.respond = original
        self.assertEqual(capacity.cleanup(self.station, None, self.ledger, log=self.log.append), [])
        self.assertFalse(self.station.records)


def packets(count, *, skip=(), fps=30):
    """ffprobe packet entries for `count` frames at `fps`, keyframe every second, minus the `skip` indices."""
    return [{"pts_time": f"{index / fps:.6f}", "flags": "K__" if index % fps == 0 else "___"}
            for index in range(count) if index not in skip]


class SmartVideoProbeTests(unittest.TestCase):
    def test_packet_continuity_reports_the_issue_964_gap_and_a_non_keyframe_start(self):
        whole = capacity.packet_continuity(packets(270))
        self.assertTrue(whole["first_packet_keyframe"])
        self.assertAlmostEqual(whole["max_gap_seconds"], 1 / 30, places=5)
        self.assertTrue(capacity.gap_ok(whole["max_gap_seconds"]))
        skipped = capacity.packet_continuity(packets(270, skip=range(2, 30)))
        self.assertFalse(capacity.gap_ok(skipped["max_gap_seconds"]))
        self.assertTrue(capacity.gap_ok(capacity.packet_continuity(packets(270, skip=(5, 6)))["max_gap_seconds"]))
        self.assertFalse(capacity.packet_continuity(packets(270)[1:])["first_packet_keyframe"])
        with self.assertRaises(capacity.CapacityError):
            capacity.packet_continuity([{"pts_time": "N/A"}])

    def test_webrtc_probe_measures_delivered_cadence(self):
        for frames, skip, passes in ((601, (), True), (601, range(1, 600, 2), False)):
            with self.subTest(skip=skip):
                process = mock.Mock(returncode=0)
                process.communicate.return_value = (json.dumps({"packets": packets(frames, skip=skip)}), "")
                result = capacity.finish_webrtc_probe(process)
                self.assertEqual(result["pass"], passes)
        self.assertAlmostEqual(result["delivered_fps"], 15.0, places=1)

    def test_probe_requires_decodable_full_length_video_with_requested_dimensions(self):
        good = {"streams": [{"codec_type": "video", "width": 1280, "height": 720, "nb_read_frames": "270",
                             "codec_name": "h264"}], "format": {"duration": "9"}, "packets": packets(270)}
        variants = [(good, "", True), ({**good, "format": {"duration": "2"}}, "", False),
                    ({**good, "streams": []}, "", False), (good, "corrupt frame", False),
                    ({**good, "streams": [{**good["streams"][0], "width": 640}]}, "", False)]
        for data, stderr, valid in variants:
            with self.subTest(data=data, stderr=stderr), mock.patch.object(capacity.subprocess, "run") as run:
                run.return_value = mock.Mock(returncode=0, stderr=stderr, stdout=json.dumps(data))
                if valid:
                    probe = capacity.probe_smart_clip(capacity.Api("http://MS3:8000"),
                                                      {"record_uuid": "clip"}, capacity.Config(eventrec=True), "ffprobe")
                    self.assertEqual(probe["decoded_frames"], 270)
                    self.assertTrue(probe["fps_pass"])
                    self.assertTrue(probe["continuity_pass"])
                    self.assertIn("-count_frames", run.call_args.args[0])
                else:
                    with self.assertRaises(capacity.CapacityError):
                        capacity.probe_smart_clip(capacity.Api("http://MS3:8000"),
                                                 {"record_uuid": "clip"}, capacity.Config(eventrec=True), "ffprobe")

    def test_probe_keeps_decodable_low_fps_clip_as_measurement(self):
        data = {"streams": [{"codec_type": "video", "width": 1280, "height": 720, "nb_read_frames": "245",
                             "codec_name": "h264"}], "format": {"duration": "9"}, "packets": packets(245)}
        with mock.patch.object(capacity.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stderr="", stdout=json.dumps(data))
            probe = capacity.probe_smart_clip(capacity.Api("http://MS3:8000"),
                                              {"record_uuid": "clip"}, capacity.Config(eventrec=True), "ffprobe")
        self.assertFalse(probe["fps_pass"])
        self.assertAlmostEqual(probe["decoded_fps"], 245 / 9)


class CapacityPlanTests(unittest.TestCase):
    def test_model_names_match_flavors_and_identify_the_aip(self):
        for flavor_id in flavors.flavor_ids(ROOT):
            with self.subTest(flavor=flavor_id):
                manifest = json.loads((ROOT / "flavors" / flavor_id / "flavor.json").read_text())
                self.assertEqual(capacity.MODEL_NAMES[flavor_id], manifest["display_name"])
        legacy = {"name": "Yolov7 tiny", "primary_processor": {"model": "Yolov7"}}
        self.assertTrue(capacity.model_matches(legacy, "yolov7"))
        self.assertFalse(capacity.model_matches(FakeStation.AIP, "yolov7"))
        self.assertTrue(capacity.model_matches(FakeStation.AIP, "yolo26"))
        conflicting = {**FakeStation.AIP, "primary_processor": {"model": "YOLOv7-tiny"}}
        self.assertFalse(capacity.model_matches(conflicting, "yolo26"))

    def test_summary_keeps_camera_sets_versions_and_verification_distinct(self):
        def records(camera, version, verified, count):
            identity = {"model": "yolo26", "profile": "base", "camera_tag": camera,
                        "aip": {**FakeStation.AIP, "version": version}, "model_verified": verified}
            trials = [{"kind": "trial", **identity, "n": n, "pass": n <= count, "min_fps": 30.0 if n <= count else 25.0}
                      for n in (count, count + 1)]
            return [*trials, {"kind": "capacity", **identity, "capacity": count, "bounded": True, "trials": trials}]

        text = capacity.summarize([*records("720p", "1.0", True, 8), *records("1080p", "1.0", True, 6),
                                   *records("1080p", "2.0", True, 5), *records("1080p", "2.0", False, 4)])
        self.assertIn("Camera set: 720p", text)
        self.assertIn("Camera set: 1080p", text)
        for row in ("yolo26 | AIPV1EMI9026 1.0 | 8", "yolo26 | AIPV1EMI9026 1.0 | 6",
                    "yolo26 | AIPV1EMI9026 2.0 | 5", "yolo26 (unverified) | AIPV1EMI9026 2.0 | 4"):
            self.assertIn(row, text)

    def test_supported_capacity_counts_failed_rechecks_below_the_search_result(self):
        def trial(n, passed):
            return {"n": n, "pass": passed}

        self.assertEqual(capacity.supported_capacity([trial(6, True), trial(7, True), trial(8, False)]),
                         (7, True, False))
        self.assertEqual(capacity.supported_capacity([trial(6, True), trial(7, True), trial(8, False),
                                                      trial(7, False), trial(6, True)]), (6, True, True))
        self.assertEqual(capacity.supported_capacity([trial(4, True), trial(5, True)]), (5, False, False))
        self.assertIsNone(capacity.supported_capacity([trial(1, False)]))

    def test_summary_uses_rechecks_but_excludes_invalid_repeat_attempts(self):
        identity = {"model": "yolo26", "profile": "base", "camera_tag": "720p",
                    "aip": FakeStation.AIP, "model_verified": True}
        records = [{"kind": "capacity", **identity, "capacity": 7, "bounded": True}]
        records += [{"kind": "trial", **identity, "n": n, "pass": passed, "min_fps": fps}
                    for n, passed, fps in ((6, True, 30), (7, True, 29.7), (8, False, 26))]
        records += [
            {"kind": "fixed", **identity, "trial": {"n": 7, "pass": False, "min_fps": 28.9}},
            {"kind": "fixed", **identity, "trial": {
                "n": 6, "pass": False, "min_fps": 29.1, "attempts": [
                    {"n": 6, "pass": True, "min_fps": 29.1},
                    {"n": 6, "pass": False, "min_fps": 30, "cleanup_errors": ["media still listed"]},
                ]}},
        ]
        text = capacity.summarize(records)
        self.assertIn("| yolo26 | AIPV1EMI9026 1.0.0 | 6†", text)
        self.assertIn("A higher count passed in at least one trial but failed in another", text)
        self.assertIn("| 6 | 29.10 | invalid |", text)
        self.assertIn("Excluded 1 invalid trial", text)

    def test_published_tables_describe_profiles_and_flag_contested_counts(self):
        def run(profile, outcomes):
            return [{"kind": "trial", "model": "yolo26", "profile": profile, "camera_tag": "720p",
                     "n": n, "pass": passed, "min_fps": fps} for n, passed, fps in outcomes]

        records = [*run("base", ((6, True, 30.1), (7, True, 29.2), (7, False, 28.8))),
                   *run("webrtc+evt-osd", ((6, True, 30.0), (7, False, 27.5)))]
        self.assertEqual(capacity.describe("webrtc-osd-1M"), "WebRTC + OSD, 1 Mbps")
        table = capacity.camera_table(records, ("yolo26",),
                                      [("720p", "base"), ("720p", "webrtc+evt-osd"), ("1080p", "base")])
        self.assertIn("| Model | 720p, no output | 720p, WebRTC + Smart Video recording + OSD | 1080p, no output |",
                      table)
        self.assertIn("| YOLO26n | 6† | 6 | - |", table)
        self.assertIn("† A higher count passed in one trial and failed in another", table)
        headroom = capacity.headroom_table(records, ("yolo26",), profiles=("base", "webrtc+evt-osd"))
        self.assertIn("| YOLO26n | 7 | 29.20 (passed), 28.80 | 27.50 |", headroom)
        with self.assertRaises(capacity.CapacityError):
            capacity.headroom_table([*records, *run("evt-osd", ((5, True, 30.0), (6, False, 28.0)))],
                                    ("yolo26",), profiles=("base", "webrtc+evt-osd", "evt-osd"))

    def test_output_failure_at_one_is_not_reported_as_a_camera_limit(self):
        identity = {"model": "yolo26", "profile": "evt-osd", "camera_tag": "720p"}
        records = [
            {"kind": "capacity", **identity, "capacity": 0, "bounded": True},
            {"kind": "trial", **identity, "n": 1, "pass": False, "min_fps": 30.09,
             "reasons": ["Smart Video clip must contain H.264 at 29+ decoded FPS; got h264 at 26.9 FPS"]},
        ]
        table = capacity.camera_table(records, ("yolo26",), [("720p", "evt-osd"), ("1080p", "evt-osd")])
        self.assertIn("| YOLO26n | Not qualified at 1 | - |", table)
        self.assertIn("Not qualified at 1", capacity.summarize(records))

    def test_plan_covers_every_packaged_flavor_and_prunes_factorial(self):
        self.assertEqual(set(capacity.MODELS), set(flavors.flavor_ids(ROOT)))
        factorial = capacity.full_factorial()
        totals = capacity.plan_totals()
        self.assertGreater(factorial["raw"], factorial["distinct"])
        self.assertLess(totals["trials"], factorial["distinct"])
        self.assertEqual(totals["searches"], 12)
        for name in (*capacity.SCREENING, *capacity.CAPACITY_PROFILES):
            capacity.validate_live_profile(capacity.parse_profile(name))
        self.assertIn("0 Pilot", [row[0] for row in capacity.plan()])

    def test_published_tables_cannot_mix_devices_or_historical_protocols(self):
        trial = {"kind": "trial", "model": "yolo26", "profile": "base", "camera_tag": "720p",
                 "n": 2, "pass": True, "min_fps": 30.0}
        for other in ({"device": "ex3"}, {"protocol": capacity.PROTOCOL}):
            with self.subTest(other=other):
                with self.assertRaisesRegex(capacity.CapacityError, "one device"):
                    capacity.camera_table([trial, {**trial, **other}])

    def test_published_capacity_results_have_raw_boundary_evidence(self):
        records = capacity.load(ROOT / "doc/benchmarks/orin-nx-16gb-jp6.0/capacity.jsonl")
        trials = [row["trial"] if row["kind"] == "fixed" else row
                  for row in records if row["kind"] in ("trial", "fixed")]
        self.assertTrue(trials)
        for trial in trials:
            attempts = trial.get("attempts") or [trial]
            for attempt in attempts:
                with self.subTest(pipeline=attempt["pipeline"]):
                    capacity.require_valid_trial(attempt)
                    judged = capacity.evaluate(attempt["samples"], attempt["n"])
                    self.assertEqual(judged["pass"], attempt["pass"])
                    self.assertAlmostEqual(judged["min_fps"], attempt["min_fps"])
            self.assertEqual(trial["pass"], all(attempt["pass"] for attempt in attempts))
            self.assertAlmostEqual(trial["min_fps"], min(attempt["min_fps"] for attempt in attempts))
        for result in (row for row in records if row["kind"] == "capacity"):
            with self.subTest(model=result["model"], profile=result["profile"]):
                boundary = {trial["n"]: trial for trial in trials
                            if (trial["run"], trial["model"], trial["profile"]) ==
                            (result["run"], result["model"], result["profile"])}
                self.assertTrue(boundary[result["capacity"]]["pass"])
                if result["bounded"]:
                    self.assertFalse(boundary[result["capacity"] + 1]["pass"])
        published = {"yolo26": 7, "yolov10": 6, "yolov7": 4}
        cells = [(camera_tag, model, profile) for model in published
                 for camera_tag, profile in (("720p", "base"), ("1080p", "base"))]
        for camera_tag, model, profile in cells:
            with self.subTest(camera=camera_tag, model=model, profile=profile):
                evidence = [row["trial"] if row["kind"] == "fixed" else row for row in records
                            if row["kind"] in ("trial", "fixed")
                            and (row["camera_tag"], row["model"], row["profile"]) == (camera_tag, model, profile)]
                count, bounded, _ = capacity.supported_capacity(evidence)
                self.assertEqual((count, bounded), (published[model], True))
                passing_boundary = [trial for trial in evidence if trial["n"] == count and trial["pass"]]
                self.assertGreaterEqual(len(passing_boundary), 2)
                for trial in passing_boundary:
                    for attempt in trial.get("attempts") or [trial]:
                        self.assertEqual((attempt["warmup_seconds"], attempt["window_seconds"]), (45, 120))

    def test_output_diagnostics_keep_startup_failures_separate_from_capacity(self):
        data = json.loads((ROOT / "doc/benchmarks/orin-nx-16gb-jp6.0/capacity-outputs.json").read_text())
        successes, failures = [], []
        for check in data["checks"]:
            probe, record = check["probe"], check["result"]
            trial = record["trial"]
            with self.subTest(pipeline=probe["pipeline_pk"]):
                self.assertEqual((probe["profile"], probe["n"]), (record["profile"], trial["n"]))
                self.assertEqual((trial["warmup_seconds"], trial["window_seconds"]), (30, 90))
                if probe["harness_rc"] == 0:
                    capacity.require_valid_trial(trial)
                    judged = capacity.evaluate(trial["samples"], trial["n"])
                    self.assertTrue(trial["pass"] and judged["pass"])
                    self.assertAlmostEqual(judged["min_fps"], trial["min_fps"])
                    successes.append(check)
                else:
                    self.assertFalse(trial["pass"])
                    with self.assertRaises(capacity.CapacityError):
                        capacity.require_valid_trial(trial)
                    failures.append(check)
                for path, output in probe["mtx_paths"].items():
                    self.assertTrue(probe["paths_matched_by_pk"])
                    self.assertTrue(path.startswith(probe["pipeline_pk"]))
                    if output["ffprobe"]["rc"] == 0:
                        stream = output["ffprobe"]["stream"]
                        self.assertEqual((stream["width"], stream["height"]),
                                         capacity.parse_profile(probe["profile"]).tiler)
        self.assertTrue(successes and failures)

    def test_capacity_artifacts_contain_no_private_station_references(self):
        root = ROOT / "doc/benchmarks"
        for path in [root / "capacity/README.md", *sorted(root.glob("orin-*/capacity.*")),
                     *sorted(root.glob("orin-*/capacity-*.json*"))]:
            with self.subTest(file=path.name):
                assert_no_private_references(self, path.read_text(), str(path.relative_to(root)))


if __name__ == "__main__":
    unittest.main()
