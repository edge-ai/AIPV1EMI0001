import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
if NUMPY_AVAILABLE:
    importlib.import_module("numpy")


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is required for callback tests")
class CallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = {
            name: types.ModuleType(name)
            for name in (
                "aiprocessors",
                "aiprocessors.callback",
                "aiprocessors.callback.custom",
                "common",
                "common.logmanager",
            )
        }
        modules["aiprocessors.callback.custom"].CustomCallback = type("CustomCallback", (), {})
        cls.logger = Mock()
        modules["common.logmanager"].get_logger = lambda **_kwargs: cls.logger
        spec = importlib.util.spec_from_file_location("aip_callback_test", ROOT / "callback.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        cls.callback_class = module.UserCustomCallback

    def setUp(self):
        self.logger.reset_mock()
        self.callback = self.callback_class()
        self.callback.pgie_classes = ["car", "person", "truck"]
        self.callback.input_sources = {}
        self.callback.initialize()

    @staticmethod
    def frame(source_id=0, number=1, classes=(), analytics=None):
        return {
            "source_id": source_id,
            "frame_num": number,
            "num_obj_meta": len(classes),
            "ntp_timestamp": number * 1000,
            "obj_meta_list": [{"class_id": class_id} for class_id in classes],
            "frame_user_meta_list": [] if analytics is None else [analytics],
        }

    def batch(self, max_frames, *frames):
        return self.callback.parse_batch_meta({
            "batch_meta": {
                "max_frames_in_batch": max_frames,
                "num_frames_in_batch": len(frames),
                "frame_meta_list": list(frames),
            }
        })

    def merge(self, *batches):
        return self.callback.merge_and_analyze(list(batches), list(range(len(batches))))

    def test_intermittent_analytics_are_aggregated_in_either_order(self):
        analytics = {
            "objInROIcnt": {"yard": 2},
            "objLCCurrCnt": {"entry": 1},
        }
        present = self.batch(1, self.frame(number=1, analytics=analytics))
        absent = self.batch(1, self.frame(number=2))

        for batches in ((present, absent), (absent, present)):
            with self.subTest(order=batches):
                result = self.merge(*batches)[0]["analytics_obj_cnt"]
                self.assertEqual(result["objInROIcnt"]["yard"],
                                 {"max_val": 2, "min_val": 2, "sum_val": 2})
                self.assertEqual(result["objLCCurrCnt"]["entry"],
                                 {"max_val": 1, "min_val": 1, "sum_val": 1})

    def test_analytics_labels_use_only_observed_values(self):
        first = self.batch(1, self.frame(analytics={
            "objInROIcnt": {"yard": 3},
            "objLCCurrCnt": {"entry": 2},
        }))
        second = self.batch(1, self.frame(number=2, analytics={
            "objInROIcnt": {"yard": 0, "gate": 4},
            "objLCCurrCnt": {"exit": 1},
        }))
        result = self.merge(first, second)[0]["analytics_obj_cnt"]
        self.assertEqual(result["objInROIcnt"], {
            "yard": {"max_val": 3, "min_val": 0, "sum_val": 3},
            "gate": {"max_val": 4, "min_val": 4, "sum_val": 4},
        })
        self.assertEqual(result["objLCCurrCnt"], {
            "entry": {"max_val": 2, "min_val": 2, "sum_val": 2},
            "exit": {"max_val": 1, "min_val": 1, "sum_val": 1},
        })
        self.assertEqual(self.merge(self.batch(1, self.frame(number=3)))[0]["analytics_obj_cnt"],
                         {"objInROIcnt": {}, "objLCCurrCnt": {}, "objLCCumCnt": {}})

    def test_cumulative_line_crossings_report_latest_running_total(self):
        frame = self.batch(1, self.frame(number=1, analytics={
            "objLCCurrCnt": {"entry": 1},
            "objLCCumCnt": {"entry": 41},
        }))
        self.assertEqual(frame[0]["objLCCumCnt"], {"entry": 41})
        later = self.batch(1, self.frame(number=2, analytics={
            "objLCCurrCnt": {"entry": 2},
            "objLCCumCnt": {"entry": 43},
        }))
        result = self.merge(frame, later)[0]["analytics_obj_cnt"]
        self.assertEqual(result["objLCCumCnt"], {
            "entry": {"max_val": 43, "min_val": 41, "last_val": 43},
        })
        self.assertEqual(result["objLCCurrCnt"]["entry"]["sum_val"], 3)
        self.assertIsNone(self.batch(2, self.frame())[1]["objLCCumCnt"])

    def test_missing_source_has_unknown_counts_not_zero(self):
        batch = self.batch(2, self.frame(classes=(0, 0, 1)))
        self.assertTrue(batch[0]["available"])
        self.assertFalse(batch[1]["available"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = self.merge(batch)
        self.assertTrue(result[0]["available"])
        self.assertEqual(result[0]["obj_counter"]["car"]["max_val"], 2)
        missing = result[1]
        self.assertFalse(missing["available"])
        self.assertIsNone(missing["frame_number_from"])
        self.assertIsNone(missing["frame_number_to"])
        self.assertTrue(all(value is None for statistics in missing["obj_counter"].values()
                            for value in statistics.values()))
        json.dumps(result, allow_nan=False)

    def test_missing_frames_do_not_override_observed_frames_or_timestamps(self):
        first = self.batch(2, self.frame(0, 10, classes=(0, 0)))
        middle = self.batch(2, self.frame(1, 11, classes=(1,)))
        last = self.batch(2, self.frame(0, 12, classes=(0,)))
        result = self.merge(first, middle, last)
        source = result[0]
        self.assertTrue(source["available"])
        self.assertEqual((source["frame_number_from"], source["frame_number_to"]), (10, 12))
        self.assertEqual((source["ntp_timestamp_from"], source["ntp_timestamp_to"]), (10000, 12000))
        self.assertEqual(source["obj_counter"]["car"], {
            "max_val": 2, "min_val": 1, "mean_val": 2, "median_val": 2, "count_val": 3,
        })
        self.assertEqual(result[1]["obj_counter"]["person"]["max_val"], 1)

    def test_window_handles_different_batch_capacities(self):
        first = self.batch(2, self.frame(0, 1), self.frame(1, 1, classes=(2,)))
        second = self.batch(1, self.frame(0, 2, classes=(0,)))
        result = self.merge(first, second)
        self.assertEqual(result[0]["obj_counter"]["car"]["max_val"], 1)
        self.assertEqual(result[1]["obj_counter"]["truck"]["max_val"], 1)

    def test_only_configured_sparse_sources_are_reported(self):
        self.callback.input_sources = {2: "Entrance", 5: "Exit"}
        first = self.batch(8, self.frame(2, 1, classes=(0,)))
        self.assertEqual([frame["source_id"] for frame in first], [2, 5])
        self.assertTrue(first[0]["available"])
        self.assertFalse(first[1]["available"])
        second = self.batch(2, self.frame(5, 2, classes=(1,)))
        result = self.merge(first, second)
        self.assertEqual(set(result), {2, 5})
        self.assertEqual(result[2]["obj_counter"]["car"]["max_val"], 1)
        self.assertEqual(result[5]["obj_counter"]["person"]["max_val"], 1)

    def test_configured_sources_exclude_unused_batch_capacity(self):
        self.callback.input_sources = {0: "Only camera"}
        result = self.merge(self.batch(8, self.frame(0, 1, classes=(0,))))
        self.assertEqual(set(result), {0})
        self.assertTrue(result[0]["available"])

    def test_unknown_class_and_source_ids_are_reported_not_miscounted(self):
        batch = self.batch(2, self.frame(0, classes=(-1, 0, 3)),
                           self.frame(-1, classes=(1,)))
        self.assertEqual(batch[0]["obj_counter"], [1, 0, 0])
        self.assertFalse(batch[1]["available"])
        self.assertGreaterEqual(self.logger.warning.call_count, 2)

    def test_empty_window_and_missing_class_labels_are_serializable(self):
        self.assertEqual(self.merge(), {})
        self.callback.pgie_classes = None
        result = self.merge(self.batch(1, self.frame()))
        self.assertEqual(result[0]["obj_counter"], {})
        json.dumps(result, allow_nan=False)

    def test_invalid_or_incomplete_metadata_is_reported(self):
        for payload in (None, {}, {"batch_meta": {"max_frames_in_batch": -1,
                                                  "num_frames_in_batch": 0,
                                                  "frame_meta_list": []}}):
            with self.subTest(payload=payload):
                self.assertEqual(self.callback.parse_batch_meta(payload), [])
        incomplete = self.frame()
        del incomplete["obj_meta_list"]
        self.assertFalse(self.batch(1, incomplete)[0]["available"])
        self.assertGreaterEqual(self.logger.warning.call_count, 4)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for callback-to-flow test")
    def test_real_callback_output_preserves_missing_data_through_flow(self):
        result = self.merge(self.batch(2, self.frame(0, classes=(0, 0, 1))))
        script = """
            const fs = require("node:fs");
            const flow = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
            const body = flow.find((node) => node.id === "aipv1emi0001normalize").func;
            const normalize = new Function("msg", "node", "env", body);
            let input = "";
            process.stdin.on("data", (chunk) => input += chunk);
            process.stdin.on("end", () => {
                const message = { payload: JSON.parse(input), eais: { pipelineId: "test" } };
                const result = normalize(message, { warn: (warning) => { throw Error(warning); } },
                    { get: () => undefined });
                process.stdout.write(JSON.stringify(result.payload));
            });
        """
        process = subprocess.run(
            ["node", "-e", script, str(ROOT / "logic/flows.json")],
            input=json.dumps(result, allow_nan=False), capture_output=True, text=True, check=True,
        )
        data = json.loads(process.stdout)
        self.assertEqual(data["sources"][0]["total"], 3)
        self.assertFalse(data["sources"][1]["available"])
        self.assertIsNone(data["sources"][1]["counts"]["car"])
        self.assertIsNone(data["totals"]["objects"])


if __name__ == "__main__":
    unittest.main()
