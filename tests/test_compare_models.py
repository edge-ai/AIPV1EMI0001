import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import compare_models  # noqa: E402
import flavors  # noqa: E402

BATCHES = (1, 4, 8)
VARIANTS = ("ONNX Runtime CUDA FP32", "TensorRT FP32", "TensorRT FP16")


def latency_report(scale):
    results = []
    for index, variant in enumerate(VARIANTS):
        for batch in BATCHES:
            median = scale * batch * (3.0 - index)
            results.append({"variant": variant, "batch": batch, "median_ms": median, "ms_per_image": median / batch,
                            "serial_images_per_second": batch * 1000 / median, "speedup_vs_onnxruntime": 3.0 / (3.0 - index)})
    return {"versions": {"tensorrt_python": "8.6.2.3", "onnxruntime": "1.19.0"}, "results": results}


def accuracy_report(fp32, fp16):
    def metrics(value):
        return {"map_50_95": value, "ap_50": value + 0.15, "aip_map_50_95": value + 0.02,
                "ap_50_95_by_class": {"car": value + 0.1, "person": value, "truck": value - 0.05}}
    results = [{"variant": variant, "batch": batch, "metrics": metrics(value)}
               for variant, value in (("TensorRT FP32", fp32), ("TensorRT FP16", fp16)) for batch in (1, 8)]
    return {"dataset": {"image_count": 5000}, "platform": {"l4t": "# R36 (release), REVISION: 3.0"}, "results": results}


class CompareModelsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def publish(self, flavor_id, scale, fp32, fp16):
        (self.path / f"{flavor_id}-latency.json").write_text(json.dumps(latency_report(scale)))
        (self.path / f"{flavor_id}-accuracy.json").write_text(json.dumps(accuracy_report(fp32, fp16)))

    def test_collects_the_measured_flavors_in_repository_order(self):
        self.publish("yolo26", 1.0, 0.399, 0.399)
        self.publish("yolov7", 2.0, 0.371, 0.370)
        entries = compare_models.collect(self.path, ROOT)
        self.assertEqual([entry["flavor"]["id"] for entry in entries],
                         [flavor_id for flavor_id in flavors.flavor_ids(ROOT) if flavor_id in ("yolov7", "yolo26")])

    def test_stops_when_nothing_was_measured(self):
        with self.assertRaises(compare_models.CompareError):
            compare_models.collect(self.path, ROOT)

    def test_a_flavor_with_only_one_report_is_left_out(self):
        self.publish("yolo26", 1.0, 0.399, 0.399)
        (self.path / "yolov7-latency.json").write_text(json.dumps(latency_report(2.0)))
        self.assertEqual([entry["flavor"]["id"] for entry in compare_models.collect(self.path, ROOT)], ["yolo26"])

    def test_tables_copy_the_measured_numbers(self):
        self.publish("yolo26", 1.0, 0.399, 0.397)
        entries = compare_models.collect(self.path, ROOT)
        accuracy = compare_models.accuracy_table(entries)
        self.assertIn("| YOLO26n | 40.1 | 39.70 | 54.70 | 41.70 | 49.70 | 39.70 | 34.70 |", accuracy)
        self.assertIn("| YOLO26n | 8 | 39.90 | 39.70 | -0.20 |", compare_models.parity_table(entries))
        self.assertIn("| YOLO26n | 8 | 24.0 ms | 16.0 ms | 8.0 ms |", compare_models.precision_table(entries))
        self.assertIn("1.00 (1000 img/s, 3.0x ORT)", compare_models.latency_table(entries))

    def test_a_flavor_without_a_published_figure_shows_not_available(self):
        self.publish("yolov7", 2.0, 0.371, 0.370)
        entry = compare_models.collect(self.path, ROOT)[0]
        self.assertIsNone(entry["flavor"]["reference"]["coco_map50_95"])
        self.assertIn("| YOLOv7-tiny | n/a | 37.00 |", compare_models.accuracy_table([entry]))

    def test_the_report_states_where_and_how_it_was_measured(self):
        self.publish("yolo26", 1.0, 0.399, 0.399)
        text = compare_models.render_markdown(compare_models.collect(self.path, ROOT))
        for expected in ("R36 (release), REVISION: 3.0", "TensorRT 8.6.2.3", "ONNX Runtime 1.19.0", "5000 COCO val2017 images",
                         "latency uses synthetic GPU-resident inputs", "separate runs",
                         "mAP 50:95 is the metric Ultralytics reports",
                         "- **YOLO26n** (40.1): Ultralytics, mAP val 50-95 (e2e) column"):
            self.assertIn(expected, text)

    def test_a_missing_variant_is_reported(self):
        self.publish("yolo26", 1.0, 0.399, 0.399)
        report = json.loads((self.path / "yolo26-latency.json").read_text())
        report["results"] = [item for item in report["results"] if item["variant"] != "TensorRT FP16"]
        (self.path / "yolo26-latency.json").write_text(json.dumps(report))
        with self.assertRaises(compare_models.CompareError):
            compare_models.latency_table(compare_models.collect(self.path, ROOT))

    def test_the_command_writes_the_markdown_and_tolerates_a_missing_chart_library(self):
        self.publish("yolo26", 1.0, 0.399, 0.399)
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = compare_models.main([str(self.path), "--root", str(ROOT)])
        self.assertEqual(status, 0, errors.getvalue())
        self.assertTrue((self.path / "comparison.md").read_text().startswith("# Model comparison"))

    def test_measurement_conditions_are_copied_into_the_report_when_present(self):
        self.publish("yolo26", 1.0, 0.399, 0.399)
        entries = compare_models.collect(self.path, ROOT)
        self.assertNotIn("Measurement conditions", compare_models.render_markdown(entries, compare_models.conditions(self.path)))
        (self.path / "conditions.md").write_text("A video pipeline shared the GPU.\n")
        text = compare_models.render_markdown(entries, compare_models.conditions(self.path))
        self.assertIn("## Measurement conditions\n\nA video pipeline shared the GPU.\n", text)

    def test_the_summary_table_copies_accuracy_and_the_smallest_and_largest_batch(self):
        self.publish("yolo26", 1.0, 0.399, 0.397)
        self.publish("yolov7", 2.0, 0.371, 0.370)
        table = compare_models.summary_table(compare_models.collect(self.path, ROOT))
        self.assertIn("FP16 ms per image, batch 1 | FP16 ms per image, batch 8 |", table)
        self.assertIn("| YOLOv7-tiny | 37.00 (n/a) | 39.00 | 2.0 | 2.0 |", table)
        self.assertIn("| YOLO26n | 39.70 (40.1) | 41.70 | 1.0 | 1.0 |", table)

    def test_the_command_fails_without_measurements(self):
        errors = StringIO()
        with redirect_stderr(errors):
            self.assertEqual(compare_models.main([str(self.path), "--root", str(ROOT)]), 1)
        self.assertIn("ERROR", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
