import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
AVAILABLE = all(importlib.util.find_spec(name) for name in
                ("numpy", "cv2", "onnxruntime", "pycocotools"))

if AVAILABLE:
    import cv2
    import numpy as np
    import accuracy
    import benchmark


@unittest.skipUnless(AVAILABLE, "accuracy evaluation dependencies are not installed")
class AccuracyTests(unittest.TestCase):
    def dataset(self, directory):
        root = Path(directory)
        images = []
        annotations = []
        for index, (category, color) in enumerate(((3, 80), (1, 160), (8, 240)), start=1):
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[:, :, 2] = color
            filename = f"{index}.png"
            self.assertTrue(cv2.imwrite(str(root / filename), image))
            images.append({"id": index, "file_name": filename, "width": 100, "height": 100})
            annotations.append({"id": index, "image_id": index, "category_id": category,
                                "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0})
        path = root / "labels.json"
        path.write_text(json.dumps({
            "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"},
                           {"id": 8, "name": "truck"}],
        }))
        return path

    def test_preprocess_and_inverse_box(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            pixels = np.zeros((100, 200, 3), np.uint8)
            pixels[:, :, 2] = 255
            self.assertTrue(cv2.imwrite(str(path), pixels))
            image = {"id": 1, "file_name": path.name, "width": 200, "height": 100}
            tensor, transform = accuracy.prepare_image(directory, image)
            self.assertEqual(tensor.shape, (3, 640, 640))
            self.assertEqual(transform, (3.2, 3.2, 200, 100))
            self.assertEqual(tensor[:, 0, 0].tolist(), [1.0, 0.0, 0.0])
            self.assertEqual(float(tensor[:, 400, :].max()), 0.0)
            self.assertEqual(accuracy.original_box([160, 80, 480, 240], transform), [50, 25, 100, 50])
            self.assertEqual(accuracy.original_box([-10, -10, 640, 320], transform), [0, 0, 200, 100])
            self.assertIsNone(accuracy.original_box([10, 10, 10, 30], transform))
            with self.assertRaisesRegex(ValueError, "dimensions"):
                accuracy.prepare_image(directory, {**image, "width": 201})

    def test_raw_nms_is_class_aware_and_uses_center_size(self):
        boxes = np.zeros((benchmark.CANDIDATES, 4), np.float32)
        scores = np.zeros((benchmark.CANDIDATES, 3), np.float32)
        boxes[0] = boxes[1] = [320, 160, 320, 160]
        scores[0, :2] = [0.9, 0.5]
        scores[1, 0] = 0.8
        predictions = accuracy.raw_detections(boxes, scores, 7, (3.2, 3.2, 200, 100),
                                              {"car": 3, "person": 1, "truck": 8})
        self.assertEqual([p["category_id"] for p in predictions], [3, 1])
        self.assertEqual(predictions[0]["bbox"], [50, 25, 100, 50])
        boxes[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            accuracy.raw_detections(boxes, scores, 7, (3.2, 3.2, 200, 100),
                                    {"car": 3, "person": 1, "truck": 8})

    def test_dataset_mapping_fingerprint_selection_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = self.dataset(directory)
            coco, images, categories, dataset = accuracy.select_dataset(labels, directory, 2)
            self.assertEqual([image["id"] for image in images], [1, 2])
            self.assertEqual(categories, {"car": 3, "person": 1, "truck": 8})
            self.assertEqual(dataset["non_crowd_ground_truth_by_class"],
                             {"car": 1, "person": 1, "truck": 0})
            self.assertEqual(len(dataset["image_content_sha256"]), 64)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                accuracy.select_dataset(labels, directory, 4)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                accuracy.image_path(directory, {"file_name": "../other.png"})
            self.assertEqual(len(coco.imgs), 3)

    def test_metrics_use_coco_matching_and_operating_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = self.dataset(directory)
            coco, images, categories, _ = accuracy.select_dataset(labels, directory, 0)
            predictions = [
                {"image_id": index, "category_id": category, "bbox": [10, 10, 20, 20], "score": 0.9}
                for index, category in ((1, 3), (2, 1), (3, 8))
            ]
            predictions.extend([
                {"image_id": 1, "category_id": 3, "bbox": [60, 60, 20, 20], "score": 0.8},
                {"image_id": 2, "category_id": 1, "bbox": [60, 60, 20, 20], "score": 0.2},
            ])
            result = accuracy.evaluate(coco, images, categories, predictions)
            self.assertAlmostEqual(result["map_50_95"], 1.0)
            car = result["classes"]["car"]
            person = result["classes"]["person"]
            self.assertEqual((car["ground_truth"], car["true_positives"], car["false_positives"]),
                             (1, 1, 1))
            self.assertEqual((car["precision_at_035_iou_050"], car["recall_at_035_iou_050"]), (0.5, 1.0))
            self.assertEqual((person["precision_at_035_iou_050"], person["recall_at_035_iou_050"]), (1.0, 1.0))
            truncated = accuracy.evaluate(coco, images, categories, predictions, standard_ap=False)
            self.assertIsNone(truncated["map_50_95"])
            self.assertEqual(truncated["classes"]["car"]["recall_at_035_iou_050"], 1.0)
            empty = accuracy.evaluate(coco, images, categories, [])
            self.assertEqual(empty["map_50_95"], 0.0)
            self.assertEqual(empty["classes"]["car"]["false_negatives"], 1)
            self.assertIsNone(empty["classes"]["car"]["precision_at_035_iou_050"])

    def test_per_class_ap_uses_category_ids_not_model_label_order(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = self.dataset(directory)
            coco, images, categories, _ = accuracy.select_dataset(labels, directory, 0)
            predictions = [{"image_id": i, "category_id": category,
                            "bbox": [10, 10, 20, 20], "score": 0.9}
                           for i, category in ((1, 3), (3, 8))]
            classes = accuracy.evaluate(coco, images, categories, predictions)["classes"]
            self.assertAlmostEqual(classes["car"]["ap_50_95"], 1.0)
            self.assertAlmostEqual(classes["person"]["ap_50_95"], 0.0)
            self.assertAlmostEqual(classes["truck"]["ap_50_95"], 1.0)
            self.assertEqual(classes["person"]["recall_at_035_iou_050"], 0.0)

    def test_report_displays_per_class_ap_precision_recall_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = self.dataset(directory)
            coco, images, categories, _ = accuracy.select_dataset(labels, directory, 0)
            predictions = [{"image_id": i, "category_id": category,
                            "bbox": [10, 10, 20, 20], "score": 0.9}
                           for i, category in ((1, 3), (2, 1), (3, 8))]
            metrics = accuracy.evaluate(coco, images, categories, predictions)
            deployment = accuracy.evaluate(coco, images, categories, predictions, standard_ap=False)
            report = {"results": [
                {"variant": accuracy.FP32, "batch": 1, "metrics": metrics},
                {"variant": accuracy.DEPLOYMENT, "batch": 1, "metrics": deployment},
            ], "fp16_parity_vs_trt_fp32": {"1": accuracy.compare(metrics, metrics)}}
            table = accuracy.result_table(report)
            self.assertIn("| Class | AP 50:95 | Precision | Recall | TP | FP | FN |", table)
            self.assertIn("| TensorRT FP32 | 1 | car | 100.00% | 100.00% | 100.00% | 1 | 0 | 0 |", table)
            self.assertIn("| Deployment engine | 1 | car | n/a | 100.00% | 100.00% | 1 | 0 | 0 |", table)

    def test_fp16_parity_gate_checks_map_and_every_class(self):
        def metrics(map_value, recalls):
            return {"map_50_95": map_value, "classes": {
                name: {"recall_at_035_iou_050": value} for name, value in recalls.items()}}
        base = metrics(0.5, {"car": 0.5, "person": 0.8, "truck": 0.6})
        good = metrics(0.497, {"car": 0.495, "person": 0.8, "truck": 0.6})
        bad_map = metrics(0.49, {"car": 0.5, "person": 0.8, "truck": 0.6})
        bad_recall = metrics(0.5, {"car": 0.48, "person": 0.8, "truck": 0.6})
        no_trucks = metrics(0.5, {"car": 0.5, "person": 0.8, "truck": None})
        self.assertEqual(accuracy.compare(base, good)["status"], "pass")
        self.assertEqual(accuracy.compare(base, bad_map)["status"], "fail")
        self.assertEqual(accuracy.compare(base, bad_recall)["status"], "fail")
        self.assertEqual(accuracy.compare(base, no_trucks)["status"], "not_evaluable")

    def test_same_images_and_padding_across_variants(self):
        class FakeRuntime:
            def infer(self, label, inputs):
                batch = inputs.shape[0]
                boxes = np.zeros((batch, benchmark.CANDIDATES, 4), np.float32)
                scores = np.zeros((batch, benchmark.CANDIDATES, 3), np.float32)
                for index in range(batch):
                    class_index = round(float(inputs[index, 0, 0, 0]) * 255 / 80) - 1
                    if class_index >= 0:
                        boxes[index, 0] = [128, 128, 128, 128]
                        scores[index, 0, class_index] = 0.9
                return {"raw_boxes": boxes, "raw_scores": scores}

        with tempfile.TemporaryDirectory() as directory:
            labels = self.dataset(directory)
            coco, images, categories, _ = accuracy.select_dataset(labels, directory, 0)
            rows = []
            for label, batch in ((accuracy.ORT, 1), (accuracy.FP32, 4), (accuracy.FP16, 8)):
                predictions = accuracy.infer_dataset(FakeRuntime(), label, batch, images, directory, categories)
                self.assertEqual(len(predictions), 3)
                rows.append(accuracy.evaluate(coco, images, categories, predictions))
            self.assertEqual(rows[0], rows[1])
            self.assertEqual(rows[1], rows[2])
            self.assertAlmostEqual(rows[0]["map_50_95"], 1.0)

    def test_deployment_outputs_are_decoded_without_claiming_ap(self):
        outputs = {
            "num_dets": np.array([[2]], np.int32),
            "det_boxes": np.array([[[64, 64, 192, 192], [0, 0, 20, 20]]], np.float32),
            "det_scores": np.array([[0.9, 0.2]], np.float32),
            "det_classes": np.array([[1, 0]], np.int32),
        }
        predictions = accuracy.deployment_detections(
            outputs, 0, 1, (6.4, 6.4, 100, 100), {"car": 3, "person": 1, "truck": 8})
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["category_id"], 1)
        self.assertEqual(predictions[0]["bbox"], [10, 10, 20, 20])
        outputs["det_classes"][0, 0] = 4
        with self.assertRaisesRegex(ValueError, "Invalid deployment"):
            accuracy.deployment_detections(outputs, 0, 1, (6.4, 6.4, 100, 100),
                                           {"car": 3, "person": 1, "truck": 8})

    def test_external_predictions_are_validated_and_score_floor_is_disclosed(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = self.dataset(directory)
            path = Path(directory) / "predictions.json"
            predictions = [{"image_id": i, "category_id": category,
                            "bbox": [10, 10, 20, 20], "score": 0.9}
                           for i, category in ((1, 3), (2, 1), (3, 8))]
            path.write_text(json.dumps(predictions))
            output = Path(directory) / "score"
            args = ["score", "--annotations", str(labels), "--images", directory,
                    "--predictions", str(path), "--score-floor", "0.35", "--output", str(output)]
            self.assertEqual(accuracy.main(args), 0)
            result = json.loads((output / "results.json").read_text())
            self.assertIsNone(result["metrics"]["map_50_95"])
            self.assertEqual(result["metrics"]["classes"]["car"]["recall_at_035_iou_050"], 1.0)
            path.write_text(json.dumps([{**predictions[0], "image_id": 999}]))
            with self.assertRaisesRegex(ValueError, "unknown image"):
                accuracy.main(args[:-1] + [str(Path(directory) / "bad")])
            nan_args = args.copy()
            nan_args[8] = "nan"
            nan_args[-1] = str(Path(directory) / "nan")
            with self.assertRaisesRegex(ValueError, "score floor"):
                accuracy.main(nan_args)

    def test_disagreement_examples_rank_ties_and_draw_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = self.dataset(directory)
            first = [{"image_id": i, "category_id": category,
                      "bbox": [10, 10, 20, 20], "score": 0.9}
                     for i, category in ((1, 3), (2, 1), (3, 8))]
            second = [{**first[0], "bbox": [60, 60, 20, 20]},
                      {**first[1], "bbox": [60, 60, 20, 20]}, first[2]]
            fp32, fp16 = Path(directory) / "fp32.json", Path(directory) / "fp16.json"
            fp32.write_text(json.dumps(first))
            fp16.write_text(json.dumps(second))
            output = Path(directory) / "examples"
            self.assertEqual(accuracy.main(["examples", "--annotations", str(labels), "--images", directory,
                                            "--fp32", str(fp32), "--fp16", str(fp16),
                                            "--output", str(output), "--count", "2"]), 0)
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["1.jpg", "2.jpg"])
            self.assertEqual(accuracy.disagreement_count([first[2]], [first[2]]), 0)


if __name__ == "__main__":
    unittest.main()
