#!/usr/bin/env python3
"""Evaluate the raw model and deployment engine against COCO-format labels.

The raw ONNX and TensorRT engines use the same low-threshold postprocessing for
an mAP comparison. The deployment engine has an embedded 0.35 score threshold,
so only its operating-point precision and recall are reported.
"""
import argparse
import contextlib
import ctypes
import hashlib
import io
import json
import math
import platform
import sys
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import benchmark


ROOT = Path(__file__).resolve().parents[1]
LABEL_FILE = ROOT / "models/AIPV1EMI0001_Yolo_Detector/labels-rev3.txt"
ORT = "ONNX Runtime FP32"
FP32 = "TensorRT FP32"
FP16 = "TensorRT FP16"
DEPLOYMENT = "Deployment engine"
INPUT_SIZE = 640
SCORE_FLOOR = 0.001
OPERATING_SCORE = 0.35
NMS_IOU = 0.65
MAX_CANDIDATES = 4096
MAX_DETECTIONS = 100


@lru_cache(maxsize=1)
def class_names():
    names = tuple(LABEL_FILE.read_text().splitlines())
    if len(names) != 3 or set(names) != {"car", "person", "truck"}:
        raise ValueError(f"Unexpected model label order: {names}")
    return names


def quiet_coco(annotations):
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(annotations))
    coco.dataset.setdefault("info", {})
    return coco


def select_dataset(annotations, images_dir, limit):
    coco = quiet_coco(annotations)
    if limit < 0:
        raise ValueError("--limit must be zero (all images) or a positive number")
    names = class_names()
    category_ids = {}
    for name in names:
        matches = [cat["id"] for cat in coco.cats.values() if cat["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"Dataset must contain exactly one '{name}' category: {matches}")
        category_ids[name] = matches[0]
    ids = sorted(coco.imgs)
    if not ids:
        raise ValueError("The dataset has no images")
    if limit > len(ids):
        raise ValueError(f"--limit {limit} exceeds the {len(ids)} available images")
    if limit:
        ids = ids[:limit]
    images = coco.loadImgs(ids)
    root = Path(images_dir).resolve()
    digest = hashlib.sha256()
    for image in images:
        path = image_path(root, image)
        digest.update(f'{image["id"]}\t{image["file_name"]}\t{benchmark.sha256(path)}\n'.encode())
    counts = {
        name: sum(not annotation.get("iscrowd", 0) for annotation in coco.loadAnns(
            coco.getAnnIds(imgIds=ids, catIds=[category_id])))
        for name, category_id in category_ids.items()
    }
    return coco, images, category_ids, {
        "annotations_sha256": benchmark.sha256(annotations),
        "image_content_sha256": digest.hexdigest(),
        "image_ids": ids,
        "image_count": len(images),
        "total_dataset_images": len(coco.imgs),
        "selection": "first image IDs in ascending order" if limit else "all image IDs",
        "non_crowd_ground_truth_by_class": counts,
        "category_ids": category_ids,
    }


def image_path(root, image):
    name = Path(image["file_name"])
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(f"Unsafe dataset image name: {name}")
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or unsafe dataset image: {name}")
    return path


def prepare_image(root, image):
    path = image_path(root, image)
    pixels = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if pixels is None:
        raise ValueError(f"Cannot decode image: {path.name}")
    height, width = pixels.shape[:2]
    if (width, height) != (image["width"], image["height"]):
        raise ValueError(f"Image dimensions differ from annotations: {path.name}")
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_width = max(1, min(INPUT_SIZE, round(width * scale)))
    new_height = max(1, min(INPUT_SIZE, round(height * scale)))
    rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    canvas[:new_height, :new_width] = rgb
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return tensor, (new_width / width, new_height / height, width, height)


def original_box(corners, transform):
    scale_x, scale_y, width, height = transform
    x1, y1, x2, y2 = (float(value) for value in corners)
    x1, x2 = np.clip([x1 / scale_x, x2 / scale_x], 0, width)
    y1, y2 = np.clip([y1 / scale_y, y2 / scale_y], 0, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def raw_detections(boxes, scores, image_id, transform, category_ids):
    names = class_names()
    if boxes.shape != (benchmark.CANDIDATES, 4) or scores.shape != (benchmark.CANDIDATES, len(names)):
        raise ValueError(f"Unexpected raw outputs: {boxes.shape}, {scores.shape}")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Raw inference returned non-finite boxes or scores")
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("Raw model scores must be probabilities between zero and one")
    anchors, classes = np.where(scores >= SCORE_FLOOR)
    values = scores[anchors, classes]
    order = np.argsort(-values, kind="stable")[:MAX_CANDIDATES]
    anchors, classes, values = anchors[order], classes[order], values[order]
    selected = []
    for class_index, name in enumerate(names):
        positions = np.flatnonzero(classes == class_index)
        if not len(positions):
            continue
        centers = boxes[anchors[positions]]
        xywh = np.column_stack((centers[:, 0] - centers[:, 2] / 2,
                                centers[:, 1] - centers[:, 3] / 2,
                                centers[:, 2], centers[:, 3]))
        valid = (xywh[:, 2] > 0) & (xywh[:, 3] > 0)
        positions, xywh = positions[valid], xywh[valid]
        if not len(positions):
            continue
        indices = cv2.dnn.NMSBoxes(xywh.tolist(), values[positions].tolist(),
                                   SCORE_FLOOR - 1e-8, NMS_IOU)
        for index in np.asarray(indices).reshape(-1):
            position, box = positions[int(index)], xywh[int(index)]
            corners = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            original = original_box(corners, transform)
            if original is not None:
                selected.append({
                    "image_id": image_id, "category_id": category_ids[name],
                    "bbox": original, "score": float(values[position]),
                })
    selected.sort(key=lambda prediction: -prediction["score"])
    return selected[:MAX_DETECTIONS]


def deployment_detections(outputs, index, image_id, transform, category_ids):
    names = class_names()
    count = int(outputs["num_dets"][index, 0])
    if not 0 <= count <= MAX_DETECTIONS:
        raise ValueError(f"Invalid deployment detection count: {count}")
    selected = []
    for slot in range(count):
        box = outputs["det_boxes"][index, slot]
        score = float(outputs["det_scores"][index, slot])
        category = int(outputs["det_classes"][index, slot])
        if not np.isfinite(box).all() or not math.isfinite(score) or not 0 <= category < len(names):
            raise ValueError("Invalid deployment detection")
        if score < OPERATING_SCORE:
            continue
        original = original_box(box, transform)
        if original is not None:
            selected.append({"image_id": image_id, "category_id": category_ids[names[category]],
                             "bbox": original, "score": score})
    return selected


def mean_valid(values):
    values = np.asarray(values)
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else None


def detection_coco(coco, predictions):
    with contextlib.redirect_stdout(io.StringIO()):
        if predictions:
            return coco.loadRes(predictions)
        result = COCO()
        result.dataset = {
            "images": list(coco.dataset["images"]),
            "categories": list(coco.dataset["categories"]),
            "annotations": [],
        }
        result.createIndex()
        return result


def evaluate(coco, images, category_ids, predictions, *, standard_ap=True):
    evaluator = COCOeval(coco, detection_coco(coco, predictions), "bbox")
    evaluator.params.imgIds = [image["id"] for image in images]
    evaluator.params.catIds = list(category_ids.values())
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]
    overall = {
        "map_50_95": mean_valid(precision[:, :, :, 0, -1]) if standard_ap else None,
        "ap_50": mean_valid(precision[0, :, :, 0, -1]) if standard_ap else None,
        "classes": {},
    }
    iou_index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.5))[0])
    for name, category_id in category_ids.items():
        category_index = evaluator.params.catIds.index(category_id)
        ground_truth, true_positives, false_positives = 0, 0, 0
        for item in evaluator.evalImgs:
            if item is None or item["category_id"] != category_id or item["aRng"] != evaluator.params.areaRng[0]:
                continue
            ground_truth += int(np.count_nonzero(np.logical_not(item["gtIgnore"])))
            chosen = np.asarray(item["dtScores"]) >= OPERATING_SCORE
            matches = np.asarray(item["dtMatches"])[iou_index] > 0
            ignored = np.asarray(item["dtIgnore"])[iou_index].astype(bool)
            true_positives += int(np.count_nonzero(chosen & matches & ~ignored))
            false_positives += int(np.count_nonzero(chosen & ~matches & ~ignored))
        total = true_positives + false_positives
        overall["classes"][name] = {
            "ap_50_95": mean_valid(precision[:, :, category_index, 0, -1]) if standard_ap else None,
            "ground_truth": ground_truth,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": ground_truth - true_positives,
            "precision_at_035_iou_050": true_positives / total if total else None,
            "recall_at_035_iou_050": true_positives / ground_truth if ground_truth else None,
        }
    return overall


def compare(fp32, fp16, max_map_drop_pp=0.5, max_recall_drop_pp=1.0):
    if fp32["map_50_95"] is None or fp16["map_50_95"] is None:
        return {"status": "not_evaluable", "reason": "mAP is unavailable"}
    recalls = {}
    for name in class_names():
        a = fp32["classes"][name]["recall_at_035_iou_050"]
        b = fp16["classes"][name]["recall_at_035_iou_050"]
        if a is None or b is None:
            return {"status": "not_evaluable", "reason": f"No labelled {name} ground truth"}
        recalls[name] = (a - b) * 100
    map_drop = (fp32["map_50_95"] - fp16["map_50_95"]) * 100
    passed = (map_drop <= max_map_drop_pp + 1e-12
              and all(drop <= max_recall_drop_pp + 1e-12 for drop in recalls.values()))
    return {
        "status": "pass" if passed else "fail",
        "map_drop_percentage_points": map_drop,
        "recall_drop_percentage_points": recalls,
        "max_map_drop_percentage_points": max_map_drop_pp,
        "max_recall_drop_percentage_points": max_recall_drop_pp,
        "meaning": "FP16 parity relative to TensorRT FP32, not an absolute accuracy guarantee",
    }


class Runtimes:
    def __init__(self, raw_onnx, engines, work_dir, trtexec):
        self.trt, self.trt_info = benchmark.load_tensorrt(work_dir, trtexec)
        self.ort = ort.InferenceSession(
            str(raw_onnx), providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0})])
        self.ort.disable_fallback()
        self.providers = self.ort.get_providers()
        if not self.providers or self.providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"ONNX Runtime CUDA provider is required: {self.providers}")
        if self.ort.get_provider_options()["CUDAExecutionProvider"]["use_tf32"] != "0":
            raise RuntimeError("ONNX Runtime TF32 must be disabled")
        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        if not self.trt.init_libnvinfer_plugins(self.logger, ""):
            raise RuntimeError("Could not register TensorRT plugins")
        self.runtime = self.trt.Runtime(self.logger)
        self.engines = {}
        self.contexts = {}
        for label, path in engines.items():
            engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
            if engine is None:
                raise RuntimeError(f"Cannot deserialize TensorRT engine: {path}")
            self.engines[label] = engine
            self.contexts[label] = engine.create_execution_context()
            if self.contexts[label] is None:
                raise RuntimeError(f"Cannot create TensorRT execution context: {path}")
            self.check_engine(label, engine)
        self.cuda = ctypes.CDLL("libcudart.so.12")
        self.stream = ctypes.c_void_p()
        self.check_cuda(self.cuda.cudaStreamCreateWithFlags(ctypes.byref(self.stream), 1),
                        "cudaStreamCreateWithFlags")
        self.buffers = {}

    def check_cuda(self, code, name):
        if code:
            raise RuntimeError(f"{name} failed with CUDA error {code}")

    def check_engine(self, label, engine):
        float_type, int_type = self.trt.DataType.FLOAT, self.trt.DataType.INT32
        outputs = ({"num_dets": int_type, "det_boxes": float_type,
                    "det_scores": float_type, "det_classes": int_type}
                   if label == DEPLOYMENT else {"raw_boxes": float_type, "raw_scores": float_type})
        expected = {"images": (self.trt.TensorIOMode.INPUT, float_type)}
        expected.update({name: (self.trt.TensorIOMode.OUTPUT, dtype) for name, dtype in outputs.items()})
        actual = {
            engine.get_tensor_name(index): (engine.get_tensor_mode(engine.get_tensor_name(index)),
                                             engine.get_tensor_dtype(engine.get_tensor_name(index)))
            for index in range(engine.num_io_tensors)
        }
        if actual != expected:
            raise ValueError(f"Unexpected {label} engine tensors: {actual}")

    def output_buffers(self, label, batch):
        key = (label, batch)
        if key not in self.buffers:
            if label == DEPLOYMENT:
                specs = {"num_dets": ((batch, 1), np.int32),
                         "det_boxes": ((batch, MAX_DETECTIONS, 4), np.float32),
                         "det_scores": ((batch, MAX_DETECTIONS), np.float32),
                         "det_classes": ((batch, MAX_DETECTIONS), np.int32)}
            else:
                specs = {name: ((batch, benchmark.CANDIDATES, width), np.float32)
                         for name, width in benchmark.OUTPUTS.items()}
            self.buffers[key] = {
                name: ort.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype), "cuda", 0)
                for name, (shape, dtype) in specs.items()
            }
        return self.buffers[key]

    def infer(self, label, input_data):
        batch = input_data.shape[0]
        gpu_input = ort.OrtValue.ortvalue_from_numpy(input_data, "cuda", 0)
        outputs = self.output_buffers(label, batch)
        if label == ORT:
            binding = self.ort.io_binding()
            binding.bind_ortvalue_input("images", gpu_input)
            for name, value in outputs.items():
                binding.bind_ortvalue_output(name, value)
            self.ort.run_with_iobinding(binding)
        else:
            context = self.contexts[label]
            if not context.set_input_shape("images", input_data.shape):
                raise RuntimeError(f"{label} does not accept input shape {input_data.shape}")
            if not context.set_tensor_address("images", gpu_input.data_ptr()):
                raise RuntimeError(f"Cannot bind {label} input")
            for name, value in outputs.items():
                if tuple(context.get_tensor_shape(name)) != tuple(value.shape()):
                    raise ValueError(f"Unexpected {label} output shape for {name}")
                if not context.set_tensor_address(name, value.data_ptr()):
                    raise RuntimeError(f"Cannot bind {label} output {name}")
            if context.infer_shapes():
                raise RuntimeError(f"Unresolved {label} tensor shapes")
            if not context.execute_async_v3(self.stream.value):
                raise RuntimeError(f"{label} inference failed")
        self.check_cuda(self.cuda.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
        return {name: value.numpy() for name, value in outputs.items()}

    def close(self):
        self.check_cuda(self.cuda.cudaStreamDestroy(self.stream), "cudaStreamDestroy")


def infer_dataset(runtime, label, batch, images, images_dir, category_ids):
    predictions = []
    for start in range(0, len(images), batch):
        group = images[start:start + batch]
        inputs = np.zeros((batch, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        transforms = []
        for index, image in enumerate(group):
            inputs[index], transform = prepare_image(images_dir, image)
            transforms.append(transform)
        outputs = runtime.infer(label, inputs)
        for index, image in enumerate(group):
            if label == DEPLOYMENT:
                detections = deployment_detections(outputs, index, image["id"],
                                                    transforms[index], category_ids)
            else:
                detections = raw_detections(outputs["raw_boxes"][index], outputs["raw_scores"][index],
                                            image["id"], transforms[index], category_ids)
            predictions.extend(detections)
        if (start + len(group)) % 100 == 0 or start + len(group) == len(images):
            print(f"{label}, batch {batch}: {start + len(group)}/{len(images)} images", flush=True)
    return predictions


def load_predictions(path, images, category_ids):
    predictions = json.loads(Path(path).read_text())
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a COCO detection JSON array")
    image_ids = {image["id"] for image in images}
    allowed_categories = set(category_ids.values())
    cleaned = []
    for prediction in predictions:
        if not isinstance(prediction, dict) or set(("image_id", "category_id", "bbox", "score")) - prediction.keys():
            raise ValueError("Each prediction needs image_id, category_id, bbox and score")
        image_id, category_id = prediction["image_id"], prediction["category_id"]
        box, score = prediction["bbox"], prediction["score"]
        if image_id not in image_ids or category_id not in allowed_categories:
            raise ValueError(f"Prediction has an unknown image or category: {image_id}, {category_id}")
        if (not isinstance(box, list) or len(box) != 4
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in box)
                or box[2] <= 0 or box[3] <= 0):
            raise ValueError(f"Invalid COCO xywh prediction box: {box}")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid detection confidence: {score}")
        cleaned.append({"image_id": image_id, "category_id": category_id, "bbox": box, "score": score})
    return cleaned


def score_predictions(args):
    if not math.isfinite(args.score_floor) or not 0 <= args.score_floor <= OPERATING_SCORE:
        raise ValueError("Prediction score floor must be between zero and the 0.35 operating threshold")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    coco, images, category_ids, dataset = select_dataset(args.annotations, args.images, args.limit)
    predictions = load_predictions(args.predictions, images, category_ids)
    metrics = evaluate(coco, images, category_ids, predictions,
                       standard_ap=args.score_floor <= SCORE_FLOOR)
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "predictions_sha256": benchmark.sha256(args.predictions),
        "declared_prediction_score_floor": args.score_floor,
        "metrics": metrics,
        "limitations": [
            "Image IDs and labels must match the captured frames; missing predictions are counted as misses.",
            "The declared confidence floor and completeness of pipeline capture cannot be verified from the JSON.",
            "Training dataset and checkpoint provenance are undocumented; evaluation-set overlap is unknown.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["| Class | AP 50:95 | Precision at 0.35 | Recall at 0.35 | TP | FP | FN |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def percent(value):
        return "n/a" if value is None else f"{100 * value:.2f}%"
    for name, row in metrics["classes"].items():
        lines.append(f'| {name} | {percent(row["ap_50_95"])} | '
                     f'{percent(row["precision_at_035_iou_050"])} | '
                     f'{percent(row["recall_at_035_iou_050"])} | {row["true_positives"]} | '
                     f'{row["false_positives"]} | {row["false_negatives"]} |')
    lines.append("\nPrecision and recall use IoU 0.50 and COCO crowd/ignore rules. "
                 "AP is omitted if the declared prediction floor exceeds 0.001.")
    (output / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved accuracy results in {output}")
    return 0


def box_iou(left, right):
    ax, ay, aw, ah = left
    bx, by, bw, bh = right
    width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = width * height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0


def disagreement_count(first, second):
    first = sorted((p for p in first if p["score"] >= OPERATING_SCORE),
                   key=lambda prediction: -prediction["score"])
    second = [p for p in second if p["score"] >= OPERATING_SCORE]
    unmatched = set(range(len(second)))
    misses = 0
    for prediction in first:
        overlaps = [(box_iou(prediction["bbox"], second[index]["bbox"]), index)
                    for index in unmatched if prediction["category_id"] == second[index]["category_id"]]
        if not overlaps or max(overlaps)[0] < 0.5:
            misses += 1
            continue
        unmatched.remove(max(overlaps)[1])
    return misses + len(unmatched)


def examples(args):
    if args.count < 1:
        raise ValueError("--count must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    coco, images, category_ids, _ = select_dataset(args.annotations, args.images, args.limit)
    fp32 = load_predictions(args.fp32, images, category_ids)
    fp16 = load_predictions(args.fp16, images, category_ids)
    by_image = {image["id"]: ([], []) for image in images}
    for index, predictions in enumerate((fp32, fp16)):
        for prediction in predictions:
            by_image[prediction["image_id"]][index].append(prediction)
    ranked = [(disagreement_count(*by_image[image["id"]]), image) for image in images]
    ranked = [(count, image) for count, image in ranked if count > 0]
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    if not ranked:
        print("No FP32/FP16 disagreements at confidence 0.35 and IoU 0.50; no examples written.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    names = {category_id: name for name, category_id in category_ids.items()}
    for count, image in ranked[:args.count]:
        pixels = cv2.imread(str(image_path(args.images, image)),
                            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if pixels is None or pixels.shape[:2] != (image["height"], image["width"]):
            raise ValueError(f"Cannot decode annotated image {image['file_name']}")
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image["id"]], catIds=list(category_ids.values())))
        layers = [
            (annotations, (0, 200, 0), "GT"),
            (by_image[image["id"]][0], (255, 120, 0), "FP32"),
            (by_image[image["id"]][1], (0, 0, 255), "FP16"),
        ]
        for detections, color, label in layers:
            for detection in detections:
                if label != "GT" and detection["score"] < OPERATING_SCORE:
                    continue
                x, y, width, height = detection["bbox"]
                pt1, pt2 = (round(x), round(y)), (round(x + width), round(y + height))
                cv2.rectangle(pixels, pt1, pt2, color, 2)
                text = f'{label} {names[detection["category_id"]]}'
                if label != "GT":
                    text += f' {detection["score"]:.2f}'
                cv2.putText(pixels, text, (pt1[0], max(12, pt1[1] - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        destination = output / f'{image["id"]}.jpg'
        if not cv2.imwrite(str(destination), pixels):
            raise RuntimeError(f"Cannot write {destination}")
        print(f'{destination}: {count} unmatched detections')
    print("Green: ground truth; blue: FP32; red: FP16. Review each image, including source-image licence.")
    return 0


def result_table(report):
    def percent(value):
        return "n/a" if value is None else f"{100 * value:.2f}%"

    lines = ["| Variant | Batch | mAP 50:95 | AP 50 | Car recall | Person recall | Truck recall |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        metrics = row["metrics"]
        recalls = [metrics["classes"][name]["recall_at_035_iou_050"]
                   for name in ("car", "person", "truck")]
        lines.append(f'| {row["variant"]} | {row["batch"]} | {percent(metrics["map_50_95"])} | '
                     f'{percent(metrics["ap_50"])} | ' + " | ".join(percent(v) for v in recalls) + " |")
    lines.append("\nRecall uses confidence >= 0.35 and IoU >= 0.50. "
                 "Deployment mAP is omitted because its embedded NMS filters low scores.")
    lines.extend(["", "| Variant | Batch | Class | AP 50:95 | Precision | Recall | TP | FP | FN |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|"])
    for row in report["results"]:
        for name, metrics in row["metrics"]["classes"].items():
            lines.append(f'| {row["variant"]} | {row["batch"]} | {name} | '
                         f'{percent(metrics["ap_50_95"])} | {percent(metrics["precision_at_035_iou_050"])} | '
                         f'{percent(metrics["recall_at_035_iou_050"])} | {metrics["true_positives"]} | '
                         f'{metrics["false_positives"]} | {metrics["false_negatives"]} |')
    lines.append("\nPer-class precision and recall use confidence >= 0.35 and IoU >= 0.50.")
    lines.extend(["", "| Batch | FP16 parity | mAP drop (pp) | Car recall drop (pp) | "
                  "Person recall drop (pp) | Truck recall drop (pp) |",
                  "|---:|---|---:|---:|---:|---:|"])
    for batch, comparison in report["fp16_parity_vs_trt_fp32"].items():
        if comparison["status"] == "not_evaluable":
            lines.append(f'| {batch} | Not evaluable: {comparison["reason"]} | n/a | n/a | n/a | n/a |')
            continue
        drops = comparison["recall_drop_percentage_points"]
        lines.append(f'| {batch} | {comparison["status"]} | '
                     f'{comparison["map_drop_percentage_points"]:.2f} | '
                     + " | ".join(f"{drops[name]:.2f}" for name in ("car", "person", "truck")) + " |")
    lines.append("\nA negative drop is an improvement. Parity is relative to FP32, not an absolute "
                 "accuracy guarantee.")
    return "\n".join(lines)


def run(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if len(set(args.batches)) != len(args.batches) or any(b not in benchmark.BATCHES for b in args.batches):
        raise ValueError(f"Batches must be unique values from {benchmark.BATCHES}")
    if (not math.isfinite(args.max_map_drop_pp) or not math.isfinite(args.max_recall_drop_pp)
            or args.max_map_drop_pp < 0 or args.max_recall_drop_pp < 0):
        raise ValueError("Allowed accuracy drops must be non-negative")
    coco, images, category_ids, dataset = select_dataset(args.annotations, args.images, args.limit)
    output.mkdir(parents=True, exist_ok=True)
    engines = {FP32: Path(args.fp32_engine), FP16: Path(args.fp16_engine)}
    if args.deployment_engine:
        engines[DEPLOYMENT] = Path(args.deployment_engine)
    artifacts = {"raw_onnx": benchmark.sha256(args.onnx), "model_labels": benchmark.sha256(LABEL_FILE),
                 **{label: benchmark.sha256(path) for label, path in engines.items()}}
    if args.source_onnx:
        artifacts["source_onnx"] = benchmark.sha256(args.source_onnx)
    runtime = Runtimes(args.onnx, engines, output, args.trtexec)
    results = []
    try:
        for batch in args.batches:
            for label in (ORT, *engines):
                predictions = infer_dataset(runtime, label, batch, images, args.images, category_ids)
                filename = {ORT: "ort", FP32: "trt-fp32", FP16: "trt-fp16",
                            DEPLOYMENT: "deployment"}[label] + f"-b{batch}.json"
                (output / filename).write_text(json.dumps(predictions, allow_nan=False) + "\n")
                metrics = evaluate(coco, images, category_ids, predictions, standard_ap=label != DEPLOYMENT)
                results.append({"variant": label, "batch": batch, "predictions": filename,
                                "detection_count": len(predictions), "metrics": metrics})
                del predictions
        comparisons = {}
        for batch in args.batches:
            rows = {row["variant"]: row["metrics"] for row in results if row["batch"] == batch}
            comparisons[str(batch)] = compare(rows[FP32], rows[FP16], args.max_map_drop_pp,
                                               args.max_recall_drop_pp)
        report = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "artifact_sha256": artifacts,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "opencv": cv2.__version__, "onnxruntime": ort.__version__,
                         "pycocotools": version("pycocotools"), **runtime.trt_info},
            "platform": {"machine": platform.machine(), "l4t": benchmark.l4t_release()},
            "onnxruntime_providers": runtime.providers,
            "method": {
                "input": "RGB, aspect-preserving top-left resize to 640x640, black padding, /255",
                "raw_postprocess": "center-size boxes, score >= 0.001, top 4096 candidates, "
                                   "class-aware OpenCV NMS IoU 0.65, at most 100 per image",
                "deployment": "actual TensorRT engine with embedded EfficientNMS; direct inference only",
                "metrics": "COCO bbox AP at IoU 0.50:0.95 and 0.50; precision/recall at "
                           "confidence >= 0.35 and IoU 0.50, following COCO crowd/ignore rules",
                "batch_padding": "last partial batch is zero-padded; padded outputs are ignored",
                "limitations": [
                    "Training dataset and checkpoint provenance are undocumented; evaluation-set overlap is unknown.",
                    "OpenCV preprocessing and NMS are not proven identical to DeepStream preprocessing or the plugin.",
                    "Direct engine evaluation does not exercise the DeepStream parser, tracking or callbacks.",
                    "A small ordered subset is a smoke test, not a representative accuracy estimate.",
                    "Engine hashes are recorded; the tool cannot prove how supplied engines were built.",
                ],
            },
            "results": results,
            "fp16_parity_vs_trt_fp32": comparisons,
        }
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (output / "results.md").write_text(result_table(report) + "\n")
    finally:
        runtime.close()
    print("\n" + result_table(report))
    print(f"\nSaved accuracy results in {output}")
    if args.enforce_tolerance and any(item["status"] != "pass" for item in comparisons.values()):
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def dataset_arguments(command):
        command.add_argument("--annotations", required=True, help="COCO instances JSON")
        command.add_argument("--images", required=True, help="directory containing the annotated images")
        command.add_argument("--limit", type=int, default=0, help="first N sorted image IDs; 0 evaluates all")
        command.add_argument("--output", required=True, help="new, empty output directory")

    run_parser = commands.add_parser("run", help="compare labelled model and deployment-engine inference")
    dataset_arguments(run_parser)
    run_parser.add_argument("--onnx", required=True, help="NMS-free raw ONNX graph")
    run_parser.add_argument("--fp32-engine", required=True, help="TensorRT FP32 engine built from --onnx")
    run_parser.add_argument("--fp16-engine", required=True, help="TensorRT FP16 engine built from --onnx")
    run_parser.add_argument("--deployment-engine", help="actual deployment engine with EfficientNMS")
    run_parser.add_argument("--source-onnx", help="deployment ONNX, recorded by hash")
    run_parser.add_argument("--batches", nargs="+", type=int, default=list(benchmark.BATCHES))
    run_parser.add_argument("--trtexec", default=benchmark.DEFAULT_TRTEXEC)
    run_parser.add_argument("--max-map-drop-pp", type=float, default=0.5)
    run_parser.add_argument("--max-recall-drop-pp", type=float, default=1.0)
    run_parser.add_argument("--enforce-tolerance", action="store_true",
                            help="exit non-zero if FP16 parity fails or cannot be evaluated")

    score_parser = commands.add_parser("score", help="evaluate externally captured COCO-format predictions")
    dataset_arguments(score_parser)
    score_parser.add_argument("--predictions", required=True, help="COCO-format detection JSON array")
    score_parser.add_argument("--score-floor", required=True, type=float,
                              help="lowest confidence the capture could have emitted")

    examples_parser = commands.add_parser("examples", help="draw FP32/FP16 disagreements")
    dataset_arguments(examples_parser)
    examples_parser.add_argument("--fp32", required=True, help="FP32 predictions JSON")
    examples_parser.add_argument("--fp16", required=True, help="FP16 predictions JSON")
    examples_parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args(argv)
    return {"run": run, "score": score_predictions, "examples": examples}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
