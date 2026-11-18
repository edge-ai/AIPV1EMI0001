| Variant | Batch | mAP 50:95 (80 classes) | AP 50 | mAP 50:95 (car, person, truck) | Car recall | Person recall | Truck recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX Runtime FP32 | 8 | 38.33% | 53.53% | 40.56% | 38.53% | 54.24% | 28.50% |
| TensorRT FP32 | 8 | 38.33% | 53.53% | 40.56% | 38.53% | 54.24% | 28.50% |
| TensorRT FP16 | 8 | 38.34% | 53.55% | 40.54% | 38.58% | 54.20% | 28.74% |

mAP 50:95 is the COCO metric Ultralytics publishes. Recall uses confidence >= 0.35 and IoU >= 0.50. Deployment mAP is omitted because its embedded NMS filters low scores.

| Variant | Batch | Class | AP 50:95 | Precision | Recall | TP | FP | FN |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ONNX Runtime FP32 | 8 | car | 37.18% | 85.04% | 38.53% | 739 | 130 | 1179 |
| ONNX Runtime FP32 | 8 | person | 52.49% | 93.18% | 54.24% | 5845 | 428 | 4932 |
| ONNX Runtime FP32 | 8 | truck | 32.02% | 76.62% | 28.50% | 118 | 36 | 296 |
| TensorRT FP32 | 8 | car | 37.18% | 85.04% | 38.53% | 739 | 130 | 1179 |
| TensorRT FP32 | 8 | person | 52.49% | 93.18% | 54.24% | 5845 | 428 | 4932 |
| TensorRT FP32 | 8 | truck | 32.02% | 76.62% | 28.50% | 118 | 36 | 296 |
| TensorRT FP16 | 8 | car | 37.17% | 85.16% | 38.58% | 740 | 129 | 1178 |
| TensorRT FP16 | 8 | person | 52.46% | 93.07% | 54.20% | 5841 | 435 | 4936 |
| TensorRT FP16 | 8 | truck | 31.98% | 76.28% | 28.74% | 119 | 37 | 295 |

Per-class precision and recall use confidence >= 0.35 and IoU >= 0.50.

| Batch | FP16 parity | mAP drop (pp) | Car recall drop (pp) | Person recall drop (pp) | Truck recall drop (pp) |
|---:|---|---:|---:|---:|---:|
| 8 | pass | -0.01 | -0.05 | 0.04 | -0.24 |

A negative drop is an improvement. Parity is relative to FP32, not an absolute accuracy guarantee.
