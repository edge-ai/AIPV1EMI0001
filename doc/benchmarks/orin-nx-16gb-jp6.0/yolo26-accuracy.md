| Variant | Batch | mAP 50:95 (80 classes) | AP 50 | mAP 50:95 (car, person, truck) | Car recall | Person recall | Truck recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX Runtime FP32 | 8 | 39.93% | 55.54% | 40.68% | 38.89% | 57.32% | 28.99% |
| TensorRT FP32 | 8 | 39.93% | 55.54% | 40.68% | 38.89% | 57.32% | 28.99% |
| TensorRT FP16 | 8 | 39.92% | 55.57% | 40.67% | 39.00% | 57.34% | 29.23% |

mAP 50:95 is the COCO metric Ultralytics publishes. Recall uses confidence >= 0.35 and IoU >= 0.50. Deployment mAP is omitted because its embedded NMS filters low scores.

| Variant | Batch | Class | AP 50:95 | Precision | Recall | TP | FP | FN |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ONNX Runtime FP32 | 8 | car | 37.13% | 82.89% | 38.89% | 746 | 154 | 1172 |
| ONNX Runtime FP32 | 8 | person | 52.71% | 90.76% | 57.32% | 6177 | 629 | 4600 |
| ONNX Runtime FP32 | 8 | truck | 32.21% | 76.43% | 28.99% | 120 | 37 | 294 |
| TensorRT FP32 | 8 | car | 37.13% | 82.89% | 38.89% | 746 | 154 | 1172 |
| TensorRT FP32 | 8 | person | 52.71% | 90.76% | 57.32% | 6177 | 629 | 4600 |
| TensorRT FP32 | 8 | truck | 32.21% | 76.43% | 28.99% | 120 | 37 | 294 |
| TensorRT FP16 | 8 | car | 37.02% | 82.83% | 39.00% | 748 | 155 | 1170 |
| TensorRT FP16 | 8 | person | 52.69% | 90.87% | 57.34% | 6179 | 621 | 4598 |
| TensorRT FP16 | 8 | truck | 32.32% | 76.58% | 29.23% | 121 | 37 | 293 |

Per-class precision and recall use confidence >= 0.35 and IoU >= 0.50.

| Batch | FP16 parity | mAP drop (pp) | Car recall drop (pp) | Person recall drop (pp) | Truck recall drop (pp) |
|---:|---|---:|---:|---:|---:|
| 8 | pass | 0.00 | -0.10 | -0.02 | -0.24 |

A negative drop is an improvement. Parity is relative to FP32, not an absolute accuracy guarantee.
