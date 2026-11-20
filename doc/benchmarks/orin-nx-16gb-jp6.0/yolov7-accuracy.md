| Variant | Batch | mAP 50:95 (80 classes) | AP 50 | mAP 50:95 (car, person, truck) | Car recall | Person recall | Truck recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX Runtime FP32 | 8 | 37.15% | 54.64% | 39.43% | 51.82% | 64.73% | 36.47% |
| TensorRT FP32 | 8 | 37.15% | 54.64% | 39.43% | 51.82% | 64.73% | 36.47% |
| TensorRT FP16 | 8 | 37.13% | 54.64% | 39.41% | 51.77% | 64.73% | 36.23% |
| Deployment engine | 8 | n/a | n/a | n/a | 51.77% | 64.74% | 36.23% |

mAP 50:95 is the COCO metric Ultralytics publishes. Recall uses confidence >= 0.35 and IoU >= 0.50. Deployment mAP is omitted because its embedded NMS filters low scores.

| Variant | Batch | Class | AP 50:95 | Precision | Recall | TP | FP | FN |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ONNX Runtime FP32 | 8 | car | 36.63% | 75.70% | 51.82% | 994 | 319 | 924 |
| ONNX Runtime FP32 | 8 | person | 50.52% | 87.29% | 64.73% | 6976 | 1016 | 3801 |
| ONNX Runtime FP32 | 8 | truck | 31.14% | 60.64% | 36.47% | 151 | 98 | 263 |
| TensorRT FP32 | 8 | car | 36.63% | 75.70% | 51.82% | 994 | 319 | 924 |
| TensorRT FP32 | 8 | person | 50.52% | 87.29% | 64.73% | 6976 | 1016 | 3801 |
| TensorRT FP32 | 8 | truck | 31.14% | 60.64% | 36.47% | 151 | 98 | 263 |
| TensorRT FP16 | 8 | car | 36.65% | 75.57% | 51.77% | 993 | 321 | 925 |
| TensorRT FP16 | 8 | person | 50.50% | 87.32% | 64.73% | 6976 | 1013 | 3801 |
| TensorRT FP16 | 8 | truck | 31.09% | 60.48% | 36.23% | 150 | 98 | 264 |
| Deployment engine | 8 | car | n/a | 75.40% | 51.77% | 993 | 324 | 925 |
| Deployment engine | 8 | person | n/a | 87.31% | 64.74% | 6977 | 1014 | 3800 |
| Deployment engine | 8 | truck | n/a | 60.48% | 36.23% | 150 | 98 | 264 |

Per-class precision and recall use confidence >= 0.35 and IoU >= 0.50.

| Batch | FP16 parity | mAP drop (pp) | Car recall drop (pp) | Person recall drop (pp) | Truck recall drop (pp) |
|---:|---|---:|---:|---:|---:|
| 8 | pass | 0.02 | 0.05 | 0.00 | 0.24 |

A negative drop is an improvement. Parity is relative to FP32, not an absolute accuracy guarantee.
