| Variant | Batch | Median ms | p95 ms | Round medians ms | ms/image | Serial images/s | Speed-up vs ORT |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX Runtime CUDA FP32 | 1 | 18.95 | 19.22 | 18.93-18.98 | 18.95 | 52.8 | 1.00x |
| TensorRT FP32 | 1 | 12.25 | 12.35 | 12.24-12.26 | 12.25 | 81.6 | 1.55x |
| TensorRT FP16 | 1 | 5.31 | 5.34 | 5.30-5.31 | 5.31 | 188.4 | 3.57x |
| ONNX Runtime CUDA FP32 | 4 | 56.95 | 57.48 | 56.88-57.06 | 14.24 | 70.2 | 1.00x |
| TensorRT FP32 | 4 | 38.50 | 38.86 | 38.47-38.54 | 9.63 | 103.9 | 1.48x |
| TensorRT FP16 | 4 | 15.04 | 15.24 | 14.96-15.05 | 3.76 | 266.0 | 3.79x |
| ONNX Runtime CUDA FP32 | 8 | 106.07 | 107.24 | 106.03-106.14 | 13.26 | 75.4 | 1.00x |
| TensorRT FP32 | 8 | 72.03 | 72.77 | 71.92-72.17 | 9.00 | 111.1 | 1.47x |
| TensorRT FP16 | 8 | 27.38 | 28.57 | 27.31-27.95 | 3.42 | 292.2 | 3.87x |

GPU before: {'gpu_load_percent': 0.0, 'gpu_frequency_mhz': 306.0, 'gpu_temperature_c': 58.625}; after: {'gpu_load_percent': 49.1, 'gpu_frequency_mhz': 918.0, 'gpu_temperature_c': 65.406}. Raw model latency, one request at a time; not pipeline FPS.
