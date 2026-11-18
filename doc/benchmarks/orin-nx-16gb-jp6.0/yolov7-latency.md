| Variant | Batch | Median ms | p95 ms | Round medians ms | ms/image | Serial images/s | Speed-up vs ORT |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX Runtime CUDA FP32 | 1 | 24.40 | 24.66 | 24.38-24.41 | 24.40 | 41.0 | 1.00x |
| TensorRT FP32 | 1 | 17.99 | 18.22 | 17.97-18.03 | 17.99 | 55.6 | 1.36x |
| TensorRT FP16 | 1 | 8.40 | 8.48 | 8.40-8.41 | 8.40 | 119.0 | 2.90x |
| ONNX Runtime CUDA FP32 | 4 | 90.76 | 93.42 | 90.62-91.09 | 22.69 | 44.1 | 1.00x |
| TensorRT FP32 | 4 | 64.52 | 65.44 | 64.45-64.57 | 16.13 | 62.0 | 1.41x |
| TensorRT FP16 | 4 | 21.25 | 22.60 | 21.17-21.85 | 5.31 | 188.2 | 4.27x |
| ONNX Runtime CUDA FP32 | 8 | 154.69 | 158.86 | 154.57-155.09 | 19.34 | 51.7 | 1.00x |
| TensorRT FP32 | 8 | 124.36 | 125.40 | 124.28-124.48 | 15.55 | 64.3 | 1.24x |
| TensorRT FP16 | 8 | 37.93 | 38.71 | 37.90-37.96 | 4.74 | 210.9 | 4.08x |

GPU before: {'gpu_load_percent': 0.0, 'gpu_frequency_mhz': 306.0, 'gpu_temperature_c': 64.218}; after: {'gpu_load_percent': 0.0, 'gpu_frequency_mhz': 612.0, 'gpu_temperature_c': 70.062}. Raw model latency, one request at a time; not pipeline FPS.
