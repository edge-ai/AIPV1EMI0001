| Variant | Batch | Median ms | p95 ms | Round medians ms | ms/image | Serial images/s | Speed-up vs ORT |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX Runtime CUDA FP32 | 1 | 18.37 | 18.61 | 18.35-18.42 | 18.37 | 54.4 | 1.00x |
| TensorRT FP32 | 1 | 12.65 | 12.71 | 12.65-12.66 | 12.65 | 79.0 | 1.45x |
| TensorRT FP16 | 1 | 5.50 | 5.54 | 5.50-5.51 | 5.50 | 181.7 | 3.34x |
| ONNX Runtime CUDA FP32 | 4 | 61.37 | 61.83 | 61.27-61.47 | 15.34 | 65.2 | 1.00x |
| TensorRT FP32 | 4 | 40.29 | 40.90 | 40.27-40.31 | 10.07 | 99.3 | 1.52x |
| TensorRT FP16 | 4 | 16.38 | 16.45 | 16.37-16.39 | 4.09 | 244.2 | 3.75x |
| ONNX Runtime CUDA FP32 | 8 | 112.21 | 113.61 | 112.02-112.75 | 14.03 | 71.3 | 1.00x |
| TensorRT FP32 | 8 | 76.51 | 77.18 | 76.49-76.54 | 9.56 | 104.6 | 1.47x |
| TensorRT FP16 | 8 | 30.10 | 30.27 | 30.03-30.12 | 3.76 | 265.8 | 3.73x |

GPU before: {'gpu_load_percent': 0.0, 'gpu_frequency_mhz': 306.0, 'gpu_temperature_c': 61.531}; after: {'gpu_load_percent': 92.8, 'gpu_frequency_mhz': 918.0, 'gpu_temperature_c': 67.687}. Raw model latency, one request at a time; not pipeline FPS.
