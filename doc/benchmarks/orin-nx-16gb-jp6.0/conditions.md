Latency was remeasured on **2026-11-18 with the video pipeline stopped and an idle GPU baseline**. Accuracy was not rerun.

### Device and load

- Orin NX 16 GB, JetPack 6.0 / L4T R36.3, TensorRT 8.6.2.3, ONNX Runtime 1.19.0, Python 3.10.12, NumPy 2.2.6.
- Power mode was **MAXN**. No services, power modes or clock settings were changed by the benchmark. GPU frequency limits remained 306-918 MHz; recorded round-boundary readings were 918 MHz, with GPU temperatures from 61.1 to 75.1 C.
- GPU load was sampled once per second for 20 seconds before each flavor, before the alternating-engine check, and after all measurements: **100 samples, all 0%**. These baselines describe the absence of background GPU activity, not GPU load while inference itself was running. The samples and timestamps are in [`idle-conditions.json`](idle-conditions.json).

### What was measured

Each flavor was measured with ONNX Runtime CUDA FP32, TensorRT FP32 and TensorRT FP16 at batch sizes 1, 4 and 8. Each backend/batch has three interleaved rounds, 20 warm-up requests per round, and 50 timed requests per round. The main tables report the median of the 150 timed requests; the raw JSON also records round medians and p95.

The timing boundary remains `cudaDeviceSynchronize -> perf_counter_ns -> inference call -> cudaDeviceSynchronize`, with synthetic GPU-resident input and preallocated GPU outputs. It includes the Python call and synchronization overhead but excludes image preprocessing, data transfers and the video pipeline. YOLOv7 is measured on its raw graph **without EfficientNMS**; these are not timings of its deployed NMS-containing engine.

**The engines were rebuilt, not reused.** The previous temporary engines had been removed. Official weights were verified against the unchanged SHA-256 pins in `flavors/`, then re-exported with the pinned exporter recipes and rebuilt with the same TensorRT flags: `--noTF32`, optional `--fp16`, 1024 MiB workspace, and min/opt/max batch 1/8/8 at 640 pixels. Export used PyTorch 2.15.0, ONNX 1.23.2, Ultralytics 8.4.215 for YOLO26/YOLOv10, and the pinned YOLOv7 source with ONNX Simplifier 0.8.1.

The new ONNX and engine hashes differ from the previous run and are recorded in each latency JSON. **The accuracy files still describe the earlier engines** evaluated on all 5,000 COCO val2017 images. Their mAP values have not been changed or claimed as a fresh accuracy validation of the rebuilt engines.

### Previous contention run versus idle rerun

The previous run had another video pipeline sharing the GPU, with a separately observed background load averaging about 55% (24-89%). Its raw latency reports are preserved in [`previous-contention/`](previous-contention/).

| Model | Previous batch 1 (ms/image) | Idle batch 1 (ms/image) | Reduction | Previous batch 8 (ms/image) | Idle batch 8 (ms/image) | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26n | 7.56 | 5.31 | 29.8% | 5.83 | 3.42 | 41.3% |
| YOLOv10n | 8.92 | 5.50 | 38.3% | 6.25 | 3.76 | 39.8% |
| YOLOv7-tiny | 12.72 | 8.40 | 34.0% | 8.66 | 4.74 | 45.2% |

The new timings are lower for every model. **These differences do not isolate the cost of the stopped pipeline:** rebuilding can select different TensorRT tactics, and the exported graphs are not byte-identical. This is a comparison of two recorded runs, not a controlled same-engine contention experiment.

### Alternating-engine check

After the main measurements, the three rebuilt FP16 engines were timed in alternating order for five rounds at batches 1 and 8 using `trtexec --warmUp=500 --duration=4 --iterations=50 --noDataTransfers`. The table reports the median of the five per-run GPU-compute medians; all 30 measurements are in [`gpu-crosscheck.json`](gpu-crosscheck.json).

| Model | Batch 1 (GPU ms) | Batch 8 (GPU ms per batch of 8) |
|---|---:|---:|
| YOLO26n | 5.09 | 26.91 |
| YOLOv10n | 5.27 | 29.66 |
| YOLOv7-tiny | 8.12 | 37.42 |

Every round kept the same ordering: YOLO26n, YOLOv10n, then YOLOv7-tiny. GPU-only times use a different timing boundary from the per-request tables; do not mix the two.

### Published figures remain context

Ultralytics publishes **4.13 ms/image** for YOLO26n TensorRT FP16 on an Orin NX 16GB ([NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/), Ultralytics 8.4.33, inference without pre- and post-processing). Our idle rerun gives **5.09 ms GPU-only** and **5.31 ms per request**, numerically about 23% and 29% higher. The idle rerun narrowed the recorded gap, but did not establish latency parity.

Matching TensorRT versions, power/clock settings, export/build choices and timing boundaries have not been established; the published benchmark table does not establish that its device was idle. These numerical gaps do not identify a cause. The T4 figures on the model pages use a different GPU.

The Jetson guide's reproduction command uses `coco128.yaml`, not full COCO val2017. Its 0.4789 YOLO26n FP16 mAP is not comparable to our earlier full-COCO 0.3992; use the model page's **40.1 end-to-end** reference instead.

### Repeating the latency measurement

Stop only workloads you own or have permission to stop, verify the GPU baseline, and use a **new work directory** for each build-and-measure run:

```bash
for flavor in yolo26 yolov10 yolov7; do
    python3 tools/export_model.py "$flavor"
    python3 tools/measure.py "$flavor" --only latency \
        --work "tmp/measure-idle/$flavor" --results tmp/benchmarks-idle
done
```

Keep the resulting ONNX graphs and raw FP32/FP16 engines for future retiming with `tools/benchmark.py run`; reusing the exact engines avoids the rebuild confound. Do not present retained accuracy results as a new evaluation when updating only latency. After reviewing and copying the new records into this folder, update these conditions and regenerate the report with `python3 tools/compare_models.py doc/benchmarks/orin-nx-16gb-jp6.0`.
