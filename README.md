# AIPV1EMI0001 — YOLOv7 Tiny AiProcessor

![AIPV1EMI0001 — real-time object detection for traffic and industrial environments](doc/assets/hero.jpg)

An [EdgeAI Station](https://edgematrix.com) AiProcessor (AIP) providing general-purpose object detection using a reparameterized [YOLOv7-tiny](https://github.com/WongKinYiu/yolov7) model, running on NVIDIA DeepStream with Triton Inference Server.

## What it does

Detects 3 generic object classes in video streams:

- `car`
- `person`
- `truck`

Tracking is handled by the NVIDIA NvDCF multi-object tracker. The AIP supports the `ROI` (region of interest) and `LCC` (line crossing counter) analytic plugins, and up to 8 input streams.

## Repository layout

| Path | Description |
|---|---|
| `manifest.json` | AIP descriptor (ID, version, hardware targets, processors, analytics) |
| `store.json` | EdgeAI Store listing metadata |
| `callback.py` | User callback — transforms DeepStream batch metadata into AIP metadata |
| `logic/` | Importable Node-RED project: callback subscription and count normalization |
| `dashboard/` | Offline Vue 3 UIBuilder SPA with HLS, class, ROI and line-crossing views |
| `configs/` | DeepStream inference config (`nvinferserver` / Triton) and NvDCF tracker config |
| `models/AIPV1EMI0001_Yolo_Detector/` | Triton model repository: `config.pbtxt`, labels. Model weights are **not** stored here — they are downloaded from Hugging Face (see below) |
| `src/nvdsinfer_customparser/` | Custom bounding-box parser plugin (`libnvds_infercustomparser.so`) for `nvinfer` |
| `bin/build.sh` | Builds the custom parser plugin and the TensorRT engine with `trtexec` |
| `localdev.ipynb` | Step-by-step notebook: download weights → explicit build → optional backend benchmark → package → register → test → install dashboard |
| `doc/` | User manual and integration notes |
| `.github/workflows/release.yml` | CI: on `v*` tag, packages the AIP archive and publishes a GitHub release |

## Requirements

- EdgeAI Station (EX3 / EX5 / EX7 hardware targets) or a Jetson host (Jetson only — x86 NVIDIA GPU hosts are not supported) with:
  - DeepStream 7.0+
  - TensorRT (`trtexec`)
  - CUDA toolkit (to build the custom parser plugin)

## Model weights

The trained weights are not stored in this repository. They are published on Hugging Face at
[`Edgematrix-JP/AIPV1EMI0001`](https://huggingface.co/Edgematrix-JP/AIPV1EMI0001):

| File | Description |
|---|---|
| `yolov7tiny-640.onnx` | **Primary artifact** — ONNX export with `EfficientNMS_TRT`, input for the engine build |
| `yolov7-fp16-1x8x8-jp6-l4t36.3-arm64.engine` | Prebuilt TensorRT FP16 engine (JetPack 6 / L4T R36.3 / aarch64 only) |
| `SHA256SUMS` | Checksums for download verification |

The weights are licensed under **GPL-3.0**, matching the upstream YOLOv7 training code.

The `localdev.ipynb` notebook downloads the ONNX and verifies it automatically.

## Build, package & deploy

The full workflow is a step-by-step guide in the [`localdev.ipynb`](localdev.ipynb) notebook:
downloading the weights, building the runtime artifacts (or using the prebuilt engine on
JetPack 6 / TensorRT 8.6.2), packaging with `eais-tools`, registering the AIP, running a test
pipeline that checks the callback output, and installing the Node-RED dashboard.

The v6.0.0 notebook makes command failures stop execution and verifies the exported
`AIPV1EMI0001_dist.tgz` against `dist/`, file by file with SHA-256. Cleanup preserves both
the package and `dist/`. Both the ONNX-build and prebuilt-engine paths were executed
top to bottom on a Jetson Orin NX 16GB with JetPack 6.0 /
L4T R36.3 / TensorRT 8.6.2 / DeepStream 7.0, with `eais-tools` 0.9.2.
Allow 30–60 minutes to compile an engine on a busy Orin NX; the prebuilt option avoids this.
The runs verified package contents and checksums, inference, callback metadata, and
dashboard dependency installation. Activating and viewing the dashboard remains a manual step.
Use a new `NR_PROJECT` name when testing on a station that already has the default
Node-RED project; the notebook refuses to overwrite it unless explicitly configured.

The current notebook exposes the parser's `make` command and the deployment engine's
`trtexec` flags, with guidance on FP16, workspace and dynamic batch profiles. Section **2A**
compares direct ONNX Runtime CUDA FP32 inference with TensorRT FP32 (`--noTF32`) and FP16
engines **using one timing harness**: the same NMS-free graph, one shared GPU-resident
input per batch (1/4/8), preallocated GPU outputs, one request at a time, and
`cudaDeviceSynchronize` before and after every timed call (see
[Comparable ONNX/TensorRT measurements](#comparable-onnxtensorrt-measurements)).
`trtexec` only builds the engines; the TensorRT Python runtime executes them.
Numerical output differences are recorded, not treated as detection-accuracy validation.
Benchmark models, logs and JSON results stay under `tmp/inference-benchmark/`, outside
the AIP package; the deployment graph retains NMS. The benchmark runs locally without
a model server and does not restart station services.
EAIS Tools remain the workflow for packaging and pipeline tests.
Set `RUN_BENCHMARKS = False` in step 1 to skip the two extra engine builds and measurements.

**Packaging safeguards.** The notebook explicitly passes `.include` to EAIS Tools and
verifies its resolved file list before building. All entries are individual files:
EAIS Tools 0.9.2 does not honor `.gitignore` or `!` exclusion rules, and recursively
includes directories. It also copies `configs/` and model version `1/` without source
filtering. The notebook requires the exact intended distribution file set and hashes,
then verifies the exported archive against that snapshot. Unexpected files cause a
failure, not silent removal.

**Configuration validation.** Mandatory static checks use EAIS's Triton and DeepStream
validators, plus tracker YAML/library-path checks. Triton validation targets the
deployment file directly: AIP-wide validation also scans old experiment configs.
**Runtime acceptance uses `eais-tools pipeline test`**, exercising the actual
station-managed inference, NvDCF tracker and callback with display output disabled.
It requires detected objects and valid callback metadata, not merely a start response
or timeout. EAIS Tools 0.9.2's `configs ds check` does not exercise `nvtracker`,
so the full pipeline test is required to validate tracking.
Before testing/cleanup, ownership checks prevent the EAIS CLI from replacing or
deleting existing test resources belonging to another run.

### Comparable ONNX/TensorRT measurements

A fair comparison needs **the same timing boundary** for every backend.
[NVIDIA's timing definitions](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html#trtexec)
show why `trtexec` cannot simply be placed next to a Python ONNX Runtime loop: its GPU
Compute Time excludes host dispatch, its Host Latency sums transfer and GPU times, and it
pipelines submissions. Section 2A therefore follows Ultralytics's normal prediction path:
[its inference-only timing block](https://github.com/ultralytics/ultralytics/blob/80c34c050921abd29970d8e3d2a2a41ec6538c98/ultralytics/engine/predictor.py#L380-L397)
uses [the same synchronized wall-clock profiler](https://github.com/ultralytics/ultralytics/blob/80c34c050921abd29970d8e3d2a2a41ec6538c98/ultralytics/utils/ops.py#L20-L76)
around each backend. (Its separate
[`ProfileModels.profile_onnx_model`](https://github.com/ultralytics/ultralytics/blob/80c34c050921abd29970d8e3d2a2a41ec6538c98/ultralytics/utils/benchmarks.py#L512-L585)
runs ONNX on CPU, so it is not a GPU-to-GPU reference.)

The notebook's harness, in one Python process:

- ONNX Runtime uses only `CUDAExecutionProvider` with `use_tf32=0`, matching the
  TensorRT engines' `--noTF32`; TensorRT FP16 is built with `--fp16 --noTF32`.
- The **actual `.engine` files** are executed by TensorRT's Python runtime
  (`execute_async_v3`). ONNX Runtime's TensorRT provider is not used: it would build
  its own engine instead of measuring the supplied one.
- All variants read the same GPU input buffer and write preallocated GPU outputs.
- Per request: `cudaDeviceSynchronize` → `perf_counter_ns` → one call →
  `cudaDeviceSynchronize`. 20 warm-up and 50 timed requests per round, 3 rounds, with
  backend order shuffled each round.
- Results: median, p95 and mean latency, ms/image, serial images/s and speed-up versus
  ONNX Runtime at the same batch; outputs are compared after timing. GPU load/frequency
  and software versions are recorded; clocks and power mode are not changed.

MS8 has TensorRT 8.6.2.3 libraries but no TensorRT Python bindings. The notebook
does not install anything system-wide: it downloads `python3-libnvinfer` at exactly
the installed `libnvinfer8` version from the configured NVIDIA Jetson APT repository,
checks its hash and exact dependencies, extracts it under `tmp/` and adds it to the
kernel's `sys.path`. A current PyPI `tensorrt` wheel would not match the station.

Reference run of the Section 2A cells (Orin NX 16 GB, L4T R36.3, TensorRT 8.6.2,
ONNX Runtime 1.19.0; median per-request latency, speed-up versus ONNX Runtime):

| Batch | ONNX Runtime CUDA FP32 | TensorRT FP32 | TensorRT FP16 |
|---|---|---|---|
| 1 | 49.8 ms | 38.7 ms (1.29×) | 12.0 ms (4.15×) |
| 4 | 181.9 ms | 125.7 ms (1.45×) | 34.6 ms (5.26×) |
| 8 | 317.3 ms | 244.5 ms (1.30×) | 63.3 ms (5.01×) |

The station was running its normal workload (GPU load 55–90 %, clock 918 MHz), so
absolute times are higher than on an idle device; compare ratios, and re-run for
your own station. TensorRT FP32 matched ONNX Runtime closely (max raw score difference
≈ 3×10⁻⁷). FP16 raw scores differed by at most 4×10⁻⁴; some raw box coordinates moved
by up to 14 px. The input is random, so this is a numerical sanity check, not an
accuracy evaluation.

The tag-triggered release workflow compiles the ARM64 parser in a pinned public DeepStream
7.0 SDK image, adds the checksum-verified prebuilt engine, and exports an archive with a
`dist/` root, matching `eais-tools aip export`. This image lacks EAIS Tools: CI uses a
strict assembly fallback with explicit `.include` files and exact file/hash checks,
not the EAIS CLI. Its output was compared with the real station builder; the notebook
remains the workflow that executes EAIS validation/build/export. A manual workflow run validates packaging
and saves an Actions artifact without publishing a release.

Run it in the station's Jupyter environment (App-Builder) from a clone at
`/data/aip/AIPV1EMI0001`:

```bash
cd /data/aip && git clone https://github.com/edge-ai/AIPV1EMI0001.git
```

## Integrated Node-RED dashboard

The AIP package includes a Node-RED project under [`logic/`](logic/) and its
UIBuilder application under [`dashboard/`](dashboard/). Together they provide:

- live HLS playback from `/ms3/pipeline/<pipeline-id>/hls`;
- per-class counts for `car`, `person`, and `truck`;
- per-source filtering;
- ROI and line-crossing counters when those analytics are configured.

Step 6 of [`localdev.ipynb`](localdev.ipynb) installs it as the Node-RED project
`nodered-aipv1emi0001-dashboard`, wired to a pipeline. To install it by hand:
copy `logic/` to a new project under the Node-RED projects directory, copy the
contents of `dashboard/` into that project's `uibuilder/` directory and run
`npm ci --omit=dev` there, then set `AIP_PIPELINE_ID` in the **AIPV1EMI0001
Logic** tab's Environment properties and deploy. The Node-RED nodes
(`@edge-ai/node-red-contrib-eais`, `node-red-contrib-uibuilder`) are
preinstalled on EdgeAI Stations. Open the dashboard at
`https://<station>/ms5/aipv1emi0001_dash/` and assign `aipv1emi0001_dash` to
users from Accounts management.

`flows_cred.json` is deliberately not distributed. If the project needs
credentials, configure Node-RED's `credentialSecret` through the station's
deployment environment, then let Node-RED generate the local credential file.
Never commit or redistribute that file.

The callback payload itself remains the source contract. Enable the flow's
disabled **inspect complete callback msg** debug node after MS3 or callback-node
upgrades and compare the live message with [`doc/USER_MANUAL.md`](doc/USER_MANUAL.md).

## Attribution

The model architecture is **YOLOv7** by WongKinYiu et al.:

> Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao.
> *"YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors."*
> CVPR 2023. <https://github.com/WongKinYiu/yolov7> — GPL-3.0 License.

This is a general-purpose YOLOv7-tiny integration adapted by EdgeMatrix for EdgeAI Station,
exposing `car`, `person` and `truck`. It is not a parking-lot-specific model.
The exact upstream checkpoint provenance of the published export is not documented here.

## Contributors

This AI Processor was originally developed at EdgeMatrix by:

- [@antowan](https://github.com/antowan)
- [@aohta-emi](https://github.com/aohta-emi)
- [@yosukehara](https://github.com/yosukehara)

## License

- **Source code** in this repository: Apache-2.0 — see [LICENSE](LICENSE). It contains no
  YOLOv7 source code.
- **Model weights** (ONNX and TensorRT engines on Hugging Face, and the engine inside the packaged
  AIP): GPL-3.0, because they were trained with the
  [GPL-3.0 YOLOv7 code](https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md).
  `manifest.json` therefore declares `Apache-2.0 AND GPL-3.0-only`.
  The packaged weights' licence text is included as [LICENSE.weights](LICENSE.weights).

Third-party files keep their own MIT licence notices:
[`src/nvdsinfer_customparser/`](src/nvdsinfer_customparser/) and
[`configs/config_tracker_NvDCF_perf.yml`](configs/config_tracker_NvDCF_perf.yml), both from NVIDIA.
