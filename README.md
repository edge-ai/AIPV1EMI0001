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
| `localdev.ipynb` | Step-by-step notebook: download weights → build → package → register → test → install dashboard |
| `doc/` | User manual and integration notes |
| `.github/workflows/release.yml` | CI: on `v*` tag, packages the AIP archive and publishes a GitHub release; internal store publishing requires secrets |

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

The tag-triggered release workflow compiles the ARM64 parser in a pinned public DeepStream
7.0 SDK image, adds the checksum-verified prebuilt engine, and exports an archive with a
`dist/` root, matching `eais-tools aip export`. A manual workflow run validates packaging
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
