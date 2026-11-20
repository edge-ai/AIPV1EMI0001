# AIPV1EMI0001 · Multi-model object detection for the EdgeAI Station

This repository is a working **AI Processor (AIP)** for the [EdgeAI Station](https://edgematrix.com) and a tutorial for building one. It detects and tracks **cars, people and trucks** in camera feeds or video files. You can count objects in a region, count crossings over a line, and see the results by camera on the included Node-RED dashboard.

![YOLO object detection for traffic and industrial environments](doc/assets/hero.jpg)

## How it works

**One AI Processor, one detector per build.** The three folders under [`flavors/`](flavors/) are recipes, not separate AIPs. The v7.0.0 release uses YOLO26n; building another flavor changes the detector, not the `AIPV1EMI0001` ID. The top-level `configs/` and `models/` are the [layout the station reads](doc/USER_MANUAL.md#where-model-settings-live), not a legacy detector.

NVIDIA DeepStream decodes the video, runs the selected TensorRT engine through Triton, and tracks detections across frames. The same [callback](callback.py) turns those detections into counts for the dashboard. All three models use official pretrained weights; no training dataset is needed.

## Choose a model

YOLO26n is the default and a good first build. YOLOv10n scored almost identically on the three application classes; the compact anchor-based model is the original flavor. Each model has its own build notebook.

| Model and build notebook | mAP 50:95, 80 COCO classes (published) | mAP 50:95, car / person / truck | FP16 ms per image | 720p30 cameras, no output | 720p30 cameras, all outputs |
|---|---:|---:|---:|---:|---:|
| [YOLO26n](notebooks/03-build-yolo26.ipynb) | 39.92 (40.1) | 40.67 | 5.3 | 7 | 7 |
| [YOLOv10n](notebooks/02-build-yolov10.ipynb) | 38.34 (38.5) | 40.54 | 5.5 | 6 | 6 |
| [Tiny anchor-based model](notebooks/01-build-yolov7.ipynb) | 37.13 (n/a) | 39.41 | 8.4 | 4 | 4 |

Measured on one **Jetson Orin NX 16 GB** (JetPack 6.0, TensorRT 8.6, FP16 engines) from 17 to 19 November 2026. Accuracy uses all 5,000 COCO val2017 images; the authors' published score is in brackets. Inference time is for one image on an idle GPU, without decoding or tracking. A camera count passed when every 30 FPS H.264 camera averaged at least **29 processed FPS**. *All outputs* means continuous recording, WebRTC, HLS and event recording with on-screen boxes; they used part of the margin but didn't lower the counts. 1080p cameras without output gave the same counts.

The [comparison notebook](notebooks/04-compare-models.ipynb) shows every table, the chart and the test dates without running code. The [capacity report](doc/benchmarks/orin-nx-16gb-jp6.0/capacity.md) has the raw trials and known limitations, including one-camera WebRTC and recording failures on the test build.

## Get started

**New to the station?** [Follow one frame from camera to count](notebooks/00-presentation.ipynb). This introduction needs no hardware.

**To build**, open an App-Builder (Jupyter) terminal on a development station. The reference stack is JetPack 6.0, TensorRT 8.6 and DeepStream 7.0. If the project isn't already installed:

```bash
cd /data/aip && git clone https://github.com/edge-ai/AIPV1EMI0001.git
```

Open the build notebook for your chosen model under `AIPV1EMI0001/notebooks/`. It takes you from pretrained weights to an installable package, then tests it on sample video. Don't build over an AIP that is already in use.

**To install a package or connect the dashboard**, see the [user manual](doc/USER_MANUAL.md).

## Prebuilt TensorRT engines

Three SHA-256-verified **v7.0.0** engines are available from the [Hugging Face model repository](https://huggingface.co/Edgematrix-JP/AIPV1EMI0001/tree/ae3bd8c672013b3bb04a9af262130dcbaa41a109/orin-nx-16gb-jp6.0). These exact binaries were used in the camera-capacity tests. They target a Jetson Orin NX 16 GB with JetPack 6.0 / TensorRT 8.6; rebuild for a different GPU or TensorRT version.

| Model | Download | SHA-256 |
|---|---|---|
| YOLO26n | [TensorRT engine](https://huggingface.co/Edgematrix-JP/AIPV1EMI0001/resolve/ae3bd8c672013b3bb04a9af262130dcbaa41a109/orin-nx-16gb-jp6.0/yolo26-fp16-1x8x8.engine) | `0cb26f248c60759893471f9966d293b62faa812a7d98bd69183d0c99a7f96802` |
| YOLOv10n | [TensorRT engine](https://huggingface.co/Edgematrix-JP/AIPV1EMI0001/resolve/ae3bd8c672013b3bb04a9af262130dcbaa41a109/orin-nx-16gb-jp6.0/yolov10-fp16-1x8x8.engine) | `8add86b6a31ec672381cf5b23d56f62c8cfcb0f30812d1591d039cc67300c85f` |
| Tiny anchor-based model | [TensorRT engine](https://huggingface.co/Edgematrix-JP/AIPV1EMI0001/resolve/ae3bd8c672013b3bb04a9af262130dcbaa41a109/orin-nx-16gb-jp6.0/yolov7-fp16-1x8x8.engine) | `c9da0f95ec3f10cc2d5cd213dbfc364a932478a4e884b1d873cf69919842fa5c` |

The [user manual](doc/USER_MANUAL.md#tensorrt-engine-files) covers engine compatibility and release artifacts.

## Licenses

The [source code](LICENSE) is Apache-2.0, with NVIDIA components retaining their MIT notices. The [model weights](licenses/) have separate terms: GPL-3.0 for the tiny anchor-based model; AGPL-3.0 for YOLOv10n and YOLO26n. See [project and model credits](doc/README.md#credits).
