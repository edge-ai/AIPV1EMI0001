# AIPV1EMI0001: a YOLO object detector for the EdgeAI Station

![AIPV1EMI0001: real-time object detection for traffic and industrial environments](doc/assets/hero.jpg)

A small, complete **AI Processor** (AIP) for the [EdgeAI Station](https://edgematrix.com), built as tutorial material.
It detects `car`, `person` and `truck` in up to 8 video streams, tracks them, counts them in regions and across lines,
and shows the result on a dashboard. It runs on NVIDIA DeepStream and Triton Inference Server.

The detector is **your choice of three YOLO models**, with the same classes and the same application around them.
Each one has its own step-by-step notebook, so you can learn how an AI Processor is built and compare what each model costs on the station.

## Choose a model

| Flavor | Model | Detection head | License of the weights | Notebook |
|---|---|---|---|---|
| `yolov7` | [YOLOv7-tiny](https://github.com/WongKinYiu/yolov7) | Non-maximum suppression (NMS) inside the engine | GPL-3.0 | [`01-build-yolov7`](notebooks/01-build-yolov7.ipynb) |
| `yolov10` | [YOLOv10n](https://docs.ultralytics.com/models/yolov10/) | NMS-free | AGPL-3.0 | [`02-build-yolov10`](notebooks/02-build-yolov10.ipynb) |
| `yolo26` | [YOLO26n](https://docs.ultralytics.com/models/yolo26/) (default) | NMS-free | AGPL-3.0 | [`03-build-yolo26`](notebooks/03-build-yolo26.ipynb) |

The AI Processor ships one detector at a time; the flavors are alternatives, not three detectors in one package.
All three start from the publishers' official **COCO weights**, and the AI Processor reports three of the 80 COCO classes.

Measured on an Orin NX 16 GB with TensorRT FP16. Accuracy uses 5,000 COCO val2017 images and **mAP 50:95**, the metric Ultralytics publishes for its models; latency is measured separately:

| Model | mAP 50:95, 80 classes (published) | mAP 50:95, car / person / truck | FP16 ms per image, batch 1 | FP16 ms per image, batch 8 |
|---|---:|---:|---:|---:|
| YOLO26n | 39.92 (40.1) | 40.67 | 5.3 | 3.4 |
| YOLOv10n | 38.34 (38.5) | 40.54 | 5.5 | 3.8 |
| YOLOv7-tiny | 37.13 (n/a) | 39.41 | 8.4 | 4.7 |

The 80-class accuracy is within 0.2 points of the published figures: the end-to-end (NMS-free) value Ultralytics gives for YOLO26n, and the YOLOv10 paper value for YOLOv10n.
There is no comparable official figure for YOLOv7-tiny, and the three-class column is this repository's own metric.
Latency was remeasured on **2026-11-18 with the video pipeline stopped**: all 100 baseline GPU-load samples were 0%.
The latency rerun used freshly rebuilt engines; the accuracy column retains the earlier evaluation, not a new accuracy test.
These are raw model times, not DeepStream pipeline throughput or a matched comparison with published Jetson benchmarks
([conditions and previous timings](doc/benchmarks/orin-nx-16gb-jp6.0/conditions.md)).

The full tables, with the chart and the method, are in [`doc/benchmarks/orin-nx-16gb-jp6.0/comparison.md`](doc/benchmarks/orin-nx-16gb-jp6.0/comparison.md).
[`04-compare-models`](notebooks/04-compare-models.ipynb) explains how to read them and how to repeat the measurement on your own station.

## Quick start

On the station, open the **App-Builder** (Jupyter) terminal:

```bash
cd /data/aip && git clone https://github.com/edge-ai/AIPV1EMI0001.git
```

Then open `/data/aip/AIPV1EMI0001/notebooks/03-build-yolo26.ipynb` (or the notebook of another flavor) and run it from top to bottom.
It exports the model, builds the parser and the TensorRT engine, packages the AI Processor with EAIS Tools, scans it and tests it in a pipeline.
Built and measured on an Orin NX 16 GB with JetPack 6.0 (L4T R36.3), TensorRT 8.6 and DeepStream 7.0. The engine build is the long step: about 30 minutes for YOLOv7-tiny on a station that was also running another pipeline.

## Learn

| Notebook | What it is for |
|---|---|
| [`00-presentation`](notebooks/00-presentation.ipynb) | Start here: what an AI Processor is, how this one is designed, measured and validated. No code to run |
| [`01-build-yolov7`](notebooks/01-build-yolov7.ipynb), [`02-build-yolov10`](notebooks/02-build-yolov10.ipynb), [`03-build-yolo26`](notebooks/03-build-yolo26.ipynb) | The hands-on build, one per flavor: same steps, one model each |
| [`04-compare-models`](notebooks/04-compare-models.ipynb) | Speed and accuracy of the three flavors side by side, and how to measure your own |

## Repository map

| Path | What it contains |
|---|---|
| `notebooks/` | The tutorial notebooks above |
| `flavors/` | One folder per model: `flavor.json` (weights, export recipe, license) and its Triton and DeepStream configuration |
| `manifest.json`, `store.json`, `configs/`, `models/` | The active flavor, as EAIS Tools reads it (`python3 tools/flavors.py use <flavor>` writes them) |
| `callback.py` | Turns DeepStream detections into the metadata other station services receive |
| `src/nvdsinfer_customparser/` | C++ parser that converts each model's output tensor into DeepStream detections ([details](src/nvdsinfer_customparser/README)) |
| `bin/build.sh` | Builds the parser and the TensorRT engine for a flavor |
| `tools/` | Model export, benchmark and COCO accuracy tools, and the package checks |
| `logic/`, `dashboard/` | Node-RED flow and dashboard, installed with `bin/install-nodered-project.sh` |
| `doc/` | [User manual](doc/USER_MANUAL.md) and [benchmarks](doc/benchmarks/) |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

## Dashboard

The package includes a Node-RED project ([`logic/`](logic/)) and its dashboard ([`dashboard/`](dashboard/)): live video, per-class counts, per-source
filtering, and region and line-crossing counters. It needs `@edgematrix/node-red-contrib-eais` **0.7.10** on the station. Installation, user access and the
callback payload are described in the [user manual](doc/USER_MANUAL.md) and in [`logic/README.md`](logic/README.md).

## Licenses

- **Source code** of this repository: Apache-2.0 ([`LICENSE`](LICENSE)). It contains no YOLO training or model code.
- **Model weights** keep the license of their publisher, and the package declares it in `manifest.json`: YOLOv7 weights are GPL-3.0, YOLOv10 and YOLO26 weights are
  AGPL-3.0 ([`licenses/`](licenses/)). Read the license of the flavor before you distribute a package, or ask Ultralytics about an enterprise license.
- The parser in `src/nvdsinfer_customparser/` and `configs/config_tracker_NvDCF_perf.yml` come from NVIDIA and keep their MIT notices.

## Credits

Models: [YOLOv7](https://github.com/WongKinYiu/yolov7) by Chien-Yao Wang, Alexey Bochkovskiy and Hong-Yuan Mark Liao (CVPR 2023);
[YOLOv10](https://github.com/THU-MIG/yolov10) by Ao Wang et al. (NeurIPS 2024); YOLO26 by [Ultralytics](https://docs.ultralytics.com/models/yolo26/).

Originally developed at EdgeMatrix by [@antowan](https://github.com/antowan), [@aohta-emi](https://github.com/aohta-emi) and [@yosukehara](https://github.com/yosukehara).
