# Changelog

## v7.0.0 - 2026-11-17

- feat!: Support three interchangeable detector flavors, YOLOv7-tiny, YOLOv10n and YOLO26n, each in its own `flavors/<id>/` folder; `python3 tools/flavors.py use <id>` selects the one the package carries (`manifest.json`, `store.json`, the inference config, `config.pbtxt` and `LICENSE.weights` follow). The default is YOLO26n
- feat!: Replace the custom Hugging Face weights with the publishers' official COCO weights, fetched from pinned URLs and verified by SHA-256 in `tools/export_model.py`. The classes are unchanged: COCO `car`, `person` and `truck`
- fix: The earlier 12.64% COCO mAP 50:95 came from custom three-class weights scored against COCO labels, not from the evaluation, which reproduces the published figures for YOLO26n (Ultralytics, end-to-end head) and YOLOv10n (YOLOv10 paper) to within 0.2 points. The official weights score about 37 to 40 on the full COCO validation set (80 classes) and about 39 to 41 on the three AI Processor classes
- feat: Add `NvDsInferParseYoloEndToEnd` for the NMS-free YOLOv10 and YOLO26 output and extend the shared parser to map COCO class ids to the AI Processor classes; test both parsers with a stubbed DeepStream header
- feat: Build the parser and the TensorRT engine of any flavor with `bin/build.sh <flavor>`
- feat: Add `tools/measure.py` to measure latency and COCO accuracy of a flavor on a station and publish the results, and `tools/compare_models.py` to compare the measured flavors
- feat: Make `tools/benchmark.py` and `tools/accuracy.py` flavor-aware; report mAP 50:95, the metric Ultralytics uses, with per-class values for the three classes
- docs!: Replace `localdev.ipynb`, `presentation.ipynb` and `accuracy.ipynb` with `notebooks/`: a presentation, one build notebook per flavor, and a model comparison. Each notebook credits its author
- docs: Rewrite the README as a short introduction; move the build, measurement and accuracy steps into the notebooks
- docs: Replace the YOLOv7 reference benchmark with one latency and one accuracy result per flavor in `doc/benchmarks/orin-nx-16gb-jp6.0/`, with a comparison table and chart
- docs: Remeasure latency on 2026-11-18 with the video pipeline stopped and rebuilt engines; record idle GPU baselines and alternating-engine checks, preserve the previous contention timings, and keep the earlier accuracy evaluation separate
- docs: Remove the conference-specific wording from the notebooks
- ci: Publish the engine of the default flavor from its `engine_asset`; the release stops until that flavor's engine is published
- test: Cover flavors, the build script, the parser and the notebooks, and check that the README and the notebooks quote the published measurements

## v6.1.0 - 2026-11-10

- docs: Rewrite `localdev.ipynb` as a command-first build guide based on `eais-tools`, with a build sequence diagram; keep architecture explanations in the presentation
- fix: Distinguish missing callback frames from zero detections and aggregate only reported analytics values
- feat: Forward DeepStream's cumulative line-crossing count (`objLCCumCnt`) and show it on the dashboard next to the instant count
- feat: Detect stream and merged callback delivery automatically; in stream mode keep each input's latest frame and mark samples older than 2 s unavailable
- feat: Add a dashboard video source selector that controls the pipeline's shared tiler and follows changes made in the station web UI
- style: Restyle the dashboard with the EdgeAI visual identity and bundle the Inter font so it renders on offline stations
- ci: Check pull requests and pin the Hugging Face model revision used by notebooks and releases
- docs: Add `presentation.ipynb`, a code-free talk companion on engineering Edge AI applications, with the reference benchmark table and chart
- feat: Add standalone `bin/install-nodered-project.sh` to install the Node-RED flow and dashboard for a pipeline, outside the build notebook
- feat: Add `tools/benchmark.py all` to prepare, build and measure the benchmark with one optional notebook command
- docs: Use native EAIS Tools build/export in the notebook; keep additional package safeguards in release CI only
- feat: Move the package checks and the benchmark harness to `tools/check_package.py` and `tools/benchmark.py`; publish the reference results under `doc/benchmarks/`
- ci: Validate the release package and archive with `tools/check_package.py`
- test: Add offline tests for the notebooks, tools and published benchmark results
- fix: Preserve model files and TensorRT timing caches that existed before a notebook run
- fix: Scan only this AI Processor without force-unregistering model-server registrations
- docs: Refresh the inference benchmark with pipelines stopped and report observed CPU/GPU node placement
- feat: Add a labelled detection accuracy tutorial and evaluator alongside the inference latency benchmark
- docs: Publish the full COCO validation reference, including per-class detection metrics and limitations
- ci: Install CPU evaluation dependencies so accuracy regression tests run in release validation
- feat: Compare ONNX Runtime CUDA and TensorRT FP32/FP16 using a shared timing harness
- fix: Validate configuration and distribution file hashes with EAIS Tools
- ci: Publish release archives and checksums to GitHub Releases
- feat: Automatically select dashboard HLS or WebRTC playback from pipeline settings, matching the station web UI without a transport selector; counts remain the latest received metadata
- fix: Exclude manual snapshots and unused browser command wiring from the dashboard
- chore: Pin the public `@edgematrix/node-red-contrib-eais` 0.7.10 and isolate the Node-RED test runtime from the distributable project
- fix: Select the callback pipeline directly to avoid the 0.7.6 editor's environment-reference issue
- docs: Describe the general-purpose EdgeAI Station integration without parking-dataset training claims
- fix: Rename the ONNX download to `yolov7tiny-640.onnx` across Hugging Face, the notebook and build script

## v6.0.0 - 2026-10-27

First public release.

- feat: YOLOv7-tiny car/person/truck detector for DeepStream 7.0 with embedded Triton inference and NvDCF tracking
- feat: Step-by-step Jupyter notebook to rebuild, package, register and test the AI Processor, with a prebuilt-engine option
- feat: Bundled Node-RED object-counting logic and an offline Vue 3 UIBuilder dashboard
- feat: Release workflow builds the ARM64 parser in a pinned public DeepStream SDK image and publishes the archive with its SHA-256 checksum
- docs: Code is licensed Apache-2.0 and model weights GPL-3.0; the runtime package includes both licence texts
