# Changelog

## Unreleased

- benchmark: Compare all three models on EX3 and EX5 using one shared 720p30 feed. Publish 33 paired results with matching OSD, WebRTC and Smart Video settings, including bitrate and output-size changes. Report processing, publisher delivery and saved-clip quality separately.
- benchmark: Publish EX3 camera counts and protocol-v2 one-camera output checks for all three EX5 YOLO flavors. Report processed FPS separately from WebRTC delivery and Smart Video clip quality.
- chore: Stop tracking the selected flavor's inference config and `config.pbtxt` copies; `tools/flavors.py use` and `bin/build.sh` generate them from `flavors/<id>/`, and the release workflow regenerates them and checks that the committed manifest, store metadata and weights license match
- docs: Show every recorded result in the comparison notebook as text, so GitHub displays it without running code: accuracy, inference time, live camera counts with and without video outputs, output settings and what was not tested
- docs: Add 80-class mAP, inference time and EX5 no-output camera counts to the README results table, with links to the EX3/EX5 output-qualification matrix
- docs: Explain that the three flavors are build recipes for one AI Processor and that the top-level `configs/` and `models/` hold the selected flavor
- docs: Replace the Mermaid diagrams, which GitHub doesn't render in notebooks, with an SVG pipeline diagram and a one-line build flow
- test: Generate the published camera, headroom and output-screening tables from `capacity.jsonl` and check that the README and the comparison notebook quote them
- docs: State one camera-count rule and one set of one-camera output checks in the EX3 and EX5 reports, the EX3/EX5 matrix, the README and the comparison notebook. Both reports now follow the same section order and publish their output checks in one table format; the matrix lists each station's test conditions. EX5 YOLOv10n at 720p is now marked 6†, like EX3 YOLO26n, because seven cameras passed 1 of 2 attempts
- docs: Remove measurement dates from the public documentation
- test: Recompute every published camera count and one-camera output row from the evidence files using the shared rule, and check that each document quotes the same rule, output checks and † notes

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
- docs: Remeasure latency with the video pipeline stopped and rebuilt engines; record idle GPU baselines and alternating-engine checks, preserve the previous contention timings, and keep the earlier accuracy evaluation separate
- docs: Publish SHA-256-verified v7.0.0 TensorRT FP16 engines for all three model flavors from the Orin NX capacity tests; pin each immutable Hugging Face URL and checksum for release packaging
- docs: Replace the single-model hero illustration with a multi-model AIP image
- docs: Remove the conference-specific wording from the notebooks
- ci: Download and verify the active flavor's prebuilt engine from `engine_asset` when packaging a release; the release stops until that engine is published
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
