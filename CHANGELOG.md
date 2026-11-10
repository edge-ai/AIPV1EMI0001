# Changelog

## v6.1.0 - Unreleased

- docs: Rewrite `localdev.ipynb` as a command-first build guide based on `eais-tools`, with a build sequence diagram; keep architecture explanations in the presentation
- fix: Distinguish missing callback frames from zero detections and aggregate only reported analytics values
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
