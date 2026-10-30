# Changelog

## Unreleased

- feat: Compare ONNX Runtime CUDA and TensorRT FP32/FP16 using a shared timing harness
- fix: Validate configuration and distribution file hashes with EAIS Tools
- ci: Publish release archives and checksums to GitHub Releases
- fix: Clearly distinguish latest callback counts from delayed HLS playback; defer inferred synchronization
- fix: Exclude manual snapshots and unused browser command wiring from the dashboard
- chore: Pin `@edge-ai/node-red-contrib-eais` 0.6.24 for Node-RED integration
- docs: Describe the general-purpose EdgeAI Station integration without parking-dataset training claims
- fix: Rename the ONNX download to `yolov7tiny-640.onnx` across Hugging Face, the notebook and build script

## v6.0.0 - 2026-10-27

First public release.

- feat: YOLOv7-tiny car/person/truck detector for DeepStream 7.0 with embedded Triton inference and NvDCF tracking
- feat: Step-by-step Jupyter notebook to rebuild, package, register and test the AI Processor, with a prebuilt-engine option
- feat: Bundled Node-RED object-counting logic and an offline Vue 3 UIBuilder dashboard
- feat: Release workflow builds the ARM64 parser in a pinned public DeepStream SDK image and publishes the archive with its SHA-256 checksum
- docs: Code is licensed Apache-2.0 and model weights GPL-3.0; the runtime package includes both licence texts
