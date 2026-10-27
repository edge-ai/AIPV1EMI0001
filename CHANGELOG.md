# Changelog

## Unreleased

- docs: Describe the general-purpose EdgeAI Station integration without parking-dataset training claims
- fix: Rename the ONNX download to `yolov7tiny-640.onnx` across Hugging Face, the notebook and build script

## v6.0.0

- fix: Notebook stops on command failures, verifies the exported archive, and preserves the package during cleanup
- fix: Propagate parser compilation failures instead of potentially reusing a stale library
- docs: Include the model weights' GPL-3.0 licence text in the runtime package
- ci: Build the ARM64 parser with a pinned public DeepStream SDK image and preserve individual `.include` file paths
- ci: Match release tags to the manifest version, use the eais-tools archive layout, and publish package checksums
- feat: Bundle Node-RED object-counting logic and an offline Vue 3 UIBuilder dashboard
- docs: Model weights are licensed GPL-3.0 (YOLOv7 training code); manifest declares `Apache-2.0 AND GPL-3.0-only`
- fix: Ship the NvDCF tracker config referenced by the manifest (the tracker previously ran on built-in defaults)
- docs: Notebook covers prerequisites, prebuilt-engine option, callback contract check and dashboard install
- ci: Skip internal store upload and Slack notification when their secrets are not configured

## v5.1.0

- feat: Use internal triton for inference

## v5.0.0

- feat: Add analytic_plugins field to the manifest file

## v4.1.1

- upgrade: Recompile TRT engine file for DeepStream 7.0

## v4.1.0

- add: Jupyter Notebook for local build of the AiProcessor archive

## v4.0.1

- update: AiProcessor logo

## v4.0.0

- upgrade!: Migration to DeepStream 6.4

## v3.0.0

- update!: Improved callback function

## v2.2.0 - v2.3.0

- feat: Update callback functions again

## v2.1.0

- feat: Update callback functions
- fix: Makefile match libnvds_infercustomparser.so version

## v2.0.0 - v2.0.7

- upgrade(breaking change): change metadata adapter implementation
- fix: missing resources folder with logo
- fix: add missing .pbtxt file
- fix: add custom lib parser libnvds_infercustomparser.so
- fix: broken imports
- fix: rebuild engine file for DS 6.2
- fix: change the model config file
- fix: update custom lib parsers that were not being updated as intended
- feat: Adapt to the latest abstract class structure

## v1.0.0

- Initial version
