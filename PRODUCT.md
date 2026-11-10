# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

EdgeAI Station operators and demo users who need to confirm that a deployed
video analytics pipeline is running and understand its object counts in real
time.

## Product Purpose

AIPV1EMI0001 is a self-contained AI Processor vertical for real-time YOLOv7
object detection. Its bundled Node-RED logic and dashboard make the live HLS
stream, class counts, ROI counts, and line-crossing counts directly observable
from the station web UI.

## Positioning

The detector, callback contract, Node-RED normalization logic, and operational
dashboard are distributed together as one deployable AIP package rather than
as separately maintained demo resources.

## Operating Context

The AIP runs on an EdgeAI Station pipeline. Operators open its assigned
UIBuilder dashboard from the station web UI while monitoring one or more video
sources. Node-RED receives callback data from MS3, normalizes it, and streams
the result to the browser.

## Capabilities and Constraints

- Detects `car`, `person`, and `truck`.
- Supports up to 8 input streams.
- Supports ROI and line-crossing analytics when configured on the pipeline.
- Uses the authenticated MS3 HLS endpoint for video.
- Must work offline; browser dependencies are installed locally through
  UIBuilder and no CDN is permitted.
- The dashboard is a Vue 3 single-page application with no build step.
- Callback payload shape varies with MS3 merge and push configuration, so
  Node-RED owns normalization and the browser consumes a stable envelope.
- Snapshot controls are excluded: a new live snapshot would not match the
  delayed HLS frame currently displayed.

## Evidence on Hand

- `callback.py` defines the source payload.
- `doc/USER_MANUAL.md` documents passthrough and merge examples.
- `manifest.json` defines classes and supported analytics.
- `resources/logo.png` is the existing AIP identity asset.

## Product Principles

- Show operational truth from live pipeline data.
- Keep the AIP vertical portable between stations.
- Remain usable in air-gapped deployments.
- Prefer immediate scanability over analytical complexity.
- Treat browser-originated input as untrusted.
