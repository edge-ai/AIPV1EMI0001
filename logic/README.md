# AIPV1EMI0001 Node-RED logic

This Node-RED project subscribes to an AIPV1EMI0001 pipeline, normalizes its
three supported callback delivery shapes, and sends a stable metrics envelope
to the bundled UIBuilder dashboard.

## Configure

1. Open this project in Node-RED.
2. Open **AIPV1EMI0001 Logic** tab properties.
3. Set `AIP_PIPELINE_ID` to the pipeline running this AIP.
4. Deploy.

Do not select a pipeline in the callback node dropdown: that replaces the
portable `${AIP_PIPELINE_ID}` reference with a station-specific value.

`flows_cred.json` is not part of this project. If credentials are needed, set
Node-RED's `credentialSecret` through the station deployment environment and
allow Node-RED to generate the local credential file. Never commit it.

The disabled **inspect complete callback msg** node is intentional. Enable it
when validating a new MS3/callback-node version because the callback delivery
envelope is not a platform-wide schema.

## Dependencies

`package.json` declares `@edge-ai/node-red-contrib-eais` (callback and server
nodes) and `node-red-contrib-uibuilder`. The EAIS library also provides a
`pipeline` node for future business-logic commands such as snapshots; no
snapshot command path or automatic capture rule is included in this flow.

After the first deploy of a new UIBuilder instance, restart Node-RED once so
UIBuilder serves the project's `aipv1emi0001_dash` folder.

## Playback and metadata

The dashboard displays the latest received metadata independently of the delayed
HLS video. They are **not synchronized**. For batched delivery the normalizer uses
the last window in the batch; counts are window maxima where applicable, not
necessarily per-frame values. The timestamp is the normalization time, not the
video frame's capture time.

Segment-name/frame-number synchronization is not included: variable durations,
pipeline restarts, and differing source clocks make that inference unreliable.
Accurate alignment requires a shared timing contract between video and metadata.
The dashboard neither buffers callback windows for alignment nor sends commands
back to the station. A station snapshot would capture a newer frame than the one
visible in HLS, so manual snapshot controls and their wiring are excluded.

## Regression checks

From the repository root, run `node --test tests/dashboard.test.cjs`. These
dependency-free checks cover callback delivery shapes, latest-window selection,
dashboard message handling, flow references, and the integration package file set.

## Output contract

The normalizer sends:

```json
{
  "type": "aipv1emi0001.metrics",
  "timestamp": "2026-09-25T00:00:00.000Z",
  "aggregation": "latest frame or merge-window maximum",
  "pipeline": {"id": "...", "name": "..."},
  "hlsUrl": "/ms3/pipeline/<pipeline-id>/hls",
  "sources": [
    {
      "id": "0",
      "name": "Camera 1",
      "counts": {"car": 4, "person": 2, "truck": 1},
      "total": 7,
      "roi": {"parking": 3},
      "lineCrossing": {"entrance": 5}
    }
  ],
  "totals": {
    "objects": 7,
    "classes": {"car": 4, "person": 2, "truck": 1},
    "roi": {"parking": 3},
    "lineCrossing": {"entrance": 5}
  }
}
```

For merged callback data, the displayed count is `max_val` from the latest
merge window. For passthrough data, it is the latest frame count.
