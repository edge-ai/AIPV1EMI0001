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
