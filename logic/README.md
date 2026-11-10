# AIPV1EMI0001 Node-RED logic

This Node-RED project subscribes to an AIPV1EMI0001 pipeline, normalizes its
three supported callback delivery shapes, and sends a stable metrics envelope
to the bundled UIBuilder dashboard.

## Runtime requirements

The Node-RED runtime must load `@edgematrix/node-red-contrib-eais` **0.7.10**
and `node-red-contrib-uibuilder` before this project is deployed. The EAIS
package requires Node-RED 4 or later, Node.js 20 or later, and npm 10 or later.
The project manifest also pins UIBuilder 7.5.0. The installed Node-RED version
can require a newer Node.js version than the EAIS package itself.

The EAIS package declares Node-RED as an optional peer, so it does not install
a separate runtime. The local integration-test harness separately pins Node-RED
**4.1.13** and is checked with Node.js **22**. Install that harness only in a
development checkout; it does not install or upgrade the separately managed
station runtime.

Some stations have the legacy `@edge-ai/node-red-contrib-eais` installed. It
registers the same node types as the public package: **do not install both
variants in one Node-RED instance**. Have the station administrator migrate
the runtime using the station-supported procedure and arrange any restart at
an approved time. Confirm the loaded package and version in **Manage palette**
before deploying this flow.

On a fresh, unmanaged Node-RED instance with no legacy EAIS package, install
the nodes from the Node-RED user directory and restart Node-RED:

```bash
npm install --save-exact --registry=https://registry.npmjs.org/ \
  @edgematrix/node-red-contrib-eais@0.7.10 node-red-contrib-uibuilder@7.5.0
```

Copying this project's `package.json` records its dependencies; it does not
by itself install or activate them in the Node-RED runtime. The notebook's
`bin/install-nodered-project.sh` installs browser libraries in `uibuilder/` only.

## Configure

1. Open this project in Node-RED.
2. Open **AIPV1EMI0001 callback** and select the pipeline running this AIP.
3. Deploy. Repeat these steps when switching pipelines.

The checked-in flow deliberately leaves the pipeline selection empty;
`bin/install-nodered-project.sh --pipeline <id>` preselects it in the installed copy. Select a pipeline
in the callback node rather than putting an environment-variable reference in
that field. When Node-RED admin authentication is enabled, the pipeline selector
requires `flows.read` permission, and refreshing the callback metadata cache
requires `flows.write`.

`flows_cred.json` is not part of this project. If credentials are needed, set
Node-RED's `credentialSecret` through the station deployment environment and
allow Node-RED to generate the local credential file. Never commit it.

The disabled **inspect complete callback msg** node is intentional. Enable it
when validating a new MS3/callback-node version because the callback delivery
envelope is not a platform-wide schema.

The EAIS library also provides a `pipeline` node for future business-logic
commands such as snapshots; no snapshot command path or automatic capture rule
is included in this flow. If a new UIBuilder instance is not served after
deployment, follow the station's UIBuilder instructions; a Node-RED restart
may be required.

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

From the repository root, run `node --test tests/dashboard.test.cjs tests/missing-metrics.test.cjs`. These
dependency-free checks cover callback delivery shapes, latest-window selection,
dashboard message handling, flow references, and the integration package file set.
To exercise the pinned EAIS node in a local Node-RED runtime with mocked station
services, run the following in a development checkout:

```bash
npm ci --prefix logic --ignore-scripts
npm ci --prefix tests/node-red-runtime --include=dev --ignore-scripts
node --test tests/node-red-runtime.test.cjs
```

The second install is a test-only runtime, not a station deployment. The test
does not exercise a live station.

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
      "available": true,
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
merge window. For passthrough data, it is the latest frame count. A source
with no observed frames has `available: false` and `null` class counts and
total; the all-source totals are also `null` when any source is unavailable.
The dashboard displays these values as `N/A`. Analytics labels are reported
only when their values were observed; absence does not prove that an analytic
is unconfigured. Totals combine reported per-source values and need not
represent a simultaneous frame. The timestamp is not a freshness alarm.
