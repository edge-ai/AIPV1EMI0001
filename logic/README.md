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
by itself install or activate them in the Node-RED runtime. The standalone
`bin/install-nodered-project.sh` installs browser libraries in `uibuilder/` only.

## Install the project

Deployment is independent of `localdev.ipynb`. From the AIP root in a station
App-Builder terminal, run:

```bash
bash bin/install-nodered-project.sh --pipeline <pipeline-id>
```

Choose the ID of an existing pipeline running this AIP. The installer requires
`python3`, `jq`, `npm`, `git`, `tar`, npm registry access and write access to `/data/projects`
(override with `NODE_RED_PROJECTS_DIR`). `--project <name>` chooses another project
name; `--force` replaces an existing copy after preparation succeeds. It does not
open the project or restart Node-RED.

## Configure

1. Open this project in Node-RED.
2. Open **AIPV1EMI0001 callback** and select the pipeline running this AIP.
3. Deploy. Repeat these steps when switching pipelines.

The checked-in flow deliberately leaves the pipeline selection empty;
`bin/install-nodered-project.sh --pipeline <id>` preselects it in the installed
copy. Select a pipeline in the callback node rather than putting an environment-variable reference in
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

## Video and metadata

The dashboard reads the pipeline's output settings from MS3 and automatically
uses HLS or WebRTC, with the same HLS precedence as the station web UI.
Enable the required output in the pipeline; there is no dashboard transport selector.
The normalizer automatically distinguishes per-frame source records from merged
statistics. Stream callbacks can list all inputs while providing a frame for only
some of them; null placeholders do not erase another input's latest valid frame.
Updates are immediate, with no deliberate display delay. On each callback,
retained stream samples older than two seconds are marked unavailable. This
expiry check is message-driven, not a timer that detects a completely silent feed.
Merged statistics use maxima from the last delivered merge window, without
carrying counts over from earlier windows. A delivery-mode or pipeline change
clears the retained stream state.

The timestamp is the normalization time, not the video frame's capture time.
No frame-level alignment is applied. The dashboard directly controls the shared
video tiler through MS3; no browser commands are wired back into this Node-RED flow.

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
  "delivery": "stream",
  "aggregation": "Latest frame per source",
  "pipeline": {"id": "...", "name": "..."},
  "sources": [
    {
      "id": "0",
      "name": "Camera 1",
      "available": true,
      "counts": {"car": 4, "person": 2, "truck": 1},
      "total": 7,
      "roi": {"parking": 3},
      "lineCrossing": {"entrance": 0},
      "lineCrossingCumulative": {"entrance": 42}
    }
  ],
  "totals": {
    "objects": 7,
    "classes": {"car": 4, "person": 2, "truck": 1},
    "roi": {"parking": 3},
    "lineCrossing": {"entrance": 0},
    "lineCrossingCumulative": {"entrance": 42}
  }
}
```

`delivery` is `stream` for per-frame records and `batch` for merged records;
`aggregation` is respectively `Latest frame per source` or `Merge-window maximum`.
Push envelopes are unwrapped to their last record before interpreting its data.
For merged callback data, the displayed count is `max_val` from the latest
merge window. For stream data, it is the latest valid frame count per input. A source
with no observed frames has `available: false` and `null` class counts and
total; the all-source totals are also `null` when any source is unavailable.
The dashboard displays these values as `N/A`. `lineCrossing` is the instant
count (`objLCCurrCnt`: crossings in the latest frame, or the window `max_val`);
`lineCrossingCumulative` is DeepStream's running total per line (`objLCCumCnt`,
the counter drawn on the video), using the window's `last_val` in batch delivery.
It requires AIP 6.1.0 or later; with older callbacks it is empty and shown as
`N/A`. Analytics labels are reported
only when their values were observed; absence does not prove that an analytic
is unconfigured. Totals combine reported per-source values and need not
represent a simultaneous frame. Stream state belongs to the normalizer's node
context; it is not a frame-synchronization buffer. The timestamp is not a
freshness alarm.
