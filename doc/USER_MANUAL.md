# AIPV1EMI0001 — User Manual

YOLOv7-tiny object detector for the EdgeAI Station: real-time detection of
**`car`**, **`person`** and **`truck`** in up to 8 input streams, with ROI
(region of interest) and LCC (line crossing counter) analytics.

## Requirements

- EdgeAI Station (EX3 / EX5 / EX7 targets) — Jetson platforms only, x86 NVIDIA GPU hosts are not supported
- JetPack 6 / L4T R36.3 for the prebuilt engine
- A valid AIP license on the station

## Installation

1. Download the release archive (`AIPV1EMI0001-<version>-<jetpack>-arm64-all.tar.gz`)
   from the [GitHub releases](https://github.com/edge-ai/AIPV1EMI0001/releases).
2. Upload it to the station: **AI Processors** page → **Upload**, or via the API
   (`POST /aiprocessor/upload`).
3. Create a pipeline that uses the AIP. No Triton registration is needed: inference
   runs inside the DeepStream pipeline (`nvinferserver`).

To build from source instead, follow [`localdev.ipynb`](../localdev.ipynb) step by step.

## Configuration

No `options.json` — the detector runs with the defaults baked into
`configs/config_infer_primary_detector_yolov7.txt` (confidence threshold 0.35, NMS IoU 0.65)
and `configs/config_tracker_NvDCF_perf.yml` (NvDCF tracker parameters).

## Node-RED dashboard

The AIP archive contains a Node-RED project in `logic/` and an offline Vue 3
UIBuilder application in `dashboard/`. The dashboard shows:

- the pipeline video stream (HLS or WebRTC, selected automatically from the
  enabled pipeline outputs, with HLS precedence as in the station web UI);
- live `car`, `person`, and `truck` counts;
- a combined or per-source view;
- ROI and line-crossing counts when those analytics are configured.

To activate the dashboard, follow these steps. Step 8 of
[`localdev.ipynb`](../localdev.ipynb) runs step 2 for its test pipeline:

1. Make sure the Node-RED runtime loads `@edgematrix/node-red-contrib-eais`
   0.7.10 and UIBuilder. Do not install the legacy
   `@edge-ai/node-red-contrib-eais` alongside it: both register the same node
   types. Have the station administrator arrange the migration and any restart;
   see [`logic/README.md`](../logic/README.md).
2. From the extracted AIP package, inside the Node-RED container, run
   `bash bin/install-nodered-project.sh --pipeline <pipeline-id>`. It creates
   the project `nodered-aipv1emi0001-dashboard` under `/data/projects/`
   (override with `NODE_RED_PROJECTS_DIR`), installs the dashboard's browser
   libraries and makes the project a git repository. Add `--force` to replace
   an existing copy. It does not install Node-RED runtime nodes.
3. In the Node-RED editor, open the project (☰ → **Projects** → **Open**).
   Check that **AIPV1EMI0001 callback** selects the pipeline running this AIP,
   and deploy.
4. Open `https://<station>/ms5/aipv1emi0001_dash/` and assign
   `aipv1emi0001_dash` to the required users in Accounts management.

No CDN is used. Vue and `hls.js` are installed from the bundled lock file and
served through UIBuilder's local vendor route.

`flows_cred.json` is intentionally excluded from the archive. For credentials,
set Node-RED's `credentialSecret` through the station's deployment environment
and allow Node-RED to create the credential file locally. Do not commit or
redistribute that file.

## Callback output (metadata contract)

Downstream Node-RED flows consume the callback output. The payload shape depends on
the pipeline's merge/push configuration:

### Passthrough mode (merge disabled)

The probe result is published directly — one message per batch, a list with one
entry per input source:

```json
[
  {
    "source_id": 0,
    "available": true,
    "frame_num": 1234,
    "num_obj_meta": 3,
    "ntp_timestamp": 1760000000000,
    "obj_counter": [2, 1, 0],
    "objInROIcnt": {"roi_label": 2},
    "objLCCurrCnt": {"line_label": 1}
  }
]
```

- `obj_counter` — per-class detection counts in this frame, ordered as
  `["car", "person", "truck"]` (the `labels-rev3.txt` order).
- `objInROIcnt` / `objLCCurrCnt` — per-label counts from the ROI / LCC analytics
  plugins (`{}` when no analytics values were reported for the frame).

### Merge mode (merge enabled, default 1 s window)

`merge_and_analyze()` aggregates the window per source ID:

```json
{
  "0": {
    "source_id": 0,
    "available": true,
    "frame_number_from": 1200,
    "frame_number_to": 1234,
    "ntp_timestamp_from": 1759999999000,
    "ntp_timestamp_to": 1760000000000,
    "obj_counter": {
      "car":    {"max_val": 4, "min_val": 1, "mean_val": 2, "median_val": 2, "count_val": 68},
      "person": {"max_val": 2, "min_val": 0, "mean_val": 1, "median_val": 1, "count_val": 30},
      "truck":  {"max_val": 1, "min_val": 0, "mean_val": 0, "median_val": 0, "count_val": 5}
    },
    "analytics_obj_cnt": {
      "objInROIcnt":  {"roi_label":  {"max_val": 2, "min_val": 0, "sum_val": 40}},
      "objLCCurrCnt": {"line_label": {"max_val": 1, "min_val": 0, "sum_val": 3}}
    }
  }
}
```

`available` is true if the source had at least one frame in the window. If it
had no frames, timestamps and each class statistic are `null` and analytics maps
are empty. In passthrough mode, the missing source has `available: false`,
`null` frame fields, an `obj_counter` array of `null` values and `null`
analytics fields. Missing frames within an otherwise available window are
excluded; they do not become zero observations. On station runtimes that
provide configured input IDs, only those IDs are reported, even when they are
sparse or the batch capacity is larger. Without that mapping, the callback
falls back to the batch's reported source slots.

Class `count_val` is the sum of per-frame counts, not a unique-object count;
the mean and median are rounded to the nearest integer. ROI and line-crossing
statistics include only values reported for that label. A label absent from the
window is omitted, which does not establish whether an analytic is configured.
The dashboard's all-source totals sum per-source maxima, not necessarily
simultaneous observations; if a source is unavailable those totals are `N/A`.

With periodic push enabled, multiple merge results are batched into time-windowed
records (`since` / `until` / `data`). Inspect a live message with a Node-RED debug
node before writing downstream bindings. If callbacks stop entirely, no new
availability update is emitted: monitor the metadata timestamp and station
pipeline health separately.

## Model

- Architecture: YOLOv7-tiny (WongKinYiu et al., CVPR 2023, GPL-3.0) — see README attribution
- Integration: an EdgeAI Station adaptation exposing three object classes. The
  trainer, training dataset, intended domain, and complete checkpoint provenance
  have not been independently documented.
- Artifacts: `yolov7tiny-640.onnx` with an `EfficientNMS_TRT` node, built into a TensorRT FP16 engine
  (dynamic batch 1–8, 640×640 input)
- Tracker: NVIDIA NvDCF
