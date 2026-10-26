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
3. Register the AIP (`POST /aiprocessor/{pk}/register` or the UI button).

To build from source instead, follow [`localdev.ipynb`](../localdev.ipynb) step by step.

## Configuration

No `options.json` — the detector runs with the defaults baked into
`configs/config_infer_primary_detector_yolov7.txt` (confidence threshold 0.35, NMS IoU 0.65)
and `configs/config_tracker_NvDCF_perf.yml` (NvDCF tracker parameters).

## Node-RED dashboard

The AIP archive contains a Node-RED project in `logic/` and an offline Vue 3
UIBuilder application in `dashboard/`. The dashboard shows:

- the pipeline HLS stream;
- live `car`, `person`, and `truck` counts;
- a combined or per-source view;
- ROI and line-crossing counts when those analytics are configured.

To activate the dashboard (step 6 of [`localdev.ipynb`](../localdev.ipynb) automates
steps 1–4):

1. Copy the bundled `logic/` directory to a new Node-RED project
   (`/data/projects/<name>/` on the station) and make it a git repository.
2. Copy the bundled `dashboard/` contents to that project's `uibuilder/`
   directory.
3. Run `npm ci --omit=dev` in the project's `uibuilder/` directory. The Node-RED
   nodes themselves are preinstalled on EdgeAI Stations.
4. Set `AIP_PIPELINE_ID` in the **AIPV1EMI0001 Logic** tab's Environment
   properties.
5. In the Node-RED editor, open the project (☰ → **Projects** → **Open**) and
   deploy. The dashboard is served at `https://<station>/ms5/aipv1emi0001_dash/`.
6. Assign `aipv1emi0001_dash` to the required users in Accounts management.

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
  plugins (`{}` when no analytics are configured on the pipeline).

### Merge mode (merge enabled, default 1 s window)

`merge_and_analyze()` aggregates the window per source ID:

```json
{
  "0": {
    "source_id": 0,
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

With periodic push enabled, multiple merge results are batched into time-windowed
records (`since` / `until` / `data`). Inspect a live message with a Node-RED debug
node before writing downstream bindings.

## Model

- Architecture: YOLOv7-tiny (WongKinYiu et al., CVPR 2023, MIT) — see README attribution
- Weights: custom-trained by EdgeMatrix on a parking-lot dataset, reparameterized,
  exported to ONNX with an `EfficientNMS_TRT` node, built into a TensorRT FP16 engine
  (dynamic batch 1–8, 640×640 input)
- Tracker: NVIDIA NvDCF
