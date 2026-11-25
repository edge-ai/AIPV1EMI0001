# Camera capacity: EX5 and EX3

Compare YOLO26n, YOLOv10n and YOLOv7-tiny on each device using **WebRTC as the
only live video output** and **Smart Video as the only recording feature**.
HLS and continuous recording are excluded and must not be enabled. Their old
trials, combined-profile trials, and diagnostics have been removed from the
published evidence, not relabeled as supported profiles. Retained measurements
keep their original dates, settings and verdicts.

## Results and gaps

Camera counts below are per pipeline for 720p30 H.264 RTSP inputs. OSD is on
in the output columns. EX5 counts are the **18-19 November no-output baseline**.
EX3 rows are the **patched v0.31.0 / protocol-v2 campaigns on 25 November**,
including the later YOLO26n boundary repeats. The [released EX3 baseline](../orin-nano-8gb-jp6.0/capacity.md#released-stack-results-24-november-2026)
uses older checks and is reported separately, not mixed into these rows.

| Device | Model | No output | WebRTC | Smart Video | WebRTC + Smart Video |
|---|---|---:|---:|---:|---:|
| EX5, Orin NX 16 GB | YOLO26n | 7 (baseline) | Not measured | Not measured | Not measured |
| EX5, Orin NX 16 GB | YOLOv10n | 6 (baseline) | Not measured | Not measured | Not measured |
| EX5, Orin NX 16 GB | YOLOv7-tiny | 4 (baseline) | Not measured | Not measured | Not measured |
| EX3, Orin Nano 8 GB | YOLO26n | 5† | Not qualified at 1 | Not qualified at 1 | Not qualified at 1 |
| EX3, Orin Nano 8 GB | YOLOv10n | 5 | Not qualified at 1 | Not qualified at 1 | Not qualified at 1 |
| EX3, Orin Nano 8 GB | YOLOv7-tiny | 3 | Not qualified at 1 | Not qualified at 1 | Not qualified at 1 |

The EX5 no-output limits come from the
[18-19 November 2026 measurements](../orin-nx-16gb-jp6.0/capacity.md).
The [EX3 report](../orin-nano-8gb-jp6.0/capacity.md) contains the detailed
results and raw evidence. **Not qualified at 1** means video-quality checks
failed even with one camera, while detection stayed near 30 FPS. It means
neither "supports one camera" nor "cannot detect on one camera." No passing
output capacity was established. `†` means six YOLO26n cameras passed only
3 of 7 attempts; five passed both campaigns. **Not measured** means no
qualification under the current output protocol, not zero capacity or a
failure. Older EX5 publisher screening is not a substitute.

The patched one-camera Smart-Video-only runs produced **22 of 24 clips below
29 decoded FPS**. This is a recording-frame deficit, distinct from the old
clip-start keyframe gap. See the [EX3 interpretation](../orin-nano-8gb-jp6.0/capacity.md#what-the-recording-failures-mean).
EX5 output profiles remain unmeasured under this protocol. EX5 accuracy and
inference-time measurements also must not be presented as EX3 measurements.

### Station checks and EX3 results

[Read-only preflight observations](preflight-2026-11-24.json):

- **EX5:** reachable, MAXN mode; one existing dashboard pipeline was running.
  New exclusive-load tests were deferred rather than stopping someone else's
  workload.
- **EX3:** the initial read-only preflight found missing per-input processed
  FPS and WebRTC support on `ms3-36.3.0-v0.24.0`. After the station was upgraded,
  the comparison ran on Orin Nano 8 GB, 15 W, L4T R36.3.0 and
  `ms3-36.3.0-v0.31.0`. No existing service or unrelated pipeline
  configuration was stopped or changed. See the [EX3 report](../orin-nano-8gb-jp6.0/capacity.md).

EX5 output profiles still require an exclusive test window. Do not replace
per-source processed FPS with decoded arrival FPS or a pipeline average.

## What each trial does

| CLI profile | Live output | Recording | OSD |
|---|---|---|---|
| `base` | None | Off | Off |
| `webrtc-osd` | WebRTC publisher | Off | On |
| `evt-osd` | None | Smart Video | On |
| `webrtc+evt-osd` | WebRTC publisher | Smart Video | On |

`evt` is the API's event-recording sink, exposed as **Smart Video recording**
in the web UI; it is not the signaling service's continuous recorder.
The harness explicitly requests the UI defaults: **3,000 ms pre-duration,
6,000 ms post-duration and 10,000 ms maximum buffer duration**. It triggers a
clip every **30 seconds** during sampling, after a 45-second warm-up has filled
the pre-record buffer. These are periodic events, not a continuous event storm.
Changing the trigger frequency or pre/post times requires a separately labeled
comparison.

Output mosaic: 1280x720. WebRTC: 2 Mbps. Smart Video: 4 Mbps. The recorded
settings, trigger acknowledgments, decoded clip frame counts, dimensions and
durations accompany each trial. After the sample window, `ffprobe -count_frames`
must decode each H.264 clip without errors at at least 29 decoded FPS; total duration must be 9 seconds within
a 2-second keyframe-alignment tolerance. Since protocol `webrtc-smart-video-v2`, the clip must also start on a
keyframe and the maximum packet gap is 0.10 s (three frame intervals, plus 5 ms timing tolerance), so a clip
that skips a GOP after its first keyframe fails even when its average frame rate passes. This does not prove the exact split
of frames before and after a trigger; verify that with timestamped footage
in the deployment pilot. Clip decoding runs after FPS sampling so it does
not contaminate the measured window.

Since `webrtc-smart-video-v2`, a WebRTC trial also reads the published stream
back from MediaMTX (`rtsp://…:8554/{pipeline}_2`, `--live-ffprobe`) for the
first 20 seconds of the sample window. It passes only if that stream delivers
at least **29 FPS** with the same packet-gap limit. An MS3 that drops preview frames
to protect inference therefore cannot pass on inference FPS alone.

The automated WebRTC profile measures the encoder and publisher **without a
connected browser viewer**. It must not be described as browser-playback performance.
For the deployment pilot, also negotiate WebRTC and confirm decoded frames in
a browser. Record viewer count and use a separate result set for viewer load.

## Measurement and deployment procedure

1. Capture device model, power mode, software versions, engine SHA-256 hashes,
   source clip/codec/resolution, and active workload inventory. Do not change
   power modes or stop unrelated pipelines implicitly.
2. Build each FP16 engine on its target GPU and validate the AIP/parser. Use
   isolated test AIP IDs rather than overwriting an installed application.
   The EX5 prebuilt engines are not validated EX3 engines. EX3 has no hardware
   video encoder: record the selected software encoder and CPU load, but do
   not predict a numerical penalty before measurement.
3. Run a one-camera pilot for every model and output profile. Verify detections,
   requested output settings, Smart Video clips and WebRTC playback. Startup
   or encoder errors are blockers, not zero-camera throughput results.
4. Search all four profiles for every model on each device. Each camera must
   average at least **29 processed FPS** over **120 seconds** after a
   **45-second warm-up**, with zero reconnects and inference stalls. Use
   per-source `pipeline_perf_fps_average_10sec`, not pre-mux arrival FPS.
5. Independently repeat each passing boundary and its next camera count.
   Screen OSD off/on, bitrate and mosaic resolution on YOLO26n; label such
   fixed-count probes as screening, not independently measured maxima.
   Probe 1080p inputs separately. Do not infer one device's limits from another.
6. Verify cleanup of all owned cameras, pipelines and clips. Retain raw
   evidence, including invalid attempts, before updating this table.

`python3 tools/capacity.py plan` reports the per-device workload estimate.
This is a multi-hour measurement, plus target-specific builds and manual
deployment checks. Binary search assumes monotonic throughput; a passing
upper search ceiling is a lower bound, not a maximum. Single-camera failures
must be investigated separately.

## Harness

Run on the station host (where `tegrastats` is available), using discovered
internal service addresses. Replace placeholders with the isolated test setup:

```sh
python3 tools/capacity.py plan
python3 tools/capacity.py preflight --ms3 http://MS3:8000
python3 tools/capacity.py search \
  --ms3 http://MS3:8000 --device ex3-orin-nano-8gb-15w \
  --model yolo26 --aip AIPV1EMI90XX \
  --rtsp-template 'rtsp://VCAM:PORT/cam{index:02d}' \
  --camera-tag 720p --profile base --out ex3-capacity.jsonl
python3 tools/capacity.py fixed \
  --ms3 http://MS3:8000 --device ex3-orin-nano-8gb-15w \
  --model yolo26 --aip AIPV1EMI90XX \
  --rtsp-template 'rtsp://VCAM:PORT/cam{index:02d}' \
  --camera-tag 720p --profile webrtc+evt-osd --n 2 \
  --out ex3-capacity.jsonl
python3 tools/capacity.py summarize ex3-capacity.jsonl
python3 tools/capacity.py cleanup --ms3 http://MS3:8000 \
  --ledger ex3-capacity.jsonl.owned.json
```

Smart Video trials require `ffprobe` (`--ffprobe` selects its executable).
WebRTC trials need an ffprobe that can reach MediaMTX as MS3 does
(`--live-ffprobe`, for example a wrapper around `docker exec ms3_videostreaming ffprobe`).
They do not require a signaling URL. Legacy `rec`/`hls` profile parsing remains
only for old ownership-ledger recovery; new trials and published tables reject
these profiles. Removing benchmark evidence must not prevent safe cleanup of
previously created resources.

Every run creates names under `cap-RUN-...` and maintains an
`OUT.owned.json` ledger. Recovery deletes only matching owned resources. Smart
Video cleanup stops the pipeline, requests asynchronous deletion scoped to
that pipeline, and polls until no clips remain. A failed stop or unfinished
deletion keeps the ledger entry. Do not start another run with that results
file until recovery completes. Keep result files separate for each device,
stack version and power/thermal setup; published tables reject mixed devices
or historical/new protocols.

## Recording diagnostic follow-up (not a capacity result)

The AIP comparison does not implement the MS3 output or recorder fixes. They
are tracked separately in [edge-ai/aibox-webserver-backend-ms3#967](https://github.com/edge-ai/aibox-webserver-backend-ms3/pull/967)
and [edge-ai/gst-plugins-emi#28](https://github.com/edge-ai/gst-plugins-emi/pull/28).
The matched EX3 experiment below has **not been run**; existing campaigns
changed both output code and recorder code and cannot isolate a regression.

Use one YOLO26n 720p30 scene on an exclusively reserved EX3, keeping the image,
fixed recorder plugin, power mode, 4 Mbps bitrate and 3/6/10-second clip settings
constant. Compare stock and patched **output code**, each with Smart Video
drawing on and off; record exact revisions. Use the same 45-second warm-up,
120-second sample, four triggers and v2 verdicts in all four arms.

Trace frame PTS and monotonic arrival time at queue input/output, private-copy
output, OSD output, encoder input/output and parser output. Match the saved
interval to MP4 packets and record inference FPS alongside it. Queue overruns
alone are not exact drop counts and do not identify the slow component.
Compare with an uninstrumented repeat if the tracer changes performance.

If the consumer catches up between bursts, evaluate a bounded buffer **after**
the private copy without enlarging the shared pre-copy queue. If throughput is
consistently insufficient, optimize the measured slow stage instead. If all
parser pictures arrive but the clip loses them, investigate the recorder.
Keep the keyframe fix and inference isolation; never manufacture a pass by
rewriting timestamps or lowering the quality thresholds.
