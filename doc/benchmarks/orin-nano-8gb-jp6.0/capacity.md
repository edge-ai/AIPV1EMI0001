# Camera capacity on EX3

With no video output, the **EX3 / Orin Nano 8 GB** supported **5 cameras for
YOLO26n, 5 for YOLOv10n and 3 for YOLOv7-tiny**. The [patched-stack
results](#patched-stack-results) explain why six is not a reliable YOLO26n
limit. All trials used 720p30 H.264 RTSP inputs, 15 W mode, L4T R36.3.0
and version 7.0.0 test AIPs. For a capacity trial to pass, every camera must
average at least 29 processed FPS over 120 seconds after a 45-second warm-up.

## Released-stack results (24 November 2026)

OSD is enabled for output profiles. WebRTC was measured with the publisher
running and **no connected viewer**. Smart Video is the UI-configured event
recording feature, with 3 seconds pre-roll, 6 seconds post-roll, a 10-second
maximum buffer and one trigger every 30 seconds. HLS and continuous recording
were not tested.

| Model | No output | WebRTC | Smart Video | WebRTC + Smart Video |
|---|---:|---|---:|---|
| YOLO26n | **5** | Fails at 1 camera | **3†** | Unstable |
| YOLOv10n | **5** | Fails at 1 camera | **4†** | Unstable |
| YOLOv7-tiny | **3** | Fails at 1 camera | **3** | **1†** |

No-output counts passed a separate boundary recheck; the next count failed.
For Smart Video, YOLOv7-tiny passed at three and failed at four. YOLOv10n
passed four and failed five because one recorded H.264 clip decoded at
27.29 FPS, despite the inference rate exceeding 30 FPS. Its first one-camera
pilot also failed clip validation, but a later confirmation passed; `†` marks
that contradictory pilot evidence. YOLO26n passed three;
an earlier four-camera search failed on a 27.29 FPS clip, while the independent
four-camera boundary recheck passed with 30.14 minimum processed FPS and
valid clips. `†` marks that conflicting four-camera evidence; three is the
conservative count pending another repeat.

WebRTC alone failed the 120-second, 29-FPS requirement even with one camera:
the measured minimums were 27.58 FPS for YOLO26n, 27.72 for YOLOv10n and
27.85 for YOLOv7-tiny. YOLOv7-tiny's shorter one-camera pilot passed, so it
does not override the long-window failure.

Combined WebRTC and Smart Video was not stable enough to publish as a maximum.
YOLOv7-tiny passed one camera in the search; the repeated two-camera search
failed, but a separate two-camera boundary recheck passed at 29.37 FPS with
valid clips. YOLOv10n passed a one-camera pilot but failed a later one-camera
clip check; its four- and five-camera boundary checks also failed. YOLO26n
passed one short pilot, but the long-window one-camera test failed at 28.64
FPS; its two-camera trial failed the clip check and its four- and five-camera
rechecks failed processed FPS. Treat these as contested outcomes, not as proof
that combined output is impossible.

## Patched-stack results

Two campaigns on 25 November repeated the comparison on the same EX3 with
two changes to `ms3-36.3.0-v0.31.0`: slow outputs could drop frames instead
of blocking inference, and Smart Video started each clip on a keyframe
without discarding the following frames. These are not stock-image results.

The no-output pass rule was unchanged. Protocol `webrtc-smart-video-v2`
also checked the **delivered video**: at least 29 FPS from the WebRTC
publisher's MediaMTX stream, at least 29 decoded FPS for each Smart Video
clip, and no packet gap over 0.10 s (plus 5 ms timing tolerance) in either.
Clips also had to start on a keyframe.

| Model | No output | WebRTC | Smart Video | WebRTC + Smart Video |
|---|---:|---|---|---|
| YOLO26n | **5†** | Fails at 1 camera | Fails at 1 camera | Fails at 1 camera |
| YOLOv10n | **5** | Fails at 1 camera | Fails at 1 camera | Fails at 1 camera |
| YOLOv7-tiny | **3** | Fails at 1 camera | Fails at 1 camera | Fails at 1 camera |

With outputs enabled, inference stayed near 30 FPS per camera, but the video
did not keep up. Across the two campaigns, the MediaMTX stream delivered
21.29–27.10 FPS with packet gaps up to 0.20 s. The 72 Smart Video clips
decoded at 22.92–29.78 FPS. All started on a keyframe; the largest gap was
0.167 s, not the roughly one-second gap seen before the recording fix.
The MediaMTX RTSP measurement is **not browser playback**. No browser viewer
was connected and lower output resolutions, frame rates and bitrates were
not tested.

### Why YOLO26n is rated for five cameras, not six

At five cameras, the slowest camera averaged **30.18** and **30.15 FPS** in
the two fixed-stack campaigns. At six, only **3 of 7 attempts passed**.
Each value below is the lowest of the six per-camera means over its own
120-second window, not an instantaneous minimum or a pipeline average.

| Six-camera attempts, in run order | Slowest camera (processed FPS) | Passed |
|---|---|---:|
| [First campaign](capacity-fixed-ms3.jsonl) | 28.81 fail, 28.22 fail | 0/2 |
| [Second campaign](capacity-fixed-ms3-recheck.jsonl) | 29.24 pass, 27.85 fail | 1/2 |
| [Boundary-only repeat](capacity-fixed-ms3-boundary.jsonl) | 29.17 pass, 28.77 fail, 29.28 pass | 2/3 |

The last three runs used fresh pipelines on 25–26 November (JST), with OSD
and all outputs off. Each yielded 24 samples with all six inputs streaming
and no reconnects or inference stalls. In the failed repeat, **all six**
camera means were below 29 FPS (28.77–28.94). Six can pass, but not
repeatably under the same conditions; **five is the qualified limit**.
No other model, camera count or output profile was rerun in this check.

The first two JSONL files contain 44 records each. Both six-camera attempts
from each campaign are in the search trial's `attempts` array; the
`capacity` record summarizes the same search and is not another trial.
The boundary-only file contains three independent `fixed` records with
per-camera means, timestamps, verdicts and aggregate telemetry.

## Released-stack boundary measurements

The table gives the slowest-camera processed FPS for independent checks at the
last supported count and the next count. Smart Video rows also require every
clip to be valid H.264 at 29+ decoded FPS; clip failures are called out below.

| Model | Profile | Passing count, min FPS | Next checked count, min FPS | Outcome |
|---|---|---:|---:|---|
| YOLO26n | No output | 5, 30.16 | 6, 27.17 | Confirmed |
| YOLOv10n | No output | 5, 30.25 | 6, 25.49 | Confirmed |
| YOLOv7-tiny | No output | 3, 30.34 | 4, 25.83 | Confirmed |
| YOLO26n | Smart Video | 3, 30.14 | 4, 30.14 | Contested: independent 4-camera check passed; earlier clip failed at 27.29 decoded FPS |
| YOLOv10n | Smart Video | 4, 30.15 | 5, 30.30 | Contested: one-camera pilot clip failed, later confirmation passed; 5 failed a clip at 27.29 decoded FPS |
| YOLOv7-tiny | Smart Video | 3, 31.07 | 4, 25.30 | Confirmed; 4 also had a clip at 27.29 decoded FPS |
| YOLOv7-tiny | WebRTC + Smart Video | 1, 30.18 | 2, 29.37 | Contested: search at 2 failed; separate 2-camera recheck passed |
| YOLOv10n | WebRTC + Smart Video | — | 4, 28.94; 5, 28.46 | No stable count established |
| YOLO26n | WebRTC + Smart Video | — | 4, 28.18; 5, 28.18 | No stable count established |

Smart Video captures were generally 30 FPS, but intermittent clips decoded at
27.29 FPS (or 27.2 FPS in pilots). Both the per-camera inference rate and clip
quality must pass; one cannot substitute for the other. The recorded clip
duration is subject to a 2-second keyframe-alignment tolerance around the
9-second pre/post total.

The EX3 has no hardware video encoder; the measured output cases therefore
include its software-encoding workload. No NVENC activity was reported.
The benchmark used the same 720p30 source clip for the camera inputs, so these
results do not represent varied or busy scenes.

## Method and evidence

Each capacity trial used 45 seconds of warm-up and 120 seconds of sampling,
with per-input processed-FPS checks every 5 seconds. A passing count requires
every input to remain connected, no inference stalls or reconnects, and at
least 29 processed FPS per camera. Smart Video clips were decoded and checked
after sampling to avoid perturbing the FPS window. WebRTC used a 1280x720
mosaic at 2 Mbps; Smart Video used 4 Mbps. No browser viewer was connected.

The sanitized [`capacity.jsonl`](capacity.jsonl) contains 81 pilot, search,
and independent boundary records, including failed attempts. It omits sample-
by-sample telemetry and internal station identifiers; aggregate per-camera
FPS, CPU/GPU utilization, output settings, clip properties, timestamps and
failure reasons are retained. The campaign was limited to the AIPs' eight
input maximum; it did not test 1080p.

The initial read-only preflight is retained at
[`../capacity/preflight-2026-11-24.json`](../capacity/preflight-2026-11-24.json).
It predates the station upgrade and is not the test software version. Test
pipeline resources were cleaned up: no benchmark-owned pipeline configurations
or ownership ledgers remained. The isolated test AIPs remain registered for
reuse. Existing station services and unrelated pipeline configurations were
left untouched.

The [capacity matrix](../capacity/README.md) compares these results with EX5.
EX5 output profiles have not been remeasured under this protocol, and EX3
1080p inputs, connected WebRTC viewers, multiple simultaneous pipelines,
long-duration thermal behavior and varied scenes remain untested.
