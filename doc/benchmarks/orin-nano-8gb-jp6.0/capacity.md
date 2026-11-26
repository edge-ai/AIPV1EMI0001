# Camera capacity on EX3

With no video output, the **EX3 / Orin Nano 8 GB** supported **5 cameras for
YOLO26n, 5 for YOLOv10n and 3 for YOLOv7-tiny**. With video output enabled,
detection kept up at one camera, but the video did not meet the delivery or
recording requirements. All trials used 720p30 H.264 RTSP inputs, 15 W mode, L4T R36.3.0
and version 7.0.0 test AIPs. For a capacity trial to pass, every camera must
average at least 29 processed FPS over 120 seconds after a 45-second warm-up.

## Camera capacity and video quality

The 25 November trials used protocol v2. WebRTC must deliver at least 29 FPS
at the publisher; every Smart Video clip must decode at 29 FPS or more and
start on a keyframe. Neither output may have a packet gap over 100 ms (plus
5 ms timing tolerance). Output trials enabled the on-screen display (OSD).

| Model | No output | WebRTC | Smart Video | WebRTC + Smart Video |
|---|---:|---|---|---|
| YOLO26n | **5†** | Not qualified at 1 | Not qualified at 1 | Not qualified at 1 |
| YOLOv10n | **5** | Not qualified at 1 | Not qualified at 1 | Not qualified at 1 |
| YOLOv7-tiny | **3** | Not qualified at 1 | Not qualified at 1 | Not qualified at 1 |

**Not qualified at 1** means the delivered video failed the one-camera check.
It does not limit the no-output camera counts.

With outputs enabled, inference stayed near 30 FPS per camera, but the video
did not keep up. Across the two campaigns, the MediaMTX stream delivered
21.29–27.10 FPS with packet gaps up to 0.20 s. The 72 Smart Video clips
decoded at 22.92–29.78 FPS. All started on a keyframe; the largest gap was
0.167 s.
The MediaMTX RTSP measurement is **not browser playback**. No browser viewer
was connected and lower output resolutions, frame rates and bitrates were
not tested.

### What the recording failures mean

For Smart Video alone with one camera, the two runs contain
eight clips per model. Detection and saved-video rate differ:

| Model | Processed FPS across runs | Decoded clip FPS | Clips below 29 FPS |
|---|---:|---:|---:|
| YOLO26n | 30.08-30.09 | 26.90-29.45 | 7/8 |
| YOLOv10n | 30.12-30.14 | 26.79-29.00 | 7/8 |
| YOLOv7-tiny | 30.09-30.12 | 26.13-28.78 | 8/8 |

Rates in the table are rounded; pass/fail uses the full-precision values.
For example, the YOLOv10n clip displayed as 29.00 is just above 29.
Across the three models, **22 of 24 clips** were below target. In the
[YOLO26n recheck](capacity-protocol-v2-recheck.jsonl), detection ran at
30.09 FPS while four 9.033333-second clips contained 253, 243, 247 and 256
pictures (271 expected). Their packet counts equal the decoded-frame counts:
pictures are missing in the files, not merely slow to decode. Every clip
started on a keyframe; the largest gap was 100 ms.

These measurements show that the saved clips contain fewer frames than
expected. They do not identify where in the recording pipeline the frames
were lost.

### Why YOLO26n is rated for five cameras, not six

At five cameras, the slowest camera averaged **30.18** and **30.15 FPS** in
the two runs. At six, only **3 of 7 attempts passed**.
Each value below is the lowest of the six per-camera means over its own
120-second window, not an instantaneous minimum or a pipeline average.

| Six-camera attempts, in run order | Slowest camera (processed FPS) | Passed |
|---|---|---:|
| [First campaign](capacity-protocol-v2.jsonl) | 28.81 fail, 28.22 fail | 0/2 |
| [Second campaign](capacity-protocol-v2-recheck.jsonl) | 29.24 pass, 27.85 fail | 1/2 |
| [Boundary-only repeat](capacity-boundary-v2.jsonl) | 29.17 pass, 28.77 fail, 29.28 pass | 2/3 |

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

## Earlier measurements (protocol v1)

On 24 November, each camera still had to average 29 processed FPS, and every
Smart Video clip had to decode at 29 FPS or more. These trials did not check
delivered WebRTC FPS, clip starts or packet gaps. Their output counts are
therefore reported separately from the v2 results.

| Model | No output | WebRTC | Smart Video | WebRTC + Smart Video |
|---|---:|---|---:|---|
| YOLO26n | **5** | Inference below target at 1 | **3†** | Unstable |
| YOLOv10n | **5** | Inference below target at 1 | **4†** | Unstable |
| YOLOv7-tiny | **3** | Inference below target at 1 | **3** | **1†** |

† Repeat trials disagreed; the table uses the conservative count. With
WebRTC alone, one-camera trials averaged 27.58 processed FPS for YOLO26n,
27.72 for YOLOv10n and 27.85 for YOLOv7-tiny over the full 120 seconds.
With both outputs, YOLO26n passed a short pilot but fell to 28.64 FPS in
the full one-camera trial; YOLOv10n passed its pilot but later failed a
one-camera clip check.

### Boundary measurements

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
after sampling to avoid perturbing the FPS window. Smart Video requested 3
seconds before and 6 seconds after each trigger, with a 10-second buffer
and one trigger every 30 seconds. WebRTC used a 1280x720
mosaic at 2 Mbps; Smart Video used 4 Mbps. No browser viewer was connected.

The [24 November dataset](capacity.jsonl) contains 81 pilot, search,
and boundary records, including failed attempts. Records include per-camera
mean FPS, CPU/GPU utilization, output settings, clip properties and failure
reasons. The search was limited to the AIP's maximum of eight inputs.

The [capacity matrix](../capacity/README.md) compares these results with EX5.
EX5 passed the protocol-v2 output checks at one camera; multi-camera output
limits have not been established. EX3 1080p inputs, connected WebRTC viewers,
multiple simultaneous pipelines, long-duration thermal behavior and varied
scenes remain untested.
