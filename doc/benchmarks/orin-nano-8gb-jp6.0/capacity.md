# Camera capacity on EX3

With video outputs off, one EX3 (Orin Nano 8 GB) sustained **5 cameras with
YOLO26n, 5 with YOLOv10n and 3 with YOLOv7-tiny** at 720p; 1080p inputs were
not tested. In the original 720p, OSD-on output screen, detection kept up
with one camera, but no output profile met the WebRTC and Smart Video checks.
That does not rule out a lower-resolution output: the separate
[same-camera comparison](../capacity/matched.md) tests 360p, 720p and 1080p
outputs while keeping the input at 720p.

## Measured no-output limits

| Model | 720p30 cameras, no output | 1080p30 cameras, no output |
|---|---:|---:|
| YOLO26n | **5†** | Not tested |
| YOLOv10n | **5** | Not tested |
| YOLOv7-tiny | **3** | Not tested |

Every camera must average at least 29 processed FPS over 120 seconds after a
45-second warm-up, with no reconnects or inference stalls. A count is reported
when it passed at least two attempts and failed none, every lower count tested
also passed, and at least one attempt at the next count failed. † means a
higher count passed in some attempts and failed in others; the table keeps the
lower count.

† YOLO26n: six 720p cameras passed 3 of 7 attempts.

These are near-real-time experimental limits for one pipeline, not guarantees
of lossless 30 FPS. Do not divide idle inference throughput by 30 to infer a
camera limit.

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

The last three runs used fresh pipelines with OSD and all outputs off. Each
yielded 24 samples with all six inputs streaming and no reconnects or inference stalls. In the failed repeat, **all six**
camera means were below 29 FPS (28.77–28.94). Six can pass, but not
repeatably under the same conditions; **five is the qualified limit**.
No other model, camera count or output profile was rerun in this check.

The first two JSONL files contain 44 records each. Both six-camera attempts
from each campaign are in the search trial's `attempts` array; the
`capacity` record summarizes the same search and is not another trial.
The boundary-only file contains three independent `fixed` records with
per-camera means, timestamps, verdicts and aggregate telemetry.

## Protocol v2: one-camera output checks

No output profile qualified at one camera. Detection kept up: every run
averaged at least 29 processed FPS during the 120-second sample after a
45-second warm-up. The delivered video did not.

WebRTC must deliver at least 29 FPS, measured at the publisher for the first 20
seconds of sampling. Every Smart Video clip must be H.264 at the mosaic size,
last 9 seconds (±2 seconds), decode at 29 FPS or more and start on a keyframe.
Neither output may have a packet gap over 100 ms (plus 5 ms timing tolerance).

| Model | Profile | Runs | Processed FPS (minimum) | WebRTC publisher | Smart Video clips | Result |
|---|---|---:|---:|---|---|---|
| YOLO26n | WebRTC | 2 | 30.12 | 22.27–22.44 FPS; 200 ms maximum gap | — | Not qualified |
| YOLO26n | Smart Video | 2 | 30.08 | — | 1/8 at 29+ FPS (26.90–29.45); 8/8 keyframe starts; 100 ms maximum gap | Not qualified |
| YOLO26n | WebRTC + Smart Video | 2 | 30.00 | 21.29–21.84 FPS; 200 ms maximum gap | 0/8 at 29+ FPS (25.90–28.78); 8/8 keyframe starts; 167 ms maximum gap | Not qualified |
| YOLOv10n | WebRTC | 2 | 30.09 | 22.34–24.39 FPS; 133 ms maximum gap | — | Not qualified |
| YOLOv10n | Smart Video | 2 | 30.12 | — | 1/8 at 29+ FPS (26.79–29.00); 8/8 keyframe starts; 133 ms maximum gap | Not qualified |
| YOLOv10n | WebRTC + Smart Video | 2 | 30.00 | 23.03–24.54 FPS; 200 ms maximum gap | 1/8 at 29+ FPS (27.12–29.23); 8/8 keyframe starts; 167 ms maximum gap | Not qualified |
| YOLOv7-tiny | WebRTC | 2 | 30.14 | 23.39–23.44 FPS; 167 ms maximum gap | — | Not qualified |
| YOLOv7-tiny | Smart Video | 2 | 30.09 | — | 0/8 at 29+ FPS (26.13–28.78); 8/8 keyframe starts; 133 ms maximum gap | Not qualified |
| YOLOv7-tiny | WebRTC + Smart Video | 2 | 30.14 | 22.29–23.14 FPS; 167 ms maximum gap | 0/8 at 29+ FPS (25.79–28.45); 8/8 keyframe starts; 167 ms maximum gap | Not qualified |

Each profile ran twice at one camera, once per campaign, with the street clip
used for the no-output trials; Smart Video searches first tried three cameras,
which also failed. Output profiles enable the on-screen display (OSD). A
separate WebRTC sweep at one to three cameras delivered 22.04–27.10 FPS.
Across both campaigns, all 72 saved clips were H.264 and started on a keyframe;
they decoded at 22.92–29.78 FPS with packet gaps up to 167 ms.

The WebRTC check reads the publisher's stream from MediaMTX over RTSP; it is
**not browser playback**. This screen had no browser viewer and did not vary
the output resolution, target frame rate or bitrate. The
[matched EX3/EX5 results](../capacity/matched.md) show what happens when
both stations use the same input video and output settings.

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

## Earlier measurements (protocol v1)

Each camera still had to average 29 processed FPS, and every Smart Video clip
had to decode at 29 FPS or more. These trials did not check
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

## Test conditions

The trials used Orin Nano 8 GB in 15 W power mode, the highest mode L4T R36.3
offers on this module, with JetPack 6.0 / L4T R36.3.0. The records do not
include DeepStream or TensorRT versions. Batch size was `min(cameras, 8)`;
NvDCF tracking was on, ROI/LCC analytics off, and inference ran on every frame.
TCP RTSP cameras looped copies of one 720p30 street clip. Node-RED and
WebSocket metadata outputs remained on.

WebRTC used a 1280x720 mosaic at 2 Mbps. Smart Video used 4 Mbps and requested
3 seconds before and 6 seconds after each trigger, with a 10-second buffer and
one trigger every 30 seconds. Clips were decoded after sampling, so the check
did not disturb the FPS window. The EX3 has no hardware video encoder, so the
output profiles include software encoding; no NVENC activity was reported.

The isolated test AIPs were version 7.0.0. Per-trial records retain AIP
identity and version, not engine build settings or hashes.

## Limitations

All trials used repeated copies of one street scene at 720p, so these results
do not represent varied or busy scenes. 1080p inputs, connected WebRTC viewers,
multiple simultaneous pipelines and long thermal soak were not tested. Searches
stopped at the AIP's maximum of eight inputs.

## Evidence

- [Protocol v2, first campaign](capacity-protocol-v2.jsonl) and
  [second campaign](capacity-protocol-v2-recheck.jsonl): searches, search
  summaries and the WebRTC sweep for the three models.
- [YOLO26n six-camera repeats](capacity-boundary-v2.jsonl): three independent
  no-output checks.
- [Protocol v1 trials](capacity.jsonl): 81 pilot, search, and boundary records,
  including failed attempts.
- [Capacity matrix](../capacity/README.md): these results next to EX5.

Records include per-camera mean FPS, CPU/GPU utilization, output settings,
clip properties and failure reasons.

To print the summary tables for one campaign file:

```sh
python3 tools/capacity.py summarize doc/benchmarks/orin-nano-8gb-jp6.0/capacity-protocol-v2.jsonl
```
