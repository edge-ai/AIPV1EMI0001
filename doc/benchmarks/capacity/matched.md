# Same-camera video output on EX3 and EX5

This test asks whether one camera can sustain both detection and the requested
video output. Both stations read the same 120-second loop of traffic footage,
with the same settings for each of three models and eleven profiles.
**Both processed about 30 FPS in every run.** With video output enabled,
EX3 passed 2 of 30 cases and EX5 passed 28 of 30. The other cases failed on
the **delivered stream or saved clips**, not on detection.

| What met the threshold | EX3 | EX5 |
|---|---:|---:|
| Input processing (all runs) | 33/33 | 33/33 |
| WebRTC publisher (runs with WebRTC enabled) | 2/24 | 24/24 |
| Smart Video (individual clips) | 3/48 | 45/48 |
| Processing and every requested output (runs with video output) | 2/30 | 28/30 |

**Which setting helped on EX3?** At 1280×720 and a 2 Mbps target, none of
the three models delivered 29 FPS with WebRTC alone, with or without overlays.
For example, YOLO26n processed 30.14 FPS in both runs, but delivered 24.09 FPS
with overlays off and 22.44 with them on. Neither the 1 nor the 4 Mbps target
met the threshold at 720p with overlays. Changing **only the output size** to
640×360 at 2 Mbps gave 29.45 FPS with YOLO26n and 29.30 with YOLOv10n; both
passed. YOLOv7-tiny delivered 28.50 and missed. At 1920×1080, the three EX3
publishers delivered 14.45–17.13 FPS. The camera input stayed 720p throughout.

**What missed on EX5?** WebRTC delivered 30.00 FPS in all 24 publisher runs.
The two misses were recordings: with YOLO26n, WebRTC and Smart Video together
with OSD on passed 3 of 4 clips; with YOLOv7-tiny, Smart Video alone with
OSD off passed 2 of 4. Each of the three affected clips decoded at 27.29 FPS
and had a 933 ms gap.

These are one-camera observations, **not camera limits**. The
[multi-camera capacity searches](README.md#cameras-per-pipeline) used
different scenes and are reported separately.

## Conditions

| | Shared setting |
|---|---|
| Input | One TCP RTSP publisher, streaming an H.264 High, yuv420p, 1280×720 file at 30 FPS in real time, without re-encoding |
| Source identity | File video bitrate 2,161,488 bits/s; SHA-256 `1b8bcc7b27fba4806be975f1ab96517a7f4b2cc0be67272a44f39368414048a5` |
| Inference | AIP 7.0.0, batch size 1 requested, inference every frame, tracking on, analytics off, Node-RED and WebSocket metadata on |
| Software | JetPack 6.0 / L4T R36.3.0, DeepStream 7.0, TensorRT 8.6.2.3, GStreamer 1.20.3, ffprobe 4.4.2 |
| Sampling | 45-second warm-up, then 120 seconds per run; processed FPS sampled every 5 seconds |
| Outputs | [Exact per-profile settings](README.md#matched-one-camera-test); bitrate is the encoder target, not measured network throughput |

The EX3 is an Orin Nano 8 GB at 15 W with software video encoding; the EX5 is
an Orin NX 16 GB in MAXN mode with hardware video encoding. The source file and
settings match, but the runs do not start on a synchronized frame: the
20-second publisher probes and triggered clips can cover different parts of
the same loop. This
comparison does not isolate the effect of the encoder or of any single hardware
difference. The station API returned the saved input, processor and output
settings, but not the effective batch size.

## Measurements

33 of 33 matched one-camera cases; one run per station in each case.

| Model | Profile | ex3-orin-nano-8gb-15w | ex5-orin-nx-16gb-maxn |
|---|---|---|---|
| YOLOv7-tiny | `base` | Pass; 30.14 processed FPS | Pass; 30.13 processed FPS |
| YOLOv7-tiny | `webrtc` | Fail; 30.11 processed FPS; 23.54 delivered FPS, 133 ms max gap | Pass; 30.11 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLOv7-tiny | `webrtc-osd` | Fail; 30.14 processed FPS; 24.09 delivered FPS, 133 ms max gap | Pass; 30.15 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLOv7-tiny | `evt` | Fail; 30.15 processed FPS; 0/4 clips passed | Fail; 30.05 processed FPS; 2/4 clips passed |
| YOLOv7-tiny | `evt-osd` | Fail; 30.14 processed FPS; 1/4 clips passed | Pass; 30.04 processed FPS; 4/4 clips passed |
| YOLOv7-tiny | `webrtc+evt-osd` | Fail; 30.14 processed FPS; 22.84 delivered FPS, 167 ms max gap; 0/4 clips passed | Pass; 30.11 processed FPS; 30.00 delivered FPS, 33 ms max gap; 4/4 clips passed |
| YOLOv7-tiny | `webrtc-osd-1M` | Fail; 30.15 processed FPS; 22.94 delivered FPS, 167 ms max gap | Pass; 30.15 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLOv7-tiny | `webrtc-osd-4M` | Fail; 30.15 processed FPS; 24.64 delivered FPS, 167 ms max gap | Pass; 30.11 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLOv7-tiny | `webrtc-osd-640x360` | Fail; 30.15 processed FPS; 28.50 delivered FPS, 100 ms max gap | Pass; 30.12 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLOv7-tiny | `webrtc-osd-1920x1080` | Fail; 30.14 processed FPS; 17.13 delivered FPS, 267 ms max gap | Pass; 30.15 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLOv7-tiny | `webrtc+evt` | Fail; 30.15 processed FPS; 23.28 delivered FPS, 133 ms max gap; 0/4 clips passed | Pass; 30.12 processed FPS; 30.00 delivered FPS, 33 ms max gap; 4/4 clips passed |
| YOLOv10n | `base` | Pass; 30.10 processed FPS | Pass; 30.00 processed FPS |
| YOLOv10n | `webrtc` | Fail; 30.05 processed FPS; 23.79 delivered FPS, 133 ms max gap | Pass; 30.15 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLOv10n | `webrtc-osd` | Fail; 30.07 processed FPS; 22.69 delivered FPS, 133 ms max gap | Pass; 30.15 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLOv10n | `evt` | Fail; 30.14 processed FPS; 0/4 clips passed | Pass; 30.14 processed FPS; 4/4 clips passed |
| YOLOv10n | `evt-osd` | Fail; 30.15 processed FPS; 0/4 clips passed | Pass; 30.00 processed FPS; 4/4 clips passed |
| YOLOv10n | `webrtc+evt-osd` | Fail; 30.06 processed FPS; 22.04 delivered FPS, 167 ms max gap; 0/4 clips passed | Pass; 30.15 processed FPS; 30.00 delivered FPS, 33 ms max gap; 4/4 clips passed |
| YOLOv10n | `webrtc-osd-1M` | Fail; 30.05 processed FPS; 22.89 delivered FPS, 167 ms max gap | Pass; 30.09 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLOv10n | `webrtc-osd-4M` | Fail; 30.09 processed FPS; 22.99 delivered FPS, 133 ms max gap | Pass; 30.12 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLOv10n | `webrtc-osd-640x360` | Pass; 30.11 processed FPS; 29.30 delivered FPS, 67 ms max gap | Pass; 30.15 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLOv10n | `webrtc-osd-1920x1080` | Fail; 30.14 processed FPS; 16.56 delivered FPS, 200 ms max gap | Pass; 30.14 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLOv10n | `webrtc+evt` | Fail; 30.14 processed FPS; 23.59 delivered FPS, 200 ms max gap; 0/4 clips passed | Pass; 30.11 processed FPS; 30.00 delivered FPS, 33 ms max gap; 4/4 clips passed |
| YOLO26n | `base` | Pass; 30.15 processed FPS | Pass; 30.10 processed FPS |
| YOLO26n | `webrtc` | Fail; 30.14 processed FPS; 24.09 delivered FPS, 133 ms max gap | Pass; 30.13 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLO26n | `webrtc-osd` | Fail; 30.14 processed FPS; 22.44 delivered FPS, 133 ms max gap | Pass; 30.14 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLO26n | `evt` | Fail; 30.12 processed FPS; 1/4 clips passed | Pass; 30.15 processed FPS; 4/4 clips passed |
| YOLO26n | `evt-osd` | Fail; 30.15 processed FPS; 0/4 clips passed | Pass; 30.00 processed FPS; 4/4 clips passed |
| YOLO26n | `webrtc+evt-osd` | Fail; 30.14 processed FPS; 22.17 delivered FPS, 100 ms max gap; 1/4 clips passed | Fail; 30.15 processed FPS; 30.00 delivered FPS, 33 ms max gap; 3/4 clips passed |
| YOLO26n | `webrtc-osd-1M` | Fail; 30.14 processed FPS; 23.84 delivered FPS, 100 ms max gap | Pass; 30.11 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLO26n | `webrtc-osd-4M` | Fail; 30.15 processed FPS; 23.14 delivered FPS, 133 ms max gap | Pass; 30.08 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLO26n | `webrtc-osd-640x360` | Pass; 30.12 processed FPS; 29.45 delivered FPS, 100 ms max gap | Pass; 30.10 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLO26n | `webrtc-osd-1920x1080` | Fail; 30.02 processed FPS; 14.45 delivered FPS, 200 ms max gap | Pass; 30.09 processed FPS; 30.00 delivered FPS, 33 ms max gap |
| YOLO26n | `webrtc+evt` | Fail; 30.10 processed FPS; 23.64 delivered FPS, 133 ms max gap; 0/4 clips passed | Pass; 30.15 processed FPS; 30.00 delivered FPS, 33 ms max gap; 4/4 clips passed |

## Reading the results

**Processed FPS** is the camera's mean processing rate, not the frame rate of
the resulting video. **Delivered FPS** measures packets at the WebRTC
publisher over 20 seconds, not playback in a browser. **Clips passed** counts
the Smart Video clips that met the decoded-frame-rate, keyframe and gap checks.
Every enabled output and the processing check must pass. See the
[thresholds](README.md#one-camera-output-checks).

Each row is one valid observation on each station, not a repeatability claim.
Each has 24 processing samples; each recording run has four clips. Two EX5
attempts lacked usable probe or telemetry data and were repeated. Neither
counts as a video-quality result; measured FPS and gap failures do count.

## Reproduce the table

The [EX3 evidence](matched-ex3.jsonl) and [EX5 evidence](matched-ex5.jsonl)
contain the source fingerprint, checked pipeline settings, processing summary
and enabled video measurements for each row. No station access is required
to compare them:

```sh
python3 tools/capacity.py compare-matched \
  doc/benchmarks/capacity/matched-ex3.jsonl \
  doc/benchmarks/capacity/matched-ex5.jsonl
```
