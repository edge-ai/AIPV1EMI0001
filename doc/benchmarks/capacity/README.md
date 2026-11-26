# Camera capacity: EX5 and EX3

How many 720p30 H.264 cameras can one pipeline process? The benchmark tests each
detector with no video output, a WebRTC publisher, Smart Video event recording,
and both outputs together.

## Cameras per pipeline

**No output** is comparable across devices: both tests required at least 29
processed FPS on every camera over 120 seconds. **Video output** is not yet
comparable. The [EX3 trials](../orin-nano-8gb-jp6.0/capacity.md) measured frames
delivered by the publisher and decoded from saved clips. The [EX5 output
trials](../orin-nx-16gb-jp6.0/capacity.md#output-screening) checked publisher
operation and recording triggers, not whether the video met the same quality
requirements.

| Device | Model | No output | WebRTC | Smart Video | WebRTC + Smart Video |
|---|---|---:|---:|---:|---:|
| EX5, Orin NX 16 GB | YOLO26n | 7 (baseline) | Not measured | Not measured | Not measured |
| EX5, Orin NX 16 GB | YOLOv10n | 6 (baseline) | Not measured | Not measured | Not measured |
| EX5, Orin NX 16 GB | YOLOv7-tiny | 4 (baseline) | Not measured | Not measured | Not measured |
| EX3, Orin Nano 8 GB | YOLO26n | 5† | Not qualified at 1 | Not qualified at 1 | Not qualified at 1 |
| EX3, Orin Nano 8 GB | YOLOv10n | 5 | Not qualified at 1 | Not qualified at 1 | Not qualified at 1 |
| EX3, Orin Nano 8 GB | YOLOv7-tiny | 3 | Not qualified at 1 | Not qualified at 1 | Not qualified at 1 |

† Five YOLO26n cameras passed both EX3 runs; six passed only 3 of 7 attempts.

**Not qualified at 1** means detection kept up with one camera but the video
did not. For example, with Smart Video alone, [22 of 24 clips](../orin-nano-8gb-jp6.0/capacity.md#what-the-recording-failures-mean)
fell below 29 decoded FPS. **Not measured** means there is no EX5 result for
that output under protocol v2. The [earlier EX3 measurements](../orin-nano-8gb-jp6.0/capacity.md#earlier-measurements-protocol-v1)
used different output checks and are not included in the table.

## What each trial does

| CLI profile | Live output | Recording | OSD |
|---|---|---|---|
| `base` | None | Off | Off |
| `webrtc-osd` | WebRTC publisher | Off | On |
| `evt-osd` | None | Smart Video | On |
| `webrtc+evt-osd` | WebRTC publisher | Smart Video | On |

OSD means on-screen display (for example, detection boxes). Both outputs use a
1280×720 mosaic. After 45 seconds of warm-up, the harness samples processed FPS
per camera for 120 seconds. A pass requires zero reconnects and inference stalls.

**Smart Video** runs at 4 Mbps. A trigger every 30 seconds requests 3 seconds
before the event and 6 seconds after it, with a 10-second buffer. After sampling,
`ffprobe` decodes each H.264 clip. Each clip must average at least 29 decoded FPS
and last 9 seconds (±2 seconds for keyframe alignment). Protocol v2 also requires
a keyframe at the start and limits packet gaps to 100 ms plus 5 ms timing
tolerance.

**WebRTC** runs at 2 Mbps. Protocol v2 reads the publisher's stream from
MediaMTX for the first 20 seconds of sampling. It must deliver at least 29 FPS
with the same packet-gap limit. This measures the publisher, not browser playback.

## Reproducing a trial

Use an idle station, an FP16 engine built for its GPU, a separate test AIP ID,
and a reproducible RTSP source. Record the hardware, power mode, software and
engine versions alongside the results. EX3 encodes video in software; EX5 has
a hardware encoder.

Run on the station host with `tegrastats` available. Substitute the streaming
service URL, test AIP ID and camera URLs for the placeholders:

```sh
python3 tools/capacity.py plan
python3 tools/capacity.py preflight --ms3 http://MS3:8000
python3 tools/capacity.py search \
  --ms3 http://MS3:8000 --device ex3-orin-nano-8gb-15w \
  --model yolo26 --aip AIPV1EMI90XX \
  --rtsp-template 'rtsp://VCAM:PORT/cam{index:02d}' \
  --camera-tag 720p --profile base --max-n 8 --out ex3-capacity.jsonl
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

`--ffprobe` selects the executable for inspecting saved clips. For WebRTC,
use `--live-ffprobe` if the station host cannot reach MediaMTX directly (for
example, point it to a wrapper that runs `ffprobe` in the streaming service
container). If a run is interrupted,
`cleanup` uses its ledger to remove only that run's resources. Finish cleanup
before reusing the result file.

Use separate result files for each device, stack version and power mode. Check
the highest passing camera count and the next higher count independently with
`fixed`. A passing count at the search ceiling is a lower bound, not a maximum.
