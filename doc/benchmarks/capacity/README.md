# Camera capacity: EX5 and EX3

How many 720p30 H.264 cameras can one pipeline process? The benchmark tests each
detector with no video output, a WebRTC publisher, Smart Video event recording,
and both outputs together. To compare EX3 and EX5 with **the same video and
pipeline settings**, see the [measured one-camera results](matched.md) and the
[matched test setup](#matched-one-camera-test).

## Cameras per pipeline

These earlier measurements used common pass rules, but **not the same input
video**. The stations differ in module, power mode, video encoder and scene (see
[Station conditions](#station-conditions)), so each row shows what one station
sustained under its own conditions. EX3 did not qualify its output profiles at
one camera; EX5 passed each profile at one camera, but its multi-camera output
limits have not been established.

| Device | Model | No output | WebRTC | Smart Video | WebRTC + Smart Video |
|---|---|---:|---:|---:|---:|
| EX5, Orin NX 16 GB | YOLO26n | 7 | At least 1 (screened) | At least 1 (screened) | At least 1 (screened) |
| EX5, Orin NX 16 GB | YOLOv10n | 6† | At least 1 (screened) | At least 1 (screened) | At least 1 (screened) |
| EX5, Orin NX 16 GB | YOLOv7-tiny | 4 | At least 1 (screened) | At least 1 (screened) | At least 1 (screened) |
| EX3, Orin Nano 8 GB | YOLO26n | 5† | Not qualified at 1 | Not qualified at 1 | Not qualified at 1 |
| EX3, Orin Nano 8 GB | YOLOv10n | 5 | Not qualified at 1 | Not qualified at 1 | Not qualified at 1 |
| EX3, Orin Nano 8 GB | YOLOv7-tiny | 3 | Not qualified at 1 | Not qualified at 1 | Not qualified at 1 |

† EX5 YOLOv10n: seven 720p cameras passed 1 of 2 attempts. EX3 YOLO26n: six
720p cameras passed 3 of 7 attempts.

**Not qualified at 1** means detection kept up with one camera but the video
did not. For example, with Smart Video alone, [22 of 24 clips](../orin-nano-8gb-jp6.0/capacity.md#what-the-recording-failures-mean)
fell below 29 decoded FPS. **At least 1 (screened)** means one camera passed
processing and video-quality checks; it is not the maximum camera count. The
[EX5 output results](../orin-nx-16gb-jp6.0/capacity.md#protocol-v2-one-camera-output-checks)
are one-camera checks only. The [earlier EX3 measurements](../orin-nano-8gb-jp6.0/capacity.md#earlier-measurements-protocol-v1)
used different output checks and are not included in the table.

## How a result qualifies

### No-output camera counts

Every camera must average at least 29 processed FPS over 120 seconds after a
45-second warm-up, with no reconnects or inference stalls. A count is reported
when it passed at least two attempts and failed none, every lower count tested
also passed, and at least one attempt at the next count failed. † means a
higher count passed in some attempts and failed in others; the table keeps the
lower count.

### One-camera output checks

WebRTC must deliver at least 29 FPS, measured at the publisher for the first 20
seconds of sampling. Every Smart Video clip must be H.264 at the mosaic size,
last 9 seconds (±2 seconds), decode at 29 FPS or more and start on a keyframe.
Neither output may have a packet gap over 100 ms (plus 5 ms timing tolerance).

The WebRTC check reads the publisher's stream from MediaMTX over RTSP; it
measures the publisher, not browser playback.

## What each trial does

| CLI profile | Live output | Recording | OSD |
|---|---|---|---|
| `base` | None | Off | Off |
| `webrtc-osd` | WebRTC publisher | Off | On |
| `evt-osd` | None | Smart Video | On |
| `webrtc+evt-osd` | WebRTC publisher | Smart Video | On |

OSD means on-screen display (for example, detection boxes). The harness requests
batch size `min(cameras, 8)`, NvDCF tracking and inference on every frame, with
ROI/LCC analytics off, Node-RED and WebSocket metadata outputs on, and TCP RTSP
inputs.

The default output is a 1280×720 mosaic. **WebRTC** requests 2 Mbps and
**Smart Video** requests 4 Mbps; a trigger every 30 seconds requests 3 seconds before the event
and 6 seconds after it, with a 10-second buffer. After sampling, `ffprobe`
decodes each saved clip, so the check does not disturb the FPS window.

## Station conditions

| Condition | EX5 | EX3 |
|---|---|---|
| Module | Orin NX 16 GB | Orin Nano 8 GB |
| Power mode | MAXN, the highest on L4T R36.3 | 15 W, the highest on L4T R36.3 |
| Software | JetPack 6.0 / L4T R36.3, DeepStream 7.0, TensorRT 8.6.2.3 | JetPack 6.0 / L4T R36.3.0; DeepStream and TensorRT versions not recorded |
| Video encoder | Hardware | Software (no hardware encoder) |
| No-output cameras | Staggered 720p30 and 1080p30 copies of one street clip | 720p30 copies of a different street clip |
| Output-check camera | One 720p30 tollgate RTSP camera | One copy of the same street clip |
| Output runs per profile | 1 | 2 |
| One-camera output settings in records | OSD on, 1280×720, WebRTC target 2 Mbps; Smart Video buffer/trigger times | Same settings; neither screen includes an OSD-off counterpart |
| Test AIPs | 7.0.0; engine SHA-256 captured after the tests | 7.0.0; engine hashes not recorded |

Because the module, encoder and scenes differ, the two stations' results are
not a controlled, scene-identical comparison.

## Matched one-camera test

The [paired results and evidence](matched.md) separate processed FPS from the
video delivered or recorded. This is a **fixed workload**, not a search for
the maximum camera count. For each detector, point both idle stations at the
**same RTSP publisher**. Loop
one encoded H.264 1280×720, 30 FPS file in real time using stream copy: no
resizing, frame-rate conversion or second encoder. Hash the **encoded file at
the publisher** with `sha256sum`. Two feeds tagged “720p” can still have
different scenes, bitrates and keyframe spacing.

The harness probes the source from each station before registering a camera.
Every case uses one TCP input, 200 ms input latency, no frame-rate cap, batch
size 1 requested, inference on every frame, analytics off and both metadata
outputs on. Each run uses 45 seconds of warm-up and 120 seconds of sampling at
5-second intervals. Run all cases once on **each** box for **each** model:

| CLI profile | OSD | WebRTC target | Smart Video target | Output mosaic |
|---|---|---:|---:|---|
| `base` | Off | Off | Off | No video output |
| `webrtc` / `webrtc-osd` | Off / on | 2 Mbps | Off | 1280×720 |
| `evt` / `evt-osd` | Off / on | Off | 4 Mbps | 1280×720 |
| `webrtc+evt` / `webrtc+evt-osd` | Off / on | 2 Mbps | 4 Mbps | 1280×720 |
| `webrtc-osd-1M` / `webrtc-osd-4M` | On | 1 / 4 Mbps | Off | 1280×720 |
| `webrtc-osd-640x360` / `webrtc-osd-1920x1080` | On | 2 Mbps | Off | 640×360 / 1920×1080 |

The last four cases change **one setting at a time** from `webrtc-osd`. Input
size never changes when output size changes. OSD on draws boxes and labels,
not a clock or masks. WebRTC uses slot 2, control-rate setting 1 and a
one-second keyframe interval. Smart Video uses control-rate setting 1,
I-frame interval 5, a 10-second buffer and the same 3-second pre-roll,
6-second post-roll and 30-second trigger interval in every recording case.
Bitrates are **encoder targets**, not measured network throughput.

On each station, set `MS3` to its streaming API, `DEVICE` to its public
hardware/power-mode label, and `AIP` to the isolated test AIP for `MODEL`.
Use the **same** `SOURCE` URL, `SOURCE_SHA256`, model version and harness
revision on both. Give each station its own fresh results file; repeat the
command with the other two model/AIP pairs, appending to that station's file.

```sh
python3 tools/capacity.py matched --ms3 "$MS3" --device "$DEVICE" \
  --model "$MODEL" --aip "$AIP" --source "$SOURCE" \
  --source-sha256 "$SOURCE_SHA256" --out "$DEVICE.jsonl"
```

`--ffprobe` must be able to read saved clips; use `--live-ffprobe` if a
different executable is needed to reach the input and MediaMTX. On a station
with `ffprobe` only inside MS3, pass `--ffprobe "$PWD/tools/capacity-ffprobe.sh"`.
If MediaMTX shares MS3's Docker network rather than listening on the host,
set `CAPACITY_MTX_HOST=mediamtx` for that command. This changes only how the
probe reaches publisher slot 2, not the source URL or encoder settings. The harness
reads the saved pipeline settings back from the station before starting it and
checks the actual publisher/clip dimensions against the requested mosaic. A
setup error is not a failed FPS measurement. A clean run removes only the
camera, pipeline and recordings it created.

After copying the two results files to one machine:

```sh
python3 tools/capacity.py compare-matched EX3.jsonl EX5.jsonl
```

Compare one repetition at a time. The command refuses missing counterparts,
duplicate cases, a different source fingerprint, mismatched settings, or
trials with setup or cleanup errors. It reports how many of the 33 cases
are present; a matched subset is not the complete screen. The operator verifies
the file digest at the publisher; the RTSP probe checks the arriving stream's
format, not its file digest or a synchronized starting frame.

If a trial is invalid, preserve the original results and use `cleanup` for any
resources still listed in its ledger. Rerun only the missing or invalid case
with `matched --profile "$PROFILE"` and a fresh `--out` file. Compare one valid row per
case. Keep measured `pass: false` results: low delivered FPS or a long packet
gap is a result, not a reason to rerun until it passes.

## Reproducing a trial

Use an idle station, an FP16 engine built for its GPU, a separate test AIP ID,
and a reproducible RTSP source. Record the hardware, power mode, software and
engine versions alongside the results.

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
