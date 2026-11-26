# Camera capacity on EX5

With video outputs off, one EX5 (Orin NX 16 GB) sustained **7 cameras with
YOLO26n, 6 with YOLOv10n and 4 with YOLOv7-tiny** at both input resolutions.
Under protocol v2, all three models also passed one-camera checks for WebRTC,
Smart Video and both outputs together. These results establish a one-camera
lower bound, not multi-camera output limits.

## Measured no-output limits

The trials ran on 18–19 November 2026.

| Model | 720p30 cameras, no output | 1080p30 cameras, no output |
|---|---:|---:|
| YOLO26n | **7** | **7** |
| YOLOv10n | **6** | **6** |
| YOLOv7-tiny | **4** | **4** |

Each camera averaged at least **29 processed FPS** over 120 seconds after a
45-second warm-up, with no reconnects or inference stalls. Each reported count
passed at least twice and the next count failed. YOLOv10n passed once at seven
720p cameras and failed a recheck, so the reported limit is six.

These are near-real-time experimental limits for one pipeline, not guarantees
of lossless 30 FPS. Do not divide idle inference throughput by 30 to infer a
camera limit.

## Protocol v2: one-camera output checks

On 26 November 2026, each model passed all three 720p30 output profiles at one
camera. Every profile averaged at least 29 processed FPS during the 120-second
sample after a 45-second warm-up.

| Model | Profile | Processed FPS (minimum) | WebRTC publisher | Smart Video clips |
|---|---|---:|---|---|
| YOLO26n | WebRTC | 30.15 | 30.00 FPS; 33 ms maximum gap | — |
| YOLO26n | Smart Video | 30.01 | — | 4/4 H.264 clips at 30.00 FPS; keyframe start; 33 ms maximum gap |
| YOLO26n | WebRTC + Smart Video | 30.15 | 30.00 FPS; 33 ms maximum gap | 4/4 H.264 clips at 30.00 FPS; keyframe start; 33 ms maximum gap |
| YOLOv10n | WebRTC | 30.15 | 30.00 FPS; 33 ms maximum gap | — |
| YOLOv10n | Smart Video | 30.12 | — | 4/4 H.264 clips at 30.00 FPS; keyframe start; 33 ms maximum gap |
| YOLOv10n | WebRTC + Smart Video | 30.14 | 30.00 FPS; 33 ms maximum gap | 4/4 H.264 clips at 30.00 FPS; keyframe start; 33 ms maximum gap |
| YOLOv7-tiny | WebRTC | 30.14 | 30.00 FPS; 33 ms maximum gap | — |
| YOLOv7-tiny | Smart Video | 30.14 | — | 4/4 H.264 clips at 30.00 FPS; keyframe start; 33 ms maximum gap |
| YOLOv7-tiny | WebRTC + Smart Video | 30.14 | 30.00 FPS; 33 ms maximum gap | 4/4 H.264 clips at 30.00 FPS; keyframe start; 33 ms maximum gap |

Each publisher was measured for 20 seconds; each of the 24 saved clips was
checked for H.264 decoding, at least 29 FPS, a keyframe start and packet gaps
within the protocol limit. The EX5 checks used a 720p30 tollgate RTSP source.
The [EX3 checks](../orin-nano-8gb-jp6.0/capacity.md) used a different repeated
street source, so the scenes were not matched. Multi-camera output limits were
not established.

## Historical output screening (protocol v1)

With the WebRTC publisher enabled, YOLO26n processed seven 720p inputs. The
slowest camera averaged 30.45 processed FPS without OSD and 30.23 with OSD.
These are **input processing rates**, not the frame rate delivered by the
publisher. No browser viewer was connected.

The historical event-recording trials enabled the buffer and sent triggers but
did not verify clip integrity or duration. They do not establish a Smart Video
camera count under the [EX3/EX5 output checks](../capacity/README.md).

### WebRTC diagnostic evidence

The diagnostics contain **16 WebRTC-only attempts: 5 successful and
11 failed**. All eleven failures used one camera with OSD. MediaMTX reported
hardware-encoder errors, processed-FPS
metrics disappeared, and the pipeline stopped. These startup/encoder errors
are not zero-camera throughput results.

The five successful checks verified H.264 publisher output at two or seven
cameras: requested 1/2/4 Mbps and 640x360/1280x720/1920x1080 mosaics. They used
RTSP access to the publisher, not a browser viewer or the v2 packet-gap rule.
The diagnostics did not establish the cause of the one-camera failures.

## Test conditions

The no-output trials used Orin NX 16 GB, MAXN power mode, JetPack
6.0 / L4T R36.3, DeepStream 7.0 and TensorRT 8.6.2.3.
Engines were FP16-enabled, 640x640, with batch size
`min(cameras, 8)`; NvDCF tracking was on, ROI/LCC analytics off, and inference
ran on every frame. TCP RTSP cameras looped staggered copies of one street
clip. Node-RED and WebSocket metadata outputs remained on.

The isolated test AIPs were version 7.0.0, built on this target with min/opt/max
batch 1/8/8. FP16 is enabled, not forced for every layer; YOLOv7-tiny includes
EfficientNMS. Engine SHA-256 values captured after these tests:

| Model | SHA-256 |
|---|---|
| YOLO26n | `0cb26f248c60759893471f9966d293b62faa812a7d98bd69183d0c99a7f96802` |
| YOLOv10n | `8add86b6a31ec672381cf5b23d56f62c8cfcb0f30812d1591d039cc67300c85f` |
| YOLOv7-tiny | `c9da0f95ec3f10cc2d5cd213dbfc364a932478a4e884b1d873cf69919842fa5c` |

Per-trial records retain AIP identity/version, not a per-trial engine hash.
These EX5 engines must be rebuilt or independently validated on another GPU.

## Limitations

The no-output tests used repeated copies of one street scene; the v2 output
checks used one tollgate scene at one camera. No viewer fan-out, varied/busy
scenes, long thermal soak, multi-camera output limit or multi-pipeline load was
qualified. Junction temperature reached 85.8 °C in the historical no-output
trials; those tests did not isolate temperature as a cause of the mixed
YOLOv10n results at seven cameras.

## Evidence

- [Trials and samples](capacity.jsonl): 41 trials (43 attempts and 977
  samples), plus 5 search summaries.
- [Protocol v2 output checks](capacity-output-v2.jsonl): nine one-camera
  profile checks for the three models.
- [WebRTC diagnostics](capacity-outputs.json): probe settings, publisher
  observations and outcomes for all 16 checks.
- [Model comparison](comparison.md): accuracy and inference time on this EX5.

To regenerate the capacity summary:

```sh
python3 tools/capacity.py summarize doc/benchmarks/orin-nx-16gb-jp6.0/capacity.jsonl
```
