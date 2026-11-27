# Camera capacity on EX5

With video outputs off, one EX5 (Orin NX 16 GB) sustained **7 cameras with
YOLO26n, 6 with YOLOv10n and 4 with YOLOv7-tiny** at both input resolutions.
In the original one-camera output screen, all three models passed with WebRTC,
Smart Video and both outputs together. These results establish a one-camera
lower bound, not multi-camera output limits. The separate
[same-camera comparison](../capacity/matched.md) includes OSD-off, bitrate
and output-size variants on both stations.

## Measured no-output limits

| Model | 720p30 cameras, no output | 1080p30 cameras, no output |
|---|---:|---:|
| YOLO26n | **7** | **7** |
| YOLOv10n | **6†** | **6** |
| YOLOv7-tiny | **4** | **4** |

Every camera must average at least 29 processed FPS over 120 seconds after a
45-second warm-up, with no reconnects or inference stalls. A count is reported
when it passed at least two attempts and failed none, every lower count tested
also passed, and at least one attempt at the next count failed. † means a
higher count passed in some attempts and failed in others; the table keeps the
lower count.

† YOLOv10n: seven 720p cameras passed 1 of 2 attempts.

These are near-real-time experimental limits for one pipeline, not guarantees
of lossless 30 FPS. Do not divide idle inference throughput by 30 to infer a
camera limit.

## Protocol v2: one-camera output checks

Each model passed all three output profiles at one camera. Every run averaged
at least 29 processed FPS during the 120-second sample after a 45-second
warm-up, and the delivered video passed every check.

WebRTC must deliver at least 29 FPS, measured at the publisher for the first 20
seconds of sampling. Every Smart Video clip must be H.264 at the mosaic size,
last 9 seconds (±2 seconds), decode at 29 FPS or more and start on a keyframe.
Neither output may have a packet gap over 100 ms (plus 5 ms timing tolerance).

| Model | Profile | Runs | Processed FPS (minimum) | WebRTC publisher | Smart Video clips | Result |
|---|---|---:|---:|---|---|---|
| YOLO26n | WebRTC | 1 | 30.15 | 30.00 FPS; 33 ms maximum gap | — | Passed |
| YOLO26n | Smart Video | 1 | 30.01 | — | 4/4 at 29+ FPS (30.00); 4/4 keyframe starts; 33 ms maximum gap | Passed |
| YOLO26n | WebRTC + Smart Video | 1 | 30.15 | 30.00 FPS; 33 ms maximum gap | 4/4 at 29+ FPS (30.00); 4/4 keyframe starts; 33 ms maximum gap | Passed |
| YOLOv10n | WebRTC | 1 | 30.15 | 30.00 FPS; 33 ms maximum gap | — | Passed |
| YOLOv10n | Smart Video | 1 | 30.12 | — | 4/4 at 29+ FPS (30.00); 4/4 keyframe starts; 33 ms maximum gap | Passed |
| YOLOv10n | WebRTC + Smart Video | 1 | 30.14 | 30.00 FPS; 33 ms maximum gap | 4/4 at 29+ FPS (30.00); 4/4 keyframe starts; 33 ms maximum gap | Passed |
| YOLOv7-tiny | WebRTC | 1 | 30.14 | 30.00 FPS; 33 ms maximum gap | — | Passed |
| YOLOv7-tiny | Smart Video | 1 | 30.14 | — | 4/4 at 29+ FPS (30.00); 4/4 keyframe starts; 33 ms maximum gap | Passed |
| YOLOv7-tiny | WebRTC + Smart Video | 1 | 30.14 | 30.00 FPS; 33 ms maximum gap | 4/4 at 29+ FPS (30.00); 4/4 keyframe starts; 33 ms maximum gap | Passed |

Each profile ran once with one 720p30 tollgate RTSP camera. Output profiles
enable the on-screen display (OSD). The [EX3 checks](../orin-nano-8gb-jp6.0/capacity.md)
ran each profile twice with a repeated street clip, so the scenes were not
matched. Multi-camera output limits were not established.

The WebRTC check reads the publisher's stream from MediaMTX over RTSP; it is
**not browser playback**. This screen had no browser viewer and did not vary
the output resolution, target frame rate or bitrate. The
[matched EX3/EX5 results](../capacity/matched.md) show what happens when
both stations use the same input video and output settings.

## Earlier measurements (protocol v1)

With the WebRTC publisher enabled, YOLO26n processed seven 720p inputs. The
slowest camera averaged 30.45 processed FPS without OSD and 30.23 with OSD.
These are **input processing rates**, not the frame rate delivered by the
publisher. No browser viewer was connected.

The earlier event-recording trials enabled the buffer and sent triggers but
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

The trials used Orin NX 16 GB in MAXN power mode, the highest mode L4T R36.3
offers on this module, with JetPack 6.0 / L4T R36.3, DeepStream 7.0 and
TensorRT 8.6.2.3.
Engines were FP16-enabled, 640x640, with batch size
`min(cameras, 8)`; NvDCF tracking was on, ROI/LCC analytics off, and inference
ran on every frame. TCP RTSP cameras looped staggered copies of one street
clip at 720p30 and 1080p30. Node-RED and WebSocket metadata outputs remained on.

The EX5 encodes output video in hardware. The one-camera records show a
1280×720 mosaic with OSD on and a 2 Mbps target when WebRTC was enabled.
Smart Video requested 3 seconds before and 6 seconds after the trigger,
with a 10-second buffer and 30 seconds between triggers. The `bitrate`
field in `output_settings` is the WebRTC setting, not the Smart Video bitrate.

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
qualified. Junction temperature reached 85.8 °C in the no-output
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
