# Camera capacity on EX5

With video outputs off, one EX5 (Orin NX 16 GB) sustained **7 cameras with
YOLO26n, 6 with YOLOv10n and 4 with YOLOv7-tiny** at both input resolutions.
WebRTC and Smart Video have no qualified camera count under protocol v2.

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

## Output screening

WebRTC-only screening on YOLO26n at seven 720p cameras measured 30.45 FPS
without OSD and 30.23 FPS with OSD. This tested the publisher **without a
connected viewer**, not browser playback or a separate maximum search.

The event-recording trials enabled the buffer and sent triggers but did not
verify clip integrity or duration. They cannot establish a Smart Video camera
count under the [EX3/EX5 output checks](../capacity/README.md).

### WebRTC diagnostic evidence

The diagnostics contain **16 WebRTC-only attempts: 5 successful and
11 failed**. All eleven failures used one camera with OSD on the EX5
development build. MediaMTX reported hardware-encoder errors, processed-FPS
metrics disappeared, and the pipeline stopped. These startup/encoder errors
are not zero-camera throughput results.

The five successful checks verified H.264 publisher output at two or seven
cameras: requested 1/2/4 Mbps and 640x360/1280x720/1920x1080 mosaics. They used
RTSP access to the publisher, not a browser viewer or the v2 packet-gap rule.
The diagnostics did not establish the cause of the one-camera failures.

## Test conditions

The no-output trials used Orin NX 16 GB, MAXN power mode, JetPack
6.0 / L4T R36.3, DeepStream 7.0 and TensorRT 8.6.2.3. The streaming service
was a development build. Engines were FP16-enabled, 640x640, with batch size
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

Only one pipeline and one repeated street scene were measured. No viewer
fan-out, varied/busy scenes, long thermal soak or multi-pipeline load was
qualified. Junction temperature reached 85.8 °C; the trials did not isolate
temperature as a cause of the mixed YOLOv10n results at seven cameras.

## Evidence

- [Trials and samples](capacity.jsonl): 41 trials (43 attempts and 977
  samples), plus 5 search summaries.
- [WebRTC diagnostics](capacity-outputs.json): probe settings, publisher
  observations and outcomes for all 16 checks.
- [Model comparison](comparison.md): accuracy and inference time on this EX5.

To regenerate the capacity summary:

```sh
python3 tools/capacity.py summarize doc/benchmarks/orin-nx-16gb-jp6.0/capacity.jsonl
```
