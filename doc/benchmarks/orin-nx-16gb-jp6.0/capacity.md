# Camera capacity on EX5

The current comparison covers **no output, WebRTC only, Smart Video recording
only, and WebRTC with Smart Video recording**. HLS is out of scope and
continuous recording is not a ready feature. Neither belongs in the current
capacity recommendation.

## Measured no-output limits

These EX5 (Orin NX 16 GB) results were measured on 18-19 November 2026.
They are not new measurements of the currently running station.

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

## Supported output comparison: not yet measured

The old "all outputs" column combined continuous recording, WebRTC, HLS and
event recording. It cannot be relabeled as "WebRTC + Smart Video": the actual
branches, encoder work and recording verification differ.

WebRTC-only screening on YOLO26n at seven 720p cameras measured 30.45 FPS
without OSD and 30.23 FPS with OSD. This tested the publisher **without a
connected viewer**, not browser playback or a separate maximum search.
One-camera WebRTC + OSD failed in eleven later diagnostics on that development
build; a maximum of seven is not proof that every smaller count works.

Old event-recording trials enabled the buffer and sent triggers, but did not
independently verify clip integrity or pre/post duration. They are not
qualified Smart Video results. New trials explicitly configure **3 seconds
pre-roll, 6 seconds post-roll, a 10-second buffer limit and a trigger every
30 seconds**, then decode and check each resulting clip.

The [EX5/EX3 matrix and readiness checks](../capacity/README.md) show what still
needs measurement. No EX3 deployment or camera-capacity result is claimed.

## Test conditions

The retained no-output results used Orin NX 16 GB, MAXN power mode, JetPack
6.0 / L4T R36.3, DeepStream 7.0 and TensorRT 8.6.2.3. The streaming service
was a development build. Engines were FP16-enabled, 640x640, with batch size
`min(cameras, 8)`; NvDCF tracking was on, ROI/LCC analytics off, and inference
ran on every frame. TCP RTSP cameras looped staggered copies of one street
clip. Node-RED and WebSocket metadata outputs remained on.

See the [historical test conditions](capacity-historical.md#test-conditions)
for engine hashes and the [limitations](capacity-historical.md#limitations)
for thermal sensitivity, scene content and pipeline sharing.

## Evidence

- [Original trials and samples](capacity.jsonl): unchanged, including excluded profiles.
- [Original output diagnostics](capacity-outputs.json): unchanged, including failures.
- [Historical report](capacity-historical.md): original output tables, boundary
  rechecks, conditions and cleanup inventory; not the current comparison.

The accuracy and inference-time results in the [model comparison](comparison.md)
are unaffected by this change in output-test scope.
