# Camera capacity on EX5

The current comparison covers **no output, WebRTC only, Smart Video recording
only, and WebRTC with Smart Video recording**. HLS and continuous-recording
benchmark evidence has been removed, including mixed profiles that enabled
either. Supported-profile trials below retain their original measurements.

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

No output profile has been qualified on EX5 under the current v2 checks.
Retained older WebRTC and event-recording observations are screening only,
not capacity recommendations. Removing excluded profiles does not turn their
measurements into evidence for the remaining profiles.

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
needs measurement; the [EX3 report](../orin-nano-8gb-jp6.0/capacity.md) is a
separate device campaign, not a qualification of EX5.

### WebRTC diagnostic evidence

The retained diagnostics contain **16 WebRTC-only attempts: 5 successful and
11 failed**. All eleven failures used one camera with OSD on the historical
development build. MediaMTX reported hardware-encoder errors, processed-FPS
metrics disappeared, and the pipeline stopped. These startup/encoder errors
are not zero-camera throughput results.

The five successful checks verified H.264 publisher output at two or seven
cameras: requested 1/2/4 Mbps and 640x360/1280x720/1920x1080 mosaics. They used
RTSP access to the publisher, not a browser viewer or the v2 packet-gap rule.
Readiness, nominal frame rate and bitrate are not enough to qualify delivered
video. The root cause of the one-camera encoder failures remains outside this
AIP benchmark; do not infer that every count below seven works.

## Test conditions

The retained no-output results used Orin NX 16 GB, MAXN power mode, JetPack
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
qualified. Junction temperature in the retained trials reached 85.8 C;
YOLOv10n's seven-camera success/failure is consistent with, but does not prove,
thermal sensitivity. Do not transfer accuracy, inference timing or capacity
numbers to EX3. EX3 has software video encoding; this EX5 has NVENC.

## Evidence

- [Selected trials and samples](capacity.jsonl): 41 trials (43 attempts and
  977 samples) plus 5 search summaries. No-output, WebRTC and event-recording
  screening rows are retained unchanged; excluded profiles are removed as
  whole records, never relabeled.
- [Selected WebRTC diagnostics](capacity-outputs.json): all 16 WebRTC-only
  checks, including the 11 failures. Metadata identifies the filtering and
  the original source hashes; those hashes are not checksums of this subset.

The superseded combined-output report is removed. Rebuild supported-profile
summaries from the selected evidence:

```sh
python3 tools/capacity.py summarize doc/benchmarks/orin-nx-16gb-jp6.0/capacity.jsonl
```

The accuracy and inference-time results in the [model comparison](comparison.md)
are unaffected by this change in output-test scope.
