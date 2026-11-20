# RTSP camera capacity on Orin NX 16 GB

How many 30 FPS RTSP cameras can **one pipeline** process with each model,
using an FP16 TensorRT engine, before any camera drops below 29 processed FPS?

## Answer

Sources: H.264 1280x720 at 30 FPS. Outputs: 1280x720 mosaic, 2 Mbps delivery.
These are repeat-checked experimental boundaries on the tested development
build, not guarantees of lossless 30 FPS or production reliability.

| Model | No video output | Recording + WebRTC, with OSD | Recording + WebRTC + HLS + event recording, with OSD |
|---|---:|---:|---:|
| YOLO26n | **7** | **7** | **7** |
| YOLOv10n | **6** (see below) | **6** | **6** |
| YOLOv7-tiny | **4** | **4** | **4** |

With 1920x1080 sources and no video output, the counts are also
7, 6 and 4; see [1080p sources](#1080p-sources).

YOLOv10n passed once at seven cameras (29.64 FPS) and failed an independent
recheck (28.91 FPS), so the repeat-checked count is six.
**A smaller count is not automatically reliable:** one-camera WebRTC and
continuous recording failed in the output diagnostics; see the
[single-camera issue](#known-issue-single-camera-mediamtx-outputs).

What this means:

- **The three measured profiles have the same camera counts.** This suggests
  the same counts for intermediate output combinations, assuming added outputs
  do not increase capacity. Those combinations are **inferred**, not individually
  measured for every model. YOLO26n also has direct screening results below.
- **The outputs are not free.** At the first failing count, the slowest camera
  is slower with outputs enabled: YOLO26n at eight cameras drops from about
  28.1 FPS with no outputs to 27.3 with recording + WebRTC and 26.6 with all
  four; YOLOv10n at seven from about 28.9 to 28.4 and 27.5. These were not
  simultaneous or temperature-controlled trials, so they do not isolate an
  exact output cost.
- **Bitrate and mosaic changes did not push YOLO26n below the pass threshold**
  at seven cameras: requested 1 to 4 Mbps and 640x360 to 1920x1080 were screened.
  This does not establish zero cost, an independently searched maximum for
  each setting, or equivalent results for the other models.
- **Model latency is not camera capacity.** Camera count depends on decoding,
  batching, tracking and outputs as well as inference; use these counts, not
  the [latency benchmarks](comparison.md), as a starting point for deployment tests.

## How a count is decided

- Each camera's mean processed rate (the per-source
  `pipeline_perf_fps_average_10sec` gauge, sampled every 5 s) must be at least
  **29 FPS** over a **120 s** window, after a 45 s warm-up.
- The pipeline must stay Running, with every camera streaming, zero reconnects
  and no inference-stall flags.
- The reported count N is the highest measured count with no valid failure at
  or below N, and a valid failure at N + 1. Each headline passing count was
  rechecked independently. Searches automatically repeat trials within 0.5 FPS
  of the threshold; fixed-count runs need explicit rechecks.
- Setup, API, encoder/startup and cleanup errors are not throughput limits.
  Failed output diagnostics are preserved separately, not silently counted as
  zero camera capacity.

The 29 FPS threshold is about 3.3% below the nominal source rate. It shows
near-real-time processing, not a measured dropped-frame percentage or lossless
30 FPS. These are averages of moving-average gauges, not frame counters;
finite-window readings can slightly exceed 30 FPS.

## Evidence at 720p

Slowest-camera mean FPS for each boundary trial and its independent rechecks.

| Model | Outputs | Passed at the count | Failed at count + 1 |
|---|---|---|---|
| YOLO26n | none | 7: 30.51, 30.51, 30.45 | 8: 28.20, 28.10 |
| YOLO26n | recording + WebRTC, OSD | 7: 30.06, 30.13, 30.29 | 8: 27.33 |
| YOLO26n | all four, OSD | 7: 29.80, 30.14 | 8: 26.57 |
| YOLOv10n | none | 6: 30.35, 30.20 | 7: 28.91 (an earlier 7 passed at 29.64); 8: 25.85, 25.59 |
| YOLOv10n | recording + WebRTC, OSD | 6: 30.40, 30.39 | 7: 28.39 |
| YOLOv10n | all four, OSD | 6: 30.49, 30.50 | 7: 27.50 |
| YOLOv7-tiny | none | 4: 30.11, 30.07 | 5: 24.58, 24.38 |
| YOLOv7-tiny | recording + WebRTC, OSD | 4: 30.42, 30.39 | 5: 25.64 |
| YOLOv7-tiny | all four, OSD | 4: 30.18, 30.41 | 5: 25.31 |

At the reported 720p counts, mean GPU busy was about 84 to 97% for YOLO26n
and YOLOv10n, and 83 to 91% for YOLOv7-tiny. Utilization alone cannot identify
the limiting stage. Accuracy is measured separately in the
[model comparison](comparison.md); these trials measure throughput, not mAP.

## Screening the other settings

Each row is one YOLO26n trial at seven cameras. Each establishes a passing
count, not a separate maximum search. An upper bound of seven for these profiles
is inferred from the no-output eight-camera failure only if output cost is
monotonic. "WebRTC" means the encoder and publisher with **no connected viewer**.

| Outputs | Slowest camera FPS | Mean GPU busy |
|---|---:|---:|
| None | 30.51 | 91.7% |
| Continuous recording | 30.35 | 96.6% |
| WebRTC | 30.45 | 94.6% |
| WebRTC + OSD | 30.23 | 95.0% |
| HLS | 30.34 | 95.1% |
| HLS + OSD | 30.21 | 96.6% |
| Event recording | 30.29 | 95.9% |
| Event recording + OSD | 30.20 | 94.8% |
| Continuous recording + WebRTC + OSD | 30.13 | 96.2% |
| WebRTC + OSD, 1 Mbps | 30.17 | 96.3% |
| WebRTC + OSD, 4 Mbps | 30.19 | 95.2% |
| WebRTC + OSD, 640x360 mosaic | 30.09 | 95.9% |
| WebRTC + OSD, 1920x1080 mosaic | 30.10 | 96.1% |

Both WebRTC and HLS, with and without OSD, sustained seven cameras in this
screen. This does not measure a statistically significant cost difference.
OSD applies to WebRTC, HLS and event recording; continuous recordings are
written **without OSD**, even when other outputs have it. Recording with OSD
is therefore not a separate continuous-recording configuration on this build.

The screen does not justify a full Cartesian-product test sweep. The next
useful tests are targeted: the chosen model/profile with actual viewers, a
longer thermal soak, and any bitrate or resolution outside this range.
Results for untested YOLOv10n/YOLOv7-tiny bitrate or mosaic variants remain
estimates until checked. Changing mosaic dimensions is not the same as
changing every camera's input resolution.

## 1080p sources

Sources: H.264 1920x1080 at 30 FPS; no video output.

| Model | Passed at the count | Failed at count + 1 |
|---|---|---|
| YOLO26n | 7: 29.98, 30.20 | 8: 28.57 and 26.67 (one repeated trial); 9: 23.02 |
| YOLOv10n | 6: 30.38, 30.42 | 7: 27.22 |
| YOLOv7-tiny | 4: 29.24, 29.39, 29.28 | 5: 23.55 |

The counts match 720p. YOLOv7-tiny has little margin at four 1080p cameras.
Outputs were not tested with 1080p sources.

## Output verification

Separate YOLO26n diagnostics used a 30-second warm-up and up to 90 seconds of
FPS sampling. The [output evidence](capacity-outputs.json) preserves all
21 attempts, including 13 failures. Successful multi-camera probes confirmed:

| Output | Cameras | Requested / observed bitrate | Observed video |
|---|---:|---|---|
| WebRTC publisher, low bitrate | 7 | 1 / 1.24 Mbps | H.264, 1280x720, 30 FPS |
| WebRTC publisher, default | 2 | 2 / 2.10 Mbps | H.264, 1280x720, 30 FPS |
| WebRTC publisher, high bitrate | 2 | 4 / 4.19 Mbps | H.264, 1280x720, 30 FPS |
| WebRTC publisher, smaller mosaic | 2 | 2 / 2.06 Mbps | H.264, 640x360, 30 FPS |
| WebRTC publisher, larger mosaic | 2 | 2 / 2.04 Mbps | H.264, 1920x1080, 30 FPS |
| HLS + OSD | 2 | 2 / 2.06 Mbps | H.264, 1280x720, 30 FPS |
| Continuous recording, combined-output profile | 2 | 3 / 3.14 Mbps at publisher | H.264, 1280x720; MP4 files parsed |

Publisher bitrates are MediaMTX received-byte deltas over a nominal 20 seconds;
HLS bitrate is segment bytes divided by playlist duration. They are approximate
transport/container rates, not exact encoder targets. The 1 Mbps setting
delivered about 1.24 Mbps in that window. MP4 probes parsed 56.58-second and
7.17-second files; their packet counts imply about 30.8 packets/s. The reported
`90000/1` rate on the recording path is not a credible video FPS measurement.

Extracted frames were inspected: the WebRTC publisher carried labeled detection
boxes; the recording path had no OSD. No clip images are included in this
repository. These checks cover RTSP access to the publisher, HLS segment
structure and recording metadata, **not WebRTC negotiation, browser playback,
full-file decode integrity or viewer fan-out**.

The first combined-output HLS probe incorrectly counted zero segments because
its parser did not strip the playlist's directory prefix. A later HLS-only
probe confirmed six segments, 1280x720 at 30 FPS. The original observation is
retained rather than overwritten.

## Known issue: single-camera MediaMTX outputs

On this development build, YOLO26n one-camera WebRTC + OSD failed in all
11 attempts across the bitrate/mosaic settings above. Continuous recording
without OSD also failed twice at one camera. Publisher readiness was missing
or false, processed-FPS metrics disappeared, and the pipeline stopped.
Service logs include `NvVideoEncBlockReportError: BlockSide error 0x4` and
failed buffer-queue errors on the MediaMTX hardware encoder.

HLS + OSD at one camera passed (30.05 processed FPS, 1280x720, about 2.08 Mbps).
Subsequent two-camera WebRTC and seven-camera 1 Mbps WebRTC checks passed
without a service restart. This points to a single-input MediaMTX encoder-path
problem, not a general one-camera compute limit or a permanently wedged
encoder. The root cause and fix are outside this repository's benchmark;
neither other models nor every one-camera profile were tested. Do not
interpret a measured maximum of seven as proof that all counts from one to
seven work.

## Test conditions

- Orin NX 16 GB, power mode MAXN, JetPack 6.0 / L4T R36.3, DeepStream 7.0,
  TensorRT 8.6.2.3. A development build of the video-streaming service, not a
  characterized production release.
- Separate test AIPs `AIPV1EMI9026`, `AIPV1EMI9010` and `AIPV1EMI9007`,
  version 7.0.0, built from this repository's flavors with FP16 engines at
  640x640 (EfficientNMS for YOLOv7-tiny). Build logs confirm `trtexec --fp16`
  with min/opt/max batch 1/8/8 and successful builds before the trials.
  TensorRT reports FP32+FP16: FP16 is enabled, not forced on every layer.
- One pipeline at a time; batch size `min(cameras, 8)`; inference on every
  frame; NvDCF tracking; ROI and line-crossing analytics off; TCP RTSP with
  200 ms latency; Node-RED and WebSocket metadata outputs on.
- Delivery (WebRTC, HLS) at 2 Mbps unless the profile says otherwise;
  continuous recording at 3 Mbps; event recording at 4 Mbps.
- Cameras: staggered loops of one re-encoded street clip, served over RTSP from
  a separate x86 host on the lab network. This is not a crowded-scene or
  scene-diversity study.

Deployed engine SHA-256 values captured after the tests:

| Model | SHA-256 |
|---|---|
| YOLO26n | `0cb26f248c60759893471f9966d293b62faa812a7d98bd69183d0c99a7f96802` |
| YOLOv10n | `8add86b6a31ec672381cf5b23d56f62c8cfcb0f30812d1591d039cc67300c85f` |
| YOLOv7-tiny | `c9da0f95ec3f10cc2d5cd213dbfc364a932478a4e884b1d873cf69919842fa5c` |

Per-trial records retain AIP identity/version, not a per-trial engine hash.
The development streaming service and its source bind mount are an additional
reproducibility limitation; these counts do not characterize every release.

## Limitations

- **Cooling.** Junction temperature reached 88.3 C and input power 24.8 W.
  YOLOv10n at seven cameras passed at 80.6 C and failed at 85.8 C, which is
  consistent with, but does not prove, thermal sensitivity. Cooling and power
  headroom need a deployment-specific soak test.
- **Viewers.** WebRTC was measured with no viewer and HLS was not played in a
  browser. Viewers add load on the station.
- **Event recording** was triggered every 30 seconds in the sampling windows.
  These are not continuous event storms, and event-file integrity was not
  independently verified.
- **One pipeline.** Several pipelines sharing the station were not tested.
- **Content.** A single clip; scenes with many objects cost more tracking and
  drawing time.

## Final resource inventory

Checked on 2026-11-19 at 05:04 UTC: no capacity-test pipelines, registered test
cameras, ownership ledgers, harness processes or MediaMTX paths remained.
Ownership was matched to exact retained trial names before removing 24 HLS
folders and 38 empty recording folders. The seven original pipelines remained
stopped; unrelated recordings were left untouched.

The isolated virtual-camera container and network were removed; the three
shared camera containers remained running and healthy. No stack service was
restarted. The three temporary AIPs remain registered for repeat testing;
the original `AIPV1EMI0001` was not replaced. Raw diagnostic logs and reusable
camera fixtures were retained in the isolated test workspaces. Inert signaling
stopped-state entries were not edited directly.

## Raw evidence and reproduction

[`capacity.jsonl`](capacity.jsonl) holds 80 capacity/pilot trials,
82 attempts and 1,897 samples, plus 11 search summaries. Trial starts begin at
2026-11-18 13:21 UTC; the final sample window ends at 2026-11-19 03:30 UTC.
There are 78 full-window attempts (45-second warm-up, 120-second sample)
and four short pilot attempts. Every retained capacity sample shows a Running
pipeline, the expected input count, streaming links and zero reconnects.
Every verdict reproduces from the saved samples; `tests/test_capacity.py`
also checks the reported boundaries.

[`capacity-outputs.json`](capacity-outputs.json) separately preserves all
21 output diagnostics, their raw trial samples and relevant encoder messages.
It omits Docker stderr containing private paths and diagnostic-image filenames.
Its failures must not be merged into the throughput searches as zero capacity.

```text
capacity.jsonl SHA-256
8c56e047daf740c62e4928c205bb861de80406c2ab652041475428540456cfaa

capacity-outputs.json SHA-256
d8c017fcbbf471de941ede1b8cba573a1a3c37a909cddc253d648ca7814c916d
```

Rebuild the summary tables with:

```sh
python3 tools/capacity.py summarize doc/benchmarks/orin-nx-16gb-jp6.0/capacity.jsonl
```

The harness, the test plan and how the 1,296 raw combinations were reduced are
in the [capacity test plan](../capacity/README.md).
