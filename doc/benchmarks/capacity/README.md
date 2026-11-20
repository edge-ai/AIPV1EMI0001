# RTSP camera capacity test plan

This plan measures how many 30 FPS RTSP camera feeds the EdgeAI Station can sustain for each of the three TensorRT FP16 AI Processors: YOLOv7-tiny, YOLOv10n, and YOLO26n. The [live report and raw evidence](../orin-nx-16gb-jp6.0/capacity.md) cover three 720p output profiles per model, thirteen YOLO26n screening trials, independent passing-boundary rechecks, and bounded 1080p no-output results. Separate output diagnostics verified bitrate/dimensions and exposed a one-camera WebRTC/continuous-recording failure on the development build. Untested combinations remain inferences, not measured cells.

The pass criterion is **at least 29 processed FPS per input**, averaged over a 120-second sample after a 45-second warm-up, with no camera reconnects or inference stalls. This is a near-30-FPS tolerance, not a guarantee of processing every frame at exactly 30 FPS. The harness uses the per-source post-mux PERF metric (`pipeline_perf_fps_average_10sec` from the video-streaming service's `/metrics` endpoint). Decoded pre-mux arrival is recorded separately: cameras sending 30 FPS do not prove the model processes 30 FPS. A station without fresh per-camera PERF metrics cannot run this measurement; the harness does not fall back to arrival or pipeline-wide averages.

The harness records per-input FPS, pipeline status, GPU/CPU/RAM/encoder telemetry when available, and the tested settings. A search finds the highest passing camera count by binary search, assuming throughput is monotonic; if the search ceiling passes, it is a lower bound. `summarize` also includes independent rechecks: it reports the highest count with no valid failing trial at or below it and flags contradictory higher passes. This reduces the observed YOLOv10n no-output count from seven to six. Setup/API errors or incomplete cleanup invalidate a trial, including errors in a repeat attempt, rather than becoming a false capacity limit. Recheck the passing boundary and its next camera count before publishing. A throughput bound does not establish reliability at every smaller count.

## Why the test matrix is pruned

The input count scales work that is per camera: decoding, inference, tracking, and analytics. The output mosaic and its encoders are shared pipeline branches, so their cost is not multiplied by camera count. We therefore search capacity for selected output profiles and use fixed-count, one-factor screening runs to estimate the other settings. A follow-up capacity search checks the prediction with the largest observed gap.

OSD is not a single global cost in this pipeline: continuous recording has no OSD; WebRTC, HLS, and event recording can have it. OSD with no display-capable output cannot change the run and is normalized away. After pipeline startup, the WebRTC MediaMTX slot is started through the video-streaming API. Continuous recording is started and stopped through the signaling service, as the station UI does; that is what makes MediaMTX write recording files. A recording trial fails if no file was written, and the files are deleted when the trial ends. HLS, recording, and event recording are independently enabled in the harness.

Bitrate and mosaic resolution are screened at a fixed camera count before deciding whether more searches are worthwhile. Input camera resolution is checked separately (720p against 1080p), because it changes per-camera decode/scaling cost. If the screen shows a material capacity or resource-usage change, add a focused search for that factor; otherwise do not multiply every combination into the test plan.

## Stages

Run `python3 tools/capacity.py plan` to print the current plan and estimated workload. The plan begins with a one-camera pilot that validates the deployed-form AIP/parser, output startup, and harness. It then searches no-output baseline capacity for each model, screens output/OSD/bitrate/mosaic settings on one reference model, and searches two representative profiles for each model: recording plus WebRTC with OSD, and recording plus WebRTC plus HLS and event recording with OSD. One additional search validates the screening-based prediction with the largest gap. Finally, two fixed-count probes compare camera resolutions.

The initial estimate is **10 capacity searches and 17 fixed/pilot runs**, approximately **57 trials or 3.8 hours** of exclusive station time. This assumes four trials per search and four minutes per trial; binary search can require more trials depending on the configured maximum, borderline trials are repeated, and actual start/settle time can vary. The completed measurement used 80 capacity/pilot trials (82 attempts), including additional resolution boundaries and rechecks, plus 21 output diagnostics. Treat the estimate as planning guidance, not a duration guarantee.

By comparison, the full listed-factor cross product is 1,296 cells (3 models × 2 OSD states × 2 recording states × 3 delivery choices × 2 event-recording states × 3 bitrates × 3 mosaic resolutions × 2 camera resolutions). Collapsing settings that cannot affect a profile leaves 972 distinct cells before capacity searches.

## Run and results

The standard-library harness is `tools/capacity.py`. It supports `plan`, `search`, `fixed`, `summarize`, and `cleanup`. For example:

```sh
python3 tools/capacity.py plan
python3 tools/capacity.py search \
  --ms3 http://MS3:8000 \
  --model yolo26 \
  --aip AIPV1EMI90XX \
  --rtsp-template 'rtsp://VCAM:PORT/cam{index:02d}' \
  --profile base \
  --out capacity.jsonl
python3 tools/capacity.py fixed \
  --ms3 http://MS3:8000 \
  --signaling http://SIGNALING:3001/signaling \
  --model yolo26 \
  --aip AIPV1EMI90XX \
  --rtsp-template 'rtsp://VCAM:PORT/cam{index:02d}' \
  --profile rec+webrtc-osd \
  --n 4 \
  --out capacity.jsonl
python3 tools/capacity.py summarize capacity.jsonl
python3 tools/capacity.py cleanup --ms3 http://MS3:8000 --signaling http://SIGNALING:3001/signaling \
  --ledger capacity.jsonl.owned.json
```

Replace placeholders with values from the authorized test setup. `--signaling` is required by profiles with continuous recording (`rec`). Use separate camera tags for different source resolutions and keep the JSONL results with a record of the source clips, station hardware/software, and actual pipeline configuration.

The harness checks that the AIP's name or primary model matches `--model`, and rejects a conflicting primary model even if the name matches. `--unverified-model` measures it anyway and marks the results `model_verified: false`. This verifies advertised metadata, not the engine's contents or FP16 precision; independently validate the deployed engine before benchmarking. Every record carries the run ID and the AIP identity (unique ID, version, record key, name, model, and batch size). `summarize` separates camera tags and AIP versions and labels unverified results instead of silently combining them.

Each trial record keeps its raw samples: timestamped per-input processed FPS (`fps`), decoded arrival (`arrival_fps` and `arrival_raw`), link state, and status flags. Sampling durations and the FPS metric name are recorded too. Completed samples survive a later polling failure. The pass threshold is checked before rounding. In a search, a valid trial within 0.5 FPS of the pass line is repeated: it passes only if both attempts pass, reports the lower minimum FPS, and keeps both attempts. `fixed` runs only one trial; independent rechecks must be requested explicitly. Positive camera counts and finite sampling timings are required; the polling interval must not exceed the window.

Every run gets its own ID. It registers its own cameras as `cap-RUN-TAG-NN`, refuses to reuse existing ones, and names its pipelines `cap-RUN-...`. While they exist, it lists them in a ledger next to the results file (`OUT.owned.json`). On successful cleanup the pipeline/media entries and recording files are removed, and the empty ledger disappears; the signaling-service remnants below remain. If a run is interrupted or a deletion fails, `cleanup --ledger` deletes exactly what the ledger lists, together with the recordings of listed recording pipelines. It skips any entry whose current name no longer matches, and it only reports other `cap-` resources rather than deleting them. Recording files are not deleted if stopping recording fails. Cameras are retained while their pipeline's cleanup is incomplete. Clear a remaining ledger before starting another run with that results file; use only one harness process per results file and the same station for recovery.

## Scope and caveats

Use the linked live report for measured profiles, inferred combinations, output failures and test conditions. The target is an Orin NX 16 GB on JetPack 6.0 in MAXN mode, with TCP RTSP inputs. The test AIPs must use deployed-form TensorRT FP16 engines; all flavors must be validated on the station before comparison. The initial runs disable ROI/LCC analytic plugins and use a controlled camera clip, so they do not represent every scene's content density. WebRTC measures the publisher pipeline without any viewer. Recording trials write to the station's continuous-recording storage while they run; the harness attempts to delete the files afterwards, but the signaling service keeps an empty per-pipeline directory and a stopped-state entry for each recording pipeline, which it offers no API to remove. Both are inert once the pipeline is deleted. HLS output folders also require separate, ownership-verified cleanup. Recovery cleanup and the final live resource inventory must be checked separately; a saved trial is not proof that all resources have since been removed. Viewer fan-out, network conditions, audio, alternate output resolutions beyond the screened values, and other hardware/software configurations are outside the initial claim. The station's per-AIP maximum input-stream limit also bounds the reported capacity.

Do not treat idle model inference latency as end-to-end camera capacity. Publish capacity numbers only with their test profile, source resolution/content, model and engine identity, hardware/software conditions, pass criterion, and raw result file.
