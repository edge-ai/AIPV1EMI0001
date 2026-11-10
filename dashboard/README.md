# AIPV1EMI0001 UIBuilder dashboard

The dashboard is a no-build Vue 3 single-page application. Vue and hls.js are installed
locally from this directory's `package.json`, so the page works on an
air-gapped EdgeAI Station without CDN access.

The UIBuilder instance URL is `aipv1emi0001_dash`. Its source files are under
`aipv1emi0001_dash/src/`, matching the UIBuilder project layout.

The page consumes the normalized `aipv1emi0001.metrics` envelope documented in
[`../logic/README.md`](../logic/README.md) and displays live video, object
counts, ROI counts, and line crossings. Each line shows its running total since
the pipeline started, the same counter drawn on the video, next to its instant
count. The total is `N/A` when the running AIP callback predates 6.1.0.

The page reads `/ms3/pipeline/<id>` using the pipeline ID from the metrics
envelope and selects the player just like the station web UI: active HLS output
takes precedence; otherwise both MTX output and WebRTC must be enabled.
HLS uses `/ms3/pipeline/<id>/hls` with hls.js or native browser support.
WebRTC uses the station's same-origin `/signaling` API.
There is no transport selector or automatic fallback to a different output.
When neither output is enabled, the video area displays a placeholder.

Configuration is refreshed every 10 seconds without restarting unchanged playback.
**Retry video** reloads the configuration immediately and reconnects the selected
player. HLS buffering follows playlist segment durations, not a fixed seconds-based
offset. Browser dependencies are bundled locally; playback needs access to the
authenticated MS3 and signaling routes on the same station origin.

**Video source** controls the pipeline's shared tiler, not a local metrics filter.
Options use the actual `input_sources[].input_id` values from pipeline configuration;
**All sources** maps to tiler index `-1`. Changing it posts to
`/ms3/pipeline/<id>/set_tiler/<index>` and reads back `/ms3/pipeline/<id>/tiler`.
This changes the output for **all viewers**, including the station web UI.
The dashboard also reads the tiler every 2 seconds, so changes made in the station
web UI or another dashboard update its selection and count scope automatically
without sending another change request.

Counts are unavailable until the selection is confirmed, during a change, or when
the selection cannot be read. A selected input without a usable normalized sample
shows unavailable metrics, not all-source totals or zero. Failed reads retry
automatically; failed changes remain visible while the dashboard reconciles with
the backend. A pipeline change cancels pending requests and retargets polling.
Closing the dashboard stops polling and cancels pending requests.

**Latest counts** show the latest received metadata; no frame-level
synchronization with the video is applied. The normalizer detects the callback
delivery automatically. For merged records, counts are the latest window's maximum.
Push envelopes are unwrapped to their last record. In stream mode, each callback
message lists every input, but only inputs whose frame is in that DeepStream batch
carry values; the others are null placeholders. The normalizer therefore keeps each input's latest real
frame, so all-source totals combine the newest frame of each input rather than
one simultaneous frame. On each callback, an input's sample older than 2 seconds
becomes unavailable (`N/A`), never zero. There is no deliberate display delay;
this message-driven expiry does not detect a completely silent callback feed.
Confirming the tiler is not confirmation that a frame from the newly selected
source is already displayed: buffered video,
especially HLS, can still show the previous source. The station web UI itself may
need a refresh to update its own selector after another viewer changes the tiler.
There is no manual snapshot control:
business logic can use the EAIS library's snapshot commands separately.
