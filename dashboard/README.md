# AIPV1EMI0001 UIBuilder dashboard

The dashboard is a no-build Vue 3 single-page application. Vue and hls.js are installed
locally from this directory's `package.json`, so the page works on an
air-gapped EdgeAI Station without CDN access.

The UIBuilder instance URL is `aipv1emi0001_dash`. Its source files are under
`aipv1emi0001_dash/src/`, matching the UIBuilder project layout.

The page consumes the normalized `aipv1emi0001.metrics` envelope documented in
[`../logic/README.md`](../logic/README.md) and displays live video, object
counts, ROI counts, and line crossings.

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

**Latest counts** show the latest received metadata; no frame-level
synchronization with the video is applied. There is no manual snapshot control:
business logic can use the EAIS library's snapshot commands separately.
