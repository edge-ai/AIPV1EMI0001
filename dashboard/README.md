# AIPV1EMI0001 UIBuilder dashboard

The dashboard is a no-build Vue 3 single-page application. Vue and `hls.js` are
installed locally from this directory's `package.json`, so the page works on an
air-gapped EdgeAI Station without CDN access.

The UIBuilder instance URL is `aipv1emi0001_dash`. Its source files are under
`aipv1emi0001_dash/src/`, matching the UIBuilder project layout.

The page consumes the normalized `aipv1emi0001.metrics` envelope documented in
[`../logic/README.md`](../logic/README.md). It focuses on delayed HLS playback,
object counts, ROI counts, and line crossings.

There is no manual snapshot control or capture gallery. A station snapshot
captures a new live frame, not the delayed frame an operator sees in the player.
The EAIS library remains a dependency so business logic can use its snapshot
commands separately; this flow has no snapshot command wiring. No automated
capture rules or browser-frame capture are added.

**Latest counts** show the latest received metadata, independently of delayed
**Pipeline playback**. They are not synchronized, and no inferred offset or fixed
delay is applied. Reliable alignment is deferred until a shared timing contract
exists. See [`../logic/README.md`](../logic/README.md).
