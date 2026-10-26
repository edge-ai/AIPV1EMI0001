# AIPV1EMI0001 UIBuilder dashboard

The dashboard is a no-build Vue 3 single-page application. Vue and `hls.js` are
installed locally from this directory's `package.json`, so the page works on an
air-gapped EdgeAI Station without CDN access.

The UIBuilder instance URL is `aipv1emi0001_dash`. Its source files are under
`aipv1emi0001_dash/src/`, matching the UIBuilder project layout.

The page accepts only the normalized `aipv1emi0001.metrics` envelope documented
in [`../logic/README.md`](../logic/README.md). Browser input is not forwarded to
Node-RED and no snapshot command is exposed. Snapshot integration should use
the final API from edge-ai/aibox-webserver-backend-ms3#705 when available.
