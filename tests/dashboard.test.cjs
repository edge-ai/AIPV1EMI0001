const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const root = path.resolve(__dirname, "..");
const read = (file) => fs.readFileSync(path.join(root, file), "utf8");
const flows = JSON.parse(read("logic/flows.json"));
const normalizer = new Function("msg", "node", "env",
    flows.find((node) => node.id === "aipv1emi0001normalize").func);

function normalize(payload, extra = {}) {
    const warnings = [];
    const result = normalizer({ payload, eais: { pipelineId: "test-pipeline" }, ...extra },
        { warn: (message) => warnings.push(message) },
        { get: () => undefined });
    return { data: result && result.payload, warnings };
}

function dashboard(globals = {}) {
    let definition;
    vm.runInNewContext(read("dashboard/aipv1emi0001_dash/src/index.js")
        .replace(/^import .*;\n/gm, ""), {
        createApp(value) {
            definition = value;
            return { mount() {} };
        },
        nextTick: (callback) => callback(),
        markRaw: (value) => value,
        AbortController,
        ...globals,
    });
    return { ...definition.data(), ...definition.methods, $options: definition };
}

function bootstrap(loadDashboard) {
    const startup = {
        removed: false,
        role: "status",
        remove() { this.removed = true; },
        setAttribute(name, value) { this[name] = value; },
    };
    const message = { textContent: "" };
    const errors = [];
    vm.runInNewContext(read("dashboard/aipv1emi0001_dash/src/bootstrap.js")
        .replace('import("./index.js")', "loadDashboard()"), {
        document: {
            getElementById(id) {
                return id === "dashboard-startup" ? startup : message;
            },
        },
        loadDashboard,
        console: { error: (...args) => errors.push(args) },
    });
    return { startup, message, errors };
}

function videoElement(nativeHls = false) {
    return {
        srcObject: null,
        canPlayType: () => nativeHls ? "maybe" : "",
        removeAttribute(name) { delete this[name]; },
        load() {},
        pause() {},
        play: async () => {},
    };
}

class MockHls {
    static isSupported() { return true; }
    static Events = { MANIFEST_PARSED: "manifest", ERROR: "error" };
    constructor(options) { this.options = options; this.handlers = {}; }
    on(event, handler) { this.handlers[event] = handler; }
    loadSource(url) { this.url = url; }
    attachMedia(video) { this.video = video; }
    destroy() { this.destroyed = true; }
}

test("startup placeholder is removed after mount or reports a failed module load", async () => {
    const success = bootstrap(() => Promise.resolve());
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(success.startup.removed, true);
    assert.equal(success.errors.length, 0);

    const failure = new Error("Vue module unavailable");
    const failed = bootstrap(() => Promise.reject(failure));
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(failed.startup.removed, false);
    assert.equal(failed.startup.role, "alert");
    assert.match(failed.message.textContent, /check.*browser libraries.*reload/i);
    assert.equal(failed.errors.length, 1);
    assert.equal(failed.errors[0][1], failure);
});

test("passthrough class arrays, ROI and line counters normalize without timing", () => {
    const { data, warnings } = normalize({
        source_id: 0, obj_counter: [2, 1, 0],
        objInROIcnt: { yard: 2 }, objLCCurrCnt: { entry: 1 },
    });
    assert.deepEqual(warnings, []);
    assert.deepEqual(data.sources[0].counts, { car: 2, person: 1, truck: 0 });
    assert.deepEqual(data.totals, {
        objects: 3, classes: { car: 2, person: 1, truck: 0 },
        roi: { yard: 2 }, lineCrossing: { entry: 1 },
    });
    assert.equal(data.sources[0].frameFrom, undefined);
    assert.equal(data.pipeline.id, "test-pipeline");
    assert.equal(data.hlsUrl, undefined);
});

test("merged sources use maxima and enriched pipeline/source names", () => {
    const { data } = normalize(JSON.stringify({ data: {
        0: { source_id: 0, obj_counter: { car: { max_val: 4, mean_val: 2 } },
            analytics_obj_cnt: { objInROIcnt: { yard: { max_val: 3 } } } },
        1: { source_id: 1, obj_counter: { car: { max_val: 1 } } },
    } }), {
        metadata: { pipeline: { uuid: "named-pipeline", name: "Demo" },
            mediaSources: { 0: { name: "Gate" } } },
    });
    assert.equal(data.totals.objects, 5);
    assert.equal(data.totals.roi.yard, 3);
    assert.equal(data.sources[0].name, "Gate");
    assert.equal(data.pipeline.name, "Demo");
    assert.equal(data.pipeline.id, "named-pipeline");
});

test("batched callbacks display only the last window, including after frame reset", () => {
    const record = (frame, count) => ({ data: {
        0: { source_id: 0, frame_number_from: frame, frame_number_to: frame + 29,
            obj_counter: { car: { max_val: count } } },
    } });
    const { data } = normalize([record(300, 9), record(0, 2)]);
    assert.equal(data.sources[0].frameFrom, 0);
    assert.equal(data.totals.objects, 2);
    assert.equal(data.windows, undefined);
    assert.equal(data.fps, undefined);
});

test("malformed or empty callbacks warn rather than fabricate counts", () => {
    for (const payload of ["not json", {}, [], null]) {
        const { data, warnings } = normalize(payload);
        assert.equal(data, null);
        assert.ok(warnings.length > 0);
    }
});

test("a missing pipeline ID is reported without inventing one", () => {
    const { data, warnings } = normalize({ source_id: 0, obj_counter: [1, 0, 0] },
        { eais: undefined });
    assert.equal(data.pipeline.id, "");
    assert.match(warnings.join(" "), /pipeline ID/i);
});

test("dashboard updates immediately without video, resets removed scope, and bounds history", () => {
    const app = dashboard();
    app.selectedSource = "missing";
    for (let count = 0; count <= 100; count++) {
        app.applyMessage({ payload: {
            type: "aipv1emi0001.metrics", timestamp: "2026-10-27T00:00:00Z",
            sources: [{ id: "0", counts: { car: count } }],
            totals: { objects: count }, pipeline: { id: "demo" },
        } });
    }
    assert.equal(app.totals.objects, 100);
    assert.equal(app.selectedSource, "all");
    assert.equal(app.messages.length, 100);
    assert.equal(app.lastUpdate.toISOString(), "2026-10-27T00:00:00.000Z");
    app.applyMessage({ payload: { type: "aipv1emi0001.snapshot", ok: true } });
    app.applyMessage({ payload: { type: "aipv1emi0001.metrics" } });
    assert.equal(app.totals.objects, 100);
    assert.equal(app.windows, undefined);
    assert.equal(app.syncTimer, undefined);
});

test("dashboard negotiates WebRTC through the station signaling API", async () => {
    const requests = [];
    class PeerConnection {
        constructor(configuration) {
            this.configuration = configuration;
            this.iceGatheringState = "complete";
            this.connectionState = "new";
        }
        addTransceiver() {}
        async createOffer() { return { type: "offer", sdp: "offer-sdp" }; }
        async setLocalDescription(offer) { this.localDescription = offer; }
        async setRemoteDescription(answer) { this.remoteDescription = answer; }
        close() { this.closed = true; }
    }
    const video = videoElement();
    const app = dashboard({
        RTCPeerConnection: PeerConnection,
        fetch: async (url, options = {}) => {
            requests.push({ url, options });
            if (url === "/ms3/pipeline/demo-pipeline") {
                return { ok: true, json: async () => ({
                    hls_output_sink_settings: { active: false },
                    mtx_output_sink_settings: { active: true, webrtc_enabled: true },
                }) };
            }
            assert.equal(options.method, "POST", `unexpected request to ${url}`);
            return { ok: true, json: async () => ({ answer_sdp: "answer-sdp" }) };
        },
    });
    app.pipeline = { id: "demo-pipeline" };
    app.$refs = { video };

    await app.connectVideo();

    assert.deepEqual(requests.map(({ url }) => url), [
        "/ms3/pipeline/demo-pipeline",
        "/signaling/rtsp-sink-and-whep",
    ]);
    assert.equal(app.videoMode, "webrtc");
    assert.equal(requests[0].options.credentials, "same-origin");
    assert.equal(requests[0].options.cache, "no-store");
    assert.deepEqual(JSON.parse(requests[1].options.body), {
        pipeline_id: "demo-pipeline",
        offer_sdp: "offer-sdp",
    });
    assert.equal(app.peerConnection.configuration, undefined);
    assert.equal(JSON.stringify(app.peerConnection.remoteDescription), JSON.stringify({
        type: "answer",
        sdp: "answer-sdp",
    }));
    const peer = app.peerConnection;
    peer.ontrack({ streams: [{}] });
    peer.connectionState = "connected";
    peer.onconnectionstatechange();
    assert.equal(app.videoState, "loading");
    video.onplaying();
    assert.equal(app.videoState, "playing");
    app.destroyVideo();
    assert.equal(peer.closed, true);
});

test("player selection matches the web UI, including HLS precedence and disabled outputs", async () => {
    for (const [hlsActive, mtxActive, webrtcEnabled, expected] of [
        [true, true, true, "hls"],
        [true, false, false, "hls"],
        [false, true, true, "webrtc"],
        [false, false, true, null],
        [false, true, false, null],
        [false, false, false, null],
    ]) {
        const requests = [];
        const app = dashboard({
            Hls: MockHls,
            fetch: async (url) => {
                requests.push(url);
                return { ok: true, json: async () => ({
                    hls_output_sink_settings: { active: hlsActive },
                    mtx_output_sink_settings: { active: mtxActive, webrtc_enabled: webrtcEnabled },
                }) };
            },
        });
        app.pipeline.id = "demo";
        app.$refs = { video: videoElement() };
        app.attachWebRtc = () => { app.webRtcStarted = true; };
        await app.connectVideo();
        assert.equal(app.videoMode, expected);
        assert.deepEqual(requests, ["/ms3/pipeline/demo"]);
        assert.equal(Boolean(app.hls), expected === "hls");
        assert.equal(Boolean(app.webRtcStarted), expected === "webrtc");
        if (!expected) {
            assert.equal(app.videoState, "disabled");
            assert.match(app.videoError, /No video output/);
        }
    }
});

test("HLS uses the pipeline playlist and segment-based settings, without restarting on each refresh", async () => {
    const app = dashboard({
        Hls: MockHls,
        fetch: async () => ({ ok: true, json: async () => ({
            hls_output_sink_settings: { active: true, target_duration: 7 },
        }) }),
    });
    app.pipeline.id = "pipeline / id";
    const video = videoElement();
    app.$refs = { video };
    await app.connectVideo();
    const hls = app.hls;
    assert.equal(hls.url, "/ms3/pipeline/pipeline%20%2F%20id/hls");
    assert.equal(hls.options.liveSyncDuration, undefined);
    assert.equal(hls.options.liveMaxLatencyDuration, undefined);
    assert.equal(hls.options.liveSyncDurationCount, 3);
    hls.handlers.manifest();
    video.onplaying();
    assert.equal(app.videoState, "playing");
    await app.connectVideo();
    assert.equal(app.hls, hls);
    assert.equal(hls.destroyed, undefined);
    await app.connectVideo(true);
    assert.equal(hls.destroyed, true);
    assert.notEqual(app.hls, hls);
    const replacement = app.hls;
    hls.handlers.error(null, { fatal: true, details: "stale error" });
    assert.equal(app.hls, replacement);
});

test("configuration changes for the same pipeline switch and clean up the existing transport", async () => {
    let config = { hls_output_sink_settings: { active: true } };
    const app = dashboard({
        Hls: MockHls,
        fetch: async () => ({ ok: true, json: async () => config }),
    });
    app.pipeline.id = "demo";
    app.$refs = { video: videoElement() };
    app.totals.objects = 12;
    let peer;
    app.attachWebRtc = () => {
        peer = { close() { this.closed = true; } };
        app.peerConnection = peer;
    };
    await app.connectVideo();
    const hls = app.hls;
    config = {
        hls_output_sink_settings: { active: false },
        mtx_output_sink_settings: { active: true, webrtc_enabled: true },
    };
    await app.connectVideo();
    assert.equal(hls.destroyed, true);
    assert.equal(app.hls, null);
    assert.equal(app.videoMode, "webrtc");
    config = { hls_output_sink_settings: { active: true } };
    await app.connectVideo();
    assert.equal(peer.closed, true);
    assert.equal(app.peerConnection, null);
    assert.ok(app.hls);
    config = { hls_output_sink_settings: { active: false } };
    await app.connectVideo();
    assert.equal(app.hls, null);
    assert.equal(app.videoState, "disabled");
    assert.equal(app.totals.objects, 12);
});

test("native HLS is supported, and autoplay rejection exposes native controls", async () => {
    const app = dashboard({ Hls: { isSupported: () => false } });
    const video = videoElement(true);
    video.play = async () => { throw new Error("autoplay blocked"); };
    app.pipeline.id = "demo";
    app.$refs = { video };
    app.attachHls();
    assert.equal(video.src, "/ms3/pipeline/demo/hls");
    video.onloadedmetadata();
    await Promise.resolve();
    assert.equal(app.videoState, "ready");
    assert.match(app.videoError, /Press play/);
    video.onplaying();
    assert.equal(app.videoState, "playing");
    assert.equal(app.videoError, "");
    video.onerror();
    assert.equal(app.videoState, "error");
    assert.equal(video.src, undefined);
    assert.equal(video.onloadedmetadata, null);
});

test("HLS errors and unsupported browsers do not silently switch to WebRTC", () => {
    for (const supported of [true, false]) {
        const app = dashboard({ Hls: supported ? MockHls : { isSupported: () => false } });
        app.pipeline.id = "demo";
        app.$refs = { video: videoElement() };
        app.attachHls();
        if (supported) {
            const hls = app.hls;
            hls.handlers.error(null, { fatal: false, details: "transient" });
            assert.equal(app.videoState, "loading");
            hls.handlers.error(null, { fatal: true, details: "manifestLoadError" });
            assert.equal(hls.destroyed, true);
        }
        assert.equal(app.videoState, "error");
        assert.ok(app.videoError);
        assert.equal(app.peerConnection, null);
    }
});

test("pipeline configuration errors are shown without guessing a transport", async () => {
    for (const response of [
        { ok: false, text: async () => '{"detail":"Access denied"}' },
        { ok: true, json: async () => ({}) },
    ]) {
        const app = dashboard({ fetch: async () => response });
        app.pipeline.id = "demo";
        app.$refs = { video: videoElement() };
        await app.connectVideo();
        assert.equal(app.videoState, "error");
        assert.match(app.videoError, /Access denied|invalid video configuration/);
        assert.equal(app.videoMode, null);
        assert.equal(app.hls, null);
        assert.equal(app.peerConnection, null);
    }
});

test("superseded configuration responses cannot replace the current pipeline player", async () => {
    let resolveOld;
    let oldSignal;
    const app = dashboard({
        Hls: MockHls,
        fetch: (url, options) => {
            if (url.endsWith("/old")) {
                oldSignal = options.signal;
                return new Promise((resolve) => { resolveOld = resolve; });
            }
            return Promise.resolve({ ok: true, json: async () => ({
                hls_output_sink_settings: { active: false },
            }) });
        },
    });
    app.pipeline.id = "old";
    app.$refs = { video: videoElement() };
    const pending = app.connectVideo();
    app.pipeline.id = "new";
    await app.connectVideo(true);
    assert.equal(oldSignal.aborted, true);
    resolveOld({ ok: true, json: async () => ({ hls_output_sink_settings: { active: true } }) });
    await pending;
    assert.equal(app.videoState, "disabled");
    assert.equal(app.videoMode, null);
    assert.equal(app.hls, null);
});

test("periodic configuration refresh is started and stopped with the dashboard", () => {
    let refresh;
    let cleared;
    const app = dashboard({
        uibuilder: { ioConnected: false, onChange() {} },
        setInterval: (callback, interval) => { refresh = { callback, interval }; return 123; },
        clearInterval: (timer) => { cleared = timer; },
    });
    app.$refs = {};
    let refreshCount = 0;
    app.connectVideo = () => { refreshCount++; };
    app.$options.mounted.call(app);
    assert.equal(refresh.interval, 10000);
    refresh.callback();
    assert.equal(refreshCount, 1);
    app.$options.beforeUnmount.call(app);
    assert.equal(cleared, 123);
});

test("flow uses a selectable callback pipeline, has valid references, and no browser command wiring", () => {
    const ids = new Set(flows.map((node) => node.id));
    assert.equal(ids.size, flows.length);
    for (const node of flows) {
        for (const id of [...(node.wires || []).flat(), ...(node.links || []), ...(node.scope || [])]) {
            assert.ok(ids.has(id), `${node.id} references missing ${id}`);
        }
        assert.doesNotMatch(node.id, /snap/i);
    }
    const callback = flows.find((node) => node.type === "callback");
    assert.equal(callback.pipelineId, "");
    assert.equal(flows.find((node) => node.id === callback.server).type, "eais-server");
    assert.deepEqual(flows.find((node) => node.type === "uibuilder").wires, [[], []]);
});

test("project pins the public EAIS node and excludes the legacy variant", () => {
    const manifest = JSON.parse(read("logic/package.json"));
    const lock = JSON.parse(read("logic/package-lock.json"));
    const name = "@edgematrix/node-red-contrib-eais";
    const legacy = "@edge-ai/node-red-contrib-eais";
    assert.equal(manifest.dependencies[name], "0.7.10");
    assert.equal(manifest.dependencies[legacy], undefined);
    assert.equal(lock.packages[""].dependencies[name], "0.7.10");
    assert.equal(lock.packages[`node_modules/${legacy}`], undefined);
    assert.equal(lock.packages[`node_modules/${name}`].version, "0.7.10");
    assert.equal(lock.packages[`node_modules/${name}`].resolved,
        "https://registry.npmjs.org/@edgematrix/node-red-contrib-eais/-/node-red-contrib-eais-0.7.10.tgz");
    assert.equal(lock.packages[`node_modules/${name}`].peerDependenciesMeta["node-red"].optional, true);
    assert.equal(manifest.dependencies["node-red"], undefined);
    assert.equal(manifest.devDependencies?.["node-red"], undefined);
    assert.equal(lock.packages["node_modules/node-red"], undefined);
});

test("the Node-RED runtime is pinned separately for integration tests", () => {
    const manifest = JSON.parse(read("tests/node-red-runtime/package.json"));
    const lock = JSON.parse(read("tests/node-red-runtime/package-lock.json"));
    assert.equal(manifest.devDependencies["node-red"], "4.1.13");
    assert.equal(lock.packages[""].devDependencies["node-red"], "4.1.13");
    assert.equal(lock.packages["node_modules/node-red"].version, "4.1.13");
});

test("integration package includes runtime assets, excludes credentials, and describes latest counts", () => {
    const includes = read(".include").split(/\r?\n/).filter((line) => line && !line.startsWith("#"));
    for (const entry of includes) assert.ok(fs.existsSync(path.join(root, entry)), entry);
    const bundled = (file) => includes.some((entry) => entry === file
        || entry.endsWith("/") && file.startsWith(entry));
    for (const file of ["logic/flows.json", "logic/package.json", "logic/package-lock.json",
        "dashboard/package.json", "dashboard/package-lock.json",
        "dashboard/aipv1emi0001_dash/src/bootstrap.js",
        ...["html", "js", "css"].map((ext) => `dashboard/aipv1emi0001_dash/src/index.${ext}`)]) {
        assert.ok(bundled(file), `${file} not packaged`);
        assert.ok(read(file).length > 0);
    }
    assert.equal(bundled("logic/flows_cred.json"), false);
    const html = read("dashboard/aipv1emi0001_dash/src/index.html");
    assert.match(html, /Latest counts/);
    assert.match(html, /v-cloak/);
    assert.match(html, /src="\.\/bootstrap\.js"/);
    assert.match(html, /id="dashboard-startup-message"/);
    assert.doesNotMatch(html, /HLS|fallback/i);
    assert.ok(JSON.parse(read("dashboard/package.json")).dependencies["hls.js"]);
    assert.match(html, /Retry video/);
    for (const ext of ["html", "js", "css"]) {
        assert.doesNotMatch(read(`dashboard/aipv1emi0001_dash/src/index.${ext}`),
            /snapshot|syncOffset|syncEnabled|playingFrame|bufferWindows|uibuilder\.send/);
    }
});
