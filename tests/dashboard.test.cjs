const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const root = path.resolve(__dirname, "..");
const read = (file) => fs.readFileSync(path.join(root, file), "utf8");
const flows = JSON.parse(read("logic/flows.json"));
const normalizer = new Function("msg", "node", "env", "context",
    flows.find((node) => node.id === "aipv1emi0001normalize").func);

function normalize(payload, extra = {}, context = undefined) {
    const warnings = [];
    const result = normalizer({ payload, eais: { pipelineId: "test-pipeline" }, ...extra },
        { warn: (message) => warnings.push(message) },
        { get: () => undefined }, context);
    return { data: result && result.payload, warnings };
}

function memoryContext() {
    const values = new Map();
    return { get: (key) => values.get(key), set: (key, value) => values.set(key, value) };
}

// Live AIP 6.0.0 stream shape: every source is listed, but only the source whose
// frame is in the DeepStream batch carries values (no "available" field).
function streamFrame(sourceId, frame, counter) {
    return [0, 1].map((id) => id === sourceId
        ? { source_id: id, frame_num: frame, num_obj_meta: 0, ntp_timestamp: 1,
            obj_counter: counter, objInROIcnt: {}, objLCCurrCnt: {} }
        : { source_id: id, frame_num: null, num_obj_meta: null, ntp_timestamp: null,
            obj_counter: [null, null, null], objInROIcnt: null, objLCCurrCnt: null });
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
        setTimeout,
        clearTimeout,
        ...globals,
    });
    const app = { ...definition.data(), ...definition.methods, $options: definition };
    for (const [name, getter] of Object.entries(definition.computed)) {
        Object.defineProperty(app, name, { get: () => getter.call(app) });
    }
    return app;
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
        roi: { yard: 2 }, lineCrossing: { entry: 1 }, lineCrossingCumulative: {},
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

test("stream frames keep each source's latest frame so all-source totals stay available", () => {
    const context = memoryContext();
    let { data } = normalize(streamFrame(0, 10, [2, 1, 0]), {}, context);
    assert.equal(data.delivery, "stream");
    assert.equal(data.aggregation, "Latest frame per source");
    assert.equal(data.sources[1].available, false, "a source never seen is unavailable, not zero");
    assert.equal(data.totals.objects, null);

    ({ data } = normalize(streamFrame(1, 20, [5, 0, 1]), {}, context));
    assert.deepEqual(data.sources.map((source) => [source.id, source.available, source.total]),
        [["0", true, 3], ["1", true, 6]]);
    assert.equal(data.totals.objects, 9);
    assert.deepEqual(data.totals.classes, { car: 7, person: 1, truck: 1 });

    ({ data } = normalize(streamFrame(0, 11, [0, 0, 0]), {}, context));
    assert.equal(data.sources[0].total, 0, "observed zero replaces the previous frame");
    assert.equal(data.sources[1].total, 6);
    assert.equal(data.totals.objects, 6);
});

test("stream sources expire after missing frames and state is scoped per pipeline", (t) => {
    const context = memoryContext();
    let now = 1000;
    t.mock.method(Date, "now", () => now);
    normalize(streamFrame(0, 1, [1, 0, 0]), {}, context);
    let { data } = normalize(streamFrame(1, 1, [2, 0, 0]), {}, context);
    assert.equal(data.totals.objects, 3);

    now += 2500;
    ({ data } = normalize(streamFrame(1, 2, [4, 0, 0]), {}, context));
    assert.equal(data.sources[0].available, false);
    assert.equal(data.sources[0].total, null);
    assert.equal(data.sources[1].total, 4);
    assert.equal(data.totals.objects, null);
    assert.equal(data.totals.classes.car, null);

    ({ data } = normalize(streamFrame(0, 3, [1, 0, 0]), { eais: { pipelineId: "other" } }, context));
    assert.equal(data.sources[1].available, false, "another pipeline does not inherit retained frames");
});

test("6.1.0 stream placeholders marked unavailable are retained like 6.0.0 placeholders", () => {
    const context = memoryContext();
    const frame = (sourceId, counter) => streamFrame(sourceId, 1, counter)
        .map((source) => ({ ...source, available: source.source_id === sourceId }));
    normalize(frame(0, [1, 1, 1]), {}, context);
    const { data } = normalize(frame(1, [2, 0, 0]), {}, context);
    assert.equal(data.totals.objects, 5);
});

test("expired stream samples clear per-source ROI and line counts as well as object counts", (t) => {
    const context = memoryContext();
    let now = 1000;
    t.mock.method(Date, "now", () => now);
    const frame = streamFrame(0, 1, [1, 0, 0]);
    frame[0].objInROIcnt = { yard: 1 };
    frame[0].objLCCurrCnt = { gate: 2 };
    normalize(frame, {}, context);

    now += 2001;
    const { data } = normalize(streamFrame(1, 2, [2, 0, 0]), {}, context);
    assert.equal(data.sources[0].available, false);
    assert.deepEqual(data.sources[0].roi, { yard: null });
    assert.deepEqual(data.sources[0].lineCrossing, { gate: null });
});

test("cumulative line crossings follow DeepStream's running total in stream and batch delivery", (t) => {
    const context = memoryContext();
    let now = 1000;
    t.mock.method(Date, "now", () => now);
    const frame = (sourceId, number, current, cumulative) => {
        const sources = streamFrame(sourceId, number, [1, 0, 0]);
        sources[sourceId].objLCCurrCnt = { gate: current };
        sources[sourceId].objLCCumCnt = { gate: cumulative };
        return sources;
    };
    normalize(frame(0, 1, 1, 40), {}, context);
    let { data } = normalize(frame(1, 1, 0, 2), {}, context);
    assert.deepEqual(data.sources.map((source) => source.lineCrossingCumulative), [{ gate: 40 }, { gate: 2 }]);
    assert.deepEqual(data.totals.lineCrossingCumulative, { gate: 42 });
    assert.deepEqual(data.totals.lineCrossing, { gate: 1 });

    now += 2001;
    ({ data } = normalize(frame(1, 2, 0, 2), {}, context));
    assert.deepEqual(data.sources[0].lineCrossingCumulative, { gate: null }, "expired totals are unknown");
    assert.deepEqual(data.totals.lineCrossingCumulative, { gate: null });

    ({ data } = normalize({ data: { 0: { source_id: 0, frame_number_from: 1, frame_number_to: 30,
        obj_counter: { car: { max_val: 1 } },
        analytics_obj_cnt: { objLCCurrCnt: { gate: { max_val: 1, sum_val: 3 } },
            objLCCumCnt: { gate: { max_val: 45, min_val: 42, last_val: 45 } } } } } }, {}, context));
    assert.deepEqual(data.totals.lineCrossingCumulative, { gate: 45 });

    ({ data } = normalize({ source_id: 0, obj_counter: [1, 0, 0], objLCCurrCnt: { gate: 0 } }));
    assert.deepEqual(data.sources[0].lineCrossingCumulative, {}, "AIP 6.0.0 callbacks omit the running total");
});

test("batch windows are not retained as stream state", () => {
    const context = memoryContext();
    normalize(streamFrame(0, 1, [3, 0, 0]), {}, context);
    const { data } = normalize([{ since: 0, until: 1, data: {
        1: { source_id: 1, frame_number_from: 1, frame_number_to: 30,
            obj_counter: { car: { max_val: 4 }, person: { max_val: 0 }, truck: { max_val: 0 } } },
    } }], {}, context);
    assert.equal(data.delivery, "batch");
    assert.equal(data.aggregation, "Merge-window maximum");
    assert.deepEqual(data.sources.map((source) => source.id), ["1"]);
    assert.equal(data.totals.objects, 4);
    assert.equal(context.get("aipv1emi0001.streamSources"), undefined);
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

test("callbacks update immediately without replacing a missing selected source with totals", () => {
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
    assert.equal(app.selectedSource, "missing");
    assert.equal(app.scopedTotal, null);
    assert.equal(app.scopedClasses, undefined);
    assert.equal(app.scopedRoi, undefined);
    assert.equal(app.scopedLines, undefined);
    assert.equal(app.messages.length, 100);
    assert.equal(app.lastUpdate.toISOString(), "2026-10-27T00:00:00.000Z");
    app.applyMessage({ payload: { type: "aipv1emi0001.snapshot", ok: true } });
    app.applyMessage({ payload: { type: "aipv1emi0001.metrics" } });
    assert.equal(app.totals.objects, 100);
    assert.equal(app.windows, undefined);
    assert.equal(app.syncTimer, undefined);
});

function selectionDashboard(globals = {}) {
    const app = dashboard(globals);
    app.pipeline = { id: "demo" };
    app.videoSources = [{ id: "2", name: "Gate" }, { id: "7", name: "Yard" }];
    app.sources = [
        { id: "2", total: 0, counts: { car: 0 }, roi: { gate: 0 }, lineCrossing: { entry: 0 } },
        { id: "7", total: 4, counts: { car: 4 }, roi: { gate: 3 }, lineCrossing: { entry: 1 } },
    ];
    app.totals = { objects: 4, classes: { car: 4 }, roi: { gate: 3 }, lineCrossing: { entry: 1 } };
    return app;
}

const selectionResponse = (index) => ({ ok: true, json: async () => ({ status: 200, output: index }) });

test("line rows pair each running total with the instant count and never invent a total", () => {
    const app = selectionDashboard();
    const rows = () => JSON.parse(JSON.stringify(app.lineRows));
    app.sources[1].lineCrossingCumulative = { entry: 12 };
    app.selectedSource = "7";
    assert.deepEqual(rows(), [{ label: "entry", value: 12, instant: 1 }]);
    app.selectedSource = "2";
    assert.deepEqual(rows(), [{ label: "entry", value: "N/A", instant: 0 }]);
    app.selectedSource = "all";
    app.totals.lineCrossingCumulative = { entry: 12, exit: 30 };
    assert.deepEqual(rows(), [
        { label: "exit", value: 30, instant: "N/A" },
        { label: "entry", value: 12, instant: 1 },
    ]);
});

test("initial and externally changed tiler selections scope counts without writing", async () => {
    const requests = [];
    let index = 7;
    const app = selectionDashboard({
        fetch: async (url, options) => {
            requests.push({ url, options });
            return selectionResponse(index);
        },
    });
    assert.equal(app.selectedSource, "");
    assert.equal(app.scopedTotal, null);
    await app.syncVideoSelection();
    assert.equal(app.selectedSource, "7");
    assert.equal(app.selectedSourceName, "Yard");
    assert.equal(app.scopedTotal, 4);
    assert.equal(app.scopedRoi.gate, 3);
    index = 2;
    await app.syncVideoSelection();
    assert.equal(app.selectedSource, "2");
    assert.equal(app.scopedTotal, 0);
    assert.equal(app.scopedClasses.car, 0);
    assert.equal(app.scopedLines.entry, 0);
    index = -1;
    await app.syncVideoSelection();
    assert.equal(app.selectedSource, "all");
    assert.equal(app.scopedTotal, 4);
    assert.equal(app.scopedClasses.car, 4);
    assert.equal(requests.length, 3);
    for (const { url, options } of requests) {
        assert.equal(url, "/ms3/pipeline/demo/tiler");
        assert.equal(options.method, undefined);
        assert.equal(options.cache, "no-store");
        assert.equal(options.credentials, "same-origin");
    }
});

test("pipeline configuration supplies and refreshes source options without guessing array indexes", async () => {
    let inputs = [{ input_id: 2, media: { name: "Gate" } }, { input_id: 7 }];
    const app = dashboard({
        fetch: async () => ({ ok: true, json: async () => ({
            input_sources: inputs, hls_output_sink_settings: { active: false },
        }) }),
    });
    app.pipeline.id = "demo";
    app.$refs = {};
    await app.connectVideo();
    assert.deepEqual(Array.from(app.videoSources, (source) => source.id), ["2", "7"]);
    assert.deepEqual(Array.from(app.videoSources, (source) => source.name), ["Gate", "Source 7"]);
    inputs = [{ input_id: 9, media: { name: "Renamed input" } }];
    await app.connectVideo();
    assert.equal(app.videoSources[0].id, "9");
    assert.equal(app.videoSources[0].name, "Renamed input");
    inputs = [{ input_id: "1" }];
    await app.connectVideo();
    assert.equal(app.videoSources.length, 0);
    assert.equal(app.videoState, "disabled");
});

test("source changes use real non-contiguous input IDs and read back authoritative state", async () => {
    const requests = [];
    let index = 2;
    const app = selectionDashboard({
        fetch: async (url, options) => {
            requests.push({ url, options });
            if (options.method === "POST") {
                assert.equal(app.selectedSource, "");
                assert.equal(app.scopedTotal, null);
                index = Number(url.split("/").pop());
            }
            return selectionResponse(index);
        },
    });
    app.selectedSource = "2";
    await app.syncVideoSelection("7");
    assert.equal(app.selectedSource, "7");
    assert.equal(app.pendingSource, null);
    assert.equal(app.scopedTotal, 4);
    await app.syncVideoSelection("all");
    assert.equal(app.selectedSource, "all");
    assert.deepEqual(requests.map(({ url }) => url), [
        "/ms3/pipeline/demo/set_tiler/7", "/ms3/pipeline/demo/tiler",
        "/ms3/pipeline/demo/set_tiler/-1", "/ms3/pipeline/demo/tiler",
    ]);
    assert.equal(requests[0].options.credentials, "same-origin");
    await app.syncVideoSelection("1");
    assert.equal(requests.length, 4);
    assert.match(app.selectionChangeError, /not in this pipeline/);
});

test("readback wins if another viewer changes selection during a write", async () => {
    const app = selectionDashboard({ fetch: async () => selectionResponse(2) });
    await app.syncVideoSelection("7");
    assert.equal(app.selectedSource, "2");
    assert.equal(app.scopedTotal, 0);
});

test("shared source changes preserve both HLS and WebRTC playback", async () => {
    for (const mode of ["hls", "webrtc"]) {
        const app = selectionDashboard({ fetch: async () => selectionResponse(7) });
        const player = {};
        app.videoMode = mode;
        app.videoState = "playing";
        if (mode === "hls") app.hls = player;
        else app.peerConnection = player;
        await app.syncVideoSelection("7");
        assert.equal(app.selectedSource, "7");
        assert.equal(app.videoMode, mode);
        assert.equal(app.videoState, "playing");
        assert.equal(mode === "hls" ? app.hls : app.peerConnection, player);
    }
});

test("failed, stopped and invalid tiler reads hide counts instead of assuming All", async () => {
    for (const response of [
        { ok: false, text: async () => '{"detail":"Pipeline is not running"}' },
        { ok: false, text: async () => '{"detail":"Access denied"}' },
        selectionResponse(-999),
        selectionResponse("7"),
        selectionResponse(3),
        { ok: true, json: async () => null },
        { ok: true, json: async () => ({ status: 500, output: 7 }) },
    ]) {
        const app = selectionDashboard({ fetch: async () => response });
        app.selectedSource = "all";
        await app.syncVideoSelection();
        assert.equal(app.selectedSource, "");
        assert.equal(app.scopedTotal, null);
        assert.match(app.selectionError, /Video selection unavailable.*Retrying automatically/);
        assert.equal(app.selectionRequest, null);
        assert.equal(app.pendingSource, null);
    }
});

test("failed changes remain visible while polling recovers the actual source", async () => {
    for (const failure of [
        { ok: false, text: async () => '{"detail":"Access denied"}' },
        { ok: true, json: async () => ({ status: 500, output: "Change rejected" }) },
    ]) {
        const app = selectionDashboard({
            fetch: async (_url, options) => options.method === "POST" ? failure : selectionResponse(2),
        });
        app.selectedSource = "7";
        await app.syncVideoSelection("all");
        assert.equal(app.selectedSource, "");
        assert.equal(app.scopedTotal, null);
        assert.match(app.selectionChangeError, /Could not confirm.*Access denied|Change rejected/);
        await app.syncVideoSelection();
        assert.equal(app.selectedSource, "2");
        assert.equal(app.selectionError, "");
        assert.ok(app.selectionChangeError);
    }
});

test("pending changes block duplicate writes and polls, and supersede older reads", async () => {
    const requests = [];
    let resolveRead;
    let resolveWrite;
    let reads = 0;
    const app = selectionDashboard({
        fetch: (url, options) => {
            requests.push({ url, options });
            if (options.method === "POST") return new Promise((resolve) => { resolveWrite = resolve; });
            if (reads++ === 0) return new Promise((resolve) => { resolveRead = resolve; });
            return Promise.resolve(selectionResponse(7));
        },
    });
    const old = app.syncVideoSelection();
    const change = app.syncVideoSelection("7");
    assert.equal(requests[0].options.signal.aborted, true);
    assert.equal(app.pendingSource, "7");
    await app.syncVideoSelection("all");
    await app.syncVideoSelection();
    assert.equal(requests.length, 2);
    resolveWrite(selectionResponse(7));
    await change;
    assert.equal(app.selectedSource, "7");
    resolveRead(selectionResponse(-1));
    await old;
    assert.equal(app.selectedSource, "7");
});

test("pipeline changes abort selection requests and reject late old-pipeline results", async () => {
    let resolveOld;
    let oldSignal;
    const app = selectionDashboard({
        fetch: (url, options) => {
            if (url.includes("/demo/")) {
                oldSignal = options.signal;
                return new Promise((resolve) => { resolveOld = resolve; });
            }
            return Promise.resolve(selectionResponse(-1));
        },
    });
    app.connectVideo = () => {};
    const old = app.syncVideoSelection();
    app.pipeline.id = "new";
    app.$options.watch["pipeline.id"].call(app);
    assert.equal(oldSignal.aborted, true);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(app.selectedSource, "all");
    resolveOld(selectionResponse(7));
    await old;
    assert.equal(app.selectedSource, "all");
    assert.equal(app.videoSources.length, 0);
});

test("selection requests time out and unmount aborts them and clears the deadline", async () => {
    let expire;
    let timeoutMs;
    const cleared = [];
    const app = selectionDashboard({
        setTimeout: (callback, ms) => { expire = callback; timeoutMs = ms; return 123; },
        clearTimeout: (timer) => cleared.push(timer),
        clearInterval() {},
        fetch: (_url, { signal }) => new Promise((_resolve, reject) =>
            signal.addEventListener("abort", () => reject(new Error("aborted")))),
    });
    const request = app.syncVideoSelection();
    assert.equal(timeoutMs, 10000);
    expire();
    await request;
    assert.match(app.selectionError, /timed out/);
    const second = app.syncVideoSelection();
    const controller = app.selectionRequest;
    app.$refs = {};
    app.$options.beforeUnmount.call(app);
    await second;
    assert.equal(controller.signal.aborted, true);
    assert.ok(cleared.includes(123));
    assert.equal(app.selectionTimeout, null);
    assert.equal(app.selectionError, "");
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

test("configuration and shared selection polling start and stop with the dashboard", () => {
    const refreshes = [];
    const cleared = [];
    const app = dashboard({
        uibuilder: { ioConnected: false, onChange() {} },
        setInterval: (callback, interval) => {
            refreshes.push({ callback, interval });
            return refreshes.length;
        },
        clearInterval: (timer) => { cleared.push(timer); },
    });
    app.$refs = {};
    let refreshCount = 0;
    let selectionCount = 0;
    app.connectVideo = () => { refreshCount++; };
    app.syncVideoSelection = () => { selectionCount++; };
    app.$options.mounted.call(app);
    assert.deepEqual(refreshes.map(({ interval }) => interval), [10000, 2000]);
    refreshes.forEach(({ callback }) => callback());
    assert.equal(refreshCount, 1);
    assert.equal(selectionCount, 1);
    app.$options.beforeUnmount.call(app);
    assert.deepEqual(cleared, [1, 2]);
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
        "dashboard/aipv1emi0001_dash/src/fonts/InterVariable.woff2",
        "dashboard/aipv1emi0001_dash/src/fonts/OFL.txt",
        ...["html", "js", "css"].map((ext) => `dashboard/aipv1emi0001_dash/src/index.${ext}`)]) {
        assert.ok(bundled(file), `${file} not packaged`);
        assert.ok(fs.statSync(path.join(root, file)).size > 0);
    }
    assert.equal(bundled("logic/flows_cred.json"), false);
    const html = read("dashboard/aipv1emi0001_dash/src/index.html");
    assert.match(html, /Latest counts/);
    assert.match(html, /<label for="video-source">Video source<\/label>/);
    assert.doesNotMatch(html, /v-model="selectedSource"/);
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

test("dashboard theme and fonts load locally without external asset imports", () => {
    const src = "dashboard/aipv1emi0001_dash/src";
    const html = read(`${src}/index.html`);
    const css = read(`${src}/index.css`);
    for (const [, asset] of html.matchAll(/(?:src|href)="([^"]+)"/g)) {
        assert.ok(asset.startsWith("./"), `non-local asset: ${asset}`);
        assert.ok(fs.existsSync(path.join(root, src, asset)), `missing asset: ${asset}`);
    }
    assert.doesNotMatch(css, /@import\b/);
    for (const [, asset] of css.matchAll(/url\(["']?([^"')]+)["']?\)/g)) {
        assert.ok(asset.startsWith("./"), `non-local CSS asset: ${asset}`);
        assert.ok(fs.existsSync(path.join(root, src, asset)), `missing CSS asset: ${asset}`);
    }
    for (const file of ["index.js", "bootstrap.js"]) {
        assert.doesNotMatch(read(`${src}/${file}`), /["'`]\s*(?:https?:|\/\/)/i);
    }
    assert.match(css, /@font-face\s*\{[^}]*font-family:\s*"Inter"/);
    assert.match(css, /font-display:\s*swap/);
    const font = fs.readFileSync(path.join(root, src, "fonts/InterVariable.woff2"));
    assert.equal(font.subarray(0, 4).toString(), "wOF2");
    assert.equal(font.readUInt32BE(8), font.length, "complete WOFF2 file");
    assert.match(read(`${src}/fonts/OFL.txt`), /SIL OPEN FONT LICENSE Version 1\.1/);
});
