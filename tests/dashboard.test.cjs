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
    const result = normalizer({ payload, ...extra },
        { warn: (message) => warnings.push(message) },
        { get: (name) => name === "AIP_PIPELINE_ID" ? "test-pipeline" : undefined });
    return { data: result && result.payload, warnings };
}

function dashboard() {
    let definition;
    vm.runInNewContext(read("dashboard/aipv1emi0001_dash/src/index.js")
        .replace(/^import .*;\n/gm, ""), {
        createApp(value) {
            definition = value;
            return { mount() {} };
        },
    });
    return { ...definition.data(), ...definition.methods };
}

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
    assert.equal(data.hlsUrl, "/ms3/pipeline/test-pipeline/hls");
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
    assert.equal(data.hlsUrl, "/ms3/pipeline/named-pipeline/hls");
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

test("flow is portable, all references resolve, and browser command wiring is absent", () => {
    const ids = new Set(flows.map((node) => node.id));
    assert.equal(ids.size, flows.length);
    for (const node of flows) {
        for (const id of [...(node.wires || []).flat(), ...(node.links || []), ...(node.scope || [])]) {
            assert.ok(ids.has(id), `${node.id} references missing ${id}`);
        }
        assert.doesNotMatch(node.id, /snap/i);
    }
    const callback = flows.find((node) => node.type === "callback");
    assert.equal(callback.pipelineId, "${AIP_PIPELINE_ID}");
    const tab = flows.find((node) => node.id === callback.z);
    assert.equal(tab.env.find((entry) => entry.name === "AIP_PIPELINE_ID").value, "");
    assert.deepEqual(flows.find((node) => node.type === "uibuilder").wires, [[], []]);
    assert.equal(JSON.parse(read("logic/package.json")).dependencies["@edge-ai/node-red-contrib-eais"], "0.6.24");
});

test("integration package includes runtime assets, excludes credentials, and describes latest counts", () => {
    const includes = read(".include").split(/\r?\n/).filter((line) => line && !line.startsWith("#"));
    for (const entry of includes) assert.ok(fs.existsSync(path.join(root, entry)), entry);
    const bundled = (file) => includes.some((entry) => entry === file
        || entry.endsWith("/") && file.startsWith(entry));
    for (const file of ["logic/flows.json", "logic/package.json", "logic/package-lock.json",
        "dashboard/package.json", "dashboard/package-lock.json",
        ...["html", "js", "css"].map((ext) => `dashboard/aipv1emi0001_dash/src/index.${ext}`)]) {
        assert.ok(bundled(file), `${file} not packaged`);
        assert.ok(read(file).length > 0);
    }
    assert.equal(bundled("logic/flows_cred.json"), false);
    const html = read("dashboard/aipv1emi0001_dash/src/index.html");
    assert.match(html, /Latest counts/);
    assert.match(html, /not synchronized/);
    for (const ext of ["html", "js", "css"]) {
        assert.doesNotMatch(read(`dashboard/aipv1emi0001_dash/src/index.${ext}`),
            /snapshot|syncOffset|syncEnabled|playingFrame|bufferWindows|uibuilder\.send/);
    }
});
