const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const root = path.resolve(__dirname, "..");
const read = (file) => fs.readFileSync(path.join(root, file), "utf8");
const normalizer = new Function("msg", "node", "env", JSON.parse(read("logic/flows.json"))
    .find((node) => node.id === "aipv1emi0001normalize").func);

function normalize(payload) {
    const warnings = [];
    const result = normalizer({ payload, eais: { pipelineId: "test-pipeline" } },
        { warn: (warning) => warnings.push(warning) }, { get: () => undefined });
    assert.deepEqual(warnings, []);
    return result.payload;
}

function dashboard() {
    let definition;
    vm.runInNewContext(read("dashboard/aipv1emi0001_dash/src/index.js")
        .replace(/^import .*;\n/gm, ""), {
        createApp(value) {
            definition = value;
            return { mount() {} };
        },
        nextTick: (callback) => callback(),
    });
    const app = { ...definition.data(), ...definition.methods };
    for (const [name, getter] of Object.entries(definition.computed)) {
        Object.defineProperty(app, name, { get: () => getter.call(app) });
    }
    return app;
}

test("missing sources remain unavailable in per-source and all-source metrics", () => {
    const stats = (max) => ({ max_val: max, min_val: max, mean_val: max, median_val: max, count_val: max });
    const data = normalize({
        0: {
            source_id: 0, available: true, frame_number_from: 10, frame_number_to: 12,
            obj_counter: { car: stats(2), person: stats(1), truck: stats(0) },
            analytics_obj_cnt: {
                objInROIcnt: { yard: { max_val: 2 } },
                objLCCurrCnt: { entry: { max_val: 1 } },
            },
        },
        1: {
            source_id: 1, available: false, frame_number_from: null, frame_number_to: null,
            obj_counter: { car: stats(null), person: stats(null), truck: stats(null) },
            analytics_obj_cnt: { objInROIcnt: {}, objLCCurrCnt: {} },
        },
    });
    assert.equal(data.sources[0].available, true);
    assert.equal(data.sources[0].total, 3);
    assert.deepEqual(data.sources[0].counts, { car: 2, person: 1, truck: 0 });
    assert.equal(data.sources[1].available, false);
    assert.equal(data.sources[1].total, null);
    assert.deepEqual(data.sources[1].counts, { car: null, person: null, truck: null });
    assert.equal(data.totals.objects, null);
    assert.deepEqual(data.totals.classes, { car: null, person: null, truck: null });
    assert.deepEqual(data.totals.roi, { yard: null });
    assert.deepEqual(data.totals.lineCrossing, { entry: null });

    const app = dashboard();
    app.applyMessage({ payload: data });
    assert.equal(app.formatCount(app.scopedTotal), "N/A");
    assert.deepEqual(Array.from(app.classCards, (entry) => entry.value), ["N/A", "N/A", "N/A"]);
    assert.deepEqual(Array.from(app.roiRows, (entry) => entry.value), ["N/A"]);
    app.selectedSource = "0";
    assert.equal(app.formatCount(app.scopedTotal), 3);
    assert.deepEqual(Array.from(app.classCards, (entry) => entry.value), [2, 1, 0]);
    app.selectedSource = "1";
    assert.equal(app.formatCount(app.scopedTotal), "N/A");
});

test("unknown values do not become zero, while observed zero remains zero", () => {
    const data = normalize([
        { source_id: 0, available: true, obj_counter: [0, null, 2],
            objInROIcnt: { yard: null, gate: 0 } },
    ]);
    assert.deepEqual(data.sources[0].counts, { car: 0, person: null, truck: 2 });
    assert.equal(data.sources[0].total, null);
    assert.equal(data.totals.objects, null);
    assert.deepEqual(data.totals.classes, { car: 0, person: null, truck: 2 });
    assert.deepEqual(data.totals.roi, { yard: null, gate: 0 });

    const app = dashboard();
    assert.equal(app.formatCount(null), "N/A");
    assert.equal(app.formatCount(0), 0);
    app.applyMessage({ payload: data });
    assert.deepEqual(Array.from(app.classCards, (entry) => entry.value), [0, "N/A", 2]);
});

test("missing analytics labels do not invent values", () => {
    const data = normalize({
        source_id: 0, available: true, obj_counter: [0, 0, 0],
        objInROIcnt: {}, objLCCurrCnt: {},
    });
    assert.deepEqual(data.totals.roi, {});
    assert.deepEqual(data.totals.lineCrossing, {});
    const app = dashboard();
    app.applyMessage({ payload: data });
    assert.equal(app.formatCount(app.scopedTotal), 0);
    assert.equal(app.roiRows.length, 0);
    assert.equal(app.lineRows.length, 0);
});
