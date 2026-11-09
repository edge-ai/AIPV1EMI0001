const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const { createRequire } = require("node:module");
const os = require("node:os");
const path = require("node:path");
const { test } = require("node:test");

const root = path.resolve(__dirname, "..");
const modules = path.resolve(process.env.NODE_RED_TEST_MODULES || path.join(root, "logic/node_modules"));
const runtimeModules = path.resolve(process.env.NODE_RED_RUNTIME_MODULES
    || (process.env.NODE_RED_TEST_MODULES ? modules : path.join(root, "tests/node-red-runtime/node_modules")));
const aipVersion = JSON.parse(fs.readFileSync(path.join(root, "manifest.json"), "utf8")).version;

async function waitFor(predicate, description) {
    const deadline = Date.now() + 10000;
    while (!predicate()) {
        if (Date.now() >= deadline) throw new Error(`Timed out waiting for ${description}`);
        await new Promise((resolve) => setTimeout(resolve, 20));
    }
}

test("EAIS 0.7.10 supports callback selection and metrics with mocked station services",
    { timeout: 45000 }, async (t) => {
        const packagePath = path.join(modules, "@edgematrix/node-red-contrib-eais");
        assert.equal(require(path.join(packagePath, "package.json")).version, "0.7.10");
        const nodeRedPackage = path.join(runtimeModules, "node-red/package.json");
        const RED = require(path.dirname(nodeRedPackage));
        const express = createRequire(nodeRedPackage)("express");
        const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "aip-node-red-"));
        let started = false;
        let httpServer;
        t.after(async () => {
            try {
                if (started) await RED.stop();
            } finally {
                try {
                    if (httpServer?.listening) {
                        await new Promise((resolve, reject) => {
                            httpServer.close((error) => error ? reject(error) : resolve());
                        });
                    }
                } finally {
                    fs.rmSync(workspace, { recursive: true, force: true });
                }
            }
        });

        fs.symlinkSync(modules, path.join(workspace, "node_modules"), "dir");
        fs.copyFileSync(path.join(root, "logic/package.json"), path.join(workspace, "package.json"));
        const flows = JSON.parse(fs.readFileSync(path.join(root, "logic/flows.json"), "utf8"));
        flows.find((node) => node.id === "aipv1emi0001callback").pipelineId = "fixture-pipeline";
        fs.writeFileSync(path.join(workspace, "flows.json"), JSON.stringify(flows));

        const app = express();
        httpServer = http.createServer(app);
        RED.init(httpServer, {
            userDir: workspace,
            flowFile: "flows.json",
            credentialSecret: false,
            functionGlobalContext: {},
            editorTheme: { projects: { enabled: false } },
            logging: { console: { level: "off", metrics: false, audit: false } },
        });
        app.use(RED.httpAdmin);
        app.use(RED.httpNode);
        RED.settings._eaisEnvironmentDetector = {
            detect: async () => ({ isEdgeAI: false, error: null }),
            clearError() {},
            reset() {},
        };
        const callbacks = [];
        const dashboard = [];
        RED.hooks.add("onReceive.aip-test", (event, done) => {
            if (event.destination.id === "aipv1emi0001normalize") {
                callbacks.push(structuredClone(event.msg));
            } else if (event.destination.id === "aipv1emi0001uib") {
                dashboard.push(structuredClone(event.msg));
            }
            done();
        });
        await RED.start();
        started = true;
        await waitFor(() => RED.nodes.getNode("aipv1emi0001server") &&
            RED.nodes.getNode("aipv1emi0001callback"), "flow startup");

        const server = RED.nodes.getNode("aipv1emi0001server");
        const callback = RED.nodes.getNode("aipv1emi0001callback");
        assert.equal(callback.pipelineId, "fixture-pipeline");
        await server._initPromise;
        let failMetadata = false;
        server.isReady = true;
        server.transport = {};
        server.request = async (method, operation, params) => {
            assert.equal(method, "GET");
            if (operation === "PIPELINES_LIST") {
                return [{ pk: "fixture-pipeline", name: "Fixture pipeline" }];
            }
            assert.equal(operation, "PIPELINE_INFO");
            assert.equal(params.id, "fixture-pipeline");
            if (failMetadata) throw new Error("metadata unavailable");
            return {
                name: "Fixture pipeline",
                input_sources: [{ input_id: 0, media: { pk: "cam", name: "Gate", type: "camera" } }],
                ai_processor_settings: { ai_processor: { unique_id: "AIPV1EMI0001",
                    name: "Fixture AIP", version: aipVersion,
                    primary_processor: { classes: ["car", "person", "truck"] } } },
            };
        };
        await new Promise((resolve) => httpServer.listen(0, "127.0.0.1", resolve));
        const response = await fetch(`http://127.0.0.1:${httpServer.address().port}`
            + "/edgematrix/callback/aipv1emi0001callback");
        assert.equal(response.status, 200);
        assert.deepEqual(await response.json(), {
            pipelines_list: [{ pk: "fixture-pipeline", name: "Fixture pipeline" }],
            pipeline_selected: "fixture-pipeline",
        });
        let stream;
        server.createEventStream = (config) => {
            stream = config;
            return {
                isRunning: true,
                connect: async () => config.onStatusChange({ fill: "green", shape: "dot", text: "connected" }),
                close() {},
            };
        };
        server.statusMonitoring.emit("eais-status", { status: "ready" });
        await waitFor(() => stream, "callback subscription");
        assert.equal(stream.pipelineId, "fixture-pipeline");
        assert.equal(stream.command, "subscribe");
        assert.equal(stream.parseJson, true);

        const cases = [
            { payload: [{ source_id: 0, obj_counter: [2, 1, 0] }], count: 3 },
            { payload: { 0: { source_id: 0, obj_counter: { car: { max_val: 4 } } } }, count: 4 },
            { payload: [
                { data: { 0: { source_id: 0, obj_counter: { car: { max_val: 9 } } } } },
                { data: { 0: { source_id: 0, obj_counter: { car: { max_val: 2 } } } } },
            ], count: 2 },
        ];
        for (const [index, fixture] of cases.entries()) {
            stream.onMessage(fixture.payload);
            await waitFor(() => dashboard.length > index, `dashboard message ${index}`);
            assert.equal(callbacks[index].eais.node, "callback");
            assert.equal(callbacks[index].eais.pipelineId, "fixture-pipeline");
            assert.equal(dashboard[index].payload.pipeline.id, "fixture-pipeline");
            assert.equal(dashboard[index].payload.pipeline.name, "Fixture pipeline");
            assert.equal(dashboard[index].payload.sources[0].name, "Gate");
            assert.equal(dashboard[index].payload.totals.objects, fixture.count);
            assert.equal(dashboard[index].payload.hlsUrl, "/ms3/pipeline/fixture-pipeline/hls");
        }

        failMetadata = true;
        assert.equal(callback.refreshCache(), true);
        stream.onMessage([{ source_id: 0, obj_counter: [1, 0, 0] }]);
        await waitFor(() => dashboard.length === 4, "message without enrichment");
        assert.equal(callbacks[3].eais.warnings[0].code, "METADATA_FETCH_FAILED");
        assert.equal(dashboard[3].payload.pipeline.id, "fixture-pipeline");
        assert.equal(dashboard[3].payload.sources[0].name, "Source 0");
        assert.equal(dashboard[3].payload.totals.objects, 1);
    });
