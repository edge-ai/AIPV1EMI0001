import "../uibuilder/uibuilder.esm.min.js";
import { createApp, nextTick } from "../uibuilder/vendor/vue/dist/vue.esm-browser.js";
import Hls from "../uibuilder/vendor/hls.js/dist/hls.mjs";

const MAX_MESSAGES = 100;

createApp({
    data() {
        return {
            connected: false,
            sources: [],
            totals: { objects: 0, classes: {}, roi: {}, lineCrossing: {} },
            pipeline: { id: "", name: "" },
            hlsUrl: "",
            aggregation: "",
            selectedSource: "all",
            lastUpdate: null,
            messages: [],
            videoState: "waiting",
            hls: null,
        };
    },

    computed: {
        pipelineLabel() {
            return this.pipeline.name || this.pipeline.id || "Waiting for pipeline metadata";
        },

        lastUpdateLabel() {
            return this.lastUpdate
                ? new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(this.lastUpdate)
                : "No data";
        },

        selected() {
            return this.sources.find((source) => source.id === this.selectedSource) || null;
        },

        selectedSourceName() {
            return this.selected ? this.selected.name : "Unknown source";
        },

        scopedTotal() {
            return this.selected ? this.selected.total : this.totals.objects;
        },

        scopedClasses() {
            return this.selected ? this.selected.counts : this.totals.classes;
        },

        scopedRoi() {
            return this.selected ? this.selected.roi : this.totals.roi;
        },

        scopedLines() {
            return this.selected ? this.selected.lineCrossing : this.totals.lineCrossing;
        },

        classCards() {
            const preferred = ["car", "person", "truck"];
            const labels = [...new Set([...preferred, ...Object.keys(this.scopedClasses || {})])];
            return labels.map((label) => ({ label, value: Number(this.scopedClasses?.[label] || 0) }));
        },

        roiRows() {
            return this.toRows(this.scopedRoi);
        },

        lineRows() {
            return this.toRows(this.scopedLines);
        },
    },

    watch: {
        hlsUrl() {
            nextTick(() => this.attachVideo());
        },
    },

    methods: {
        toRows(values) {
            return Object.entries(values || {})
                .map(([label, value]) => ({ label, value: Number(value) || 0 }))
                .sort((left, right) => right.value - left.value || left.label.localeCompare(right.label));
        },

        applyMessage(msg) {
            const data = msg && msg.payload;
            if (!data || data.type !== "aipv1emi0001.metrics" || !Array.isArray(data.sources)) return;

            this.sources = data.sources;
            this.totals = data.totals || { objects: 0, classes: {}, roi: {}, lineCrossing: {} };
            this.pipeline = data.pipeline || { id: "", name: "" };
            this.aggregation = data.aggregation || "";
            this.lastUpdate = new Date(data.timestamp || Date.now());
            this.messages.unshift(msg);
            if (this.messages.length > MAX_MESSAGES) this.messages.length = MAX_MESSAGES;
            if (data.hlsUrl && data.hlsUrl !== this.hlsUrl) this.hlsUrl = data.hlsUrl;
            if (this.selectedSource !== "all" && !this.sources.some((source) => source.id === this.selectedSource)) {
                this.selectedSource = "all";
            }
        },

        destroyVideo() {
            if (this.hls) {
                this.hls.destroy();
                this.hls = null;
            }
            const video = this.$refs.video;
            if (video) {
                video.removeAttribute("src");
                video.load();
            }
        },

        attachVideo() {
            const video = this.$refs.video;
            if (!video || !this.hlsUrl) return;
            this.destroyVideo();
            this.videoState = "loading";

            if (video.canPlayType("application/vnd.apple.mpegurl")) {
                video.src = this.hlsUrl;
                video.addEventListener("playing", () => { this.videoState = "playing"; }, { once: true });
                video.addEventListener("error", () => { this.videoState = "error"; }, { once: true });
                video.play().catch(() => { this.videoState = "ready"; });
                return;
            }

            if (!Hls.isSupported()) {
                this.videoState = "unsupported";
                return;
            }

            this.hls = new Hls({ liveSyncDurationCount: 3, enableWorker: true });
            this.hls.loadSource(this.hlsUrl);
            this.hls.attachMedia(video);
            this.hls.on(Hls.Events.MANIFEST_PARSED, () => {
                video.play().catch(() => { this.videoState = "ready"; });
            });
            this.hls.on(Hls.Events.ERROR, (_event, details) => {
                if (details.fatal) this.videoState = "error";
            });
            video.addEventListener("playing", () => { this.videoState = "playing"; });
        },
    },

    mounted() {
        this.connected = Boolean(uibuilder.ioConnected);
        uibuilder.onChange("ioConnected", (connected) => { this.connected = Boolean(connected); });
        uibuilder.onChange("msg", (msg) => this.applyMessage(msg));
    },

    beforeUnmount() {
        this.destroyVideo();
    },
}).mount("#app");
