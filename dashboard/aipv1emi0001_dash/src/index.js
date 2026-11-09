import "../uibuilder/uibuilder.esm.min.js";
import { createApp, markRaw, nextTick } from "../uibuilder/vendor/vue/dist/vue.esm-browser.js";
import Hls from "../uibuilder/vendor/hls.js/dist/hls.mjs";

const MAX_MESSAGES = 100;
const SIGNALING_BASE = "/signaling";
const VIDEO_CONFIG_REFRESH_MS = 10000;

createApp({
    data() {
        return {
            connected: false,
            sources: [],
            totals: { objects: null, classes: {}, roi: {}, lineCrossing: {} },
            pipeline: { id: "", name: "" },
            aggregation: "",
            selectedSource: "all",
            lastUpdate: null,
            messages: [],
            videoState: "waiting",
            videoError: "",
            peerConnection: null,
            hls: null,
            videoMode: null,
            videoConfigKey: "",
            videoConfigRequest: null,
            videoRefreshTimer: null,
            videoAttempt: 0,
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
            return labels.map((label) => ({ label, value: this.formatCount(this.scopedClasses?.[label]) }));
        },

        roiRows() {
            return this.toRows(this.scopedRoi);
        },

        lineRows() {
            return this.toRows(this.scopedLines);
        },
    },

    watch: {
        "pipeline.id"() {
            nextTick(() => this.connectVideo(true));
        },
    },

    methods: {
        formatCount(value) {
            return typeof value === "number" && Number.isFinite(value) ? value : "N/A";
        },

        toRows(values) {
            return Object.entries(values || {})
                .map(([label, value]) => ({ label, value: this.formatCount(value) }))
                .sort((left, right) => {
                    const leftValue = typeof left.value === "number" ? left.value : -Infinity;
                    const rightValue = typeof right.value === "number" ? right.value : -Infinity;
                    return rightValue - leftValue || left.label.localeCompare(right.label);
                });
        },

        applyMessage(msg) {
            const data = msg && msg.payload;
            if (!data || data.type !== "aipv1emi0001.metrics" || !Array.isArray(data.sources)) return;

            this.sources = data.sources;
            this.totals = data.totals || { objects: null, classes: {}, roi: {}, lineCrossing: {} };
            this.pipeline = data.pipeline || { id: "", name: "" };
            this.aggregation = data.aggregation || "";
            this.lastUpdate = new Date(data.timestamp || Date.now());
            this.messages.unshift(msg);
            if (this.messages.length > MAX_MESSAGES) this.messages.length = MAX_MESSAGES;
            if (this.selectedSource !== "all" && !this.sources.some((source) => source.id === this.selectedSource)) {
                this.selectedSource = "all";
            }
        },

        async responseError(response) {
            const text = await response.text();
            if (!text) return response.statusText || `HTTP ${response.status}`;
            try {
                const body = JSON.parse(text);
                return body.detail || body.message || text;
            } catch {
                return text.length > 240 ? `${text.slice(0, 237)}...` : text;
            }
        },

        waitForIceGathering(peerConnection, timeoutMs = 10000) {
            if (peerConnection.iceGatheringState === "complete") return Promise.resolve();
            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    peerConnection.removeEventListener("icegatheringstatechange", onStateChange);
                    reject(new Error("timed out while preparing the connection."));
                }, timeoutMs);
                const onStateChange = () => {
                    if (peerConnection.iceGatheringState !== "complete") return;
                    clearTimeout(timeout);
                    peerConnection.removeEventListener("icegatheringstatechange", onStateChange);
                    resolve();
                };
                peerConnection.addEventListener("icegatheringstatechange", onStateChange);
                onStateChange();
            });
        },

        async connectVideo(force = false) {
            if (!this.pipeline.id) {
                this.destroyVideo();
                this.videoConfigKey = "";
                this.videoMode = null;
                this.videoState = "waiting";
                this.videoError = "";
                return;
            }
            if (this.videoConfigRequest && !force) return;
            if (force) this.destroyVideo();
            if (force || !this.videoConfigKey) {
                this.videoState = "loading";
                this.videoError = "";
            }
            const pipelineId = this.pipeline.id;
            const controller = markRaw(new AbortController());
            this.videoConfigRequest = controller;
            try {
                const response = await fetch(`/ms3/pipeline/${encodeURIComponent(pipelineId)}`, {
                    credentials: "same-origin",
                    cache: "no-store",
                    signal: controller.signal,
                });
                if (!response.ok) throw new Error(await this.responseError(response));
                const config = await response.json();
                if (controller !== this.videoConfigRequest || pipelineId !== this.pipeline.id) return;
                if (!config || typeof config !== "object"
                    || !(config.hls_output_sink_settings || config.mtx_output_sink_settings)) {
                    throw new Error("The pipeline service returned an invalid video configuration.");
                }
                // Match the station web UI: HLS takes precedence when both outputs are enabled.
                const mode = config.hls_output_sink_settings?.active ? "hls"
                    : config.mtx_output_sink_settings?.active && config.mtx_output_sink_settings?.webrtc_enabled
                        ? "webrtc" : null;
                const key = `${pipelineId}:${mode}`;
                this.videoConfigRequest = null;
                if (!force && key === this.videoConfigKey) return;
                this.destroyVideo();
                this.videoConfigKey = key;
                this.videoMode = mode;
                if (!mode) {
                    this.videoState = "disabled";
                    this.videoError = "No video output is enabled for this pipeline.";
                } else if (mode === "hls") {
                    this.attachHls();
                } else {
                    await this.attachWebRtc();
                }
            } catch (error) {
                if (controller !== this.videoConfigRequest) return;
                this.destroyVideo();
                this.videoConfigKey = "";
                this.videoMode = null;
                this.videoState = "error";
                this.videoError = `Video unavailable: ${error.message}`;
            } finally {
                if (controller === this.videoConfigRequest) this.videoConfigRequest = null;
            }
        },

        destroyVideo() {
            this.videoAttempt += 1;
            if (this.videoConfigRequest) {
                this.videoConfigRequest.abort();
                this.videoConfigRequest = null;
            }
            if (this.hls) {
                this.hls.destroy();
                this.hls = null;
            }
            if (this.peerConnection) {
                this.peerConnection.ontrack = null;
                this.peerConnection.onconnectionstatechange = null;
                this.peerConnection.close();
                this.peerConnection = null;
            }
            const video = this.$refs.video;
            if (video) {
                video.onplaying = null;
                video.onloadedmetadata = null;
                video.onerror = null;
                video.pause();
                video.srcObject = null;
                video.removeAttribute("src");
                video.load();
            }
        },

        playVideo(video, attempt) {
            video.play().catch(() => {
                if (attempt === this.videoAttempt) {
                    this.videoState = "ready";
                    this.videoError = "Press play to start the video.";
                }
            });
        },

        attachHls() {
            this.destroyVideo();
            const attempt = this.videoAttempt;
            const video = this.$refs.video;
            if (!video || !this.pipeline.id) return;
            const url = `/ms3/pipeline/${encodeURIComponent(this.pipeline.id)}/hls`;
            this.videoState = "loading";
            this.videoError = "";
            video.onplaying = () => {
                if (attempt !== this.videoAttempt) return;
                this.videoState = "playing";
                this.videoError = "";
            };
            const fail = (message) => {
                if (attempt !== this.videoAttempt) return;
                this.destroyVideo();
                this.videoState = "error";
                this.videoError = message;
            };
            video.onerror = () => fail("Video unavailable: HLS playback failed. Retry the video connection.");
            try {
                if (Hls.isSupported()) {
                    const hls = markRaw(new Hls({
                        lowLatencyMode: false,
                        backBufferLength: 30,
                        liveSyncDurationCount: 3,
                        liveMaxLatencyDurationCount: 6,
                        maxLiveSyncPlaybackRate: 1.5,
                    }));
                    this.hls = hls;
                    hls.on(Hls.Events.MANIFEST_PARSED, () => {
                        if (attempt === this.videoAttempt) this.playVideo(video, attempt);
                    });
                    hls.on(Hls.Events.ERROR, (_event, data) => {
                        if (data.fatal) fail(`Video unavailable: HLS playback failed (${data.details || data.type}). Retry the video connection.`);
                    });
                    hls.loadSource(url);
                    hls.attachMedia(video);
                } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
                    video.onloadedmetadata = () => {
                        if (attempt === this.videoAttempt) this.playVideo(video, attempt);
                    };
                    video.src = url;
                    video.load();
                } else {
                    fail("This browser does not support HLS playback.");
                }
            } catch (error) {
                fail(`Video unavailable: ${error.message}`);
            }
        },

        async attachWebRtc() {
            this.destroyVideo();
            const attempt = this.videoAttempt;
            const pipelineId = this.pipeline.id;
            const video = this.$refs.video;
            if (!video || !pipelineId) return;
            this.videoState = "loading";
            this.videoError = "";

            try {
                const peerConnection = markRaw(new RTCPeerConnection());
                this.peerConnection = peerConnection;
                peerConnection.addTransceiver("video", { direction: "recvonly" });
                peerConnection.ontrack = (event) => {
                    if (attempt !== this.videoAttempt) return;
                    video.srcObject = event.streams[0] || new MediaStream([event.track]);
                    this.playVideo(video, attempt);
                };
                peerConnection.onconnectionstatechange = () => {
                    if (attempt !== this.videoAttempt) return;
                    if (peerConnection.connectionState === "failed") {
                        this.videoState = "error";
                        this.videoError = "Video connection failed. Check that the pipeline's WebRTC output is enabled.";
                    } else if (peerConnection.connectionState === "disconnected") {
                        this.videoState = "disconnected";
                    }
                };
                video.onplaying = () => {
                    if (attempt === this.videoAttempt) {
                        this.videoState = "playing";
                        this.videoError = "";
                    }
                };

                const offer = await peerConnection.createOffer();
                await peerConnection.setLocalDescription(offer);
                await this.waitForIceGathering(peerConnection);
                if (attempt !== this.videoAttempt) return;

                const response = await fetch(`${SIGNALING_BASE}/rtsp-sink-and-whep`, {
                    method: "POST",
                    credentials: "same-origin",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        pipeline_id: pipelineId,
                        offer_sdp: peerConnection.localDescription.sdp,
                    }),
                });
                if (!response.ok) throw new Error(await this.responseError(response));
                const answer = await response.json();
                if (typeof answer.answer_sdp !== "string" || !answer.answer_sdp) {
                    throw new Error("The signaling service did not return an SDP answer.");
                }
                if (attempt !== this.videoAttempt) return;
                await peerConnection.setRemoteDescription({ type: "answer", sdp: answer.answer_sdp });
            } catch (error) {
                if (attempt !== this.videoAttempt) return;
                if (this.peerConnection) {
                    this.peerConnection.ontrack = null;
                    this.peerConnection.onconnectionstatechange = null;
                    this.peerConnection.close();
                    this.peerConnection = null;
                }
                this.videoState = "error";
                this.videoError = `Video unavailable: ${error.message}`;
            }
        },
    },

    mounted() {
        this.connected = Boolean(uibuilder.ioConnected);
        uibuilder.onChange("ioConnected", (connected) => { this.connected = Boolean(connected); });
        uibuilder.onChange("msg", (msg) => this.applyMessage(msg));
        this.videoRefreshTimer = setInterval(() => this.connectVideo(), VIDEO_CONFIG_REFRESH_MS);
    },

    beforeUnmount() {
        clearInterval(this.videoRefreshTimer);
        this.destroyVideo();
    },
}).mount("#app");
