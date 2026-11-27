#!/usr/bin/env python3
"""Find how many RTSP cameras a station sustains at 30 FPS, per model and output profile.

The harness drives the station's video-streaming API (ms3). For every trial it creates a pipeline with N
RTSP inputs, starts the outputs the profile asks for, waits for the pipeline to settle, samples the per-input
processed rate for a fixed window and tears everything down again. A trial passes when every camera averages at
least 29 FPS and no camera reconnects. `search` finds the largest passing N with a binary search; `fixed` runs
one N, which is how the cheap one-factor screening runs are done. Plain stdlib, so it runs as it is on the station:

    python3 tools/capacity.py plan
    python3 tools/capacity.py search --ms3 http://MS3:8000 --device ex5 --model yolo26 --aip AIPV1EMI9026 \
        --rtsp-template 'rtsp://VCAM:PORT/cam{index:02d}' --profile base --out capacity.jsonl
    python3 tools/capacity.py fixed  ... --device ex3 --n 4 --profile webrtc+evt-osd
    python3 tools/capacity.py summarize capacity.jsonl
    python3 tools/capacity.py compare-matched EX3.jsonl EX5.jsonl
    python3 tools/capacity.py cleanup --ms3 http://MS3:8000 --ledger capacity.jsonl.owned.json

doc/benchmarks/capacity/README.md explains why this is a pruned plan and not every combination. Every run gets its
own ID, creates its cameras and pipelines under it and lists them in a ledger next to the results file until they are
deleted again; `cleanup` deletes what a ledger lists and nothing else.
"""
import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

TARGET_FPS = 30.0
PASS_FPS = 29.0
BORDERLINE_FPS = 0.5
# The flavor display names, which the packaging tools write into the AIP name and model fields.
MODEL_NAMES = {"yolov7": "YOLOv7-tiny", "yolov10": "YOLOv10n", "yolo26": "YOLO26n"}
MODELS = tuple(MODEL_NAMES)
ENGINE_MAX_BATCH = 8
DEFAULT_MAX_N = 16
DEFAULT_BITRATE = 2_000_000
EVENTREC_BITRATE = 4_000_000
SMART_PRE_MS = 3000
SMART_POST_MS = 6000
SMART_MAX_MS = 10000
SMART_TRIGGER_SECONDS = 30
PROTOCOL = "webrtc-smart-video-v2"
MAX_FRAME_GAP_SECONDS = 3 / TARGET_FPS
GAP_TOLERANCE_SECONDS = 0.005
WEBRTC_PROBE_SECONDS = 20
# Read from inside the MS3 container, the same MediaMTX path that WebRTC viewers are served from.
WEBRTC_READ_URL = "rtsp://host.docker.internal:8554/{uuid}_{slot}"
DEFAULT_TILER = (1280, 720)
MATCHED_INPUT = {"codec_name": "h264", "width": 1280, "height": 720, "avg_frame_rate": "30/1"}
NAME_PREFIX = "cap-"
# Slot 2 is the MediaMTX source for the WebRTC publisher.
WEBRTC_SLOT = 2
# Teardown calls accept 404 so that repeating a teardown, as `cleanup` does, is harmless.
GONE_OK = (200, 201, 202, 204, 404)
MEDIA_SETTLE_SECONDS = 5
WARMUP_SECONDS = 45
WINDOW_SECONDS = 120
SAMPLE_SECONDS = 5
CONTAINERS = ("ms3_videostreaming", "mediamtx")
PERF_METRIC = "pipeline_perf_fps_average_10sec"

FACTORS = {
    "model": MODELS,
    "osd": (False, True),
    "delivery": ("none", "webrtc"),
    "Smart Video recording": (False, True),
    "delivery bitrate": (1_000_000, 2_000_000, 4_000_000),
    "output resolution": ((640, 360), (1280, 720), (1920, 1080)),
    "camera resolution": ("720p", "1080p"),
}
SEARCH_TRIALS = 4
TRIAL_MINUTES = 4


class CapacityError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    """What the pipeline does with the frames besides inference. Camera resolution is a property of the media."""

    # Retained to recover old ownership ledgers, never for trials or published tables.
    recording: bool = False
    webrtc: bool = False
    hls: bool = False
    eventrec: bool = False
    osd: bool = False
    bitrate: int = DEFAULT_BITRATE
    tiler: tuple = DEFAULT_TILER

    @property
    def displays(self):
        """The outputs that can carry on-screen display. Continuous recording never does."""
        return self.webrtc or self.hls or self.eventrec

    @property
    def name(self):
        parts = [part for part, on in (("rec", self.recording), ("webrtc", self.webrtc),
                                       ("hls", self.hls), ("evt", self.eventrec)) if on]
        name = "+".join(parts) or "base"
        if self.osd:
            name += "-osd"
        if (self.webrtc or self.hls) and self.bitrate != DEFAULT_BITRATE:
            name += f"-{self.bitrate // 1_000_000}M"
        if self.tiler != DEFAULT_TILER:
            name += f"-{self.tiler[0]}x{self.tiler[1]}"
        return name


def normalize(config):
    """Collapse settings that cannot change the pipeline, so equal pipelines get one name."""
    return replace(config, osd=config.osd and config.displays,
                   bitrate=config.bitrate if config.webrtc or config.hls else DEFAULT_BITRATE)


def parse_profile(name):
    head, *options = name.split("-")
    config = Config()
    if head != "base":
        for part in head.split("+"):
            field = {"rec": "recording", "webrtc": "webrtc", "hls": "hls", "evt": "eventrec"}.get(part)
            if field is None:
                raise CapacityError(f"unknown output {part!r} in profile {name!r}")
            config = replace(config, **{field: True})
    for option in options:
        if option == "osd":
            config = replace(config, osd=True)
        elif re.fullmatch(r"\d+M", option):
            config = replace(config, bitrate=int(option[:-1]) * 1_000_000)
        elif re.fullmatch(r"\d+x\d+", option):
            config = replace(config, tiler=tuple(int(side) for side in option.split("x")))
        else:
            raise CapacityError(f"unknown option {option!r} in profile {name!r}")
    if normalize(config) != config or config.name != name:
        raise CapacityError(f"profile {name!r} is not canonical; did you mean {normalize(config).name!r}?")
    if config.bitrate <= 0 or min(config.tiler) <= 0:
        raise CapacityError("bitrate and output dimensions must be positive")
    return config


def validate_live_profile(config):
    if config.recording or config.hls:
        raise CapacityError("HLS and continuous recording are excluded from benchmarks; use base, webrtc, evt "
                            "(Smart Video recording), or webrtc+evt, optionally with -osd")


def pipeline_setup(name, config, aip, media_uuids, *, tiler=None, matched=False):
    """The request body for POST /pipeline. `aip` is an /aiprocessors record."""
    validate_live_profile(config)
    count = len(media_uuids)
    width, height = tiler or config.tiler
    mtx = config.webrtc
    setup = {
        "name": name,
        "description": "Capacity test created by tools/capacity.py",
        "auto_reload": False,
        "inputs": [{"media_uuid": media, "analytic_setup": None,
                    "media_setup": {"latency": 200, "use_tcp_only": True, "loop": False, "framerate_limit": None}}
                   for media in media_uuids],
        "ai_processor_setup": {"uuid": aip["pk"], "inference_drop": 0, "batch_size": min(count, ENGINE_MAX_BATCH),
                               "merge_interval": 1000, "push_interval": 5000, "cache_ttl": 30,
                               "callback_options": {}},
        "tiler_setup": {"width": width, "height": height},
        "osd_setup": {"active": config.osd},
        "nodered_output_data_setup": {"active": True},
        "websocket_output_data_setup": {"active": True},
        "rtsp_output_sink_setup": {"active": False},
        "hls_output_sink_setup": {"active": False},
        "mtx_output_sink_setup": ({"active": True, "num_slots": 2,
                                   "recording_enabled": False, "webrtc_enabled": True,
                                   "rtmp_enabled": False,
                                   "webrtc_bitrate": config.bitrate} if mtx else {"active": False}),
        "eventrec_sink_setup": {"active": config.eventrec, "osd": config.osd, "bitrate": EVENTREC_BITRATE,
                               "pre_duration": SMART_PRE_MS, "post_duration": SMART_POST_MS,
                               "max_duration": SMART_MAX_MS},
        "ntp_mode": "auto",
    }
    if matched:
        setup["osd_setup"].update(clock=False, bbox=True, text=True, mask=False, analytics=0)
        setup["eventrec_sink_setup"].update(control_rate=1, iframeinterval=5)
        if mtx:
            setup["mtx_output_sink_setup"].update(webrtc_control_rate=1, webrtc_key_frame_interval_sec=1.0)
    return {"pipeline": setup}


def matched_parameters(setup):
    processor = setup["ai_processor_setup"]
    return {"input": setup["inputs"][0]["media_setup"],
            "processor": {k: v for k, v in processor.items() if k not in ("uuid", "batch_size")},
            "requested_batch_size": processor["batch_size"],
            **{name: setup[name + "_setup"] for name in
               ("tiler", "osd", "mtx_output_sink", "eventrec_sink", "nodered_output_data", "websocket_output_data")}}


def matched_readback(ms3, uuid_, setup):
    """Verify the station retained the requested settings. Keep no URLs, credentials or station IDs."""
    detail = ms3.call("GET", f"/pipeline/{uuid_}")

    def checked(label, actual, expected):
        for key, value in expected.items():
            if not isinstance(actual, dict) or key not in actual or actual[key] != value:
                raise CapacityError(f"matched setup: {label}.{key} does not match the requested {value!r}")
        return {key: actual[key] for key in expected}

    inputs = detail.get("input_sources") or []
    if len(inputs) != 1 or len(setup["inputs"]) != 1:
        raise CapacityError("matched setup requires exactly one camera")
    if ((inputs[0].get("media") or {}).get("pk") != setup["inputs"][0]["media_uuid"]
            or "analytic_settings" not in inputs[0] or inputs[0]["analytic_settings"] is not None):
        raise CapacityError("matched setup: unexpected camera or analytics enabled")
    selected = detail.get("ai_processor_settings") or {}
    checked("ai_processor", selected.get("ai_processor"), {"pk": setup["ai_processor_setup"]["uuid"]})
    expected = matched_parameters(setup)
    processor = selected.get("config") or {}
    actual = {"input": inputs[0].get("media_settings"), "processor": processor}
    result = {name: checked(name, actual.get(name, detail.get(name + "_settings")), values)
              for name, values in expected.items() if name != "requested_batch_size"}
    for name in ("rtsp_output_sink", "hls_output_sink"):
        checked(name, detail.get(name + "_settings"), {"active": False})
    # Some station versions omit batch_size from GET /pipeline even when it was requested.
    if "batch_size" in processor:
        checked("processor", processor, {"batch_size": expected["requested_batch_size"]})
    return {**result, "requested_batch_size": expected["requested_batch_size"]}


def matched_source(ffprobe, uri, sha256):
    """The digest identifies the publisher's encoded file; the RTSP probe checks the actual stream format."""
    if not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise CapacityError("supply the SHA-256 of the looped source file from its publisher")
    if not uri.startswith("rtsp://") or "{" in uri or "}" in uri:
        raise CapacityError("matched runs require one literal RTSP source, not a template")
    try:
        probe = subprocess.run([ffprobe, "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0",
                                "-show_entries", "stream=codec_name,width,height,avg_frame_rate", "-of", "json", uri],
                               capture_output=True, text=True, timeout=20, check=True)
        stream, = json.loads(probe.stdout)["streams"]
        observed = {k: stream[k] for k in ("codec_name", "width", "height", "avg_frame_rate")}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError) as error:
        raise CapacityError("could not verify the matched RTSP source with ffprobe") from error
    if observed != MATCHED_INPUT:
        raise CapacityError(f"matched source must be H.264 1280x720 at 30 FPS; got {observed}")
    return {"file_sha256": sha256, **observed}


# ---- Judging a trial ----

def processed_fps(metrics, pipeline_id):
    """Read the station's per-source post-mux PERF gauge, not arrival or batch cadence."""
    values = {}
    for line in metrics.splitlines():
        match = re.fullmatch(rf"{PERF_METRIC}\{{(.*)\}}\s+(\S+)(?:\s+\d+)?", line)
        if not match:
            continue
        labels = {name: json.loads(f'"{value}"') for name, value in
                  re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', match[1])}
        if labels.get("uuid") != pipeline_id:
            continue
        key = labels.get("input_id")
        try:
            value = float(match[2])
        except ValueError as error:
            raise CapacityError("invalid processed-FPS metric") from error
        if key is None or key in values or not math.isfinite(value) or value < 0:
            raise CapacityError("invalid or duplicate per-input processed-FPS metric")
        values[key] = value
    return values


def sample_from(inputs_fps, pipeline_fps, status, perf):
    """Join per-camera processed FPS with decoded arrival and connectivity telemetry."""
    inputs = {}
    for key, entry in (inputs_fps or {}).items():
        link = entry.get("link") or {}
        arrival = entry.get("arrival") or {}
        if str(key) not in perf:
            raise CapacityError(f"camera {key} has no fresh {PERF_METRIC} metric; cannot measure inference capacity")
        inputs[str(key)] = {"fps": perf[str(key)], "arrival_fps": float(arrival.get("average_10sec") or 0.0),
                            "arrival_raw": float(arrival.get("raw") or 0.0),
                            "state": link.get("state"), "attempt": int(link.get("attempt") or 0)}
    status = status or {}
    return {"inputs": inputs, "pipeline_fps": float((pipeline_fps or {}).get("average_10sec") or 0.0),
            "health": status.get("health"), "overall": status.get("overall"),
            "flags": [flag for flag in ("inference_stall", "low_fps", "gpu_overloaded", "cpu_overloaded")
                      if status.get(flag) == "true"]}


def evaluate(samples, count, *, pass_fps=PASS_FPS):
    """Pass when every one of `count` cameras averages `pass_fps` over the window and none reconnects."""
    if not samples:
        return {"pass": False, "min_fps": 0.0, "mean_fps": 0.0, "per_input_fps": {}, "reasons": ["no samples"]}
    per_input = {}
    states, attempts = {}, {}
    expected_keys = set(samples[0]["inputs"])
    incomplete_samples = 0
    for sample in samples:
        if len(sample["inputs"]) != count or set(sample["inputs"]) != expected_keys:
            incomplete_samples += 1
        for key, entry in sample["inputs"].items():
            per_input.setdefault(key, []).append(entry["fps"])
            states.setdefault(key, set()).add(entry["state"])
            attempts[key] = max(attempts.get(key, 0), entry["attempt"])
    means = {key: statistics.fmean(values) for key, values in sorted(per_input.items())}
    reasons = []
    if len(means) < count:
        reasons.append(f"only {len(means)} of {count} cameras reported")
    if incomplete_samples:
        reasons.append(f"{incomplete_samples} of {len(samples)} samples did not report the same {count} cameras")
    reasons += [f"camera {key} reported non-finite FPS" for key, values in per_input.items()
                if any(not math.isfinite(value) for value in values)]
    reasons += [f"camera {key} averaged {mean:g} FPS" for key, mean in means.items() if mean < pass_fps]
    reasons += [f"camera {key} left the streaming state" for key, seen in sorted(states.items()) if seen != {"streaming"}]
    reasons += [f"camera {key} reconnected {attempt} time(s)" for key, attempt in sorted(attempts.items()) if attempt]
    stalls = sum("inference_stall" in sample["flags"] for sample in samples)
    if stalls:
        reasons.append(f"inference stalled in {stalls} of {len(samples)} samples")
    if any("overall" in sample and sample["overall"] != "Running" for sample in samples):
        reasons.append("pipeline was not Running throughout the window")
    lowest = min(means.values()) if means else 0.0
    return {"pass": not reasons, "min_fps": lowest, "mean_fps": round(statistics.fmean(means.values()), 2) if means else 0.0,
            "per_input_fps": means, "reasons": reasons,
            "flag_samples": {flag: sum(flag in sample["flags"] for sample in samples)
                             for flag in ("low_fps", "gpu_overloaded", "cpu_overloaded")}}


def search_capacity(trial, max_n, *, first=None):
    """Largest N in [0, max_n] whose trial passes, assuming a pass at N implies a pass below it.

    Returns (capacity, bounded, results). `bounded` is False when even max_n passed: capacity is then a lower bound.
    """
    low, high = 0, max_n + 1
    results = []
    guess = first
    while high - low > 1:
        count = guess if guess is not None and low < guess < high else (low + high) // 2
        guess = None
        result = trial(count)
        results.append(result)
        if result["pass"]:
            low = count
        else:
            high = count
    return low, low < max_n, results


def settled(trial, count, *, margin=BORDERLINE_FPS):
    """Run a trial; repeat it once when the slowest camera lands within `margin` FPS of the pass line.

    A repeated trial passes only when both attempts pass, reports the lower of their slowest cameras and keeps both
    attempt records, so its verdict and its figures always agree.
    """
    first = trial(count)
    if first.get("error") or first.get("cleanup_errors"):
        return first
    if not (0 < first["min_fps"] and abs(first["min_fps"] - PASS_FPS) < margin):
        return first
    again = trial(count)
    combined = {key: first[key] for key in ("kind", "model", "profile", "run", "aip", "model_verified", "fps_metric",
                                             "device", "protocol")
                if key in first}
    combined.update({"n": count, "pass": first["pass"] and again["pass"],
                     "min_fps": min(first["min_fps"], again["min_fps"]),
                     "reasons": first["reasons"] + [f"repeat: {reason}" for reason in again["reasons"]],
                     "attempts": [first, again]})
    return combined


def worst_attempt(trial):
    """The attempt with the slowest camera; the trial itself unless it was repeated."""
    return min(trial.get("attempts") or [trial], key=lambda attempt: attempt["min_fps"])


def without_samples(trial):
    trial = {key: value for key, value in trial.items() if key != "samples"}
    if "attempts" in trial:
        trial["attempts"] = [without_samples(attempt) for attempt in trial["attempts"]]
    return trial


def require_valid_trial(trial):
    """An API/setup or teardown failure is not a measured capacity limit."""
    for attempt in trial.get("attempts") or [trial]:
        errors = ([attempt["error"]] if attempt.get("error") else []) + attempt.get("cleanup_errors", [])
        if errors:
            raise CapacityError("trial invalid: " + "; ".join(errors))


# ---- tegrastats ----

TEGRA_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
TEGRA_CPU = re.compile(r"CPU \[([^\]]*)\]")
TEGRA_GPU = re.compile(r"GR3D_FREQ (\d+)%")
TEGRA_ENGINE = re.compile(r"\b(NVENC|NVDEC|NVJPG|OFA|VIC)(?:_FREQ)? (\d+)")
TEGRA_TEMPERATURE = re.compile(r"\btj@(-?[\d.]+)C")
TEGRA_POWER = re.compile(r"VDD_IN (\d+)mW")


def parse_tegrastats(line):
    """A tegrastats line as numbers. Engines tegrastats omits while idle are left out."""
    reading = {}
    for key, pattern in (("ram_used_mb", TEGRA_RAM), ("gpu_percent", TEGRA_GPU), ("tj_c", TEGRA_TEMPERATURE),
                         ("power_mw", TEGRA_POWER)):
        match = pattern.search(line)
        if match:
            reading[key] = float(match.group(1))
    cpu = TEGRA_CPU.search(line)
    if cpu:
        loads = [int(core.split("%")[0]) for core in cpu.group(1).split(",") if "%" in core]
        if loads:
            reading["cpu_percent"] = statistics.fmean(loads)
    for engine, mhz in TEGRA_ENGINE.findall(line):
        reading[f"{engine.lower()}_mhz"] = float(mhz)
    return reading


def summarize_readings(readings):
    """Mean and maximum of every metric seen in a window; engines that stayed idle report 0 MHz."""
    keys = sorted({key for reading in readings for key in reading})
    summary = {}
    for key in keys:
        values = [reading.get(key, 0.0) if key.endswith("_mhz") else reading[key]
                  for reading in readings if key in reading or key.endswith("_mhz")]
        summary[key] = {"mean": round(statistics.fmean(values), 1), "max": round(max(values), 1)}
    return summary


class Tegrastats:
    """Collects tegrastats lines while a trial runs. Does nothing when tegrastats is not installed."""

    def __init__(self, command="tegrastats"):
        self.command = command
        self.readings = []
        self.process = None
        self.thread = None

    def __enter__(self):
        try:
            self.process = subprocess.Popen([self.command, "--interval", "1000"], stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return self
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()
        return self

    def collect(self):
        for line in self.process.stdout:
            reading = parse_tegrastats(line)
            if reading:
                self.readings.append((time.time(), reading))

    def __exit__(self, *exc):
        if self.process:
            self.process.terminate()
            self.process.wait()

    def between(self, start, end):
        return summarize_readings([reading for stamp, reading in list(self.readings) if start <= stamp <= end])


def container_cpu(names):
    """CPU percent of the named containers, best effort."""
    if not names:
        return {}
    try:
        output = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.CPUPerc}}", *names],
                                capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    cpu = {}
    for line in output.splitlines():
        name, _, percent = line.partition(" ")
        if percent.endswith("%"):
            cpu[name] = float(percent[:-1])
    return cpu


# ---- The station ----

class ApiError(CapacityError):
    def __init__(self, method, path, status, payload):
        super().__init__(f"{method} {path} returned {status}: {str(payload)[:300]}")
        self.status = status
        self.payload = payload


class Api:
    """A JSON HTTP API: ms3 video streaming or the signaling service."""

    def __init__(self, base, timeout=60):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def call(self, method, path, body=None, query=None, form=None, ok=(200, 201, 202)):
        url = self.base + path + ("?" + urllib.parse.urlencode(query) if query else "")
        data, headers = None, {}
        if form is not None:
            boundary = uuid.uuid4().hex
            data = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
                            for key, value in form.items()) + f"--{boundary}--\r\n".encode()
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status, text = response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            status, text = error.code, error.read().decode()
        except (urllib.error.URLError, OSError) as error:
            raise CapacityError(f"{method} {path}: {error}") from None
        try:
            payload = json.loads(text) if text else None
        except ValueError:
            payload = text
        if status not in ok:
            raise ApiError(method, path, status, payload)
        return payload


def retrying(call, *, statuses=(409, 502), deadline=90, pause=3):
    """Repeat `call` while the station answers with one of `statuses`, which outputs do while the pipeline settles."""
    end = time.monotonic() + deadline
    while True:
        try:
            return call()
        except ApiError as error:
            if error.status not in statuses or time.monotonic() > end:
                raise
            time.sleep(pause)


def find_aip(api, unique_id):
    for aip in api.call("GET", "/aiprocessors"):
        if aip.get("unique_id") == unique_id:
            return aip
    raise CapacityError(f"AI processor {unique_id} is not registered on this station")


def preflight(api, config):
    """Fail before registering cameras on an incompatible or occupied station."""
    paths = api.call("GET", "/openapi.json")["paths"]
    required = {"/pipeline/{uuid}/inputs/fps": "get", "/pipeline/{uuid}/status": "get", "/metrics": "get"}
    if config.webrtc:
        required["/pipeline/{uuid}/mtx/start"] = "post"
    if config.eventrec:
        required.update({"/pipeline/{uuid}/eventrec/trigger": "post", "/pipeline/records": "get",
                         "/pipeline/record/{uuid}": "get", "/pipeline/records/deletion": "post"})
    missing = [f"{method.upper()} {path}" for path, method in required.items()
               if method not in paths.get(path, {})]
    if missing:
        raise CapacityError("station cannot run this comparison protocol; missing " + ", ".join(missing))
    for pipeline in api.call("GET", "/pipelines"):
        state = api.call("GET", f"/pipeline/{pipeline['pk']}/status").get("overall")
        if state not in ("Stopped", "Failed"):
            raise CapacityError(f"exclusive station required: pipeline {pipeline['name']!r} is {state!r}; "
                                "no existing pipeline was stopped")


def aip_identity(aip):
    """What a result needs to say which AIP produced it."""
    primary = aip.get("primary_processor") or {}
    return {"unique_id": aip.get("unique_id"), "version": aip.get("version"), "pk": aip.get("pk"),
            "name": aip.get("name"), "model": primary.get("model"), "batch_size": primary.get("batch_size")}


def squash(text):
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def model_matches(aip, model):
    """Match metadata, rejecting a conflicting primary model even if the display name matches."""
    primary = aip.get("primary_processor") or {}
    expected, declared = squash(MODEL_NAMES[model]), squash(primary.get("model"))
    if declared and declared not in (expected, squash(model)):
        return False
    return expected in {squash(aip.get("name")), declared}


class Ledger:
    """The station resources a results file's runs created and have not deleted yet.

    Kept as JSON next to the results file and rewritten on every change, so an interrupted run still leaves an exact
    list of what to delete. The file disappears once it is empty.
    """

    KINDS = ("pipelines", "media")

    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        return {kind: dict(data.get(kind) or {}) for kind in self.KINDS}

    def add(self, kind, pk, **info):
        self.write(kind, pk, info)

    def remove(self, kind, pk):
        self.write(kind, pk, None)

    def write(self, kind, pk, info):
        data = self.read()
        if info is None:
            data[kind].pop(pk, None)
        else:
            data[kind][pk] = info
        if not any(data.values()):
            if self.path.exists():
                self.path.unlink()
            return
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        temporary.replace(self.path)


def ledger_path(results):
    return Path(str(results) + ".owned.json")


def new_run_id():
    return time.strftime("%m%d%H%M", time.gmtime()) + uuid.uuid4().hex[:4]


def recording_files(signaling, uuid_):
    """Every continuous-recording file the signaling service holds for a pipeline."""
    files, page = [], 1
    while True:
        listing = signaling.call("GET", f"/recordings/{uuid_}/files", query={"page": page, "size": 50})
        files += listing.get("files") or []
        if not listing.get("has_next"):
            return files
        page += 1


def start_outputs(ms3, signaling, uuid_, config, *, matched=False):
    """Start the WebRTC publisher. Smart Video buffers automatically and is triggered during sampling."""
    validate_live_profile(config)
    if config.webrtc:
        query = {"stream_index": WEBRTC_SLOT}
        if matched:
            query["bitrate"] = config.bitrate
        retrying(lambda: ms3.call("POST", f"/pipeline/{uuid_}/mtx/start", query=query))


def smart_records(ms3, uuid_):
    """Only this pipeline's Smart Video clips, including every result page."""
    records, page = [], 1
    while True:
        listing = ms3.call("GET", "/pipeline/records",
                          query={"pipeline_uuid": uuid_, "record_type": "eventrec", "page": page, "size": 50})
        for record in listing["records"]:
            if record["pipeline_uuid"] != uuid_ or record["record_type"] != "eventrec":
                raise CapacityError("record listing escaped the requested Smart Video pipeline scope")
            records.append(record)
        if page >= listing["pagination"]["max_page"]:
            return records
        page += 1


def probe_smart_clip(ms3, record, config, ffprobe):
    """Decode/count frames after the measurement window, without retaining scene footage."""
    url = f"{ms3.base}/pipeline/record/{urllib.parse.quote(record['record_uuid'], safe='')}"
    try:
        result = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_streams",
                                 "-show_format", "-show_entries", "packet=pts_time,flags", "-of", "json", url],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        raise CapacityError(f"Smart Video clip probe failed: {error}") from error
    if result.returncode or result.stderr.strip():
        raise CapacityError(f"Smart Video clip did not decode cleanly: {result.stderr.strip()[:300]}")
    try:
        data = json.loads(result.stdout)
        stream = next(s for s in data["streams"] if s["codec_type"] == "video")
        duration = float(data["format"]["duration"])
        frames = int(stream["nb_read_frames"])
        width, height = int(stream["width"]), int(stream["height"])
    except (ValueError, KeyError, TypeError, StopIteration) as error:
        raise CapacityError("Smart Video clip probe returned incomplete video metadata") from error
    expected = (SMART_PRE_MS + SMART_POST_MS) / 1000
    # Clip starts are keyframe-aligned; this checks total duration, not the exact trigger/frame alignment.
    if not math.isfinite(duration) or abs(duration - expected) > 2 or frames < 1:
        raise CapacityError(f"Smart Video clip has {frames} frames over {duration:g}s; expected {expected:g}s +/- 2s")
    if (width, height) != config.tiler:
        raise CapacityError(f"Smart Video clip dimensions {(width, height)} do not match {config.tiler}")
    if stream.get("codec_name") != "h264":
        raise CapacityError(f"Smart Video clip must contain H.264; got {stream.get('codec_name')}")
    continuity = packet_continuity(data.get("packets"))
    return {"record_uuid": record["record_uuid"], "duration_seconds": duration, "decoded_frames": frames,
            "decoded_fps": frames / duration, "fps_pass": frames / duration >= PASS_FPS,
            "width": width, "height": height, "codec": stream["codec_name"], **continuity,
            "continuity_pass": continuity["first_packet_keyframe"] and gap_ok(continuity["max_gap_seconds"])}


def packet_continuity(packets):
    """Timing of the video packets ffprobe listed: the largest PTS gap and whether the first one is a keyframe."""
    try:
        timed = [(float(packet["pts_time"]), "K" in packet.get("flags", "")) for packet in packets or []
                 if packet.get("pts_time") not in (None, "N/A")]
    except (TypeError, ValueError, KeyError) as error:
        raise CapacityError("ffprobe returned unreadable packet timestamps") from error
    if len(timed) < 2:
        raise CapacityError(f"need at least two timed video packets, got {len(timed)}")
    first_keyframe = timed[0][1]
    stamps = sorted(stamp for stamp, _ in timed)
    gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
    return {"packets": len(stamps), "span_seconds": round(stamps[-1] - stamps[0], 6),
            "max_gap_seconds": round(max(gaps), 6), "first_packet_keyframe": first_keyframe,
            "keyframes": sum(key for _, key in timed)}


def gap_ok(gap):
    return gap <= MAX_FRAME_GAP_SECONDS + GAP_TOLERANCE_SECONDS


def start_webrtc_probe(ffprobe, uuid_, *, seconds=WEBRTC_PROBE_SECONDS, url=WEBRTC_READ_URL):
    """Read the publisher's H.264 packets over RTSP, without a browser viewer."""
    source = url.format(uuid=uuid_, slot=WEBRTC_SLOT)
    try:
        return subprocess.Popen([ffprobe, "-v", "error", "-rtsp_transport", "tcp", "-read_intervals", f"%+{seconds:g}",
                                 "-select_streams", "v:0", "-show_entries",
                                 "packet=pts_time,flags:stream=codec_name,width,height", "-of", "json", source],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as error:
        raise CapacityError(f"WebRTC delivery probe could not start: {error}") from error


def finish_webrtc_probe(process, *, timeout=WEBRTC_PROBE_SECONDS + 60, tiler=None):
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise CapacityError("WebRTC delivery probe did not finish") from error
    if process.returncode:
        raise CapacityError(f"WebRTC delivery probe failed: {stderr.strip()[:300]}")
    try:
        data = json.loads(stdout)
        continuity = packet_continuity(data.get("packets"))
    except ValueError as error:
        raise CapacityError("WebRTC delivery probe returned invalid JSON") from error
    if tiler is not None:
        streams = data.get("streams") or []
        if len(streams) != 1 or (streams[0].get("codec_name"), streams[0].get("width"), streams[0].get("height")) != (
                "h264", *tiler):
            observed = [(s.get("codec_name"), s.get("width"), s.get("height")) for s in streams]
            raise CapacityError(f"WebRTC publisher did not deliver the requested H.264 {tiler[0]}x{tiler[1]} mosaic; "
                                f"observed {observed}")
        continuity.update(codec="h264", width=tiler[0], height=tiler[1])
    fps = (continuity["packets"] - 1) / continuity["span_seconds"] if continuity["span_seconds"] > 0 else 0.0
    return {**continuity, "delivered_fps": round(fps, 2),
            "pass": fps >= PASS_FPS and gap_ok(continuity["max_gap_seconds"])}


def verify_smart_recordings(ms3, uuid_, config, triggers, ffprobe, *, deadline=60):
    if not triggers:
        raise CapacityError("Smart Video recording was enabled but never triggered")
    end = time.monotonic() + deadline
    while True:
        records = smart_records(ms3, uuid_)
        if len({record["record_uuid"] for record in records}) >= len(triggers):
            return [probe_smart_clip(ms3, record, config, ffprobe) for record in records]
        if time.monotonic() >= end:
            raise CapacityError(f"Smart Video produced {len(records)} clips for {len(triggers)} triggers")
        time.sleep(2)


def delete_smart_recordings(ms3, uuid_, *, deadline=60):
    """Wait for async deletion before letting the ownership ledger disappear."""
    ms3.call("POST", "/pipeline/records/deletion", query={"pipeline_uuid": uuid_, "record_type": "eventrec"},
             ok=GONE_OK)
    end = time.monotonic() + deadline
    while smart_records(ms3, uuid_):
        if time.monotonic() >= end:
            raise CapacityError("Smart Video clips remain after the deletion request")
        time.sleep(2)


def delete_medias(ms3, pks, *, rounds=3):
    """Delete cameras and confirm ms3 no longer lists them. Returns {pk: error} for the ones that remain.

    ms3 has answered 200 to a camera deletion and still listed the camera afterwards, while a preview capture
    for it was running, so a deletion only counts once the camera has stayed gone for a moment.
    """
    remaining, errors = list(pks), {}
    for _ in range(rounds):
        if not remaining:
            break
        for pk in remaining:
            try:
                ms3.call("DELETE", f"/media/{pk}", ok=GONE_OK)
            except CapacityError as error:
                errors[pk] = f"deleting media {pk}: {error}"
        time.sleep(MEDIA_SETTLE_SECONDS)
        listed = {media["pk"] for media in ms3.call("GET", "/medias")}
        remaining = [pk for pk in remaining if pk in listed]
    return {pk: errors.get(pk, f"media {pk} is still listed after {rounds} deletions") for pk in remaining}


def teardown(ms3, signaling, uuid_, config):
    """Stop and delete one harness pipeline and what its outputs wrote. Returns the errors; safe to repeat."""
    errors = []

    def attempt(what, call):
        try:
            call()
            return True
        except CapacityError as error:
            errors.append(f"{what}: {error}")
            return False

    def delete_recordings():
        files = recording_files(signaling, uuid_)
        if files:
            result = signaling.call("DELETE", f"/recordings/{uuid_}/files",
                                    body={"files": files, "exclude_writing": False}) or {}
            if result.get("skipped_writing"):
                raise CapacityError(f"files still being written: {result['skipped_writing']}")
            if recording_files(signaling, uuid_):
                raise CapacityError("continuous-recording files remain after deletion")

    if config.recording:
        if signaling is None:
            errors.append("continuous recording: no signaling URL, so it was neither stopped nor deleted")
        else:
            # Stopping also removes the MediaMTX recording path the signaling service configured for this pipeline.
            if attempt("stopping continuous recording",
                       lambda: signaling.call("POST", "/recordings/stop", body={"pipeline_id": uuid_})):
                attempt("deleting continuous recordings", delete_recordings)
    def stop_pipeline():
        try:
            ms3.call("POST", f"/pipeline/{uuid_}/stop", query={"force": "true"}, ok=GONE_OK)
        except ApiError as error:
            # 422 E3307 "Pipeline is not running": ms3 already stopped it, for example when its own recovery gave up.
            if error.status != 422 or "E3307" not in str(error.payload):
                raise

    if not attempt("stopping the pipeline", stop_pipeline):
        return errors
    if config.eventrec:
        if not attempt("deleting Smart Video recordings", lambda: delete_smart_recordings(ms3, uuid_)):
            return errors
    attempt("deleting the pipeline",
            lambda: ms3.call("DELETE", f"/pipeline/{uuid_}", query={"force": "true"}, ok=GONE_OK))
    return errors


def wait_for_inputs(api, uuid_, count, *, deadline=180):
    end = time.monotonic() + deadline
    status = {}
    while time.monotonic() < end:
        status = api.call("GET", f"/pipeline/{uuid_}/status")
        if status.get("overall") == "Failed":
            raise CapacityError(f"pipeline failed to start: {status.get('detail')}")
        if status.get("overall") == "Running" and int(status.get("inputs:live", 0)) >= count:
            return
        time.sleep(3)
    raise CapacityError(f"only {status.get('inputs:live')} of {count} cameras delivered frames within {deadline} s "
                        f"(health {status.get('health')})")


class Session:
    """One harness run on a station: it creates its own cameras, runs trials, and owns what it creates.

    Everything it creates is named cap-<run>-... and listed in the ledger until it has been deleted.
    """

    def __init__(self, ms3, aip, model, ledger, *, signaling=None, verified=True, run=None, warmup=WARMUP_SECONDS,
                 window=WINDOW_SECONDS, interval=SAMPLE_SECONDS, containers=CONTAINERS, tegrastats=None,
                 log=print, device="unspecified", ffprobe="ffprobe", live_ffprobe=None):
        self.ms3, self.signaling, self.aip, self.model, self.ledger = ms3, signaling, aip, model, ledger
        self.verified = verified
        self.run = run or new_run_id()
        self.warmup, self.window, self.interval = warmup, window, interval
        self.containers, self.tegrastats, self.log = containers, tegrastats, log
        self.media = []
        self.trials = 0
        self.device, self.ffprobe = device, ffprobe
        self.live_ffprobe = live_ffprobe or ffprobe

    def provenance(self):
        return {"run": self.run, "aip": aip_identity(self.aip), "model_verified": self.verified,
                "fps_metric": PERF_METRIC, "device": self.device, "protocol": PROTOCOL}

    def create_media(self, template, count, tag):
        """Register `count` cameras from the RTSP template. Refuses to adopt cameras that already exist."""
        names = [f"{NAME_PREFIX}{self.run}-{tag}-{index:02d}" for index in range(1, count + 1)]
        taken = sorted({media["name"] for media in self.ms3.call("GET", "/medias")}.intersection(names))
        if taken:
            raise CapacityError(f"media {', '.join(taken)} already exist and are not this run's; not reusing them")
        for index, name in enumerate(names, 1):
            pk = self.ms3.call("POST", "/media", form={
                "name": name, "description": "Capacity test camera",
                "proxy_setup": json.dumps({"uri": template.format(index=index)}),
                # ms3 rejects RTSP media without external settings, even empty ones.
                "external_setup": json.dumps({})})["pk"]
            self.ledger.add("media", pk, name=name, run=self.run)
            self.media.append(pk)

    def close(self):
        """Delete this run's cameras. Returns the errors; what failed stays in the ledger."""
        if any(info["run"] == self.run for info in self.ledger.read()["pipelines"].values()):
            return ["pipeline cleanup incomplete; its cameras remain in the ledger"]
        try:
            failed = delete_medias(self.ms3, self.media)
        except CapacityError as error:
            return [f"could not confirm the cameras were deleted: {error}"]
        for pk in self.media:
            if pk not in failed:
                self.ledger.remove("media", pk)
        self.media = []
        return list(failed.values())

    def sample(self, uuid_, config, samples, triggers=None):
        """Append samples as they arrive, retaining partial evidence if polling fails."""
        begin, clock = time.time(), time.monotonic()
        trigger_at = clock
        while time.monotonic() - clock < self.window:
            if config.eventrec and time.monotonic() >= trigger_at:
                try:
                    response = self.ms3.call("POST", f"/pipeline/{uuid_}/eventrec/trigger")
                    if not isinstance(response, dict) or response.get("status") not in (200, 201):
                        raise CapacityError(f"Smart Video trigger was not acknowledged: {response}")
                    if triggers is not None:
                        triggers.append({"t": round(time.monotonic() - clock, 3), "response": response})
                except CapacityError as error:
                    raise CapacityError(f"event recording trigger failed: {error}") from error
                trigger_at += SMART_TRIGGER_SECONDS
            sample = sample_from(self.ms3.call("GET", f"/pipeline/{uuid_}/inputs/fps"),
                                 self.ms3.call("GET", f"/pipeline/{uuid_}/fps"),
                                 self.ms3.call("GET", f"/pipeline/{uuid_}/status"),
                                 processed_fps(self.ms3.call("GET", "/metrics"), uuid_))
            sample["t"] = round(time.monotonic() - clock, 3)
            samples.append(sample)
            time.sleep(min(self.interval, max(0, self.window - (time.monotonic() - clock))))
        return begin, time.time()

    def trial(self, count, config, *, source=None):
        """One pipeline with `count` cameras: start, settle, sample, tear down. Returns the trial record."""
        validate_live_profile(config)
        self.trials += 1
        name = f"{NAME_PREFIX}{self.run}-{self.model}-{config.name}-n{count}-t{self.trials:02d}"
        record = {"kind": "trial", "model": self.model, "profile": config.name, "n": count, "pipeline": name,
                  "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **self.provenance(),
                  "warmup_seconds": self.warmup, "window_seconds": self.window,
                  "interval_seconds": self.interval, "samples": [], "output_settings": asdict(config),
                  "smart_video": ({"pre_duration_ms": SMART_PRE_MS, "post_duration_ms": SMART_POST_MS,
                                   "max_duration_ms": SMART_MAX_MS, "trigger_interval_seconds": SMART_TRIGGER_SECONDS,
                                   "triggers": [], "clips": []} if config.eventrec else None)}
        uuid_ = webrtc_probe = None
        try:
            if not 1 <= count <= len(self.media):
                raise CapacityError(f"{count} cameras asked for but this run registered {len(self.media)}")
            body = pipeline_setup(name, config, self.aip, self.media[:count], matched=source is not None)
            uuid_ = self.ms3.call("POST", "/pipeline", body=body)["pk"]
            self.ledger.add("pipelines", uuid_, name=name, profile=config.name, run=self.run)
            if source is not None:
                record["matched_setup"] = {"source": source, "pipeline": matched_readback(
                    self.ms3, uuid_, body["pipeline"])}
            self.ms3.call("POST", f"/pipeline/{uuid_}/start")
            wait_for_inputs(self.ms3, uuid_, count)
            start_outputs(self.ms3, self.signaling, uuid_, config, matched=source is not None)
            self.log(f"  n={count} {config.name}: settling for {self.warmup:g} s")
            time.sleep(self.warmup)
            samples = record["samples"]
            triggers = record["smart_video"]["triggers"] if config.eventrec else None
            webrtc_probe = start_webrtc_probe(self.live_ffprobe, uuid_) if config.webrtc else None
            begin, end = self.sample(uuid_, config, samples, triggers)
            record["sample_start_unix"], record["sample_end_unix"] = begin, end
            record["containers_cpu_percent"] = container_cpu(self.containers)
            record.update(evaluate(samples, count))
            if not samples:
                raise CapacityError("sampling window produced no observations")
            record["pipeline_fps_mean"] = round(statistics.fmean(sample["pipeline_fps"] for sample in samples), 2)
            if self.tegrastats is not None:
                record["tegrastats"] = self.tegrastats.between(begin, end)
            if webrtc_probe is not None:
                delivery = record["webrtc_delivery"] = finish_webrtc_probe(
                    webrtc_probe, tiler=config.tiler if source is not None else None)
                if not delivery["pass"]:
                    record["pass"] = False
                    record["reasons"].append(
                        f"WebRTC must deliver {PASS_FPS:g}+ FPS with no gap over {MAX_FRAME_GAP_SECONDS:.2f}s; got "
                        f"{delivery['delivered_fps']:g} FPS, largest gap {delivery['max_gap_seconds']:g}s")
            if config.eventrec:
                record["smart_video"]["clips"] = verify_smart_recordings(
                    self.ms3, uuid_, config, triggers, self.ffprobe)
                for clip in record["smart_video"]["clips"]:
                    if not clip["fps_pass"]:
                        record["pass"] = False
                        record["reasons"].append(
                            f"Smart Video clip must contain H.264 at {PASS_FPS:g}+ decoded FPS; "
                            f"got {clip['codec']} at {clip['decoded_fps']:g} FPS")
                    if not clip.get("continuity_pass", True):
                        record["pass"] = False
                        record["reasons"].append(
                            f"Smart Video clip must start on a keyframe with no gap over {MAX_FRAME_GAP_SECONDS:.2f}s; "
                            f"got largest gap {clip['max_gap_seconds']:g}s, "
                            f"first packet keyframe={clip['first_packet_keyframe']}")
        except CapacityError as error:
            record.update(evaluate(record["samples"], count))
            record.update({"pass": False, "error": str(error)})
            record["reasons"].append(f"trial failed: {error}")
        finally:
            if webrtc_probe is not None and webrtc_probe.poll() is None:
                webrtc_probe.kill()
                webrtc_probe.communicate()
            if uuid_:
                cleanup_errors = teardown(self.ms3, self.signaling, uuid_, config)
                if cleanup_errors:
                    record["cleanup_errors"] = cleanup_errors
                    self.log(f"  cleanup warning, {uuid_} stays in {self.ledger.path}: " + "; ".join(cleanup_errors))
                else:
                    self.ledger.remove("pipelines", uuid_)
        self.log(f"  n={count} {config.name}: {'PASS' if record['pass'] else 'FAIL'} "
                 f"min {record['min_fps']:.2f} FPS {'; '.join(record['reasons'][:2])}")
        return record


def cleanup(ms3, signaling, ledger, log=print):
    """Delete what the ledger lists, and nothing else. Returns the errors; what failed stays listed."""
    owned = ledger.read()
    pipelines = {pipeline["pk"]: pipeline["name"] for pipeline in ms3.call("GET", "/pipelines")}
    medias = {media["pk"]: media["name"] for media in ms3.call("GET", "/medias")}
    errors = []
    blocked_runs = set()
    removed = {"pipelines": 0, "media": 0}
    deletable = []
    for kind, present in (("pipelines", pipelines), ("media", medias)):
        for pk, info in owned[kind].items():
            if pk in present and present[pk] != info["name"]:
                errors.append(f"{kind} {pk} is named {present[pk]!r}, not {info['name']!r}; left alone")
                if kind == "pipelines":
                    blocked_runs.add(info["run"])
                continue
            if kind == "media" and info["run"] in blocked_runs:
                errors.append(f"media {pk} retained because its pipeline cleanup failed")
                continue
            if kind == "media":
                deletable.append(pk)
                continue
            problems = teardown(ms3, signaling, pk, parse_profile(info["profile"]))
            if problems:
                errors += problems
                blocked_runs.add(info["run"])
            else:
                ledger.remove(kind, pk)
                removed[kind] += 1
    try:
        failed = delete_medias(ms3, deletable)
    except CapacityError as error:
        failed = {pk: f"could not confirm media {pk} was deleted: {error}" for pk in deletable}
    errors += failed.values()
    for pk in deletable:
        if pk not in failed:
            ledger.remove("media", pk)
            removed["media"] += 1
    log(f"removed {removed['pipelines']} pipelines and {removed['media']} media listed in {ledger.path}")
    names = {info["name"] for entries in owned.values() for info in entries.values()}
    others = sorted(name for name in [*pipelines.values(), *medias.values()]
                    if name.startswith(NAME_PREFIX) and name not in names)
    if others:
        log(f"left alone {len(others)} other {NAME_PREFIX}* resources this ledger does not list: " + ", ".join(others))
    return errors


# ---- Results ----

def append(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def bottleneck(trial):
    """The resource closest to saturation in a trial, from the tegrastats window of its slowest attempt."""
    measured = worst_attempt(trial).get("tegrastats") or {}
    candidates = {"GPU": measured.get("gpu_percent", {}).get("mean"), "CPU": measured.get("cpu_percent", {}).get("mean")}
    candidates = {key: value for key, value in candidates.items() if value is not None}
    if not candidates:
        return "n/a"
    name = max(candidates, key=candidates.get)
    return f"{name} {candidates[name]:.0f}%"


def supported_capacity(trials):
    """The highest count with no valid failure at or below it, across already-validated searches and rechecks.

    Returns (count, bounded, contested): `bounded` when a trial at count + 1 failed, `contested` when some higher
    count passed in at least one trial; None when no count qualifies.
    """
    passing = {trial["n"] for trial in trials if trial["pass"]}
    first_failure = min((trial["n"] for trial in trials if not trial["pass"]), default=None)
    counts = [n for n in passing if first_failure is None or n < first_failure]
    if not counts:
        return None
    count = max(counts)
    return count, first_failure == count + 1, any(n > count for n in passing)


def summarize(records):
    """Markdown tables: the supported capacity per model and profile, the fixed-N runs, and the AIPs used."""
    for record in records:
        if record["kind"] in ("trial", "fixed", "capacity"):
            validate_live_profile(parse_profile(record["profile"]))
    contexts = {(r.get("device", "historical EX5"), r.get("protocol", "historical"))
                for r in records if r["kind"] in ("trial", "fixed", "capacity")}
    if len(contexts) > 1:
        return "\n\n".join(
            f"Device: {device}; protocol: {protocol}\n" + summarize([
                r for r in records if (r.get("device", "historical EX5"), r.get("protocol", "historical")) ==
                (device, protocol)]) for device, protocol in sorted(contexts))
    capacities = [record for record in records if record["kind"] == "capacity"]
    lines = [f"Device: {device}; protocol: {protocol}" for device, protocol in sorted(contexts)]

    def identity(record):
        aip = record.get("aip") or {}
        return (record["model"], str(aip.get("unique_id") or "unknown"),
                str(aip.get("version") or "unknown"), record.get("model_verified", False))

    evidence, fixed, invalid_count = {}, [], 0
    for record in records:
        if record["kind"] in ("trial", "fixed"):
            trial = record["trial"] if record["kind"] == "fixed" else record
            try:
                require_valid_trial(trial)
            except CapacityError:
                valid = False
                invalid_count += 1
            else:
                valid = True
                key = (record.get("camera_tag", "unknown"), identity(record), record["profile"])
                evidence.setdefault(key, []).append(trial)
            if record["kind"] == "fixed":
                fixed.append((record, valid))
    if invalid_count:
        lines += [f"Excluded {invalid_count} invalid trial(s) with API/setup/cleanup errors from capacity calculations."]
    contested_any = False
    for camera_tag in sorted({record.get("camera_tag", "unknown") for record in capacities}):
        profiles = sorted({record["profile"] for record in capacities
                           if record.get("camera_tag", "unknown") == camera_tag},
                          key=lambda name: (name != "base", len(name), name))
        lines += [f"\nCamera set: {camera_tag}", "", "| Model | AIP | " + " | ".join(profiles) + " |",
                  "|---|---|" + "---:|" * len(profiles)]
        for model, unique_id, version, verified in sorted({key[1] for key in evidence if key[0] == camera_tag
                                                           and key[2] in profiles}):
            cells = []
            for profile in profiles:
                trials = evidence.get((camera_tag, (model, unique_id, version, verified), profile), [])
                found = supported_capacity(trials)
                if found is None:
                    cells.append("-" if not trials else f"Not qualified at {min(trial['n'] for trial in trials)}")
                    continue
                count, bounded, contested = found
                contested_any |= contested
                at_count = [trial for trial in trials if trial["n"] == count]
                cells.append(f"{'' if bounded else '≥'}{count}{'†' if contested else ''}"
                             f" ({bottleneck(at_count[-1])})")
            label = model + ("" if verified else " (unverified)")
            lines.append(f"| {label} | {unique_id} {version} | " + " | ".join(cells) + " |")
    if capacities:
        lines += ["", "Highest count with no failing trial at or below it, from searches and rechecks; "
                  "≥ marks a count without a failing trial at the next count.",
                  "Not qualified at N means the checks did not establish a passing capacity at that count, "
                  "not that N cameras are supported. - means no measurements."]
        if contested_any:
            lines.append("† A higher count passed in some attempts and failed in others.")
    if fixed:
        lines += ["", "| Model | AIP | Camera set | Profile | Cameras | Min processed FPS | All checks pass | GPU % | CPU % | NVENC MHz | Highest mean utilization |",
                  "|---|---|---|---|---:|---:|---|---:|---:|---:|---|"]
        for record, valid in fixed:
            trial = record["trial"]
            measured = worst_attempt(trial).get("tegrastats") or {}
            model, unique_id, version, verified = identity(record)
            lines.append("| {} | {} | {} | {} | {} | {:.2f} | {} | {} | {} | {} | {} |".format(
                model + ("" if verified else " (unverified)"), f"{unique_id} {version}",
                record.get("camera_tag", "unknown"), record["profile"], trial["n"], trial["min_fps"],
                "invalid" if not valid else "yes" if trial["pass"] else "no",
                *(measured.get(key, {}).get("mean", "-") for key in ("gpu_percent", "cpu_percent", "nvenc_mhz")),
                bottleneck(trial)))
        lines += ["", "Processed FPS is not delivered-video FPS. Passing requires all checks for that protocol; "
                  "aggregate CPU/GPU utilization does not identify the stage causing a failure."]
    aips = sorted({identity(record)
                   for record in records if record["kind"] in ("capacity", "fixed") and record.get("aip")})
    if aips:
        lines += ["", "AIPs: " + "; ".join(f"{model} = {unique_id} {version}"
                                           + ("" if verified else " (model not verified)")
                                           for model, unique_id, version, verified in aips)]
    return "\n".join(lines)


# ---- Published tables (the comparison notebook quotes these) ----

def describe(profile):
    """A supported profile in words, for example 'WebRTC + OSD, 1 Mbps'."""
    config = parse_profile(profile)
    validate_live_profile(config)
    outputs = [label for label, on in (("WebRTC", config.webrtc), ("Smart Video recording", config.eventrec)) if on]
    text = " + ".join(outputs) or "no output"
    if config.osd:
        text += " + OSD"
    if config.bitrate != DEFAULT_BITRATE:
        text += f", {config.bitrate // 1_000_000} Mbps"
    if config.tiler != DEFAULT_TILER:
        text += f", {config.tiler[0]}x{config.tiler[1]} mosaic"
    return text


def capitalized(text):
    return text[:1].upper() + text[1:]


def valid_trials(records):
    """Searched and fixed-count trials without setup, API or cleanup errors, by (camera set, model, profile)."""
    grouped = {}
    contexts = set()
    for record in records:
        if record["kind"] not in ("trial", "fixed"):
            continue
        validate_live_profile(parse_profile(record["profile"]))
        trial = record["trial"] if record["kind"] == "fixed" else record
        try:
            require_valid_trial(trial)
        except CapacityError:
            continue
        contexts.add((record.get("device", "historical EX5"), record.get("protocol", "historical")))
        if len(contexts) > 1:
            raise CapacityError("select one device and comparison protocol before generating a published table")
        key = (record.get("camera_tag", "unknown"), record["model"], record["profile"])
        grouped.setdefault(key, []).append(trial)
    return grouped


def camera_table(records, models=MODELS, columns=None):
    """Supported camera count per model and (camera set, profile), as in the capacity report's answer."""
    columns = columns or [("720p", "base"), *(("720p", profile) for profile in CAPACITY_PROFILES), ("1080p", "base")]
    grouped = valid_trials(records)
    lines = ["| Model | " + " | ".join(f"{camera_tag}, {describe(profile)}" for camera_tag, profile in columns) + " |",
             "|---|" + "---:|" * len(columns)]
    contested_any = False
    for model in models:
        cells = []
        for column in columns:
            trials = grouped.get((column[0], model, column[1]), [])
            found = supported_capacity(trials)
            if found is None:
                cells.append("-" if not trials else f"Not qualified at {min(trial['n'] for trial in trials)}")
                continue
            count, bounded, contested = found
            contested_any |= contested
            cells.append(f"{'' if bounded else '≥'}{count}{'†' if contested else ''}")
        lines.append(f"| {MODEL_NAMES[model]} | " + " | ".join(cells) + " |")
    if contested_any:
        lines += ["", "† A higher count passed in some attempts and failed in others; the table keeps the lower count."]
    return "\n".join(lines)


def headroom_table(records, models=MODELS, camera_tag="720p", profiles=None):
    """Slowest-camera FPS of every trial at the first failing count, with and without outputs."""
    profiles = profiles or ("base", *CAPACITY_PROFILES)
    grouped = valid_trials(records)
    lines = ["| Model | Cameras | " + " | ".join(capitalized(describe(profile)) for profile in profiles) + " |",
             "|---|---:|" + "---|" * len(profiles)]
    for model in models:
        evidence = [grouped.get((camera_tag, model, profile), []) for profile in profiles]
        found = [supported_capacity(trials) for trials in evidence]
        if None in found:
            raise CapacityError(f"{model}: no supported count for every profile")
        failing = {count + 1 for count, _, _ in found}
        if len(failing) != 1:
            raise CapacityError(f"{model}: the profiles fail at different counts {sorted(failing)}")
        n = failing.pop()
        cells = [", ".join(f"{trial['min_fps']:.2f}" + (" (passed)" if trial["pass"] else "")
                           for trial in trials if trial["n"] == n) or "-" for trials in evidence]
        lines.append(f"| {MODEL_NAMES[model]} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def screening_table(records, stage="screening", profiles=None):
    """One row per fixed-count screening run: which outputs, the slowest camera and the GPU load."""
    profiles = SCREENING if profiles is None else profiles
    runs = [record for record in records if record["kind"] == "fixed" and record.get("stage") == stage
            and record["profile"] in profiles]
    lines = ["| Model | Cameras | Outputs | Slowest camera FPS | Mean GPU busy |", "|---|---:|---|---:|---:|"]
    for record in runs:
        trial = record["trial"]
        require_valid_trial(trial)
        gpu = (worst_attempt(trial).get("tegrastats") or {}).get("gpu_percent", {}).get("mean")
        lines.append(f"| {MODEL_NAMES[record['model']]} | {trial['n']} | {capitalized(describe(record['profile']))} | "
                     f"{trial['min_fps']:.2f}{'' if trial['pass'] else ' (failed)'} | "
                     f"{'n/a' if gpu is None else f'{gpu:.1f}%'} |")
    return "\n".join(lines)


# ---- The plan ----

def full_factorial():
    """Every combination of the factors the question lists, before and after removing the ones that cannot differ."""
    cells = set()
    total = 0
    for osd, delivery, eventrec, bitrate, tiler in product(
            FACTORS["osd"], FACTORS["delivery"], FACTORS["Smart Video recording"],
            FACTORS["delivery bitrate"], FACTORS["output resolution"]):
        total += 1
        cells.add(normalize(Config(webrtc=delivery == "webrtc",
                                   eventrec=eventrec, osd=osd, bitrate=bitrate, tiler=tiler)))
    cameras, models = len(FACTORS["camera resolution"]), len(MODELS)
    return {"raw": total * cameras * models, "distinct": len(cells) * cameras * models}


SCREENING = ("base", "webrtc", "webrtc-osd", "evt", "evt-osd", "webrtc+evt-osd",
             "webrtc-osd-1M", "webrtc-osd-4M", "webrtc-osd-640x360", "webrtc-osd-1920x1080")
MATCHED_PROFILES = (*SCREENING, "webrtc+evt")
CAPACITY_PROFILES = ("webrtc-osd", "evt-osd", "webrtc+evt-osd")
CAMERA_PROBES = 2


def plan():
    """The staged, pruned plan as rows of (stage, what, searches, single runs)."""
    return [
        ("0 Pilot", "One camera per model and profile: inference, WebRTC publisher, Smart Video clips",
         0, len(MODELS) * (1 + len(CAPACITY_PROFILES))),
        ("1 Baseline", "Capacity search per model with no output (decode + inference + tracker)", len(MODELS), 0),
        ("2 Screening", f"One reference model at its baseline capacity: {len(SCREENING)} single runs, one factor at a "
                        "time (outputs, OSD, bitrate, output resolution)", 0, len(SCREENING)),
        ("3 Capacity", "Capacity search per model: WebRTC, Smart Video, and both (OSD on)",
         len(MODELS) * len(CAPACITY_PROFILES), 0),
        ("4 Check", "Independent rechecks at each passing boundary and the next camera count",
         0, 2 * len(MODELS) * (1 + len(CAPACITY_PROFILES))),
        ("5 Cameras", "Camera resolution probes (720p against 1080p) at baseline", 0, CAMERA_PROBES),
    ]


def plan_totals():
    rows = plan()
    searches = sum(row[2] for row in rows)
    trials = searches * SEARCH_TRIALS + sum(row[3] for row in rows)
    return {"searches": searches, "trials": trials, "hours": round(trials * TRIAL_MINUTES / 60, 1)}


def print_plan():
    factorial = full_factorial()
    print("Per device: EX5 and EX3 must be measured separately. No HLS or continuous recording.")
    print(f"Smart Video: {SMART_PRE_MS / 1000:g}s pre-roll, {SMART_POST_MS / 1000:g}s post-roll, "
          f"trigger every {SMART_TRIGGER_SECONDS}s; decode and verify clips after sampling.")
    print("Factors:", ", ".join(f"{name} x{len(values)}" for name, values in FACTORS.items()))
    print(f"Every combination: {factorial['raw']} cells; {factorial['distinct']} once settings that cannot "
          f"differ are merged; each cell is a search of about {SEARCH_TRIALS} trials of about {TRIAL_MINUTES} min "
          f"= about {round(factorial['distinct'] * SEARCH_TRIALS * TRIAL_MINUTES / 60)} h.")
    print()
    for stage, what, searches, runs in plan():
        print(f"{stage:<12} {searches:>2} searches {runs:>2} runs  {what}")
    totals = plan_totals()
    print(f"\nPlanned: {totals['searches']} searches and {totals['trials']} trials in total, "
          f"about {totals['hours']} h of exclusive station time.")


# ---- Command line ----

@contextmanager
def open_session(args, config, cameras, *, exact):
    """A Session with `cameras` registered cameras, or as many as the AIP accepts unless `exact`; deleted on exit."""
    validate_live_profile(config)
    if cameras < 1:
        raise CapacityError("camera count must be positive")
    if (not all(math.isfinite(value) for value in (args.warmup, args.window, args.interval))
            or args.warmup < 0 or args.window <= 0 or not 0 < args.interval <= args.window):
        raise CapacityError("timings must be finite: warmup >= 0 and 0 < interval <= window")
    if getattr(args, "first", None) is not None and not 1 <= args.first <= cameras:
        raise CapacityError("--first must be within the camera search range")
    if config.eventrec and shutil.which(args.ffprobe) is None:
        raise CapacityError("Smart Video trials require ffprobe to decode and verify the resulting clips")
    live_ffprobe = args.live_ffprobe or args.ffprobe
    if config.webrtc and shutil.which(live_ffprobe) is None:
        raise CapacityError("WebRTC trials require ffprobe to read the delivered stream")
    ms3 = Api(args.ms3)
    preflight(ms3, config)
    aip = find_aip(ms3, args.aip)
    verified = model_matches(aip, args.model)
    if not verified:
        primary = aip.get("primary_processor") or {}
        message = (f"AIP {args.aip} {aip.get('version')} is named {aip.get('name')!r} with model "
                   f"{primary.get('model')!r}, not {MODEL_NAMES[args.model]}")
        if not args.unverified_model:
            raise CapacityError(message + "; pass --unverified-model to measure it anyway")
        print(f"warning: {message}; results are marked model_verified=false")
    limit = int(aip.get("maximum_input_streams") or cameras)
    if limit < 1:
        raise CapacityError("AIP maximum_input_streams must be positive")
    if exact and cameras > limit:
        raise CapacityError(f"AIP {args.aip} accepts at most {limit} inputs")
    ledger = Ledger(ledger_path(args.out))
    if any(ledger.read().values()):
        raise CapacityError(f"undeleted resources in {ledger.path}; complete cleanup before another run")
    # Fail before creating station resources if the results cannot be persisted.
    with open(args.out, "a"):
        pass
    with Tegrastats() as tegrastats:
        session = Session(ms3, aip, args.model, ledger, verified=verified, device=args.device, ffprobe=args.ffprobe,
                          live_ffprobe=live_ffprobe,
                          warmup=args.warmup, window=args.window, interval=args.interval,
                          containers=tuple(name for name in args.containers.split(",") if name),
                          tegrastats=tegrastats)
        try:
            session.create_media(args.rtsp_template, min(cameras, limit), args.camera_tag)
            yield session
        finally:
            errors = session.close()
            if errors:
                raise CapacityError(f"cleanup incomplete: {'; '.join(errors)}. "
                                    f"Run cleanup with --ledger {ledger.path}")


def command_search(args):
    config = parse_profile(args.profile)
    with open_session(args, config, args.max_n, exact=False) as session:
        def trial(count):
            result = settled(lambda n: session.trial(n, config), count)
            result["stage"] = args.stage
            result["camera_tag"] = args.camera_tag
            append(args.out, result)
            require_valid_trial(result)
            return result
        max_n = len(session.media)
        capacity, bounded, results = search_capacity(trial, max_n, first=args.first)
    summary = {"kind": "capacity", "stage": args.stage, "model": args.model, "profile": config.name,
               "camera_tag": args.camera_tag, **session.provenance(), "capacity": capacity, "bounded": bounded,
               "max_n": max_n, "trials": [without_samples(result) for result in results]}
    append(args.out, summary)
    print(f"{args.model} {config.name}: {'at least ' if not bounded else ''}{capacity} cameras at {PASS_FPS:g}+ FPS")


def command_fixed(args):
    config = parse_profile(args.profile)
    with open_session(args, config, args.n, exact=True) as session:
        trial = session.trial(args.n, config)
        append(args.out, {"kind": "fixed", "stage": args.stage, "model": args.model, "profile": config.name,
                          "camera_tag": args.camera_tag, **session.provenance(), "trial": trial})
        require_valid_trial(trial)


def command_matched(args):
    source = matched_source(args.live_ffprobe or args.ffprobe, args.rtsp_template, args.source_sha256)
    profiles = (args.profile,) if args.profile else MATCHED_PROFILES
    for profile in profiles:
        config = parse_profile(profile)
        with open_session(args, config, 1, exact=True) as session:
            trial = session.trial(1, config, source=source)
            result = {"kind": "fixed", "stage": "matched-one-camera", "camera_tag": args.camera_tag,
                      "model": args.model, "profile": profile, **session.provenance(), "trial": trial}
        # Camera deletion is confirmed when open_session exits; never publish a success before then.
        append(args.out, result)
        require_valid_trial(trial)


def matched_comparison(left, right):
    """Compare measured cases only; never substitute profile defaults for missing station readbacks."""
    def index(records):
        devices = {row.get("device") for row in records}
        if len(devices) != 1 or not all(devices):
            raise CapacityError("each matched file must contain results from exactly one named device")
        cases, model_versions = {}, {}
        for row in records:
            key = row.get("model"), row.get("profile")
            trial = row.get("trial") or {}
            if (row.get("stage") != "matched-one-camera" or row.get("kind") != "fixed"
                    or key[0] not in MODELS or key[1] not in MATCHED_PROFILES or trial.get("n") != 1
                    or not row.get("model_verified") or not (row.get("aip") or {}).get("version")
                    or row.get("protocol") != PROTOCOL):
                raise CapacityError(f"{key}: not a verified matched one-camera trial")
            version = row["aip"]["version"]
            if model_versions.setdefault(key[0], version) != version:
                raise CapacityError(f"{key}: AIP version changed between profiles for this model")
            require_valid_trial(trial)
            setup = trial.get("matched_setup") or {}
            source = setup.get("source") or {}
            config = parse_profile(key[1])
            pipeline = pipeline_setup("matched", config, {"pk": ""}, [""], matched=True)["pipeline"]
            if (not re.fullmatch(r"[0-9a-f]{64}", str(source.get("file_sha256", "")))
                    or any(source.get(k) != value for k, value in MATCHED_INPUT.items())
                    or setup.get("pipeline") != matched_parameters(pipeline)):
                raise CapacityError(f"{key}: source identity or pipeline settings do not match the specification")
            timings = tuple(trial.get(k) for k in ("warmup_seconds", "window_seconds", "interval_seconds"))
            if timings != (WARMUP_SECONDS, WINDOW_SECONDS, SAMPLE_SECONDS):
                raise CapacityError(f"{key}: sampling times differ from the matched protocol")
            delivery = trial.get("webrtc_delivery") or {}
            smart = trial.get("smart_video") or {}
            if bool(delivery) != config.webrtc or bool(smart.get("clips")) != config.eventrec:
                raise CapacityError(f"{key}: observed outputs differ from the requested profile")
            if config.webrtc and (delivery.get("codec"), delivery.get("width"), delivery.get("height")) != (
                    "h264", *config.tiler):
                raise CapacityError(f"{key}: publisher format differs from the requested mosaic")
            if config.eventrec:
                recording_times = tuple(smart.get(k) for k in (
                    "pre_duration_ms", "post_duration_ms", "max_duration_ms", "trigger_interval_seconds"))
                if recording_times != (SMART_PRE_MS, SMART_POST_MS, SMART_MAX_MS, SMART_TRIGGER_SECONDS):
                    raise CapacityError(f"{key}: recording times differ from the matched protocol")
                if any((c.get("codec"), c.get("width"), c.get("height")) != ("h264", *config.tiler)
                       for c in smart["clips"]):
                    raise CapacityError(f"{key}: clip format differs from the requested mosaic")
            if key in cases:
                raise CapacityError(f"{key}: duplicate case; compare one repetition at a time")
            cases[key] = row
        return devices.pop(), cases

    a, b = index(left), index(right)
    if a[0] == b[0] or a[1].keys() != b[1].keys():
        raise CapacityError("two different devices with the same model/profile cases are required")
    lines = [f"{len(a[1])} of {len(MODELS) * len(MATCHED_PROFILES)} matched one-camera cases; "
             "one run per station in each case.", "",
             f"| Model | Profile | {a[0]} | {b[0]} |", "|---|---|---|---|"]
    source = next(iter(a[1].values()))["trial"]["matched_setup"]["source"]
    for model, profile in product(MODELS, MATCHED_PROFILES):
        if (model, profile) not in a[1]:
            continue
        config = parse_profile(profile)
        first, second = (cases[(model, profile)] for _, cases in (a, b))
        trials = [row["trial"] for row in (first, second)]
        if (first["aip"]["version"] != second["aip"]["version"]
                or trials[0]["matched_setup"] != trials[1]["matched_setup"]
                or trials[0]["matched_setup"]["source"] != source):
            raise CapacityError(f"{model} {profile}: source, pipeline settings or AIP version differ")
        cells = []
        for trial in trials:
            parts = ["Pass" if trial["pass"] else "Fail", f"{trial['min_fps']:.2f} processed FPS"]
            if config.webrtc:
                delivery = trial["webrtc_delivery"]
                parts.append(f"{delivery['delivered_fps']:.2f} delivered FPS, "
                             f"{delivery['max_gap_seconds'] * 1000:.0f} ms max gap")
            if config.eventrec:
                clips = trial["smart_video"]["clips"]
                passed = sum(clip["fps_pass"] and clip["continuity_pass"] for clip in clips)
                parts.append(f"{passed}/{len(clips)} clips passed")
            cells.append("; ".join(parts))
        lines.append(f"| {MODEL_NAMES[model]} | `{profile}` | {' | '.join(cells)} |")
    return "\n".join(lines)


def command_summarize(args):
    print(summarize(load(args.results)))


def command_cleanup(args):
    signaling = Api(args.signaling) if args.signaling else None
    errors = cleanup(Api(args.ms3), signaling, Ledger(args.ledger))
    if errors:
        raise CapacityError("cleanup incomplete: " + "; ".join(errors))


def command_plan(args):
    print_plan()


def command_preflight(args):
    config = parse_profile(args.profile)
    validate_live_profile(config)
    preflight(Api(args.ms3), config)
    print("OK: required API routes exist and existing pipelines are stopped. "
          "Fresh per-camera metrics and working outputs still require a deployment pilot.")


def parser():
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="print the pruned plan and what it saves").set_defaults(run=command_plan)
    probe = commands.add_parser("preflight", help="check station routes and workload without creating resources")
    probe.add_argument("--ms3", required=True)
    probe.add_argument("--profile", default="webrtc+evt-osd")
    probe.set_defaults(run=command_preflight)

    def station(sub, *, matched=False):
        sub.add_argument("--ms3", required=True, help="video-streaming API base URL, e.g. http://MS3:8000")
        sub.add_argument("--device", required=True, help="public hardware/power-mode label, e.g. ex3-orin-nano-8gb-15w")
        sub.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable for Smart Video clip validation")
        sub.add_argument("--live-ffprobe", help="ffprobe that can reach MediaMTX inside MS3 to read the WebRTC "
                         "slot (default: --ffprobe)")
        sub.add_argument("--model", required=True, choices=MODELS)
        sub.add_argument("--aip", required=True, help="unique ID of the AIP built for this model")
        sub.add_argument("--out", required=True,
                         help="JSON Lines results file; its ledger of undeleted resources is OUT.owned.json")
        if matched:
            sub.add_argument("--source", dest="rtsp_template", metavar="RTSP_URL", required=True,
                             help="the same looped 720p30 RTSP feed on both devices")
            sub.add_argument("--source-sha256", required=True,
                             help="SHA-256 of the encoded file in the stream-copy publisher")
            sub.add_argument("--profile", choices=MATCHED_PROFILES, help="run only this case; default: all matched cases")
            sub.set_defaults(warmup=WARMUP_SECONDS, window=WINDOW_SECONDS, interval=SAMPLE_SECONDS,
                             unverified_model=False, camera_tag="720p", containers=",".join(CONTAINERS))
            return
        sub.add_argument("--unverified-model", action="store_true",
                         help="measure even when the AIP's name and model do not match --model")
        sub.add_argument("--rtsp-template", required=True, help="camera URL with an {index} placeholder")
        sub.add_argument("--camera-tag", default="720p",
                         help="label for the camera set (media are named cap-RUN-TAG-NN)")
        sub.add_argument("--profile", default="base",
                         help="base, webrtc-osd, evt-osd (Smart Video), or webrtc+evt-osd")
        sub.add_argument("--max-n", type=int, default=DEFAULT_MAX_N)
        sub.add_argument("--warmup", type=float, default=WARMUP_SECONDS, help="seconds to settle before sampling")
        sub.add_argument("--window", type=float, default=WINDOW_SECONDS, help="seconds of sampling")
        sub.add_argument("--interval", type=float, default=SAMPLE_SECONDS, help="seconds between samples")
        sub.add_argument("--containers", default=",".join(CONTAINERS),
                         help="containers whose CPU is recorded (empty: none)")
        sub.add_argument("--stage", default="", help="label stored with the results")

    search = commands.add_parser("search", help="find the largest passing camera count")
    station(search)
    search.add_argument("--first", type=int, help="first camera count to try (default: the middle of the range)")
    search.set_defaults(run=command_search)
    fixed = commands.add_parser("fixed", help="run one camera count")
    station(fixed)
    fixed.add_argument("--n", type=int, required=True)
    fixed.set_defaults(run=command_fixed)
    matched = commands.add_parser("matched", help="run the same one-camera pipeline settings on each station")
    station(matched, matched=True)
    matched.set_defaults(run=command_matched)
    paired = commands.add_parser("compare-matched", help="check paired settings and print measured results")
    paired.add_argument("left", help="first station's matched JSONL")
    paired.add_argument("right", help="second station's matched JSONL")
    paired.set_defaults(run=lambda args: print(matched_comparison(load(args.left), load(args.right))))
    summary = commands.add_parser("summarize", help="Markdown tables from a results file")
    summary.add_argument("results")
    summary.set_defaults(run=command_summarize)
    clean = commands.add_parser("cleanup", help="delete the pipelines and media a ledger lists, and nothing else")
    clean.add_argument("--ms3", required=True)
    clean.add_argument("--signaling", help="signaling API base URL; needed when the ledger lists recording pipelines")
    clean.add_argument("--ledger", required=True, help="the OUT.owned.json file next to a results file")
    clean.set_defaults(run=command_cleanup)
    return top


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        args.run(args)
    except (CapacityError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
