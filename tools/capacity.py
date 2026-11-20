#!/usr/bin/env python3
"""Find how many RTSP cameras a station sustains at 30 FPS, per model and output profile.

The harness drives the station's video-streaming API (ms3). For every trial it creates a pipeline with N
RTSP inputs, starts the outputs the profile asks for, waits for the pipeline to settle, samples the per-input
processed rate for a fixed window and tears everything down again. A trial passes when every camera averages at
least 29 FPS and no camera reconnects. `search` finds the largest passing N with a binary search; `fixed` runs
one N, which is how the cheap one-factor screening runs are done. Plain stdlib, so it runs as it is on the station:

    python3 tools/capacity.py plan
    python3 tools/capacity.py search --ms3 http://MS3:8000 --model yolo26 --aip AIPV1EMI9026 \
        --rtsp-template 'rtsp://VCAM:PORT/cam{index:02d}' --profile base --out capacity.jsonl
    python3 tools/capacity.py fixed  ... --n 8 --profile rec+webrtc-osd --signaling http://SIGNALING:3001/signaling
    python3 tools/capacity.py summarize capacity.jsonl
    python3 tools/capacity.py cleanup --ms3 http://MS3:8000 --ledger capacity.jsonl.owned.json

doc/benchmarks/capacity/README.md explains why this is a pruned plan and not every combination. Every run gets its
own ID, creates its cameras and pipelines under it and lists them in a ledger next to the results file until they are
deleted again; `cleanup` deletes what a ledger lists and nothing else.
"""
import argparse
import json
import math
import re
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
from dataclasses import dataclass, replace
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
RECORDING_BITRATE = 3_000_000
EVENTREC_BITRATE = 4_000_000
DEFAULT_TILER = (1280, 720)
NAME_PREFIX = "cap-"
# ms3 MTX_SLOT_ROLES: slot 1 records, slot 2 serves WebRTC, slot 3 is RTMP. The signaling service starts slot 1
# itself when it starts continuous recording.
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
    "recording": (False, True),
    "delivery": ("none", "webrtc", "hls"),
    "event recording": (False, True),
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


def pipeline_setup(name, config, aip, media_uuids, *, tiler=None):
    """The request body for POST /pipeline. `aip` is an /aiprocessors record."""
    count = len(media_uuids)
    width, height = tiler or config.tiler
    mtx = config.recording or config.webrtc
    return {"pipeline": {
        "name": name,
        "description": "Capacity test created by tools/capacity.py",
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
        "hls_output_sink_setup": {"active": config.hls, "bitrate": config.bitrate, "max_files": 12,
                                  "playlist_len": 6, "target_duration": 1},
        "mtx_output_sink_setup": ({"active": True, "num_slots": 2 if config.webrtc else 1,
                                   "recording_enabled": config.recording, "webrtc_enabled": config.webrtc,
                                   "rtmp_enabled": False, "recording_bitrate": RECORDING_BITRATE,
                                   "webrtc_bitrate": config.bitrate} if mtx else {"active": False}),
        "eventrec_sink_setup": {"active": config.eventrec, "osd": config.osd, "bitrate": EVENTREC_BITRATE},
        "ntp_mode": "auto",
    }}


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
    combined = {key: first[key] for key in ("kind", "model", "profile", "run", "aip", "model_verified", "fps_metric")
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
            raise CapacityError("trial invalid; search stopped: " + "; ".join(errors))


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


def start_outputs(ms3, signaling, uuid_, config):
    """Start the outputs that stay off until asked: WebRTC, and continuous recording the way the station UI does it."""
    if config.webrtc:
        retrying(lambda: ms3.call("POST", f"/pipeline/{uuid_}/mtx/start", query={"stream_index": WEBRTC_SLOT}))
    if config.recording:
        retrying(lambda: signaling.call("POST", "/recordings/start", body={"pipeline_id": uuid_}),
                 statuses=(409, 500, 502, 503))


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

    attempt("stopping the pipeline", stop_pipeline)
    if config.eventrec:
        attempt("deleting event recordings",
                lambda: ms3.call("POST", "/pipeline/records/deletion", query={"pipeline_uuid": uuid_},
                                 ok=GONE_OK))
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
                 log=print):
        self.ms3, self.signaling, self.aip, self.model, self.ledger = ms3, signaling, aip, model, ledger
        self.verified = verified
        self.run = run or new_run_id()
        self.warmup, self.window, self.interval = warmup, window, interval
        self.containers, self.tegrastats, self.log = containers, tegrastats, log
        self.media = []
        self.trials = 0

    def provenance(self):
        return {"run": self.run, "aip": aip_identity(self.aip), "model_verified": self.verified,
                "fps_metric": PERF_METRIC}

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

    def sample(self, uuid_, config, samples):
        """Append samples as they arrive, retaining partial evidence if polling fails."""
        begin, clock = time.time(), time.monotonic()
        trigger_at = clock
        while time.monotonic() - clock < self.window:
            if config.eventrec and time.monotonic() >= trigger_at:
                try:
                    self.ms3.call("POST", f"/pipeline/{uuid_}/eventrec/trigger")
                except CapacityError as error:
                    raise CapacityError(f"event recording trigger failed: {error}") from error
                trigger_at += 30
            sample = sample_from(self.ms3.call("GET", f"/pipeline/{uuid_}/inputs/fps"),
                                 self.ms3.call("GET", f"/pipeline/{uuid_}/fps"),
                                 self.ms3.call("GET", f"/pipeline/{uuid_}/status"),
                                 processed_fps(self.ms3.call("GET", "/metrics"), uuid_))
            sample["t"] = round(time.monotonic() - clock, 3)
            samples.append(sample)
            time.sleep(min(self.interval, max(0, self.window - (time.monotonic() - clock))))
        return begin, time.time()

    def trial(self, count, config):
        """One pipeline with `count` cameras: start, settle, sample, tear down. Returns the trial record."""
        self.trials += 1
        name = f"{NAME_PREFIX}{self.run}-{self.model}-{config.name}-n{count}-t{self.trials:02d}"
        record = {"kind": "trial", "model": self.model, "profile": config.name, "n": count, "pipeline": name,
                  "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **self.provenance(),
                  "warmup_seconds": self.warmup, "window_seconds": self.window,
                  "interval_seconds": self.interval, "samples": []}
        uuid_ = None
        try:
            if not 1 <= count <= len(self.media):
                raise CapacityError(f"{count} cameras asked for but this run registered {len(self.media)}")
            uuid_ = self.ms3.call("POST", "/pipeline",
                                  body=pipeline_setup(name, config, self.aip, self.media[:count]))["pk"]
            self.ledger.add("pipelines", uuid_, name=name, profile=config.name, run=self.run)
            self.ms3.call("POST", f"/pipeline/{uuid_}/start")
            wait_for_inputs(self.ms3, uuid_, count)
            start_outputs(self.ms3, self.signaling, uuid_, config)
            self.log(f"  n={count} {config.name}: settling for {self.warmup:g} s")
            time.sleep(self.warmup)
            samples = record["samples"]
            begin, end = self.sample(uuid_, config, samples)
            record["sample_start_unix"], record["sample_end_unix"] = begin, end
            record["containers_cpu_percent"] = container_cpu(self.containers)
            record.update(evaluate(samples, count))
            if not samples:
                raise CapacityError("sampling window produced no observations")
            record["pipeline_fps_mean"] = round(statistics.fmean(sample["pipeline_fps"] for sample in samples), 2)
            if self.tegrastats is not None:
                record["tegrastats"] = self.tegrastats.between(begin, end)
            if config.recording:
                record["recording_files"] = len(recording_files(self.signaling, uuid_))
                if not record["recording_files"]:
                    raise CapacityError("continuous recording wrote no file")
        except CapacityError as error:
            record.update(evaluate(record["samples"], count))
            record.update({"pass": False, "error": str(error)})
            record["reasons"].append(f"trial failed: {error}")
        finally:
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
    capacities = [record for record in records if record["kind"] == "capacity"]
    lines = []

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
                    cells.append("-" if not trials else "<" + str(min(trial["n"] for trial in trials)))
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
                  "≥ marks a count without a failing trial at the next count."]
        if contested_any:
            lines.append("† A higher count passed in at least one trial but failed in another.")
    if fixed:
        lines += ["", "| Model | AIP | Camera set | Profile | Cameras | Min FPS | Pass | GPU % | CPU % | NVENC MHz | Bottleneck |",
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
    aips = sorted({identity(record)
                   for record in records if record["kind"] in ("capacity", "fixed") and record.get("aip")})
    if aips:
        lines += ["", "AIPs: " + "; ".join(f"{model} = {unique_id} {version}"
                                           + ("" if verified else " (model not verified)")
                                           for model, unique_id, version, verified in aips)]
    return "\n".join(lines)


# ---- The plan ----

def full_factorial():
    """Every combination of the factors the question lists, before and after removing the ones that cannot differ."""
    cells = set()
    total = 0
    for osd, recording, delivery, eventrec, bitrate, tiler in product(
            FACTORS["osd"], FACTORS["recording"], FACTORS["delivery"], FACTORS["event recording"],
            FACTORS["delivery bitrate"], FACTORS["output resolution"]):
        total += 1
        cells.add(normalize(Config(recording=recording, webrtc=delivery == "webrtc", hls=delivery == "hls",
                                   eventrec=eventrec, osd=osd, bitrate=bitrate, tiler=tiler)))
    cameras, models = len(FACTORS["camera resolution"]), len(MODELS)
    return {"raw": total * cameras * models, "distinct": len(cells) * cameras * models}


SCREENING = ("base", "rec", "webrtc", "webrtc-osd", "hls", "hls-osd", "evt", "evt-osd", "rec+webrtc-osd",
             "webrtc-osd-1M", "webrtc-osd-4M", "webrtc-osd-640x360", "webrtc-osd-1920x1080")
CAPACITY_PROFILES = ("rec+webrtc-osd", "rec+webrtc+hls+evt-osd")
CAMERA_PROBES = 2


def plan():
    """The staged, pruned plan as rows of (stage, what, searches, single runs)."""
    return [
        ("0 Pilot", "One camera, one model: deployed-form AIP, parser, outputs and harness work", 0, 2),
        ("1 Baseline", "Capacity search per model with no output (decode + inference + tracker)", len(MODELS), 0),
        ("2 Screening", f"One reference model at its baseline capacity: {len(SCREENING)} single runs, one factor at a "
                        "time (outputs, OSD, bitrate, output resolution)", 0, len(SCREENING)),
        ("3 Capacity", f"Capacity search per model for {len(CAPACITY_PROFILES)} profiles: typical and worst case",
         len(MODELS) * len(CAPACITY_PROFILES), 0),
        ("4 Check", "Predict the other profiles from the screening, then search the one with the largest gap", 1, 0),
        ("5 Cameras", "Camera resolution probes (720p against 1080p) at baseline", 0, CAMERA_PROBES),
    ]


def plan_totals():
    rows = plan()
    searches = sum(row[2] for row in rows)
    trials = searches * SEARCH_TRIALS + sum(row[3] for row in rows)
    return {"searches": searches, "trials": trials, "hours": round(trials * TRIAL_MINUTES / 60, 1)}


def print_plan():
    factorial = full_factorial()
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
    if cameras < 1:
        raise CapacityError("camera count must be positive")
    if (not all(math.isfinite(value) for value in (args.warmup, args.window, args.interval))
            or args.warmup < 0 or args.window <= 0 or not 0 < args.interval <= args.window):
        raise CapacityError("timings must be finite: warmup >= 0 and 0 < interval <= window")
    if getattr(args, "first", None) is not None and not 1 <= args.first <= cameras:
        raise CapacityError("--first must be within the camera search range")
    ms3 = Api(args.ms3)
    signaling = Api(args.signaling) if args.signaling else None
    if config.recording and signaling is None:
        raise CapacityError("continuous recording is started through the signaling service; pass --signaling")
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
        session = Session(ms3, aip, args.model, ledger, signaling=signaling, verified=verified,
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


def command_summarize(args):
    print(summarize(load(args.results)))


def command_cleanup(args):
    signaling = Api(args.signaling) if args.signaling else None
    errors = cleanup(Api(args.ms3), signaling, Ledger(args.ledger))
    if errors:
        raise CapacityError("cleanup incomplete: " + "; ".join(errors))


def command_plan(args):
    print_plan()


def parser():
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="print the pruned plan and what it saves").set_defaults(run=command_plan)

    def station(sub):
        sub.add_argument("--ms3", required=True, help="video-streaming API base URL, e.g. http://MS3:8000")
        sub.add_argument("--signaling", help="signaling API base URL, e.g. http://SIGNALING:3001/signaling; "
                                             "needed by profiles with continuous recording (rec)")
        sub.add_argument("--model", required=True, choices=MODELS)
        sub.add_argument("--aip", required=True, help="unique ID of the AIP built for this model")
        sub.add_argument("--unverified-model", action="store_true",
                         help="measure even when the AIP's name and model do not match --model")
        sub.add_argument("--rtsp-template", required=True, help="camera URL with an {index} placeholder")
        sub.add_argument("--camera-tag", default="1080p",
                         help="label for the camera set (media are named cap-RUN-TAG-NN)")
        sub.add_argument("--profile", default="base", help="output profile, for example base or rec+webrtc-osd")
        sub.add_argument("--max-n", type=int, default=DEFAULT_MAX_N)
        sub.add_argument("--warmup", type=float, default=WARMUP_SECONDS, help="seconds to settle before sampling")
        sub.add_argument("--window", type=float, default=WINDOW_SECONDS, help="seconds of sampling")
        sub.add_argument("--interval", type=float, default=SAMPLE_SECONDS, help="seconds between samples")
        sub.add_argument("--containers", default=",".join(CONTAINERS),
                         help="containers whose CPU is recorded (empty: none)")
        sub.add_argument("--stage", default="", help="label stored with the results")
        sub.add_argument("--out", required=True,
                         help="JSON Lines results file; its ledger of undeleted resources is OUT.owned.json")

    search = commands.add_parser("search", help="find the largest passing camera count")
    station(search)
    search.add_argument("--first", type=int, help="first camera count to try (default: the middle of the range)")
    search.set_defaults(run=command_search)
    fixed = commands.add_parser("fixed", help="run one camera count")
    station(fixed)
    fixed.add_argument("--n", type=int, required=True)
    fixed.set_defaults(run=command_fixed)
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
