#!/usr/bin/env python3
"""Comparable raw-inference benchmark: ONNX Runtime CUDA FP32 versus TensorRT FP32/FP16.

Every backend runs in one Python process, on the same NMS-free graph, the same
GPU-resident input and the same timing boundary per request:

    cudaDeviceSynchronize -> perf_counter_ns -> one inference call -> cudaDeviceSynchronize

Usage (from the repository root, inside the station's Jupyter container):

    python3 tools/benchmark.py raw-onnx SOURCE.onnx RAW.onnx
    python3 tools/benchmark.py run --onnx RAW.onnx \\
        --engine "TensorRT FP32=raw-fp32.engine" --engine "TensorRT FP16=raw-fp16.engine" \\
        --output RESULTS_DIR

Build the two engines from RAW.onnx with trtexec first (see localdev.ipynb, section 3).
This tool never restarts services, changes clocks or modifies the deployment graph.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BATCHES = (1, 4, 8)
WARMUP_REQUESTS, MEASURED_REQUESTS, ROUNDS = 20, 50, 3
OUTPUTS = {"raw_boxes": 4, "raw_scores": 3}
CANDIDATES = 25200
ORT_LABEL = "ONNX Runtime CUDA FP32"
DEFAULT_TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_raw_onnx(source, destination):
    """Copy the deployment ONNX without its terminal EfficientNMS_TRT node.

    The copy exposes the NMS inputs as raw_boxes [batch, candidates, 4] and
    raw_scores [batch, candidates, 3]. Weights and the source file are unchanged.
    """
    import onnx
    from onnx import TensorProto, helper, shape_inference

    source, destination = Path(source), Path(destination)
    source_sha = sha256(source)
    graph = onnx.load(source)
    nms_nodes = [n for n in graph.graph.node if n.op_type == "EfficientNMS_TRT"]
    assert len(nms_nodes) == 1, "Expected exactly one terminal NMS node"
    nms = nms_nodes[0]
    assert set(nms.output) == {v.name for v in graph.graph.output}
    assert len(nms.input) == 2, "Expected boxes and scores"
    raw_names = list(nms.input)
    graph.graph.node.remove(nms)
    del graph.graph.output[:]
    graph.graph.output.extend([
        helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", "candidates", width])
        for name, width in zip(raw_names, OUTPUTS.values())
    ])
    graph = shape_inference.infer_shapes(graph)
    renames = dict(zip(raw_names, OUTPUTS))
    for node in graph.graph.node:
        for field in (node.input, node.output):
            for index, name in enumerate(field):
                field[index] = renames.get(name, name)
    for value in list(graph.graph.output) + list(graph.graph.value_info):
        value.name = renames.get(value.name, value.name)
    assert not any(n.domain == "TRT" for n in graph.graph.node)
    onnx.checker.check_model(graph)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph, destination)
    assert sha256(source) == source_sha, "The deployment graph must remain unchanged"
    print(f"Benchmark-only graph: {destination} ({len(graph.graph.node)} nodes)")
    print("Outputs: raw_boxes [batch, candidates, 4], raw_scores [batch, candidates, 3]")
    return destination


def dpkg_version(package):
    result = subprocess.run(["dpkg-query", "-W", "-f=${Version}", package],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def unmet_exact_dependencies(depends, installed=dpkg_version):
    unmet = []
    for clause in depends.split(","):
        match = re.fullmatch(r"\s*([a-z0-9.+-]+)\s*\(\s*=\s*([^)\s]+)\s*\)\s*", clause)
        if match and installed(match[1]) != match[2]:
            unmet.append(f"{match[1]} (= {match[2]})")
    return unmet


def trtexec_version(trtexec=DEFAULT_TRTEXEC):
    banner = subprocess.run([str(trtexec), "--help"], capture_output=True, text=True).stdout
    match = re.search(r"TensorRT v(\d+)", banner)
    assert match, f"cannot read the TensorRT version from {trtexec}"
    return match[1]


def load_tensorrt(work_dir, trtexec=DEFAULT_TRTEXEC):
    """Import TensorRT Python bindings that exactly match the installed libnvinfer8.

    The station image may ship TensorRT libraries without Python bindings. A PyPI
    wheel would bring different libraries, so the matching python3-libnvinfer
    package is downloaded from the configured APT repository, verified against the
    APT index and extracted under work_dir. Nothing is installed system-wide.
    """
    trt_version = trtexec_version(trtexec)
    libnvinfer = dpkg_version("libnvinfer8")
    assert libnvinfer, "libnvinfer8 package not found"
    major, minor, patch = libnvinfer.split(".")[:3]
    assert f"{int(major) * 1000 + int(minor) * 100 + int(patch)}" == trt_version, (
        f"libnvinfer8 {libnvinfer} does not match trtexec v{trt_version}")
    info = {"trtexec": trt_version, "libnvinfer8": libnvinfer}
    try:
        import tensorrt as trt
        info["source"] = "preinstalled"
    except ModuleNotFoundError:
        package = f"python3-libnvinfer={libnvinfer}"
        root = Path(work_dir) / "tensorrt-python"
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["apt-get", "download", package], cwd=root, check=True)
        (deb,) = root.glob("python3-libnvinfer_*.deb")
        index = subprocess.run(["apt-cache", "show", package], capture_output=True,
                               text=True, check=True).stdout
        assert f"SHA256: {sha256(deb)}" in index.splitlines(), "package hash differs from APT index"
        fields = dict(line.split(": ", 1) for line in subprocess.run(
            ["dpkg-deb", "-f", str(deb), "Package", "Version", "Architecture", "Depends"],
            capture_output=True, text=True, check=True).stdout.splitlines())
        assert fields["Version"] == libnvinfer, fields
        unmet = unmet_exact_dependencies(fields["Depends"])
        assert not unmet, f"Installed TensorRT packages do not match: {unmet}"
        subprocess.run(["dpkg-deb", "-x", str(deb), str(root / "root")], check=True)
        site = (root / "root" / "usr" / "lib" /
                f"python{sys.version_info.major}.{sys.version_info.minor}" / "dist-packages")
        assert (site / "tensorrt").is_dir(), f"bindings not built for this Python: {site}"
        sys.path.insert(0, str(site))
        import tensorrt as trt
        info.update(source=f"{deb.name} (extracted, not installed)", deb_sha256=sha256(deb))
    assert trt.__version__ == f"{major}.{minor}.{patch}", (trt.__version__, libnvinfer)
    info["tensorrt_python"] = trt.__version__
    print(f"TensorRT Python {trt.__version__} from {trt.__file__}")
    return trt, info


def summarize(label, batch, rounds, reference_median=None, *,
              expected_rounds=ROUNDS, expected_requests=MEASURED_REQUESTS):
    import numpy as np

    assert len(rounds) == expected_rounds and all(len(r) == expected_requests for r in rounds)
    values = np.asarray(rounds, dtype=np.float64)
    assert np.isfinite(values).all() and (values > 0).all(), label
    median = float(np.median(values))
    return {
        "variant": label, "batch": batch, "median_ms": median,
        "p95_ms": float(np.percentile(values, 95)), "mean_ms": float(values.mean()),
        "round_medians_ms": [float(np.median(r)) for r in values],
        "ms_per_image": median / batch, "serial_images_per_second": batch * 1000 / median,
        "speedup_vs_onnxruntime": (reference_median / median) if reference_median else 1.0,
    }


def gpu_state():
    load = Path("/sys/devices/platform/gpu.0/load")
    frequency = Path("/sys/class/devfreq/17000000.gpu/cur_freq")
    return {
        "gpu_load_percent": int(load.read_text()) / 10 if load.exists() else None,
        "gpu_frequency_mhz": int(frequency.read_text()) / 1e6 if frequency.exists() else None,
    }


def l4t_release():
    release = Path("/etc/nv_tegra_release")
    if not release.exists():
        return "unknown"
    return " ".join(part.strip() for part in release.read_text().split(",")[:2])


def run_benchmark(raw_onnx, engines, output_dir, *, source_onnx=None, deployment_engine=None,
                  trtexec=DEFAULT_TRTEXEC, workspace_mib=None):
    import numpy as np
    import onnx
    import onnxruntime as ort

    raw_onnx, output_dir = Path(raw_onnx), Path(output_dir)
    engines = {label: Path(path) for label, path in engines.items()}
    assert "CUDAExecutionProvider" in ort.get_available_providers(), "ONNX Runtime CUDA provider missing"
    trt, trt_info = load_tensorrt(output_dir, trtexec)

    cudart = ctypes.CDLL("libcudart.so.12")

    def cuda_check(code, call):
        if code != 0:
            raise RuntimeError(f"{call} failed with CUDA error {code}")

    def device_sync():
        cuda_check(cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    state_before = gpu_state()
    if (state_before["gpu_load_percent"] or 0) > 10:
        print(f"WARNING: GPU already {state_before['gpu_load_percent']:.0f}% busy before the "
              "benchmark; absolute latencies will include that contention.")

    session = ort.InferenceSession(
        str(raw_onnx), providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0})])
    session.disable_fallback()
    providers = session.get_providers()
    assert providers[0] == "CUDAExecutionProvider", providers
    assert "TensorrtExecutionProvider" not in providers, providers
    ort_cuda_options = session.get_provider_options()["CUDAExecutionProvider"]
    assert ort_cuda_options["use_tf32"] == "0", ort_cuda_options

    trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    contexts = {}
    for label, path in engines.items():
        engine = trt_runtime.deserialize_cuda_engine(path.read_bytes())
        assert engine is not None, f"Cannot load {path}"
        io = {engine.get_tensor_name(i): (engine.get_tensor_mode(engine.get_tensor_name(i)),
                                          engine.get_tensor_dtype(engine.get_tensor_name(i)))
              for i in range(engine.num_io_tensors)}
        assert io == {"images": (trt.TensorIOMode.INPUT, trt.DataType.FLOAT),
                      **{name: (trt.TensorIOMode.OUTPUT, trt.DataType.FLOAT) for name in OUTPUTS}}, io
        contexts[label] = (engine, engine.create_execution_context())
    stream = ctypes.c_void_p()
    cuda_check(cudart.cudaStreamCreateWithFlags(ctypes.byref(stream), 1), "cudaStreamCreateWithFlags")

    def gpu_outputs(batch):
        return {name: ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((batch, CANDIDATES, width), np.float32), "cuda", 0)
                for name, width in OUTPUTS.items()}

    order_rng = np.random.default_rng(2026)
    results, differences, orders = [], [], []
    for batch in BATCHES:
        host_input = np.random.default_rng(42).random((batch, 3, 640, 640), dtype=np.float32)
        gpu_input = ort.OrtValue.ortvalue_from_numpy(host_input, "cuda", 0)
        outputs = {ORT_LABEL: gpu_outputs(batch)}
        binding = session.io_binding()
        binding.bind_ortvalue_input("images", gpu_input)
        for name, value in outputs[ORT_LABEL].items():
            binding.bind_ortvalue_output(name, value)
        runners = {ORT_LABEL: lambda binding=binding: session.run_with_iobinding(binding)}
        for label, (engine, context) in contexts.items():
            outputs[label] = gpu_outputs(batch)
            assert context.set_input_shape("images", host_input.shape)
            assert context.set_tensor_address("images", gpu_input.data_ptr())
            for name, value in outputs[label].items():
                assert tuple(context.get_tensor_shape(name)) == tuple(value.shape()), name
                assert context.set_tensor_address(name, value.data_ptr())
            assert not context.infer_shapes()

            def run_engine(context=context):
                if not context.execute_async_v3(stream.value):
                    raise RuntimeError("TensorRT enqueue failed")
            runners[label] = run_engine

        samples = {label: [] for label in runners}
        for _ in range(ROUNDS):
            order = list(runners)
            order_rng.shuffle(order)
            orders.append({"batch": batch, "order": order})
            for label in order:
                for _ in range(WARMUP_REQUESTS):
                    runners[label]()
                timings = []
                for _ in range(MEASURED_REQUESTS):
                    device_sync()
                    start = time.perf_counter_ns()
                    runners[label]()
                    device_sync()
                    timings.append((time.perf_counter_ns() - start) / 1e6)
                samples[label].append(timings)

        reference = {name: value.numpy() for name, value in outputs[ORT_LABEL].items()}
        for label in runners:
            for name, width in OUTPUTS.items():
                values = outputs[label][name].numpy()
                assert values.shape == (batch, CANDIDATES, width) and np.isfinite(values).all(), (label, name)
                if label != ORT_LABEL:
                    error = np.abs(values - reference[name])
                    differences.append({
                        "variant": label, "batch": batch, "output": name,
                        "mean_absolute_difference": float(error.mean()),
                        "max_absolute_difference": float(error.max())})
        ort_row = summarize(ORT_LABEL, batch, samples[ORT_LABEL])
        results.append(ort_row)
        for label in contexts:
            results.append(summarize(label, batch, samples[label], ort_row["median_ms"]))
        for row in results[-len(runners):]:
            print(f"{row['variant']:<24} batch {batch}: median {row['median_ms']:8.3f} ms, "
                  f"p95 {row['p95_ms']:8.3f} ms", flush=True)
        del binding, runners, outputs, gpu_input
    state_after = gpu_state()
    cuda_check(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")
    del contexts, session
    gc.collect()

    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "method": "per request: cudaDeviceSynchronize, perf_counter_ns, one inference call, "
                  "cudaDeviceSynchronize; shared GPU-resident input; preallocated GPU outputs",
        "source_onnx_sha256": sha256(source_onnx) if source_onnx else None,
        "raw_onnx_sha256": sha256(raw_onnx),
        "engine_sha256": {label: sha256(engine) for label, engine in engines.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "onnx": onnx.__version__, "onnxruntime": ort.__version__, **trt_info},
        "l4t": l4t_release(),
        "onnxruntime_providers": providers, "onnxruntime_cuda_use_tf32": ort_cuda_options["use_tf32"],
        "tensorrt_build_flags": {"FP32": "--noTF32", "FP16": "--fp16 --noTF32"},
        "workspace_mib": workspace_mib, "profile_min_opt_max_batch": [1, 8, 8],
        "warmup_requests": WARMUP_REQUESTS, "measured_requests_per_round": MEASURED_REQUESTS,
        "rounds": ROUNDS, "round_orders": orders,
        "gpu_state_before": state_before, "gpu_state_after": state_after,
        "results": results, "output_differences": differences,
        "deployment_engine_sha256": sha256(deployment_engine) if deployment_engine else None,
        "note": "same timing boundary for all variants; raw NMS-free model latency, not pipeline FPS",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "results.md").write_text(markdown_table(report) + "\n")
    return report


def markdown_table(report):
    table = ("| Variant | Batch | Median ms | p95 ms | Round medians ms | ms/image | "
             "Serial images/s | Speed-up vs ORT |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in report["results"]:
        rounds = row["round_medians_ms"]
        table += (f"| {row['variant']} | {row['batch']} | {row['median_ms']:.2f} | "
                  f"{row['p95_ms']:.2f} | {min(rounds):.2f}-{max(rounds):.2f} | "
                  f"{row['ms_per_image']:.2f} | {row['serial_images_per_second']:.1f} | "
                  f"{row['speedup_vs_onnxruntime']:.2f}x |\n")
    return (table + f"\nGPU before: {report['gpu_state_before']}; after: {report['gpu_state_after']}. "
            "Raw model latency, one request at a time; not pipeline FPS.")


def parse_engine(value):
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("use LABEL=PATH, e.g. 'TensorRT FP16=raw-fp16.engine'")
    return label, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    raw = commands.add_parser("raw-onnx", help="write an NMS-free copy of the ONNX graph")
    raw.add_argument("source")
    raw.add_argument("destination")
    bench = commands.add_parser("run", help="measure ONNX Runtime and the TensorRT engines")
    bench.add_argument("--onnx", required=True, help="NMS-free ONNX graph (from raw-onnx)")
    bench.add_argument("--engine", action="append", type=parse_engine, required=True,
                       help="LABEL=PATH of a TensorRT engine built from --onnx (repeatable)")
    bench.add_argument("--output", required=True, help="directory for results.json and results.md")
    bench.add_argument("--source-onnx", help="original deployment ONNX, recorded by hash")
    bench.add_argument("--deployment-engine", help="deployment engine, recorded by hash")
    bench.add_argument("--trtexec", default=DEFAULT_TRTEXEC)
    bench.add_argument("--workspace-mib", type=int)
    args = parser.parse_args(argv)

    if args.command == "raw-onnx":
        make_raw_onnx(args.source, args.destination)
        return 0
    engines = dict(args.engine)
    assert len(engines) == len(args.engine), "engine labels must be unique"
    assert ORT_LABEL not in engines, f"'{ORT_LABEL}' is reserved for the reference"
    report = run_benchmark(args.onnx, engines, args.output, source_onnx=args.source_onnx,
                           deployment_engine=args.deployment_engine, trtexec=args.trtexec,
                           workspace_mib=args.workspace_mib)
    print()
    print(markdown_table(report))
    print("\nOutput differences versus ONNX Runtime (not an accuracy score):")
    for d in report["output_differences"]:
        print(f"  {d['variant']:<14} b{d['batch']} {d['output']:<10} "
              f"mean {d['mean_absolute_difference']:.3g} max {d['max_absolute_difference']:.3g}")
    print(f"Saved: {Path(args.output) / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
