#!/usr/bin/env python3
"""Download a flavor's official COCO weights and export them to ONNX.

    python3 tools/export_model.py yolo26            # writes tmp/export/yolo26/yolo26n.onnx
    python3 tools/export_model.py yolov7 --force    # export again even if the ONNX already exists

The weights are downloaded from the URL in flavors/<id>/flavor.json and verified
against its SHA-256 before anything runs. The exporter is pinned too: an exact
Ultralytics release for YOLOv10 and YOLO26, an exact git commit of the YOLOv7
repository for YOLOv7. The resulting graph is checked against the input and output
contract that the Triton model configuration of the flavor expects.

Nothing is installed or modified outside the output directory. If the exporter or
one of its dependencies is missing, the tool stops and says what to install (see
tools/requirements-export.txt).
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import flavors

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {1: "float32", 6: "int32"}

# YOLOv7's export.py calls torch.onnx.export without the TorchScript exporter that newer
# PyTorch releases no longer use by default. onnxsim installs onnxruntime from PyPI when
# the import fails, which would shadow the station's GPU build, so refuse to continue instead.
YOLOV7_WRAPPER = """
import functools, inspect, runpy, sys
import torch
import onnxruntime  # noqa: F401  must come from the environment, never from an automatic pip install
if "dynamo" in inspect.signature(torch.onnx.export).parameters:
    torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False)
sys.argv = ["export.py"] + sys.argv[1:]
runpy.run_path("export.py", run_name="__main__")
"""


class ExportError(Exception):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_weights(flavor, directory):
    weights = flavor["weights"]
    target = Path(directory) / weights["file"]
    if not (target.is_file() and sha256(target) == weights["sha256"]):
        print(f"Downloading {weights['url']}", flush=True)
        partial = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(weights["url"], timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        actual = sha256(partial)
        if actual != weights["sha256"]:
            partial.unlink()
            raise ExportError(f"{weights['file']}: SHA-256 {actual} does not match the pinned {weights['sha256']}")
        partial.replace(target)
    print(f"Weights verified: {target} (sha256 {weights['sha256'][:12]}...)")
    return target


def export_ultralytics(flavor, weights, directory):
    package, _, pinned = flavor["exporter"]["package"].partition("==")
    try:
        installed = version(package)
    except PackageNotFoundError:
        installed = None
    if installed != pinned:
        raise ExportError(f"{package}=={pinned} is required (found {installed or 'nothing'}); "
                          "install tools/requirements-export.txt, see the build notebook")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(directory) / ".ultralytics"))
    from ultralytics import YOLO

    exported = Path(YOLO(str(weights)).export(**flavor["exporter"]["arguments"]))
    return exported


def export_yolov7(flavor, weights, directory):
    exporter = flavor["exporter"]
    source = Path(directory) / "yolov7-source"
    if not source.is_dir():
        subprocess.run(["git", "clone", "--quiet", exporter["repository"], str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--quiet", exporter["commit"]], check=True)
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != exporter["commit"]:
        raise ExportError(f"YOLOv7 checkout is at {head}, expected {exporter['commit']}")
    command = [sys.executable, "-c", YOLOV7_WRAPPER, "--weights", str(weights.resolve()),
               *exporter["arguments"]]
    subprocess.run(command, cwd=source, check=True)
    for extra in weights.parent.glob(f"{weights.stem}.torchscript*"):
        extra.unlink()
    return weights.with_suffix(".onnx")


def contract_problems(flavor, onnx_path):
    """Compare the graph inputs and outputs with what the flavor's Triton config expects.

    Dimensions the graph leaves unspecified (the batch, or the output sizes of a plugin node
    such as EfficientNMS) are accepted; every dimension the graph does state must match.
    """
    import onnx

    graph = onnx.load(onnx_path, load_external_data=False).graph

    def describe(value):
        dims = [dim.dim_value if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]
        return DTYPES.get(value.type.tensor_type.elem_type, "other"), dims

    problems = []
    inputs = {value.name: describe(value) for value in graph.input}
    if "images" not in inputs or inputs["images"][0] != "float32" or len(inputs["images"][1]) != 4:
        problems.append(f"expected a float32 input 'images' [batch, 3, height, width], found {inputs}")
    outputs = {value.name: describe(value) for value in graph.output}
    expected = {item["name"]: (item["dtype"], item["dims"]) for item in flavor["outputs"]}
    if set(outputs) != set(expected):
        problems.append(f"expected outputs {sorted(expected)}, found {sorted(outputs)}")
    for name, (dtype, dims) in expected.items():
        if name in outputs:
            actual_dtype, actual_dims = outputs[name]
            known = [(want, got) for want, got in zip(dims, actual_dims[1:]) if got is not None]
            if actual_dtype != dtype or len(actual_dims) != len(dims) + 1 or any(w != g for w, g in known):
                problems.append(f"output {name}: expected {dtype} [batch, {', '.join(map(str, dims))}], "
                                f"found {actual_dtype} {actual_dims}")
    return problems


def export(flavor, directory, force=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / flavor["onnx"]
    if target.is_file() and not force:
        print(f"{target} already exists; use --force to export again")
    else:
        weights = fetch_weights(flavor, directory)
        kind = flavor["exporter"]["kind"]
        exporter = {"ultralytics": export_ultralytics, "yolov7": export_yolov7}.get(kind)
        if exporter is None:
            raise ExportError(f"unknown exporter '{kind}'")
        exported = exporter(flavor, weights, directory)
        if exported.resolve() != target.resolve():
            shutil.move(exported, target)
    problems = contract_problems(flavor, target)
    if problems:
        raise ExportError(f"{target} does not match the {flavor['id']} contract: " + "; ".join(problems))
    print(f"ONNX ready: {target} (sha256 {sha256(target)})")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("flavor", help="flavor id, see: python3 tools/flavors.py list")
    parser.add_argument("--directory", help="output directory (default: tmp/export/FLAVOR)")
    parser.add_argument("--force", action="store_true", help="export again even if the ONNX exists")
    args = parser.parse_args(argv)
    try:
        flavor = flavors.load(ROOT, args.flavor)
        directory = args.directory or ROOT / flavors.EXPORT_DIR / flavor["id"]
        export(flavor, directory, args.force)
    except (flavors.FlavorError, ExportError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
