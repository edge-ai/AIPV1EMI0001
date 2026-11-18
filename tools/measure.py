#!/usr/bin/env python3
"""Measure one YOLO flavor on this station: latency first, then labelled accuracy.

    python3 tools/measure.py yolo26 --annotations instances_val2017.json --images val2017 \
        --results tmp/benchmarks

It runs the two tools below with the engines and ONNX files that belong together, then
copies their reports to <results>/<flavor>-latency.* and <results>/<flavor>-accuracy.*,
the layout that tools/compare_models.py reads:

    tools/benchmark.py all    builds raw FP32 and FP16 engines and times them against ONNX Runtime
    tools/accuracy.py run     scores the same engines on COCO val2017 (mAP 50:95, AP 50, per class)

The flavor's ONNX model must exist first (python3 tools/export_model.py FLAVOR). Engines and
logs stay in tmp/measure/<flavor>/ unless --work says otherwise. Pass --limit N to score only
the first N images when you only want to see that the chain works.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import flavors

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
ENGINE_DIR = "tmp/engines"


class MeasureError(Exception):
    pass


def deployment_engine(root, flavor):
    """The engine bin/build.sh installed, when it embeds NMS and so differs from the raw engines."""
    path = Path(root) / ENGINE_DIR / flavor["id"] / flavor["engine"]
    return path if flavor["family"] == "efficientnms" and path.is_file() else None


def latency_command(flavor, onnx, work, *, engine=None, trtexec=None, workspace_mib=1024):
    command = [sys.executable, str(TOOLS / "benchmark.py"), "all", "--flavor", flavor["id"],
               "--onnx", str(onnx), "--output", str(work / "latency"), "--workspace-mib", str(workspace_mib)]
    if engine:
        command += ["--deployment-engine", str(engine)]
    if trtexec:
        command += ["--trtexec", str(trtexec)]
    return command


def accuracy_command(flavor, onnx, work, annotations, images, *, engine=None, trtexec=None, limit=0,
                     batches=(8,)):
    latency = work / "latency"
    # benchmark.py cuts the EfficientNMS node into raw.onnx; the NMS-free flavors are scored as exported.
    raw_onnx = latency / "raw.onnx" if flavor["family"] == "efficientnms" else onnx
    command = [sys.executable, str(TOOLS / "accuracy.py"), "run", "--flavor", flavor["id"],
               "--onnx", str(raw_onnx), "--fp32-engine", str(latency / "raw-fp32.engine"),
               "--fp16-engine", str(latency / "raw-fp16.engine"),
               "--annotations", str(annotations), "--images", str(images), "--limit", str(limit),
               "--batches", *map(str, batches), "--output", str(work / "accuracy")]
    if engine:
        command += ["--deployment-engine", str(engine), "--source-onnx", str(onnx)]
    if trtexec:
        command += ["--trtexec", str(trtexec)]
    return command


def publish(work, results, flavor_id):
    results.mkdir(parents=True, exist_ok=True)
    for kind in ("latency", "accuracy"):
        for suffix in ("json", "md"):
            source = work / kind / f"results.{suffix}"
            if source.is_file():
                shutil.copyfile(source, results / f"{flavor_id}-{kind}.{suffix}")


def run(command):
    print("+ " + " ".join(command), flush=True)
    if subprocess.run(command).returncode:
        raise MeasureError(f"{Path(command[1]).name} failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("flavor", help="flavor id, see: python3 tools/flavors.py list")
    parser.add_argument("--annotations", help="COCO instances JSON (instances_val2017.json)")
    parser.add_argument("--images", help="directory with the annotated COCO images (val2017)")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N images (default: all)")
    parser.add_argument("--only", choices=("latency", "accuracy"), help="run just one of the two measurements")
    parser.add_argument("--results", default=str(ROOT / "tmp/benchmarks"), help="where to write the reports")
    parser.add_argument("--work", help="where to keep engines and logs (default: tmp/measure/FLAVOR)")
    parser.add_argument("--trtexec", help="trtexec binary (default: the one the tools look for)")
    parser.add_argument("--workspace-mib", type=int, default=1024, help="TensorRT builder workspace")
    args = parser.parse_args(argv)
    try:
        flavor = flavors.load(ROOT, args.flavor)
        onnx = flavors.onnx_path(ROOT, flavor)
        if not onnx.is_file():
            raise MeasureError(f"{onnx} not found; run: python3 tools/export_model.py {flavor['id']}")
        if args.only != "latency" and not (args.annotations and args.images):
            raise MeasureError("--annotations and --images are required for the accuracy measurement")
        work = Path(args.work) if args.work else ROOT / "tmp/measure" / flavor["id"]
        engine = deployment_engine(ROOT, flavor)
        if args.only != "accuracy":
            run(latency_command(flavor, onnx, work, engine=engine, trtexec=args.trtexec,
                                workspace_mib=args.workspace_mib))
        if args.only != "latency":
            run(accuracy_command(flavor, onnx, work, args.annotations, args.images, engine=engine,
                                 trtexec=args.trtexec, limit=args.limit))
        publish(work, Path(args.results), flavor["id"])
    except (flavors.FlavorError, MeasureError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Reports in {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
