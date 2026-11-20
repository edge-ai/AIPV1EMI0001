#!/usr/bin/env python3
"""Put the measured YOLO flavors side by side.

Reads two files per flavor from one directory:

    <flavor>-latency.json    results.json written by tools/benchmark.py all
    <flavor>-accuracy.json   results.json written by tools/accuracy.py run

and writes comparison.md next to them, plus comparison.png when matplotlib is installed. An optional
conditions.md in the same folder (for example: which other workloads shared the GPU) is copied into the report:

    python3 tools/compare_models.py doc/benchmarks/orin-nx-16gb-jp6.0

The numbers are copied from the measurement files; nothing is estimated.
"""
import argparse
import json
import sys
from pathlib import Path

import flavors

TENSORRT_FP16 = "TensorRT FP16"
ONNX_RUNTIME = "ONNX Runtime CUDA FP32"
CLASSES = ("car", "person", "truck")
CONDITIONS = "conditions.md"


class CompareError(Exception):
    pass


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise CompareError(f"cannot read {path}: {error}") from None


def row(report, variant, batch):
    for item in report["results"]:
        if item["variant"] == variant and item["batch"] == batch:
            return item
    raise CompareError(f"no {variant} result for batch {batch}")


def collect(directory, root=None, flavor_ids=None):
    """Return one entry per flavor, ordered as flavors/ lists them."""
    root = Path(root) if root else flavors.default_root()
    directory = Path(directory)
    entries = []
    for flavor_id in flavor_ids or flavors.flavor_ids(root):
        latency, accuracy = (directory / f"{flavor_id}-{kind}.json" for kind in ("latency", "accuracy"))
        if not latency.is_file() or not accuracy.is_file():
            continue
        entries.append({
            "flavor": flavors.load(root, flavor_id),
            "latency": read(latency),
            "accuracy": read(accuracy),
        })
    if not entries:
        raise CompareError(f"no <flavor>-latency.json and <flavor>-accuracy.json pairs in {directory}")
    return entries


def accuracy_row(entry):
    """Accuracy of the TensorRT FP16 engine at the largest batch that was evaluated."""
    rows = [item for item in entry["accuracy"]["results"] if item["variant"] == TENSORRT_FP16]
    if not rows:
        raise CompareError(f"{entry['flavor']['id']}: no {TENSORRT_FP16} accuracy result")
    return max(rows, key=lambda item: item["batch"])


def percent(value):
    return "n/a" if value is None else f"{100 * value:.2f}"


def summary_table(entries):
    """The comparison notebook's compact table: accuracy and FP16 time per image at the smallest and largest batch."""
    batches = sorted({item["batch"] for entry in entries for item in entry["latency"]["results"]})
    small, large = batches[0], batches[-1]
    lines = [f"| Model | mAP 50:95, 80 classes (published) | mAP 50:95, car / person / truck "
             f"| FP16 ms per image, batch {small} | FP16 ms per image, batch {large} |",
             "|---|---:|---:|---:|---:|"]
    for entry in entries:
        flavor, metrics = entry["flavor"], accuracy_row(entry)["metrics"]
        published = flavor["reference"]["coco_map50_95"]
        lines.append(
            f"| {flavor['display_name']} | {percent(metrics['map_50_95'])} "
            f"({'n/a' if published is None else f'{published:.1f}'}) | {percent(metrics['aip_map_50_95'])} | "
            + " | ".join(f"{row(entry['latency'], TENSORRT_FP16, batch)['ms_per_image']:.1f}"
                         for batch in (small, large)) + " |")
    return "\n".join(lines)


def accuracy_table(entries):
    lines = ["| Model | Published COCO mAP 50:95 | Measured mAP 50:95 (80 classes) | AP 50 "
             "| mAP 50:95 (car, person, truck) | AP car | AP person | AP truck |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for entry in entries:
        flavor, metrics = entry["flavor"], accuracy_row(entry)["metrics"]
        published = flavor["reference"]["coco_map50_95"]
        by_class = metrics["ap_50_95_by_class"]
        lines.append(
            f"| {flavor['display_name']} | {'n/a' if published is None else f'{published:.1f}'} | "
            f"{percent(metrics['map_50_95'])} | {percent(metrics['ap_50'])} | "
            f"{percent(metrics['aip_map_50_95'])} | "
            + " | ".join(percent(by_class[name]) for name in CLASSES) + " |")
    return "\n".join(lines)


def latency_table(entries):
    batches = sorted({item["batch"] for entry in entries for item in entry["latency"]["results"]})
    head = ["| Model |"] + [f" Batch {batch} (ms per image) |" for batch in batches]
    lines = ["".join(head), "|---|" + "---:|" * len(batches)]
    for entry in entries:
        cells = []
        for batch in batches:
            fp16, ort = row(entry["latency"], TENSORRT_FP16, batch), row(entry["latency"], ONNX_RUNTIME, batch)
            cells.append(f" {fp16['ms_per_image']:.2f} ({fp16['serial_images_per_second']:.0f} img/s, "
                         f"{fp16['speedup_vs_onnxruntime']:.1f}x ORT) |")
        lines.append(f"| {entry['flavor']['display_name']} |" + "".join(cells))
    return "\n".join(lines)


def precision_table(entries):
    batches = sorted({item["batch"] for entry in entries for item in entry["latency"]["results"]})
    lines = ["| Model | Batch | ONNX Runtime CUDA FP32 | TensorRT FP32 | TensorRT FP16 |",
             "|---|---:|---:|---:|---:|"]
    for entry in entries:
        for batch in batches:
            medians = [row(entry["latency"], variant, batch)["median_ms"]
                       for variant in (ONNX_RUNTIME, "TensorRT FP32", TENSORRT_FP16)]
            lines.append(f"| {entry['flavor']['display_name']} | {batch} | "
                         + " | ".join(f"{value:.1f} ms" for value in medians) + " |")
    return "\n".join(lines)


def parity_table(entries):
    lines = ["| Model | Batch | mAP 50:95 TensorRT FP32 | mAP 50:95 TensorRT FP16 | Difference (points) |",
             "|---|---:|---:|---:|---:|"]
    for entry in entries:
        for item in entry["accuracy"]["results"]:
            if item["variant"] != TENSORRT_FP16:
                continue
            fp32 = next(r for r in entry["accuracy"]["results"]
                        if r["variant"] == "TensorRT FP32" and r["batch"] == item["batch"])
            a, b = fp32["metrics"]["map_50_95"], item["metrics"]["map_50_95"]
            lines.append(f"| {entry['flavor']['display_name']} | {item['batch']} | {percent(a)} | {percent(b)} | "
                         f"{100 * (b - a):+.2f} |")
    return "\n".join(lines)


def environment(entries):
    first = entries[0]
    versions = first["latency"]["versions"]
    return (f"{first['accuracy']['platform']['l4t'].strip('# ').strip()}; TensorRT "
            f"{versions.get('tensorrt_python', '?')}, ONNX Runtime {versions.get('onnxruntime', '?')}")


def conditions(directory):
    """Free-text measurement conditions kept next to the results (conditions.md), or an empty string."""
    path = Path(directory) / CONDITIONS
    return path.read_text().strip() if path.is_file() else ""


def reference_list(entries):
    lines = []
    for entry in entries:
        flavor = entry["flavor"]
        reference = flavor["reference"]
        published = reference["coco_map50_95"]
        value = "n/a" if published is None else f"{published:.1f}"
        lines.append(f'- **{flavor["display_name"]}** ({value}): {reference["note"]} [Source]({reference["url"]})')
    return "\n".join(lines)


def render_markdown(entries, notes=""):
    dataset = entries[0]["accuracy"]["dataset"]
    images = dataset["image_count"]
    return "\n".join([
        "# Model comparison", "",
        f"Measured on the EdgeAI Station ({environment(entries)}). Accuracy uses {images} COCO val2017 images; "
        "latency uses synthetic GPU-resident inputs. These are separate runs with engines built for this "
        "repository (`tools/benchmark.py` and `tools/accuracy.py`); see the measurement conditions for build provenance. "
        "Regenerate this file with `python3 tools/compare_models.py`.", "",
        "## Accuracy", "",
        "mAP 50:95 is the metric Ultralytics reports on COCO val2017. Values are percentages of the "
        "TensorRT FP16 engine; the published column is the reference recorded in each flavor's `flavor.json`. "
        "Its sources:", "",
        reference_list(entries), "",
        accuracy_table(entries), "",
        "## Latency", "",
        "TensorRT FP16, median over the timed requests, in milliseconds per image "
        "(raw model, one request at a time, not DeepStream pipeline throughput).", "",
        latency_table(entries), "",
        "### All precisions", "",
        precision_table(entries), "",
        "## FP16 against FP32", "",
        parity_table(entries), "",
        *(["## Measurement conditions", "", notes, ""] if notes else []),
    ])


def render_chart(entries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [entry["flavor"]["display_name"] for entry in entries]
    batches = sorted({item["batch"] for entry in entries for item in entry["latency"]["results"]})
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    width = 0.8 / len(batches)
    for index, batch in enumerate(batches):
        values = [row(entry["latency"], TENSORRT_FP16, batch)["ms_per_image"] for entry in entries]
        bars = left.bar([x + index * width for x in range(len(names))], values, width,
                        label=f"batch {batch}", color=colors[index % len(colors)])
        left.bar_label(bars, fmt="%.1f", fontsize=7, padding=1)
    left.set_xticks([x + width * (len(batches) - 1) / 2 for x in range(len(names))], names)
    left.set_ylabel("ms per image (lower is better)")
    left.set_title("TensorRT FP16 latency")
    left.set_ylim(0, left.get_ylim()[1] * 1.15)
    left.legend(ncol=len(batches), loc="upper center")

    measured = [100 * accuracy_row(entry)["metrics"]["map_50_95"] for entry in entries]
    published = [entry["flavor"]["reference"]["coco_map50_95"] for entry in entries]
    positions = range(len(names))
    bars = right.bar([x - 0.2 for x in positions], measured, 0.4, label="measured, TensorRT FP16", color=colors[0])
    right.bar_label(bars, fmt="%.1f", fontsize=8, padding=1)
    known = [(x, value) for x, value in zip(positions, published) if value is not None]
    if known:
        bars = right.bar([x + 0.2 for x, _ in known], [value for _, value in known], 0.4,
                         label="published by the authors", color=colors[1])
        right.bar_label(bars, fmt="%.1f", fontsize=8, padding=1)
    right.set_xticks(list(positions), names)
    right.set_ylabel("COCO val2017 mAP 50:95 (higher is better)")
    right.set_title("Accuracy, 80 COCO classes")
    right.set_ylim(0, right.get_ylim()[1] * 1.15)
    right.legend(ncol=2, loc="upper center")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory", help="folder with the <flavor>-latency.json and <flavor>-accuracy.json files")
    parser.add_argument("--root", help="repository root (default: this repository)")
    args = parser.parse_args(argv)
    try:
        entries = collect(args.directory, args.root)
    except (CompareError, flavors.FlavorError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    directory = Path(args.directory)
    (directory / "comparison.md").write_text(render_markdown(entries, conditions(directory)))
    print(f"Wrote {directory / 'comparison.md'}")
    try:
        render_chart(entries, directory / "comparison.png")
        print(f"Wrote {directory / 'comparison.png'}")
    except ModuleNotFoundError:
        print("matplotlib is not installed: comparison.png was not written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
