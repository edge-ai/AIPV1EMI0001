#!/usr/bin/env python3
"""Select which YOLO flavor this AI Processor packages.

An AI Processor ships exactly one detector, so the repository keeps every flavor under
flavors/<id>/ and copies the chosen one to the places EAIS Tools reads.

    python3 tools/flavors.py list                  # flavors and which one is active
    python3 tools/flavors.py use yolo26            # make yolo26 the packaged detector
    python3 tools/flavors.py check                 # fail if the repository files do not match the active flavor
    python3 tools/flavors.py active                # print the id of the active flavor
    python3 tools/flavors.py get yolo26 engine     # print one field of flavor.json (used by shell scripts)

`use` rewrites these files from flavors/<id>/:

    manifest.json, store.json              name, description, license, model and config_plan
    configs/<infer_config>                 DeepStream nvinferserver configuration (generated, not in git)
    models/AIPV1EMI0001_Yolo_Detector/config.pbtxt   Triton model configuration (generated, not in git)
    LICENSE.weights                        licence text of the packaged weights

The two configuration copies are git-ignored so each setting has a single source in flavors/<id>/.
bin/build.sh runs `use` before every build. The committed manifest.json, store.json and
LICENSE.weights stay in git because they describe the published AI Processor.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

MODEL_DIR = "models/AIPV1EMI0001_Yolo_Detector"
INFER_CONFIG_PATTERN = re.compile(r"config_infer_primary_detector_[a-z0-9]+\.txt")
EXPORT_DIR = "tmp/export"
CANDIDATES = 25200
COCO_CLASSES = 80
# Raw (pre-NMS) outputs per batch item, used by the benchmark and accuracy tools:
#   efficientnms  YOLOv7 graph without its EfficientNMS_TRT node: every anchor with its box and class scores
#   end2end       YOLOv10 / YOLO26 graph as exported: 300 rows of x1, y1, x2, y2, score, class
RAW_OUTPUTS = {
    "efficientnms": {"raw_boxes": (CANDIDATES, 4), "raw_scores": (CANDIDATES, COCO_CLASSES)},
    "end2end": {"output0": (300, 6)},
}
FAMILIES = set(RAW_OUTPUTS)


class FlavorError(Exception):
    pass


def default_root():
    return Path(__file__).resolve().parent.parent


def flavor_ids(root):
    flavors = Path(root) / "flavors"
    return sorted(path.parent.name for path in flavors.glob("*/flavor.json"))


def load(root, flavor_id):
    path = Path(root) / "flavors" / flavor_id / "flavor.json"
    if not path.is_file():
        raise FlavorError(f"unknown flavor '{flavor_id}'; available: {', '.join(flavor_ids(root)) or 'none'}")
    flavor = json.loads(path.read_text())
    if flavor.get("id") != flavor_id:
        raise FlavorError(f"{path}: id must be '{flavor_id}'")
    if flavor.get("family") not in FAMILIES:
        raise FlavorError(f"{path}: family must be one of {sorted(FAMILIES)}")
    return flavor


def onnx_path(root, flavor):
    """Where tools/export_model.py writes the flavor's ONNX graph (git-ignored)."""
    return Path(root) / EXPORT_DIR / flavor["id"] / flavor["onnx"]


def read_json(path):
    return json.loads(Path(path).read_text())


def dump_json(data, like=None):
    text = json.dumps(data, indent=4, ensure_ascii=False)
    return text + "\n" if like is None or like.endswith("\n") else text


def render_manifest(manifest, flavor):
    manifest = json.loads(json.dumps(manifest))
    manifest["name"] = flavor["display_name"]
    manifest["description"] = flavor["description"]
    manifest["license"] = flavor["license"]
    manifest["primary_processor"]["model"] = flavor["display_name"]
    manifest["primary_processor"]["config_plan"] = flavor["infer_config"]
    return manifest


def render_store(store, flavor):
    store = json.loads(json.dumps(store))
    store["name"] = flavor["display_name"]
    store["description"] = flavor["description"]
    return store


def active_id(root):
    """Flavor whose infer config the manifest points to, or None."""
    root = Path(root)
    config_plan = read_json(root / "manifest.json")["primary_processor"]["config_plan"]
    for flavor_id in flavor_ids(root):
        if load(root, flavor_id)["infer_config"] == config_plan:
            return flavor_id
    return None


def targets(root, flavor):
    """Repository path -> expected content (text for JSON files, source path for copies)."""
    root = Path(root)
    source = root / "flavors" / flavor["id"]
    return {
        f"configs/{flavor['infer_config']}": source / flavor["infer_config"],
        f"{MODEL_DIR}/config.pbtxt": source / "config.pbtxt",
        "LICENSE.weights": root / flavor["weights_license"],
    }


def stale_configs(root, flavor):
    configs = Path(root) / "configs"
    return sorted(path.name for path in configs.glob("config_infer_primary_detector_*.txt")
                  if INFER_CONFIG_PATTERN.fullmatch(path.name) and path.name != flavor["infer_config"])


def differences(root, flavor):
    """List what differs between the repository and the given flavor."""
    root = Path(root)
    problems = []
    for relative, source in targets(root, flavor).items():
        target = root / relative
        if not source.is_file():
            problems.append(f"missing flavor source: {source.relative_to(root)}")
        elif not target.is_file() or target.read_bytes() != source.read_bytes():
            problems.append(f"{relative} differs from {source.relative_to(root)}")
    manifest = read_json(root / "manifest.json")
    if manifest != render_manifest(manifest, flavor):
        problems.append("manifest.json does not describe this flavor")
    store = read_json(root / "store.json")
    if store != render_store(store, flavor):
        problems.append("store.json does not describe this flavor")
    problems += [f"stale configs/{name}" for name in stale_configs(root, flavor)]
    return problems


def use(root, flavor_id):
    root = Path(root)
    flavor = load(root, flavor_id)
    for relative, source in targets(root, flavor).items():
        if not source.is_file():
            raise FlavorError(f"missing flavor source: {source.relative_to(root)}")
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, root / relative)
    for name in stale_configs(root, flavor):
        (root / "configs" / name).unlink()
    for name, render in (("manifest.json", render_manifest), ("store.json", render_store)):
        path = root / name
        path.write_text(dump_json(render(read_json(path), flavor), like=path.read_text()))
    return flavor


def check(root):
    root = Path(root)
    flavor_id = active_id(root)
    if flavor_id is None:
        raise FlavorError("manifest.json config_plan does not belong to any flavor")
    problems = differences(root, load(root, flavor_id))
    if problems:
        raise FlavorError(f"repository does not match the active flavor '{flavor_id}': " + "; ".join(problems))
    return flavor_id


def get(flavor, dotted):
    value = flavor
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise FlavorError(f"no field '{dotted}' in flavor {flavor['id']}")
        value = value[part]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=str(default_root()), help="AIP source folder (default: this repository)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list flavors and show the active one")
    use_parser = commands.add_parser("use", help="make a flavor the packaged detector")
    use_parser.add_argument("flavor")
    commands.add_parser("check", help="verify the repository matches the active flavor")
    commands.add_parser("active", help="print the id of the active flavor")
    get_parser = commands.add_parser("get", help="print one flavor.json field")
    get_parser.add_argument("flavor")
    get_parser.add_argument("field", help="dotted path, e.g. engine or weights.url")
    args = parser.parse_args(argv)
    root = Path(args.root)
    try:
        if args.command == "list":
            active = active_id(root)
            for flavor_id in flavor_ids(root):
                flavor = load(root, flavor_id)
                print(f"{'*' if flavor_id == active else ' '} {flavor_id:<8} {flavor['display_name']:<12} "
                      f"{flavor['family']}")
        elif args.command == "use":
            flavor = use(root, args.flavor)
            print(f"Active flavor: {flavor['id']} ({flavor['display_name']})")
        elif args.command == "check":
            print(f"OK: repository matches the active flavor '{check(root)}'")
        elif args.command == "active":
            active = active_id(root)
            if active is None:
                raise FlavorError("manifest.json config_plan does not belong to any flavor")
            print(active)
        else:
            value = get(load(root, args.flavor), args.field)
            print(value if isinstance(value, str) else json.dumps(value))
    except FlavorError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
