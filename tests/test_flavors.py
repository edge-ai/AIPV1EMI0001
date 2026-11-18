import importlib.util
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import export_model  # noqa: E402
import flavors  # noqa: E402
import measure  # noqa: E402

COPIED = ("manifest.json", "store.json", "LICENSE.weights")


def copy_repository(destination):
    destination = Path(destination)
    for name in ("flavors", "licenses", "configs", "models"):
        shutil.copytree(ROOT / name, destination / name)
    for name in COPIED:
        shutil.copy2(ROOT / name, destination / name)
    return destination


def triton_outputs(text):
    """(name, dims) of every output in a Triton config.pbtxt."""
    block = text[text.index("output ["):]
    return re.findall(r'name: "([^"]+)"\s+data_type: \w+\s+dims: \[ ([^\]]+) \]', block)


class FlavorDefinitionTests(unittest.TestCase):
    def test_the_three_flavors_are_defined(self):
        self.assertEqual(flavors.flavor_ids(ROOT), ["yolo26", "yolov10", "yolov7"])

    def test_every_flavor_pins_its_weights_and_licence(self):
        for flavor_id in flavors.flavor_ids(ROOT):
            with self.subTest(flavor=flavor_id):
                flavor = flavors.load(ROOT, flavor_id)
                weights = flavor["weights"]
                self.assertRegex(weights["sha256"], r"^[0-9a-f]{64}$")
                self.assertIn("/releases/download/", weights["url"])
                self.assertTrue(weights["url"].endswith(weights["file"]))
                self.assertTrue((ROOT / flavor["weights_license"]).is_file())
                if flavor["exporter"]["kind"] == "yolov7":
                    self.assertRegex(flavor["exporter"]["commit"], r"^[0-9a-f]{40}$")
                else:
                    self.assertRegex(flavor["exporter"]["package"], r"^ultralytics==\d+\.\d+\.\d+$")

    def test_flavors_do_not_share_engine_or_graph_names(self):
        loaded = [flavors.load(ROOT, flavor_id) for flavor_id in flavors.flavor_ids(ROOT)]
        for field in ("engine", "onnx", "infer_config"):
            values = [flavor[field] for flavor in loaded]
            self.assertEqual(len(values), len(set(values)), field)

    def test_triton_and_deepstream_configs_agree_with_the_flavor(self):
        for flavor_id in flavors.flavor_ids(ROOT):
            with self.subTest(flavor=flavor_id):
                flavor = flavors.load(ROOT, flavor_id)
                folder = ROOT / "flavors" / flavor_id
                triton = (folder / "config.pbtxt").read_text()
                self.assertIn(f'default_model_filename: "{flavor["engine"]}"', triton)
                expected = [(item["name"], ", ".join(map(str, item["dims"]))) for item in flavor["outputs"]]
                self.assertEqual(triton_outputs(triton), expected)
                infer = (folder / flavor["infer_config"]).read_text()
                self.assertIn(f'custom_parse_bbox_func: "{flavor["parser_function"]}"', infer)
                self.assertIn("num_detected_classes: 3", infer)

    def test_every_parser_function_is_exported_by_the_shared_library(self):
        source = (ROOT / "src/nvdsinfer_customparser/nvdsinfer_custombboxparser.cpp").read_text()
        for flavor_id in flavors.flavor_ids(ROOT):
            parser = flavors.load(ROOT, flavor_id)["parser_function"]
            self.assertIn(f"CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE({parser});", source)

    def test_family_and_raw_outputs_cover_the_exported_graphs(self):
        for flavor_id in flavors.flavor_ids(ROOT):
            flavor = flavors.load(ROOT, flavor_id)
            self.assertIn(flavor["family"], flavors.RAW_OUTPUTS)

    def test_the_repository_matches_its_active_flavor(self):
        self.assertIn(flavors.check(ROOT), flavors.flavor_ids(ROOT))


class FlavorSwitchTests(unittest.TestCase):
    def test_use_rewrites_every_packaged_file_and_check_follows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = copy_repository(directory)
            for flavor_id in ("yolov7", "yolov10", "yolo26", "yolov7"):
                flavor = flavors.use(root, flavor_id)
                self.assertEqual(flavors.check(root), flavor_id)
                self.assertEqual(flavors.active_id(root), flavor_id)
                self.assertEqual(sorted(p.name for p in (root / "configs").glob("config_infer_*")),
                                 [flavor["infer_config"]])
                manifest = json.loads((root / "manifest.json").read_text())
                self.assertEqual(manifest["primary_processor"]["config_plan"], flavor["infer_config"])
                self.assertIn(flavor["engine"], (root / flavors.MODEL_DIR / "config.pbtxt").read_text())
                self.assertEqual((root / "LICENSE.weights").read_bytes(),
                                 (root / flavor["weights_license"]).read_bytes())

    def test_check_reports_a_modified_triton_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = copy_repository(directory)
            pbtxt = root / flavors.MODEL_DIR / "config.pbtxt"
            pbtxt.write_text(pbtxt.read_text() + "\n# drift\n")
            with self.assertRaisesRegex(flavors.FlavorError, "config.pbtxt"):
                flavors.check(root)

    def test_unknown_flavor_is_rejected(self):
        with self.assertRaisesRegex(flavors.FlavorError, "unknown flavor"):
            flavors.load(ROOT, "yolov99")


@unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx not installed")
class ExportContractTests(unittest.TestCase):
    def graph(self, directory, outputs):
        import onnx
        from onnx import TensorProto, helper

        values = [helper.make_tensor_value_info(name, dtype, shape) for name, dtype, shape in outputs]
        image = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 640, 640])
        path = Path(directory) / "model.onnx"
        onnx.save(helper.make_model(helper.make_graph([], "g", [image], values)), path)
        return path

    def test_end2end_graph_must_have_the_300_by_6_head(self):
        from onnx import TensorProto

        flavor = flavors.load(ROOT, "yolo26")
        with tempfile.TemporaryDirectory() as directory:
            good = self.graph(directory, [("output0", TensorProto.FLOAT, ["batch", 300, 6])])
            self.assertEqual(export_model.contract_problems(flavor, good), [])
            bad = self.graph(directory, [("output0", TensorProto.FLOAT, ["batch", 84, 8400])])
            self.assertTrue(export_model.contract_problems(flavor, bad))

    def test_efficientnms_graph_must_expose_the_four_nms_outputs(self):
        from onnx import TensorProto

        flavor = flavors.load(ROOT, "yolov7")
        with tempfile.TemporaryDirectory() as directory:
            good = self.graph(directory, [
                ("num_dets", TensorProto.INT32, ["batch", 1]),
                ("det_boxes", TensorProto.FLOAT, ["batch", 100, 4]),
                ("det_scores", TensorProto.FLOAT, ["batch", 100]),
                ("det_classes", TensorProto.INT32, ["batch", 100])])
            self.assertEqual(export_model.contract_problems(flavor, good), [])
            bad = self.graph(directory, [("output0", TensorProto.FLOAT, ["batch", 300, 6])])
            self.assertTrue(export_model.contract_problems(flavor, bad))


class MeasureCommandTests(unittest.TestCase):
    work = Path("work")

    def test_nms_free_flavors_are_scored_as_exported(self):
        flavor = flavors.load(ROOT, "yolo26")
        command = measure.accuracy_command(flavor, Path("yolo26n.onnx"), self.work, "ann.json", "images")
        self.assertEqual(command[command.index("--onnx") + 1], "yolo26n.onnx")
        self.assertNotIn("--deployment-engine", command)

    def test_yolov7_is_scored_on_the_graph_without_its_nms_node(self):
        flavor = flavors.load(ROOT, "yolov7")
        command = measure.accuracy_command(flavor, Path("yolov7-tiny.onnx"), self.work, "ann.json", "images",
                                           engine=Path("deployment.engine"))
        self.assertEqual(command[command.index("--onnx") + 1], str(self.work / "latency/raw.onnx"))
        self.assertEqual(command[command.index("--deployment-engine") + 1], "deployment.engine")
        self.assertEqual(command[command.index("--source-onnx") + 1], "yolov7-tiny.onnx")

    def test_latency_and_accuracy_share_the_same_engines(self):
        flavor = flavors.load(ROOT, "yolov10")
        latency = measure.latency_command(flavor, Path("a.onnx"), self.work, workspace_mib=512)
        accuracy = measure.accuracy_command(flavor, Path("a.onnx"), self.work, "ann.json", "images")
        self.assertIn("512", latency)
        self.assertEqual(accuracy[accuracy.index("--fp16-engine") + 1], str(self.work / "latency/raw-fp16.engine"))

    def test_only_yolov7_has_a_distinct_deployment_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            for flavor_id in flavors.flavor_ids(ROOT):
                flavor = flavors.load(ROOT, flavor_id)
                engine = Path(directory) / measure.ENGINE_DIR / flavor_id / flavor["engine"]
                engine.parent.mkdir(parents=True)
                engine.write_bytes(b"engine")
                expected = engine if flavor_id == "yolov7" else None
                self.assertEqual(measure.deployment_engine(directory, flavor), expected)

    def test_publish_copies_reports_under_the_flavor_name(self):
        with tempfile.TemporaryDirectory() as directory:
            work, results = Path(directory) / "work", Path(directory) / "results"
            for kind in ("latency", "accuracy"):
                (work / kind).mkdir(parents=True)
                (work / kind / "results.json").write_text("{}")
            measure.publish(work, results, "yolo26")
            self.assertEqual(sorted(p.name for p in results.iterdir()),
                             ["yolo26-accuracy.json", "yolo26-latency.json"])


if __name__ == "__main__":
    unittest.main()
