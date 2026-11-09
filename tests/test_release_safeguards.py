import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "2f7dbaf4fa12b301e90f2e83c8e417ea34ead4cc"
ONNX_SHA256 = "5796e9e0836751e3b7a9623cc6db976ac7fd087fcfb2742d20d06dcf0a0b3560"
ENGINE_SHA256 = "670150b495a0f120d178fbbb1e3f29758ccbee710e4477bf6847f46517a873b8"


def notebook_cells(name):
    notebook = json.loads((ROOT / name).read_text())
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


class ReleaseSafeguardsTests(unittest.TestCase):
    def test_notebooks_and_release_use_immutable_verified_weights(self):
        local = "\n".join(notebook_cells("localdev.ipynb").values())
        accuracy = "\n".join(notebook_cells("accuracy.ipynb").values())
        release = (ROOT / ".github/workflows/release.yml").read_text()
        for source in (local, accuracy, release):
            self.assertIn(f"/resolve/{MODEL_REVISION}", source)
            self.assertNotIn("/resolve/main", source)
        self.assertIn(ONNX_SHA256, local)
        self.assertIn(ONNX_SHA256, accuracy)
        self.assertIn(ENGINE_SHA256, local)
        self.assertIn(ENGINE_SHA256, release)

    def test_setup_preserves_existing_station_aip_and_explains_engine_check(self):
        cells = notebook_cells("localdev.ipynb")
        markdown = "\n".join(cells.values())
        self.assertIn("[ ! -e /data/aip/AIPV1EMI0001 ]", markdown)
        self.assertNotIn("mv AIPV1EMI0001", markdown)
        self.assertIn("do not move, replace or remove an existing AIP", markdown)
        self.assertIn("download checks TensorRT 8.6.2 only", markdown)
        self.assertIn("engine-load smoke test", markdown)


if __name__ == "__main__":
    unittest.main()
