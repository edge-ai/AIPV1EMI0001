import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import flavors  # noqa: E402
from public_reference import assert_no_private_references  # noqa: E402

RELEASE = (ROOT / ".github/workflows/release.yml").read_text()


class ReleaseSafeguardsTests(unittest.TestCase):
    def test_engine_asset_is_either_complete_or_absent(self):
        for flavor_id in flavors.flavor_ids(ROOT):
            asset = flavors.load(ROOT, flavor_id)["engine_asset"]
            with self.subTest(flavor=flavor_id):
                self.assertEqual(set(asset), {"url", "sha256"})
                if asset["url"] is not None or asset["sha256"] is not None:
                    self.assertTrue(asset["url"].startswith("https://"))
                    self.assertRegex(asset["sha256"], r"^[0-9a-f]{64}$")

    def test_release_takes_the_engine_of_the_active_flavor_and_verifies_it(self):
        self.assertIn("python3 tools/flavors.py check", RELEASE)
        self.assertIn("engine_asset.url", RELEASE)
        self.assertIn("engine_asset.sha256", RELEASE)
        self.assertIn("sha256sum -c -", RELEASE)
        self.assertLess(RELEASE.index("sha256sum -c -"), RELEASE.index("Build ARM64 custom parser"))

    def test_release_stops_when_the_active_flavor_has_no_published_engine(self):
        guard = RELEASE.split('"$SHA256" = "null"', 1)[1].split("\n          fi", 1)[0]
        self.assertIn("::error::", guard)
        self.assertIn("exit 1", guard)

    def test_release_uses_immutable_inputs(self):
        self.assertNotIn("/resolve/main", RELEASE)
        self.assertNotIn("/releases/latest", RELEASE)
        self.assertRegex(RELEASE, r"PARSER_BUILD_IMAGE: nvcr\.io/nvidia/deepstream@sha256:[0-9a-f]{64}")

    def test_release_runs_every_test_and_the_package_check(self):
        self.assertIn("python3 -m unittest discover -s tests", RELEASE)
        self.assertIn("node --test tests/*.test.cjs", RELEASE)
        self.assertIn("python3 tools/check_package.py --archive", RELEASE)

    def test_weights_licence_ships_with_the_package(self):
        self.assertIn("LICENSE.weights", (ROOT / ".include").read_text())
        flavor = flavors.load(ROOT, flavors.check(ROOT))
        self.assertEqual((ROOT / "LICENSE.weights").read_bytes(), (ROOT / flavor["weights_license"]).read_bytes())
        self.assertIn("LICENSE.weights", RELEASE)

    def test_ci_installs_the_dependencies_the_tests_import(self):
        pull_request = (ROOT / ".github/workflows/pull-request.yml").read_text()
        for workflow in (pull_request, RELEASE):
            self.assertIn("python3 -m pip install -r tests/requirements-accuracy.txt", workflow)
        self.assertIn("pycocotools==2.0.10", (ROOT / "tools/requirements-accuracy.txt").read_text())

    def test_public_files_contain_no_station_identity(self):
        paths = [ROOT / name for name in ("README.md", "CHANGELOG.md", "PRODUCT.md", "bin/build.sh")]
        paths += sorted((ROOT / "doc").glob("*.md")) + sorted((ROOT / "tools").glob("*.py"))
        paths += sorted((ROOT / "tests").glob("*.py")) + sorted((ROOT / "flavors").glob("*/*"))
        for path in paths:
            with self.subTest(file=str(path.relative_to(ROOT))):
                assert_no_private_references(self, path.read_text(), path.name)

    def test_the_manifest_licence_names_the_packaged_weights(self):
        flavor = flavors.load(ROOT, flavors.check(ROOT))
        manifest = json.loads((ROOT / "manifest.json").read_text())
        self.assertEqual(manifest["license"], flavor["license"])
        self.assertRegex(manifest["license"], r"^Apache-2\.0 AND A?GPL-3\.0-only$")


if __name__ == "__main__":
    unittest.main()
