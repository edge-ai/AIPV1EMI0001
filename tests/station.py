"""Stub station used by the script and notebook tests: a temporary AIP folder and fake station commands."""
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_package  # noqa: E402
import flavors  # noqa: E402

# Station commands replaced by stubs. Each call is appended to $CALLS.
STUBS = {
    "curl": """
        echo "curl $*" >> "$CALLS"
        url=${@: -1}; name=${url##*/}; name=${name%%[?]*}
        [ -f "$FIXTURES/$name.json" ] || exit 22
        cat "$FIXTURES/$name.json"
    """,
    "timeout": """
        while [[ $1 == --* ]]; do shift; done; shift
        exec "$@"
    """,
    "trtexec": """
        echo "trtexec $*" >> "$CALLS"
        echo "&&&& RUNNING TensorRT.trtexec [TensorRT v${TRT_VERSION:-8602}]"
        for arg; do [[ $arg == --saveEngine=* ]] && echo engine > "${arg#--saveEngine=}"; done
        exit "${TRTEXEC_STATUS:-0}"
    """,
    "eais-tools": """
        echo "eais-tools $*" >> "$CALLS"
        case "$1 $2" in
            "aip build")
                [ "${EAIS_BUILD_STATUS:-0}" = 0 ] || exit "$EAIS_BUILD_STATUS"
                python3 "$STUB_DIR/eais_build.py" ;;
            "aip export")
                [ "${EAIS_EXPORT_STATUS:-0}" = 0 ] || exit "$EAIS_EXPORT_STATUS"
                tar -czf "${AIP_ID}_dist.tgz" dist ;;
            "pipeline test") exit "${PIPELINE_TEST_STATUS:-0}" ;;
            "pipeline clean") echo "[]" > "$FIXTURES/pipelines.json" ;;
        esac
    """,
    "make": 'echo "make $*" >> "$CALLS"; exit "${MAKE_STATUS:-0}"',
    "nvcc": 'echo "Cuda compilation tools, release 12.2, V12.2.140"',
    "npm": """
        echo "npm $*" >> "$CALLS"
        for package in vue hls.js; do
            mkdir -p "node_modules/$package" && echo "{}" > "node_modules/$package/package.json"
        done
    """,
}

EAIS_BUILD = f"""
import os, shutil, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / "tools")!r})
from check_package import eais_files, include_entries
shutil.rmtree("dist", ignore_errors=True)
for name in eais_files(".") | include_entries("."):
    Path("dist", name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(name, Path("dist", name))
extra = os.environ.get("EXTRA_DIST_FILE")
if extra:
    Path("dist", extra).parent.mkdir(parents=True, exist_ok=True)
    Path("dist", extra).write_text("must not ship")
"""

PACKAGE_SOURCES = ("flavors", "licenses", "configs", "models", "manifest.json", "store.json", "LICENSE.weights")


class Station:
    """Temporary AIP folder plus stub commands for running shell code like the station would."""

    def __init__(self, directory, fixtures=None):
        self.root = Path(directory) / "aip"
        self.root.mkdir()
        self.calls = Path(directory) / "calls.log"
        self.calls.touch()
        self.bin = Path(directory) / "bin"
        self.bin.mkdir()
        self.fixtures = Path(directory) / "fixtures"
        self.fixtures.mkdir()
        for name, body in STUBS.items():
            stub = self.bin / name
            stub.write_text("#!/bin/bash\n" + textwrap.dedent(body))
            stub.chmod(0o755)
        (self.bin / "eais_build.py").write_text(EAIS_BUILD)
        for name, data in (fixtures or {}).items():
            (self.fixtures / f"{name}.json").write_text(json.dumps(data))
        shutil.copytree(ROOT / "tools", self.root / "tools", ignore=shutil.ignore_patterns("__pycache__"))

    def environment(self, **env):
        return {
            **os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "CALLS": str(self.calls),
            "FIXTURES": str(self.fixtures), "STUB_DIR": str(self.bin), "AIP_ID": "AIPV1EMI0001",
            "MODEL_DIR": "models/AIPV1EMI0001_Yolo_Detector", "TRTEXEC": "trtexec", "TRTEXEC_BIN": "trtexec",
            "CLEANUP_EAIS_RESOURCES": "0", "MS3": "http://ms3", "WORKSPACE_MIB": "1024",
            "TRT_VERSION": "8602", "PIPELINE_ID": "owned-pipeline", "COPYFILE_DISABLE": "1", **env,
        }

    def run(self, source, **env):
        source = source.replace("/usr/local/cuda/bin/nvcc", "nvcc")
        return subprocess.run(["bash", "-c", source], cwd=self.root, env=self.environment(**env),
                              capture_output=True, text=True)

    def log(self):
        return self.calls.read_text().splitlines()

    def add_flavor_sources(self, flavor_id=None):
        """Copy the files that make this folder a packageable AI Processor (optionally for one flavor)."""
        for name in PACKAGE_SOURCES:
            source = ROOT / name
            if source.is_dir():
                shutil.copytree(source, self.root / name, dirs_exist_ok=True)
            else:
                shutil.copy2(source, self.root / name)
        if flavor_id:
            flavors.use(self.root, flavor_id)

    def add_build_script(self):
        """bin/build.sh with a stub make that writes the parser library like the real one."""
        (self.root / "bin").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "bin/build.sh", self.root / "bin/build.sh")
        (self.root / "src/nvdsinfer_customparser").mkdir(parents=True, exist_ok=True)
        make = self.bin / "make"
        make.write_text('#!/bin/bash\necho "make $*" >> "$CALLS"\n[ "${MAKE_STATUS:-0}" = 0 ] || exit "$MAKE_STATUS"\n'
                        'echo parser > "${@: -1}/libnvds_infercustomparser.so"\n')
        make.chmod(0o755)

    def add_onnx(self, flavor_id):
        flavor = flavors.load(ROOT, flavor_id)
        path = flavors.onnx_path(self.root, flavor)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("onnx")
        return flavor

    def add_package_sources(self, flavor_id=None):
        self.add_flavor_sources(flavor_id)
        shutil.copy2(ROOT / ".include", self.root / ".include")
        for name in check_package.eais_files(self.root) | check_package.include_entries(ROOT):
            path = self.root / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"fixture: {name}\n")
