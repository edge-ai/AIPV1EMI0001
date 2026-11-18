#!/bin/bash
#
# build.sh — build the two runtime artifacts of this AI Processor for one YOLO flavor:
#
#   1. libnvds_infercustomparser.so
#      Custom DeepStream bounding-box parser. It reads the detector output of every flavor
#      (EfficientNMS_TRT for YOLOv7, 300 NMS-free rows for YOLOv10 and YOLO26) and produces
#      NvDsInferObjectDetectionInfo for the car, person and truck classes. Compiled from
#      src/nvdsinfer_customparser/ with make; requires the CUDA toolkit and DeepStream
#      headers (present in the ms3_videostreaming and ms8_jupyter containers on the station).
#
#   2. models/AIPV1EMI0001_Yolo_Detector/1/<engine>
#      TensorRT FP16 engine built by trtexec from the flavor's ONNX model, which
#      tools/export_model.py writes to tmp/export/<flavor>/. The engine is kept in
#      tmp/engines/<flavor>/ and the one for the selected flavor is copied to models/.
#
# Usage (from anywhere):
#   bin/build.sh                    # active flavor, see: python3 tools/flavors.py list
#   bin/build.sh yolov10            # select the flavor first, then build it
#   TRTEXEC_BIN=/path/to/trtexec bin/build.sh   # override the trtexec location
#   SKIP_ENGINE=1 bin/build.sh      # parser only (prebuilt engine already in models/.../1/)
#
# --workspace=<MiB> (trtexec): maximum GPU scratch memory TensorRT may use
# during the build. Higher values let TensorRT pick faster tactics (more
# algorithms fit in memory); lower values reduce peak build-time memory but
# can exclude some tactics — the resulting engine may be slightly slower and
# the build can take longer. It does NOT change the engine's runtime memory
# footprint significantly.
#
#   - 4096 MiB works on x86 dev machines.
#   - On a 15 GB Jetson Orin station running the full EdgeAI stack, 4096 MiB
#     gets the build OOM-killed (observed on an Orin NX 16GB); 1024 MiB succeeds.
#   - Override with WORKSPACE_MIB env var if you hit OOM on a busy station.
#
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL_PREFIX="AIPV1EMI0001"
MODEL_DIR="../models/${MODEL_PREFIX}_Yolo_Detector"
WORKSPACE_MIB="${WORKSPACE_MIB:-1024}"
TRTEXEC_BIN="${TRTEXEC_BIN:-/usr/src/tensorrt/bin/trtexec}"
SKIP_ENGINE="${SKIP_ENGINE:-0}"

FLAVOR="${1:-$(python3 ../tools/flavors.py active)}"
python3 ../tools/flavors.py use "${FLAVOR}" || exit 1
EngineFile=$(python3 ../tools/flavors.py get "${FLAVOR}" engine)
OnnxModel="../tmp/export/${FLAVOR}/$(python3 ../tools/flavors.py get "${FLAVOR}" onnx)"
EngineDir="../tmp/engines/${FLAVOR}"

if [ "${SKIP_ENGINE}" != "1" ] && [ ! -f "${TRTEXEC_BIN}" ]; then
    echo "ERROR: trtexec binary not found at ${TRTEXEC_BIN}. Set TRTEXEC_BIN." >&2
    exit 1
fi

function buildCustomPlugin {
    cuda_version=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
    echo "CUDA version: ${cuda_version}"
    src_dir="../src/nvdsinfer_customparser"
    echo "Building custom parser plugin..."
    CUDA_VER=${cuda_version} make -C "${src_dir}" || return 1
    cp "${src_dir}/libnvds_infercustomparser.so" "${MODEL_DIR}/" || return 1
    echo "Custom parser plugin built successfully."
}

function buildEngineFromOnnx {
    if [ ! -f "${OnnxModel}" ]; then
        echo "ERROR: ${OnnxModel#../} not found. Run: python3 tools/export_model.py ${FLAVOR}" >&2
        return 1
    fi
    mkdir -p "${EngineDir}"

    # Fixed 640x640 input, batch 1 to 8 (the Ultralytics ONNX files have dynamic height and width).
    TRTEXEC_CMD=("${TRTEXEC_BIN}"
        "--onnx=${OnnxModel}"
        --fp16
        "--workspace=${WORKSPACE_MIB}"
        --minShapes=images:1x3x640x640
        --optShapes=images:8x3x640x640
        --maxShapes=images:8x3x640x640
        "--saveEngine=${EngineDir}/${EngineFile}"
        "--timingCacheFile=../tmp/engines/timing.cache"
        --useSpinWait)

    echo "Generating Engine file: ${EngineFile} (this can take several minutes on a Jetson Orin NX)"

    LOG_FILE="${EngineDir}/build.log"
    if "${TRTEXEC_CMD[@]}" >"${LOG_FILE}" 2>&1; then
        echo "Finished building ${EngineFile}"
    else
        echo "ERROR: Failed to build the engine for flavor \"${FLAVOR}\". Check ${LOG_FILE#../} for more information." >&2
        return 1
    fi
}

function installEngine {
    mkdir -p "${MODEL_DIR}/1"
    # The package must carry the engine of one flavor only.
    find "${MODEL_DIR}/1" -maxdepth 1 -name '*.engine' ! -name "${EngineFile}" -delete
    cp "${EngineDir}/${EngineFile}" "${MODEL_DIR}/1/${EngineFile}"
}

echo "Flavor: ${FLAVOR}"
echo "Building the custom nvdsinfer_customparser plugin file .."
buildCustomPlugin || exit 1
echo "nvdsinfer_customparser.so lib built successfully"

if [ "${SKIP_ENGINE}" = "1" ]; then
    echo "SKIP_ENGINE=1: keeping the existing TensorRT engine in models/${MODEL_PREFIX}_Yolo_Detector/1/"
else
    echo "Creating optimized TRT engine file .."
    buildEngineFromOnnx || exit 1
    installEngine || exit 1
    echo "${EngineFile} installed in models/${MODEL_PREFIX}_Yolo_Detector/1/"
fi

echo "Generation of the optimized AiProcessor files is finished with success."
