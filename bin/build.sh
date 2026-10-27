#!/bin/bash
#
# build.sh — build the two runtime artifacts of this AIP:
#
#   1. libnvds_infercustomparser.so
#      Custom DeepStream bounding-box parser (EfficientNMS_TRT output ->
#      NvDsInferObjectDetectionInfo). Compiled from src/nvdsinfer_customparser/
#      with make; requires CUDA toolkit + DeepStream headers (present in the
#      ms3_videostreaming / ms8_jupyter containers on the station).
#
#   2. models/AIPV1EMI0001_Yolo_Detector/1/yolov7-fp16-1x8x8.engine
#      TensorRT FP16 engine built by trtexec from the ONNX model
#      (models/AIPV1EMI0001_Yolo_Detector/yolov7tiny-parking-rev3-640.onnx),
#      which must be downloaded first (see localdev.ipynb / README).
#
# Usage:
#   cd bin && ./build.sh
#   TRTEXEC_BIN=/path/to/trtexec ./build.sh   # override trtexec location
#   SKIP_ENGINE=1 ./build.sh                  # parser only (prebuilt engine already in models/.../1/)
#
# --workspace=<MiB> (trtexec): maximum GPU scratch memory TensorRT may use
# during the build. Higher values let TensorRT pick faster tactics (more
# algorithms fit in memory); lower values reduce peak build-time memory but
# can exclude some tactics — the resulting engine may be slightly slower and
# the build can take longer. It does NOT change the engine's runtime memory
# footprint significantly.
#
#   - 4096 MiB is the historical default and works on x86 dev machines.
#   - On a 15 GB Jetson Orin station running the full EdgeAI stack, 4096 MiB
#     gets the build OOM-killed (observed on an Orin NX 16GB); 1024 MiB succeeds.
#   - Override with WORKSPACE_MIB env var if you hit OOM on a busy station.
#
set -e

MODEL_PREFIX="AIPV1EMI0001"
WORKSPACE_MIB="${WORKSPACE_MIB:-1024}"
TRTEXEC_BIN="${TRTEXEC_BIN:-/usr/src/tensorrt/bin/trtexec}"
SKIP_ENGINE="${SKIP_ENGINE:-0}"

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
    cp "${src_dir}/libnvds_infercustomparser.so" "../models/${MODEL_PREFIX}_Yolo_Detector/" || return 1
    echo "Custom parser plugin built successfully."
}

function buildEngineFromOnnx {
    ModelDir=$1
    EngineFile="yolov7-fp16-1x8x8.engine"
    OnnxModel="yolov7tiny-parking-rev3-640.onnx"
    mkdir -p "../models/${MODEL_PREFIX}_Yolo_Detector/1"

    TRTEXEC_CMD=("${TRTEXEC_BIN}"
        "--onnx=../models/${MODEL_PREFIX}_Yolo_Detector/${OnnxModel}"
        --fp16
        "--workspace=${WORKSPACE_MIB}"
        --minShapes=images:1x3x640x640
        --optShapes=images:8x3x640x640
        --maxShapes=images:8x3x640x640
        "--saveEngine=../models/${MODEL_PREFIX}_Yolo_Detector/1/${EngineFile}"
        --timingCacheFile=timing.cache
        --useSpinWait)

    echo "Generating Engine file: ${EngineFile}"

    LOG_FILE="buildModel${ModelDir}.log"
    LOG_FILE=${LOG_FILE//[\/]/_}
    if "${TRTEXEC_CMD[@]}" >>"${LOG_FILE}" 2>&1; then
        echo "Finished building Model ${ModelDir}"
        rm "${LOG_FILE}"
    else
        echo "ERROR: Failed to build engine for model \"$ModelDir\". Check ${LOG_FILE} for more information."
        return 1
    fi
}

echo "Building the custom nvdsinfer_customparser plugin file .."
buildCustomPlugin || exit 1
echo "nvdsinfer_customparser.so lib built successfully"

if [ "${SKIP_ENGINE}" = "1" ]; then
    echo "SKIP_ENGINE=1: keeping the existing TensorRT engine in models/${MODEL_PREFIX}_Yolo_Detector/1/"
else
    echo "Creating optimized TRT engine file .."
    buildEngineFromOnnx "${MODEL_PREFIX}_Yolo_Detector" || exit 1
    echo "${MODEL_PREFIX}_Yolo_Detector optimized TRT engine file created successfully."
fi

echo "Generation of the optimized AiProcessor files is finished with success."
