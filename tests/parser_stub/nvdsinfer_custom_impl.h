// Minimal stand-in for DeepStream's nvdsinfer_custom_impl.h, only for unit-testing the
// bounding-box parsers on machines without DeepStream. The field layout of the structs below
// follows the public DeepStream 7.0 headers (nvdsinfer.h, nvdsinfer_custom_impl.h).
#pragma once

#include <vector>

#define NVDSINFER_MAX_DIMS 8

typedef struct
{
    unsigned int numDims;
    unsigned int d[NVDSINFER_MAX_DIMS];
    unsigned int numElements;
} NvDsInferDims;

typedef struct
{
    unsigned int width;
    unsigned int height;
    unsigned int channels;
} NvDsInferNetworkInfo;

typedef struct
{
    NvDsInferDims inferDims;
    int bindingIndex;
    const char *layerName;
    void *buffer;
    int isInput;
} NvDsInferLayerInfo;

typedef struct
{
    unsigned int classId;
    float left;
    float top;
    float width;
    float height;
    float detectionConfidence;
} NvDsInferObjectDetectionInfo;

typedef struct
{
    unsigned int numClassesConfigured;
    std::vector<float> perClassPreclusterThreshold;
    std::vector<float> perClassPostclusterThreshold;
    std::vector<float> &perClassThreshold = perClassPreclusterThreshold;
} NvDsInferParseDetectionParams;

typedef bool (*NvDsInferParseCustomFunc)(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                         NvDsInferNetworkInfo const &networkInfo,
                                         NvDsInferParseDetectionParams const &detectionParams,
                                         std::vector<NvDsInferObjectDetectionInfo> &objectList);

#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(customParseFunc) \
    static void checkFunc_##customParseFunc(NvDsInferParseCustomFunc func = customParseFunc) \
    { checkFunc_##customParseFunc(); }; \
    extern "C" bool customParseFunc(std::vector<NvDsInferLayerInfo> const &outputLayersInfo, \
                                    NvDsInferNetworkInfo const &networkInfo, \
                                    NvDsInferParseDetectionParams const &detectionParams, \
                                    std::vector<NvDsInferObjectDetectionInfo> &objectList);
