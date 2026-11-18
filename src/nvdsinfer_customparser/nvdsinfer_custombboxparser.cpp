/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
 * Bounding-box parsers shared by the three YOLO flavors of this AI Processor.
 *
 * Every flavor is an official COCO-pretrained model with 80 classes, while the
 * AI Processor publishes only three: car, person and truck (labels-rev3.txt).
 * Both parsers therefore translate the COCO class id into the AIP class id and
 * drop every other class.
 *
 *   NvDsInferParseCustomEfficientNMS  YOLOv7  : TensorRT EfficientNMS outputs
 *                                               num_dets / det_boxes / det_scores / det_classes
 *   NvDsInferParseYoloEndToEnd        YOLOv10 and YOLO26 : NMS-free output0 [300, 6]
 *                                               rows of x1, y1, x2, y2, score, class
 *
 * Boxes are in network-input pixels (640 x 640); DeepStream rescales them.
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include "nvdsinfer_custom_impl.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))

static const int kNumCocoClasses = 80;

/* COCO class id -> AIP class id (index in labels-rev3.txt: car, person, truck); -1 = ignored. */
static int cocoToAip(int cocoClass)
{
    switch (cocoClass) {
        case 0: return 1; /* person */
        case 2: return 0; /* car    */
        case 7: return 2; /* truck  */
        default: return -1;
    }
}

static bool debugEnabled()
{
    const char *value = std::getenv("ENABLE_DEBUG");
    return value != NULL && std::atoi(value) != 0;
}

static float classThreshold(NvDsInferParseDetectionParams const &detectionParams, int aipClass)
{
    const std::vector<float> &thresholds = detectionParams.perClassThreshold;
    if (thresholds.empty())
        return 0.f;
    return thresholds[(size_t) aipClass < thresholds.size() ? aipClass : 0];
}

static void addObject(NvDsInferNetworkInfo const &networkInfo,
                      std::vector<NvDsInferObjectDetectionInfo> &objectList,
                      float x1, float y1, float x2, float y2, float score, int cocoClass, int aipClass)
{
    const float maxX = (float) networkInfo.width - 1;
    const float maxY = (float) networkInfo.height - 1;
    x1 = CLIP(x1, 0, maxX);
    y1 = CLIP(y1, 0, maxY);
    x2 = CLIP(x2, 0, maxX);
    y2 = CLIP(y2, 0, maxY);

    NvDsInferObjectDetectionInfo object;
    object.classId = (unsigned int) aipClass;
    object.detectionConfidence = score;
    object.left = x1;
    object.top = y1;
    object.width = x2 - x1;
    object.height = y2 - y1;

    if (debugEnabled()) {
        std::cout << "coco/aip class, conf, x/y w/h -- " << cocoClass << "/" << aipClass << " " << score
                  << " " << object.left << " " << object.top << " " << object.width << " " << object.height
                  << std::endl;
    }
    objectList.push_back(object);
}

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomEfficientNMS (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomEfficientNMS (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    if (outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers. "
                  << "Expected 4 output buffers, detected in the network: "
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    const NvDsInferLayerInfo *numDets = NULL, *boxes = NULL, *scores = NULL, *classes = NULL;
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
        const char *name = outputLayersInfo[i].layerName;
        if (strcmp(name, "num_dets") == 0) numDets = &outputLayersInfo[i];
        if (strcmp(name, "det_boxes") == 0) boxes = &outputLayersInfo[i];
        if (strcmp(name, "det_scores") == 0) scores = &outputLayersInfo[i];
        if (strcmp(name, "det_classes") == 0) classes = &outputLayersInfo[i];
    }
    if (!numDets || !boxes || !scores || !classes)
    {
        std::cerr << "Expected the outputs num_dets, det_boxes, det_scores and det_classes." << std::endl;
        return false;
    }

    const int count = ((const int *) numDets->buffer)[0];
    const float *p_bboxes = (const float *) boxes->buffer;
    const float *p_scores = (const float *) scores->buffer;
    const int *p_classes = (const int *) classes->buffer;
    const int capacity = (int) (boxes->inferDims.numElements / 4);

    for (int i = 0; i < count && i < capacity; i++) {
        const int aipClass = cocoToAip(p_classes[i]);
        if (aipClass < 0 || p_scores[i] < classThreshold(detectionParams, aipClass)) continue;
        addObject(networkInfo, objectList,
                  p_bboxes[4 * i], p_bboxes[4 * i + 1], p_bboxes[4 * i + 2], p_bboxes[4 * i + 3],
                  p_scores[i], p_classes[i], aipClass);
    }
    return true;
}

extern "C"
bool NvDsInferParseYoloEndToEnd (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseYoloEndToEnd (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo  const &networkInfo,
                                 NvDsInferParseDetectionParams const &detectionParams,
                                 std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    if (outputLayersInfo.size() != 1)
    {
        std::cerr << "Mismatch in the number of output buffers. "
                  << "Expected 1 output buffer (output0), detected in the network: "
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    if (layer.inferDims.numDims != 2 || layer.inferDims.d[1] != 6)
    {
        std::cerr << "Expected output0 with dims [N, 6] (x1, y1, x2, y2, score, class)." << std::endl;
        return false;
    }

    const float *rows = (const float *) layer.buffer;
    const unsigned int numRows = layer.inferDims.d[0];
    for (unsigned int i = 0; i < numRows; i++) {
        const float *row = rows + 6 * i;
        const int cocoClass = (int) (row[5] + 0.5f);
        const int aipClass = (cocoClass >= 0 && cocoClass < kNumCocoClasses) ? cocoToAip(cocoClass) : -1;
        if (aipClass < 0 || row[4] < classThreshold(detectionParams, aipClass)) continue;
        addObject(networkInfo, objectList, row[0], row[1], row[2], row[3], row[4], cocoClass, aipClass);
    }
    return true;
}

/* Check that the custom functions have been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomEfficientNMS);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloEndToEnd);
