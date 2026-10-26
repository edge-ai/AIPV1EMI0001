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

#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

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
    if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    const char* log_enable = std::getenv("ENABLE_DEBUG");
    static int keep_count_index = -1;
    static int bboxes_index = -1;
    static int scores_index = -1;
    static int classes_index = -1;

    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
        if (strcmp(outputLayersInfo[i].layerName, "num_dets") == 0) {
            keep_count_index = i;
        }
        if (strcmp(outputLayersInfo[i].layerName, "det_boxes") == 0) {
            bboxes_index = i;
        }
        if (strcmp(outputLayersInfo[i].layerName, "det_scores") == 0) {
            scores_index = i;
        }
        if (strcmp(outputLayersInfo[i].layerName, "det_classes") == 0) {
            classes_index = i;
        }
    }

    int* p_keep_count = (int *) outputLayersInfo[keep_count_index].buffer;
    float* p_bboxes = (float *) outputLayersInfo[bboxes_index].buffer;
    NvDsInferDims inferDims_p_bboxes = outputLayersInfo[bboxes_index].inferDims;
    float* p_scores = (float *) outputLayersInfo[scores_index].buffer;
    unsigned int* p_classes = (unsigned int *) outputLayersInfo[classes_index].buffer;
    int numElements_p_bboxes=inferDims_p_bboxes.numElements;

    const float threshold = detectionParams.perClassThreshold[0];
    float max_bbox=0;
    for (int i=0; i < numElements_p_bboxes; i++)
    {
        if ( max_bbox < p_bboxes[i] )
            max_bbox=p_bboxes[i];
    }

    if (p_keep_count[0] > 0)
    {
        assert (!(max_bbox < 2.0));
        for (int i = 0; i < p_keep_count[0]; i++) {

            if ( p_scores[i] < threshold) continue;
            assert((unsigned int) p_classes[i] < detectionParams.numClassesConfigured);

            NvDsInferObjectDetectionInfo object;
            object.classId = (int) p_classes[i];
            object.detectionConfidence = p_scores[i];
            object.left=p_bboxes[4*i];
            object.top=p_bboxes[4*i+1];
            object.width=(p_bboxes[4*i+2] - object.left);
            object.height= (p_bboxes[4*i+3] - object.top);

            if(log_enable != NULL && std::stoi(log_enable)) {
                std::cout << "label/conf/ x/y w/h -- "
                << p_classes[i] << " "
                << p_scores[i] << " "
                << object.left << " " << object.top << " " << object.width << " "<< object.height << " "
                << std::endl;
            }

            object.left=CLIP(object.left, 0, networkInfo.width - 1);
            object.top=CLIP(object.top, 0, networkInfo.height - 1);
            object.width=CLIP(object.width, 0, networkInfo.width - 1);
            object.height=CLIP(object.height, 0, networkInfo.height - 1);
            objectList.push_back(object);
        }
    }
    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomEfficientNMS);
