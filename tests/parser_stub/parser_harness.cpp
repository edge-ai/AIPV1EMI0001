// Feeds synthetic network outputs to both bounding-box parsers and prints one line per
// detection: "<parser> <aip class> <score> <left> <top> <width> <height>".
#include <cstdio>
#include <string>

#include "nvdsinfer_custom_impl.h"
#include "../../src/nvdsinfer_customparser/nvdsinfer_custombboxparser.cpp"

static NvDsInferLayerInfo layer(const char *name, void *buffer, std::vector<unsigned int> dims)
{
    NvDsInferLayerInfo info = {};
    info.layerName = name;
    info.buffer = buffer;
    info.inferDims.numDims = (unsigned int) dims.size();
    info.inferDims.numElements = 1;
    for (size_t i = 0; i < dims.size(); i++) {
        info.inferDims.d[i] = dims[i];
        info.inferDims.numElements *= dims[i];
    }
    return info;
}

static void print(const char *parser, bool ok, const std::vector<NvDsInferObjectDetectionInfo> &objects)
{
    std::printf("%s ok=%d count=%zu\n", parser, ok ? 1 : 0, objects.size());
    for (const auto &o : objects)
        std::printf("%s %u %.2f %.1f %.1f %.1f %.1f\n", parser, o.classId, o.detectionConfidence, o.left, o.top,
                    o.width, o.height);
}

int main()
{
    NvDsInferNetworkInfo network = {640, 640, 3};
    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 3;
    params.perClassPreclusterThreshold = {0.35f, 0.35f, 0.35f};

    {   // EfficientNMS: 5 detections of which dog(16), low-score person and out-of-range class are dropped.
        int num_dets[1] = {5};
        float boxes[100 * 4] = {10, 20, 110, 220,      // car     0.90
                                -5, 30, 50, 700,       // person  0.80 (clipped)
                                1, 2, 3, 4,            // dog     0.99 (ignored)
                                100, 100, 200, 200,    // truck   0.20 (below threshold)
                                300, 300, 400, 500};   // truck   0.50
        float scores[100] = {0.90f, 0.80f, 0.99f, 0.20f, 0.50f};
        int classes[100] = {2, 0, 16, 7, 7};
        std::vector<NvDsInferLayerInfo> layers = {layer("num_dets", num_dets, {1}),
                                                  layer("det_boxes", boxes, {100, 4}),
                                                  layer("det_scores", scores, {100}),
                                                  layer("det_classes", classes, {100})};
        std::vector<NvDsInferObjectDetectionInfo> objects;
        print("nms", NvDsInferParseCustomEfficientNMS(layers, network, params, objects), objects);

        std::vector<NvDsInferLayerInfo> bad = {layer("num_dets", num_dets, {1})};
        std::vector<NvDsInferObjectDetectionInfo> none;
        print("nms_bad", NvDsInferParseCustomEfficientNMS(bad, network, params, none), none);
    }

    {   // End to end: rows of x1, y1, x2, y2, score, class.
        float rows[5 * 6] = {10, 20, 110, 220, 0.90f, 2,        // car
                             -5, 30, 50, 700, 0.80f, 0,         // person (clipped)
                             1, 2, 3, 4, 0.99f, 16,             // dog (ignored)
                             100, 100, 200, 200, 0.20f, 7,      // truck below threshold
                             300, 300, 400, 500, 0.50f, 7};     // truck
        std::vector<NvDsInferLayerInfo> layers = {layer("output0", rows, {5, 6})};
        std::vector<NvDsInferObjectDetectionInfo> objects;
        print("e2e", NvDsInferParseYoloEndToEnd(layers, network, params, objects), objects);

        std::vector<NvDsInferLayerInfo> bad = {layer("output0", rows, {5, 7})};
        std::vector<NvDsInferObjectDetectionInfo> none;
        print("e2e_bad", NvDsInferParseYoloEndToEnd(bad, network, params, none), none);
    }
    return 0;
}
