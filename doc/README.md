# Documentation

| Document | Purpose |
|---|---|
| [`USER_MANUAL.md`](USER_MANUAL.md) | Install the package, deploy the Node-RED dashboard, and the callback metadata contract |
| [`../notebooks/`](../notebooks/) | The tutorial: one presentation notebook, one build notebook per YOLO flavor, and the model comparison |
| [`benchmarks/`](benchmarks/) | Reference measurements on an EdgeAI Station: latency and COCO accuracy of each flavor |
| [`../src/nvdsinfer_customparser/README`](../src/nvdsinfer_customparser/README) | The DeepStream parser shared by the flavors |
| [`../logic/README.md`](../logic/README.md) | The Node-RED project |
| [`../dashboard/README.md`](../dashboard/README.md) | The dashboard application |

Integration references for the model formats:

- [Triton model configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#auto-generated-model-configuration)
- [YOLOv7 Triton deployment notes](https://github.com/WongKinYiu/yolov7/tree/8eee99fcc5906633a1e0ecdf0fc47adf4ae7a751/deploy/triton-inference-server)
- [Ultralytics ONNX export](https://docs.ultralytics.com/modes/export/)
