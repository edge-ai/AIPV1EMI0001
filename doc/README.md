# Documentation

| Document | Purpose |
|---|---|
| [`USER_MANUAL.md`](USER_MANUAL.md) | Install the package, deploy the Node-RED dashboard, and the callback metadata contract |
| [`../notebooks/`](../notebooks/) | The tutorial: one presentation notebook, one build notebook per YOLO flavor, and the model comparison |
| [Model measurements](benchmarks/orin-nx-16gb-jp6.0/comparison.md) | COCO detection accuracy and raw-model latency on the Orin NX 16 GB |
| [RTSP camera capacity](benchmarks/orin-nx-16gb-jp6.0/capacity.md) | Processed FPS per camera with tracking, recording and video outputs; includes the limits of the tests |
| [`../src/nvdsinfer_customparser/README`](../src/nvdsinfer_customparser/README) | The DeepStream parser shared by the flavors |
| [`../logic/README.md`](../logic/README.md) | The Node-RED project |
| [`../dashboard/README.md`](../dashboard/README.md) | The dashboard application |

Integration references for the model formats:

- [Triton model configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#auto-generated-model-configuration)
- [YOLOv7 Triton deployment notes](https://github.com/WongKinYiu/yolov7/tree/8eee99fcc5906633a1e0ecdf0fc47adf4ae7a751/deploy/triton-inference-server)
- [Ultralytics ONNX export](https://docs.ultralytics.com/modes/export/)

## Credits

Models: [YOLOv7](https://github.com/WongKinYiu/yolov7) by Chien-Yao Wang, Alexey Bochkovskiy and Hong-Yuan Mark Liao; [YOLOv10](https://github.com/THU-MIG/yolov10) by Ao Wang et al.; [YOLO26](https://docs.ultralytics.com/models/yolo26/) by Ultralytics.

The original AIP was developed at EdgeMatrix by [@antowan](https://github.com/antowan), [@aohta-emi](https://github.com/aohta-emi) and [@yosukehara](https://github.com/yosukehara).
