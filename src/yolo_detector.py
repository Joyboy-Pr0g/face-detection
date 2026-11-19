import time
from ultralytics import YOLO


class YoloFaceDetector:
    def __init__(self, cfg):
        self.conf = cfg["conf"]
        self.iou = cfg["nms_iou"]
        self.imgsz = cfg["imgsz"]
        self.device = cfg["device"]

        self.model = YOLO(cfg["weights"])
        if len(self.model.names) != 1:
            raise ValueError(
                f"Expected a 1-class face model, got {self.model.names}")

    def detect(self, img_bgr):
        t0 = time.perf_counter()
        results = self.model.predict(
            img_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        t1 = time.perf_counter()

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy().tolist()
        scores = r.boxes.conf.cpu().numpy().tolist()

        return {
            "boxes": boxes,
            "scores": scores,
            "count": len(boxes),
            "time_ms": (t1 - t0) * 1000.0,
            "speed": r.speed,   # preprocess / inference / postprocess ms
        }