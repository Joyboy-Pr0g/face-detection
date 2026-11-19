import time
import cv2


class HaarDetector:
    def __init__(self, cfg):
        self.scale_factor = cfg["scale_factor"]
        self.min_neighbors = cfg["min_neighbors"]
        self.min_size = tuple(cfg["min_size"])

        self.cascade = cv2.CascadeClassifier(cfg["cascade"])
        if self.cascade.empty():
            raise FileNotFoundError(
                f"Could not load Haar cascade: {cfg['cascade']}")

    def detect(self, img_bgr):
        t0 = time.perf_counter()

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        rects = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        t1 = time.perf_counter()

        boxes = [[float(x), float(y), float(x + w), float(y + h)]
                 for (x, y, w, h) in rects]

        return {
            "boxes": boxes,
            "scores": [1.0] * len(boxes),
            "count": len(boxes),
            "time_ms": (t1 - t0) * 1000.0,
        }