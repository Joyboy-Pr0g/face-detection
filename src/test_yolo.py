import sys, cv2
from utils import load_config, load_image, draw_boxes
from yolo_detector import YoloFaceDetector

cfg = load_config("config.yaml")
det = YoloFaceDetector(cfg["yolo"])
print("classes:", det.model.names, "| task:", det.model.task)

img = load_image(sys.argv[1], cfg["image"]["max_side"])
det.detect(img)                      # warm-up run, result discarded
res = det.detect(img)
print(res["count"], "faces,", round(res["time_ms"], 1), "ms", res["speed"])
cv2.imwrite("results/yolo_out.png", draw_boxes(img, res["boxes"], (0, 0, 255)))