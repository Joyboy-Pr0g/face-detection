import sys, cv2
from utils import load_config, load_image, draw_boxes
from haar_detector import HaarDetector

cfg = load_config("config.yaml")
det = HaarDetector(cfg["haar"])

img = load_image(sys.argv[1], cfg["image"]["max_side"])
res = det.detect(img)

print(res["count"], "faces,", round(res["time_ms"], 1), "ms")

result_img = draw_boxes(img, res["boxes"])

cv2.imshow("haar", result_img)

saved = cv2.imwrite("results/haar_result.jpg", result_img)

if saved:
    print("Image saved successfully!")
else:
    print("Failed to save image.")

cv2.waitKey(0)
cv2.destroyAllWindows()