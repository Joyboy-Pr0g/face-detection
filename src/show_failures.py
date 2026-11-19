import json, os, sys
import cv2
import pandas as pd
from dataset import load_dataset
from utils import load_image, draw_boxes

det, var, name = sys.argv[1:4]
df = pd.read_csv("results/per_image.csv")
row = df[(df.detector == det) & (df.variant == var) & (df.image == name)].iloc[0]
pred = json.loads(row["boxes"])

item = next(i for i in load_dataset() if i["name"] == name)
img = load_image(item["path"])
img = draw_boxes(img, item["gt_boxes"], (0, 255, 0), 2)
img = draw_boxes(img, pred, (0, 0, 255), 2)

os.makedirs("results/failures", exist_ok=True)
out = f"results/failures/{det}_{var}_{name.replace('/', '_')}.png"
cv2.imwrite(out, img)
print(out, "| green = ground truth, red = predictions")

from evaluation import iou
for g in item["gt_boxes"]:
    best = max((iou(p, g) for p in pred), default=0.0)
    print("GT", [round(v) for v in g], "best IoU:", round(best, 2))