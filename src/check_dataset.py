import os, cv2
from dataset import load_dataset
from utils import load_image, draw_boxes

items = load_dataset()
os.makedirs("results/gt_check", exist_ok=True)
per_cond = {}
for it in items:
    per_cond.setdefault(it["cond"], []).append(it["gt_count"])
    print(f'{it["name"]:16s} {it["gt_count"]}')
    img = load_image(it["path"])
    out = draw_boxes(img, it["gt_boxes"])
    cv2.imwrite(f'results/gt_check/{it["cond"]}_{os.path.basename(it["path"])}', out)

for c, v in per_cond.items():
    print(c, "images:", len(v), "faces:", sum(v))
print("total images:", len(items), "| total faces:", sum(i["gt_count"] for i in items))