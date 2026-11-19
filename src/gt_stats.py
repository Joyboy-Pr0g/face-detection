import numpy as np
from dataset import load_dataset

by = {}
for it in load_dataset():
    for x1, y1, x2, y2 in it["gt_boxes"]:
        by.setdefault(it["cond"], []).append((x2 - x1, y2 - y1))

for c, v in by.items():
    w = np.array([a for a, _ in v])
    h = np.array([b for _, b in v])
    print(c, "| faces:", len(w),
          "| width px min/median/max:",
          round(w.min()), round(float(np.median(w))), round(w.max()),
          "| below 30 px:", int((w < 30).sum()),
          "| mean height/width:", round(float((h / w).mean()), 2))