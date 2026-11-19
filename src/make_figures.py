import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import load_config, load_image
from dataset import load_dataset
from enhance import brightness_contrast, hist_equalize, clahe

FIG_DIR = "results/figures"
PICKS = ["low_light/6", "normal/1", "far_faces/3"]
os.makedirs(FIG_DIR, exist_ok=True)

cfg = load_config("config.yaml")
e = cfg["enhance"]
variants = {
    "original": lambda im: im,
    "brightness/contrast": lambda im: brightness_contrast(
        im, e["contrast_alpha"], e["brightness_beta"]),
    "histogram equalization": hist_equalize,
    "CLAHE": lambda im: clahe(im, e["clahe_clip"], e["clahe_tile"]),
}
items = {it["name"]: it for it in load_dataset()}


def blur_faces(img, boxes):
    out = img.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        pad = 0.2 * (x2 - x1)
        a, b = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        c, d = int(min(w, x2 + pad)), int(min(h, y2 + pad))
        roi = out[b:d, a:c]
        out[b:d, a:c] = cv2.GaussianBlur(roi, (0, 0), max(4.0, (c - a) / 5))
    return out


for name in PICKS:
    it = items[name]
    img = load_image(it["path"], cfg["image"]["max_side"])
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for col, (vname, fn) in enumerate(variants.items()):
        v = fn(img)
        gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
        shown = blur_faces(v, it["gt_boxes"])
        axes[0, col].imshow(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(vname)
        axes[0, col].axis("off")
        axes[1, col].hist(gray.ravel(), bins=64, range=(0, 255))
        axes[1, col].set_title(f"mean {gray.mean():.1f}   std {gray.std():.1f}")
        axes[1, col].set_xlabel("gray level")
        axes[1, col].set_ylabel("pixels")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "hist_" + name.replace("/", "_") + ".png"),
                dpi=150)
    plt.close(fig)

df = pd.read_csv("results/per_image.csv")


def f1_of(g):
    tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


order = ["original", "bc", "he", "clahe"]
names = ["original", "B/C", "HE", "CLAHE"]
conds = ["normal", "low_light", "far_faces"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, c in zip(axes, conds):
    sub = df[df.cond == c]
    for k, det in enumerate(["haar", "yolo"]):
        vals = [f1_of(sub[(sub.detector == det) & (sub.variant == v)])
                for v in order]
        ax.bar([i + 0.4 * k for i in range(4)], vals, width=0.4, label=det)
    ax.set_xticks([i + 0.2 for i in range(4)])
    ax.set_xticklabels(names)
    ax.set_title(c)
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("F1 (IoU >= 0.5)")
axes[0].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "f1_by_condition.png"), dpi=150)
plt.close(fig)
print("figures saved in", FIG_DIR)