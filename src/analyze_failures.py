import json
import cv2
import pandas as pd
from utils import load_config, load_image
from dataset import load_dataset
from enhance import brightness_contrast, hist_equalize, clahe
from evaluation import match

cfg = load_config("config.yaml")
e = cfg["enhance"]
thr = cfg["eval"]["iou_thr"]
variants = {
    "original": lambda im: im,
    "bc": lambda im: brightness_contrast(im, e["contrast_alpha"], e["brightness_beta"]),
    "he": hist_equalize,
    "clahe": lambda im: clahe(im, e["clahe_clip"], e["clahe_tile"]),
}

res = pd.read_csv("results/per_image.csv")
pred = {(r.detector, r.variant, r.image): json.loads(r.boxes)
        for r in res.itertuples()}

img_rows, face_rows = [], []
for it in load_dataset():
    img = load_image(it["path"], cfg["image"]["max_side"])
    for vname, fn in variants.items():
        gray = cv2.cvtColor(fn(img), cv2.COLOR_BGR2GRAY)
        img_rows.append({"cond": it["cond"], "image": it["name"],
                         "variant": vname,
                         "img_mean": gray.mean(), "img_std": gray.std()})

        found = {}
        for det in ("haar", "yolo"):
            m = match(pred[(det, vname, it["name"])], it["gt_boxes"], thr)
            found[det] = {j for _, j, _ in m}

        for j, (x1, y1, x2, y2) in enumerate(it["gt_boxes"]):
            crop = gray[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)]
            face_rows.append({"cond": it["cond"], "image": it["name"],
                              "face": j, "variant": vname,
                              "width": x2 - x1, "height": y2 - y1,
                              "face_mean": crop.mean(), "face_std": crop.std(),
                              "haar_found": j in found["haar"],
                              "yolo_found": j in found["yolo"]})

img_df = pd.DataFrame(img_rows)
face_df = pd.DataFrame(face_rows)
img_df.to_csv("results/stats_images.csv", index=False)
face_df.to_csv("results/stats_faces.csv", index=False)

print("== 1. Whole image: brightness (mean) and contrast (std) ==")
print(img_df.groupby(["cond", "variant"])[["img_mean", "img_std"]]
      .mean().round(1).to_string())

print("\n== 2. Face regions: brightness and contrast by version ==")
print(face_df.groupby(["cond", "variant"])[["face_mean", "face_std"]]
      .mean().round(1).to_string())

print("\n== 3. Haar on ORIGINAL images: found vs missed faces ==")
o = face_df[face_df.variant == "original"]
print(o.groupby(["cond", "haar_found"]).agg(
    n=("width", "size"), width=("width", "mean"),
    face_mean=("face_mean", "mean"),
    face_std=("face_std", "mean")).round(1).to_string())

print("\n== 4. Fraction of faces found (recall at IoU 0.5) ==")
print(face_df.groupby(["cond", "variant"])[["haar_found", "yolo_found"]]
      .mean().round(3).to_string())
print("\ncheck: Haar original found =", int(o.haar_found.sum()),
      "of", len(o))