import random
import time

import pandas as pd

from utils import load_config, load_image, draw_boxes
from dataset import load_dataset
from haar_detector import HaarDetector
from yolo_detector import YoloFaceDetector
from enhance import brightness_contrast, hist_equalize, clahe

SEED = 0
cfg = load_config("config.yaml")
e = cfg["enhance"]
rounds = cfg["timing"]["repeats"]
warmup = cfg["timing"]["warmup_runs"]

detectors = {"haar": HaarDetector(cfg["haar"]),
             "yolo": YoloFaceDetector(cfg["yolo"])}
variants = {
    "original": lambda im: im,
    "bc": lambda im: brightness_contrast(im, e["contrast_alpha"], e["brightness_beta"]),
    "he": hist_equalize,
    "clahe": lambda im: clahe(im, e["clahe_clip"], e["clahe_tile"]),
}

items = load_dataset()
base = {it["name"]: load_image(it["path"], cfg["image"]["max_side"])
        for it in items}

first = next(iter(base.values()))
for det in detectors.values():
    for _ in range(warmup):
        det.detect(first)

jobs = [(d, v, n) for d in detectors for v in variants for n in base]
rng = random.Random(SEED)
rows = []
for rnd in range(rounds):
    rng.shuffle(jobs)
    for d, v, n in jobs:
        t0 = time.perf_counter()
        img = variants[v](base[n])
        t1 = time.perf_counter()
        res = detectors[d].detect(img)
        t2 = time.perf_counter()
        draw_boxes(img, res["boxes"])
        t3 = time.perf_counter()
        rows.append({"round": rnd, "detector": d, "variant": v, "image": n,
                     "enhance_ms": (t1 - t0) * 1000,
                     "detect_ms": res["time_ms"],
                     "draw_ms": (t3 - t2) * 1000,
                     "e2e_ms": (t3 - t0) * 1000})
    print("round", rnd + 1, "/", rounds)

df = pd.DataFrame(rows)
df.to_csv("results/timing_raw.csv", index=False)

cols = ["enhance_ms", "detect_ms", "draw_ms", "e2e_ms"]
summary = df.groupby(["detector", "variant"])[cols].agg(["mean", "std"]).round(1)
summary.to_csv("results/timing_summary.csv")
print(summary.to_string())

print()
print(df.groupby("detector")["detect_ms"].agg(["mean", "std", "median"]).round(1))