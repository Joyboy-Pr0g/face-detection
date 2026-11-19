import csv, json, os, platform, sys, time
import cv2, numpy as np, torch, ultralytics
from utils import load_config, load_image
from dataset import load_dataset
from haar_detector import HaarDetector
from yolo_detector import YoloFaceDetector
from enhance import brightness_contrast, hist_equalize, clahe
from evaluation import evaluate_image

cfg = load_config("config.yaml")
e = cfg["enhance"]
thr = cfg["eval"]["iou_thr"]
warmup = cfg["timing"]["warmup_runs"]
repeats = cfg["timing"]["repeats"]
max_side = cfg["image"]["max_side"]

detectors = {"haar": HaarDetector(cfg["haar"]),
             "yolo": YoloFaceDetector(cfg["yolo"])}
variants = {
    "original": lambda im: im,
    "bc": lambda im: brightness_contrast(im, e["contrast_alpha"], e["brightness_beta"]),
    "he": hist_equalize,
    "clahe": lambda im: clahe(im, e["clahe_clip"], e["clahe_tile"]),
}

items = load_dataset()
first = load_image(items[0]["path"], max_side)
for det in detectors.values():
    for _ in range(warmup):
        det.detect(first)

rows = []
for n, it in enumerate(items, 1):
    img = load_image(it["path"], max_side)
    for vname, fn in variants.items():
        enh_times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            enh = fn(img)
            enh_times.append((time.perf_counter() - t0) * 1000)

        for dname, det in detectors.items():
            times = []
            for _ in range(repeats):
                res = det.detect(enh)
                times.append(res["time_ms"])
            ev = evaluate_image(res["boxes"], it["gt_boxes"], thr)
            rows.append({
                "detector": dname, "variant": vname,
                "cond": it["cond"], "image": it["name"],
                "gt_count": ev["gt_count"], "pred_count": ev["pred_count"],
                "tp": ev["tp"], "fp": ev["fp"], "fn": ev["fn"],
                "count_err": ev["count_err"],
                "detect_ms_mean": round(float(np.mean(times)), 2),
                "detect_ms_std": round(float(np.std(times)), 2),
                "enhance_ms_mean": round(float(np.mean(enh_times)), 2),
                "boxes": json.dumps([[round(v, 1) for v in b] for b in res["boxes"]]),
            })
    print(f"{n}/{len(items)} {it['name']}")

os.makedirs("results", exist_ok=True)
with open("results/per_image.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

info = {
    "config": cfg,
    "python": sys.version,
    "platform": platform.platform(),
    "processor": platform.processor(),
    "opencv": cv2.__version__,
    "numpy": np.__version__,
    "torch": torch.__version__,
    "ultralytics": ultralytics.__version__,
    "timing": "detect_ms = detector call only (Haar: gray conversion + detectMultiScale; "
              "YOLO: predict incl. preprocess, inference, NMS); enhance_ms = enhancement only; "
              "image loading and drawing excluded; mean over repeats after warm-up; CPU",
}
with open("results/run_info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
print("done:", len(rows), "rows")