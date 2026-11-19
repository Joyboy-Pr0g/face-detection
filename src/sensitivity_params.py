import pandas as pd

from utils import load_config, load_image
from dataset import load_dataset
from haar_detector import HaarDetector
from yolo_detector import YoloFaceDetector
from enhance import brightness_contrast
from evaluation import evaluate_image, prf

cfg = load_config("config.yaml")
thr = cfg["eval"]["iou_thr"]
e = cfg["enhance"]
haar = HaarDetector(cfg["haar"])
yolo = YoloFaceDetector(cfg["yolo"])

items = load_dataset()
orig = {it["name"]: load_image(it["path"], cfg["image"]["max_side"])
        for it in items}

sample = next(iter(orig.values()))
for _ in range(cfg["timing"]["warmup_runs"]):
    haar.detect(sample)
    yolo.detect(sample)

hd = cfg["haar"]
haar_default = {"scale_factor": hd["scale_factor"],
                "min_neighbors": hd["min_neighbors"],
                "min_size": hd["min_size"][0]}
yd = cfg["yolo"]
yolo_default = {"conf": yd["conf"], "nms_iou": yd["nms_iou"]}

haar_grid = {"scale_factor": [1.05, 1.2, 1.3],
             "min_neighbors": [3, 8],
             "min_size": [20, 40]}
yolo_grid = {"conf": [0.15, 0.40, 0.60, 0.80],
             "nms_iou": [0.30, 0.60]}


def set_haar(over):
    p = dict(haar_default)
    p.update(over)
    haar.scale_factor = p["scale_factor"]
    haar.min_neighbors = p["min_neighbors"]
    haar.min_size = (p["min_size"], p["min_size"])


def set_yolo(over):
    p = dict(yolo_default)
    p.update(over)
    yolo.conf = p["conf"]
    yolo.iou = p["nms_iou"]


def score(det, imgs, cond=None):
    tp = fp = fn = 0
    errs, times = [], []
    for it in items:
        if cond is not None and it["cond"] != cond:
            continue
        res = det.detect(imgs[it["name"]])
        ev = evaluate_image(res["boxes"], it["gt_boxes"], thr)
        tp += ev["tp"]
        fp += ev["fp"]
        fn += ev["fn"]
        errs.append(ev["count_err"])
        times.append(res["time_ms"])
    p, r, f1 = prf(tp, fp, fn)
    return {"TP": tp, "FP": fp, "FN": fn, "precision": round(p, 3),
            "recall": round(r, 3), "F1": round(f1, 3),
            "MAE": round(sum(errs) / len(errs), 3),
            "detect_ms": round(sum(times) / len(times), 1)}


rows = []


def add(dname, experiment, setting, det, imgs, conds):
    for c in conds:
        rows.append({"detector": dname, "experiment": experiment,
                     "setting": setting, "cond": c or "all",
                     **score(det, imgs, c)})


haar_cfgs = [("default", {})] + [(f"{k}={v}", {k: v})
                                 for k, vs in haar_grid.items() for v in vs]
for name, over in haar_cfgs:
    set_haar(over)
    add("haar", "param", name, haar, orig, [None, "far_faces"])
set_haar({})

yolo_cfgs = [("default", {})] + [(f"{k}={v}", {k: v})
                                 for k, vs in yolo_grid.items() for v in vs]
for name, over in yolo_cfgs:
    set_yolo(over)
    add("yolo", "param", name, yolo, orig, [None])
set_yolo({})

ablation = {
    "original": lambda im: im,
    "alpha_only": lambda im: brightness_contrast(im, e["contrast_alpha"], 0),
    "beta_only": lambda im: brightness_contrast(im, 1.0, e["brightness_beta"]),
    "alpha_and_beta": lambda im: brightness_contrast(
        im, e["contrast_alpha"], e["brightness_beta"]),
}
for name, fn in ablation.items():
    imgs = {k: fn(v) for k, v in orig.items()}
    add("haar", "bc_ablation", name, haar, imgs, [None, "low_light"])
    add("yolo", "bc_ablation", name, yolo, imgs, [None, "low_light"])

df = pd.DataFrame(rows)
df.to_csv("results/sensitivity_params.csv", index=False)
for (d, ex), g in df.groupby(["detector", "experiment"]):
    print(d, ex)
    print(g.drop(columns=["detector", "experiment"]).to_string(index=False))
    print()