import json
import pandas as pd
from dataset import load_dataset
from evaluation import evaluate_image, prf

df = pd.read_csv("results/per_image.csv")
gt = {it["name"]: it["gt_boxes"] for it in load_dataset()}

rows = []
for c in ["all"] + sorted(df.cond.unique()):
    sub = df if c == "all" else df[df.cond == c]
    for thr in (0.3, 0.4, 0.5):
        for (det, var), g in sub.groupby(["detector", "variant"]):
            tp = fp = fn = 0
            for img, b in zip(g.image, g.boxes):
                r = evaluate_image(json.loads(b), gt[img], thr)
                tp += r["tp"]
                fp += r["fp"]
                fn += r["fn"]
            p, rc, f1 = prf(tp, fp, fn)
            rows.append({"cond": c, "iou_thr": thr, "detector": det,
                         "variant": var, "TP": tp, "FP": fp, "FN": fn,
                         "precision": round(p, 3), "recall": round(rc, 3),
                         "F1": round(f1, 3)})

out = pd.DataFrame(rows)
out.to_csv("results/summary_iou_sensitivity.csv", index=False)
print(out[out.variant == "original"].to_string(index=False))