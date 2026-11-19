import os
import pandas as pd

df = pd.read_csv("results/per_image.csv")


def agg(g):
    tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return pd.Series({"images": len(g), "gt": g.gt_count.sum(),
                      "pred": g.pred_count.sum(), "TP": tp, "FP": fp, "FN": fn,
                      "precision": round(p, 3), "recall": round(r, 3),
                      "F1": round(f1, 3),
                      "MAE": round(g.count_err.mean(), 3),
                      "detect_ms": round(g.detect_ms_mean.mean(), 1),
                      "enhance_ms": round(g.enhance_ms_mean.mean(), 1)})


overall = df.groupby(["detector", "variant"]).apply(agg)
by_cond = df.groupby(["detector", "cond", "variant"]).apply(agg)
overall.to_csv("results/summary_overall.csv")
by_cond.to_csv("results/summary_by_condition.csv")
print(overall.to_string())

orig = df[df.variant == "original"].set_index(["detector", "image"])["boxes"]
for (det, var), g in df[df.variant != "original"].groupby(["detector", "variant"]):
    changed = sum(b != orig[(det, img)] for img, b in zip(g.image, g.boxes))
    print(det, var, "images with different boxes vs original:", changed, "/", len(g))

bad = df[(df.fp > 0) | (df.fn > 0)]
bad = bad[["detector", "variant", "image", "gt_count", "pred_count", "tp", "fp", "fn"]]
bad.to_csv("results/failures.csv", index=False)
print("failure rows:", len(bad))