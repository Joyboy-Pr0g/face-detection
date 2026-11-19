def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(pred, gt, thr=0.5):
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            v = iou(p, g)
            if v >= thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)

    used_p, used_g, matches = set(), set(), []
    for v, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matches.append((i, j, v))
    return matches


def evaluate_image(pred, gt, thr=0.5):
    tp = len(match(pred, gt, thr))
    return {"tp": tp,
            "fp": len(pred) - tp,
            "fn": len(gt) - tp,
            "pred_count": len(pred),
            "gt_count": len(gt),
            "count_err": abs(len(pred) - len(gt))}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1