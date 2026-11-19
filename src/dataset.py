import os, glob
from PIL import Image


def load_dataset(img_root="dataset/images", lbl_root="dataset/labels"):
    paths = sorted(
        glob.glob(os.path.join(img_root, "*", "*.jpg")),
        key=lambda p: (os.path.dirname(p),
                       int(os.path.splitext(os.path.basename(p))[0])))
    items = []
    for p in paths:
        rel = os.path.relpath(p, img_root)
        cond = os.path.dirname(rel)
        stem = os.path.splitext(os.path.basename(rel))[0]
        lbl = os.path.join(lbl_root, cond, stem + ".txt")
        if not os.path.exists(lbl):
            raise FileNotFoundError(f"missing label file: {lbl}")

        w, h = Image.open(p).size
        gt = []
        with open(lbl) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, xc, yc, bw, bh = map(float, parts)
                gt.append([(xc - bw / 2) * w, (yc - bh / 2) * h,
                           (xc + bw / 2) * w, (yc + bh / 2) * h])

        items.append({"cond": cond, "name": f"{cond}/{stem}", "path": p,
                      "gt_boxes": gt, "gt_count": len(gt)})
    return items