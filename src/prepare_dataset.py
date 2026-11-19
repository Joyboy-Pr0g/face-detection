import os, glob, cv2
from utils import load_config, load_image

cfg = load_config("config.yaml")
max_side = cfg["image"]["max_side"]
src_root, dst_root = "dataset/raw", "dataset/images"

for cond in sorted(os.listdir(src_root)):
    os.makedirs(os.path.join(dst_root, cond), exist_ok=True)
    for path in sorted(glob.glob(os.path.join(src_root, cond, "*"))):
        img = load_image(path, max_side)
        stem = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(dst_root, cond, stem + ".jpg")
        cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(cond, stem, img.shape[1], "x", img.shape[0])