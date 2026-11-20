import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_image(path, max_side=1280):
    pil = Image.open(path)
    pil = ImageOps.exif_transpose(pil)       
    pil = pil.convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:                            
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img


def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    out = img.copy()                         
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, thickness)
    return out