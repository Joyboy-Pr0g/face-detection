import os, sys, cv2
import numpy as np
from utils import load_config, load_image
from enhance import hist_equalize_gray, brightness_contrast, hist_equalize, clahe

cfg = load_config("config.yaml")
e = cfg["enhance"]
img = load_image(sys.argv[1], cfg["image"]["max_side"])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mine = hist_equalize_gray(gray)
ref = cv2.equalizeHist(gray)
diff = np.abs(mine.astype(int) - ref.astype(int))
print("max abs diff vs cv2.equalizeHist:", diff.max(),
      "| pixels differing:", int((diff > 0).sum()))

os.makedirs("results/dev", exist_ok=True)
name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
variants = {
    "orig": img,
    "bc": brightness_contrast(img, e["contrast_alpha"], e["brightness_beta"]),
    "he": hist_equalize(img),
    "clahe": clahe(img, e["clahe_clip"], e["clahe_tile"]),
}
for k, v in variants.items():
    cv2.imwrite(f"results/dev/{name}_{k}.png", v)
    print(k, "mean:", round(float(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY).mean()), 1),
          "std:", round(float(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY).std()), 1))