import cv2
import numpy as np
from utils import load_image
from dataset import load_dataset
from enhance import hist_equalize_gray

worst = 0
for it in load_dataset():
    gray = cv2.cvtColor(load_image(it["path"]), cv2.COLOR_BGR2GRAY)
    d = np.abs(hist_equalize_gray(gray).astype(int) - cv2.equalizeHist(gray).astype(int))
    worst = max(worst, int(d.max()))
print("images:", len(load_dataset()), "| max abs difference over all images:", worst)