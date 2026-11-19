import cv2
import numpy as np


def brightness_contrast(img, alpha, beta):
    out = img.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def hist_equalize_gray(gray):
    hist = np.bincount(gray.ravel(), minlength=256)
    cdf = hist.cumsum()
    cdf_min = cdf[np.nonzero(cdf)[0][0]]
    n = gray.size
    if n == cdf_min:                      # flat image, nothing to equalize
        return gray.copy()
    lut = np.round((cdf - cdf_min) / (n - cdf_min) * 255)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[gray]


def _apply_on_luminance(img_bgr, func):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = func(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def hist_equalize(img_bgr):
    return _apply_on_luminance(img_bgr, hist_equalize_gray)


def clahe(img_bgr, clip_limit, tile):
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile))
    return _apply_on_luminance(img_bgr, c.apply)