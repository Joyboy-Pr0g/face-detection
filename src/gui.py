import tkinter as tk
from tkinter import ttk, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

from utils import load_config, load_image, draw_boxes
from enhance import brightness_contrast, hist_equalize, clahe
from haar_detector import HaarDetector
from yolo_detector import YoloFaceDetector

PANEL_W, PANEL_H = 300, 400
YOLO_COLOR = (0, 0, 255)
HAAR_COLOR = (0, 255, 0)


def fit_max_side(img, max_side):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    return img


class App:
    def __init__(self, root):
        self.root = root
        root.title("Digital Image Processing - Face Detection Project")
        self.cfg = load_config("config.yaml")
        self.img = None
        self.yolo = None
        self.haar = None
        self.counts = {}

        e = self.cfg["enhance"]
        self.use_he = tk.BooleanVar(value=False)
        self.use_clahe = tk.BooleanVar(value=False)
        self.use_bc = tk.BooleanVar(value=False)
        self.beta = tk.IntVar(value=e["brightness_beta"])
        self.alpha = tk.DoubleVar(value=e["contrast_alpha"])

        self.method = tk.StringVar(value="yolo")
        self.p_scale = tk.DoubleVar()
        self.p_neigh = tk.IntVar()
        self.p_minsize = tk.IntVar()
        self.p_conf = tk.DoubleVar()
        self.p_nms = tk.DoubleVar()
        self.reset_params()

        self.status = tk.StringVar(value="")
        self.count_text = tk.StringVar(value="-")
        self.yolo_text = tk.StringVar(value="YOLO: -")
        self.haar_text = tk.StringVar(value="Viola-Jones: -")

        self.blank = tk.PhotoImage(width=PANEL_W, height=PANEL_H)
        self._build_controls()
        self._build_panels()
        self._build_results()

        self.detect_btn.state(["disabled"])
        root.after(200, self.load_models)

    def _build_controls(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(fill="x")

        src = ttk.LabelFrame(bar, text="1  Image source", padding=6)
        src.pack(side="left", padx=4, fill="y")
        ttk.Button(src, text="Open image...",
                   command=self.open_image).pack(fill="x")
        ttk.Button(src, text="Capture from camera",
                   command=self.capture).pack(fill="x", pady=(6, 0))

        pre = ttk.LabelFrame(bar, text="2  Preprocessing", padding=6)
        pre.pack(side="left", padx=4, fill="y")
        ttk.Checkbutton(pre, text="Histogram equalization", variable=self.use_he,
                        command=self.on_he).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(pre, text="CLAHE", variable=self.use_clahe,
                        command=self.on_clahe).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(pre, text="Brightness / contrast", variable=self.use_bc,
                        command=self.refresh).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(pre, text="Brightness (beta)").grid(row=3, column=0, sticky="w")
        tk.Scale(pre, from_=-100, to=100, orient="horizontal", length=160,
                 variable=self.beta,
                 command=lambda _v: self.refresh()).grid(row=3, column=1)
        ttk.Label(pre, text="Contrast (alpha)").grid(row=4, column=0, sticky="w")
        tk.Scale(pre, from_=0.5, to=2.5, resolution=0.1, orient="horizontal",
                 length=160, variable=self.alpha,
                 command=lambda _v: self.refresh()).grid(row=4, column=1)

        det = ttk.LabelFrame(bar, text="3  Detection", padding=6)
        det.pack(side="left", padx=4, fill="y")
        ttk.Radiobutton(det, text="YOLO", variable=self.method, value="yolo",
                        command=self.update_count).pack(anchor="w")
        ttk.Radiobutton(det, text="Viola-Jones", variable=self.method, value="haar",
                        command=self.update_count).pack(anchor="w")
        self.detect_btn = ttk.Button(det, text="Detect", command=self.detect)
        self.detect_btn.pack(fill="x", pady=(6, 0))

        par = ttk.LabelFrame(bar, text="4  Parameters (defaults from config.yaml)",
                             padding=6)
        par.pack(side="left", padx=4, fill="y")
        rows = [
            ("Haar scaleFactor", self.p_scale, 1.01, 2.0, 0.01, "%.2f"),
            ("Haar minNeighbors", self.p_neigh, 1, 20, 1, "%d"),
            ("Haar minSize (px)", self.p_minsize, 10, 200, 5, "%d"),
            ("YOLO conf", self.p_conf, 0.05, 0.95, 0.05, "%.2f"),
            ("YOLO NMS IoU", self.p_nms, 0.1, 0.9, 0.05, "%.2f"),
        ]
        for r, (label, var, lo, hi, step, fmt) in enumerate(rows):
            ttk.Label(par, text=label).grid(row=r, column=0, sticky="w")
            ttk.Spinbox(par, from_=lo, to=hi, increment=step, textvariable=var,
                        width=7, format=fmt).grid(row=r, column=1, padx=4)
        ttk.Button(par, text="Reset to config",
                   command=self.reset_params).grid(row=len(rows), column=0,
                                                   columnspan=2, sticky="ew",
                                                   pady=(4, 0))

    def _build_panels(self):
        grid = ttk.Frame(self.root, padding=6)
        grid.pack(fill="both", expand=True)
        self.labels = {}
        names = ["Original", "Preprocessed", "YOLO result", "Viola-Jones result"]
        for col, name in enumerate(names):
            box = ttk.LabelFrame(grid, text=name, padding=4)
            box.grid(row=0, column=col, padx=4, sticky="n")
            lbl = tk.Label(box, image=self.blank)
            lbl.pack()
            self.labels[name] = lbl

    def _build_results(self):
        bar = ttk.LabelFrame(self.root, text="Results", padding=6)
        bar.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(bar, text="Faces (selected method):").pack(side="left")
        ttk.Label(bar, textvariable=self.count_text,
                  font=("Segoe UI", 20, "bold")).pack(side="left", padx=(4, 24))
        ttk.Label(bar, textvariable=self.yolo_text).pack(side="left", padx=12)
        ttk.Label(bar, textvariable=self.haar_text).pack(side="left", padx=12)
        ttk.Label(bar, textvariable=self.status).pack(side="right")

    def reset_params(self):
        h, y = self.cfg["haar"], self.cfg["yolo"]
        self.p_scale.set(h["scale_factor"])
        self.p_neigh.set(h["min_neighbors"])
        self.p_minsize.set(h["min_size"][0])
        self.p_conf.set(y["conf"])
        self.p_nms.set(y["nms_iou"])

    def load_models(self):
        self.status.set("Loading models and warming up...")
        self.root.update_idletasks()
        self.haar = HaarDetector(self.cfg["haar"])
        self.yolo = YoloFaceDetector(self.cfg["yolo"])
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(self.cfg["timing"]["warmup_runs"]):
            self.haar.detect(dummy)
            self.yolo.detect(dummy)
        self.detect_btn.state(["!disabled"])
        self.status.set("Ready")

    def apply_params(self):
        scale = float(self.p_scale.get())
        neigh = int(self.p_neigh.get())
        msize = int(self.p_minsize.get())
        conf = float(self.p_conf.get())
        nms = float(self.p_nms.get())
        if not (scale > 1.0 and neigh >= 0 and msize >= 1
                and 0 < conf < 1 and 0 < nms <= 1):
            raise ValueError("parameter out of range")
        self.haar.scale_factor = scale
        self.haar.min_neighbors = neigh
        self.haar.min_size = (msize, msize)
        self.yolo.conf = conf
        self.yolo.iou = nms

    def preprocess(self, img):
        e = self.cfg["enhance"]
        out = img
        if self.use_bc.get():
            out = brightness_contrast(out, self.alpha.get(), self.beta.get())
        if self.use_he.get():
            out = hist_equalize(out)
        elif self.use_clahe.get():
            out = clahe(out, e["clahe_clip"], e["clahe_tile"])
        return out

    def to_photo(self, bgr):
        h, w = bgr.shape[:2]
        s = min(PANEL_W / w, PANEL_H / h)
        small = cv2.resize(bgr, (int(w * s), int(h * s)),
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def show(self, name, bgr):
        photo = self.to_photo(bgr)
        self.labels[name].configure(image=photo)
        self.labels[name].image = photo

    def clear_results(self):
        for name in ("YOLO result", "Viola-Jones result"):
            self.labels[name].configure(image=self.blank)
            self.labels[name].image = self.blank
        self.counts = {}
        self.count_text.set("-")
        self.yolo_text.set("YOLO: -")
        self.haar_text.set("Viola-Jones: -")

    def update_count(self):
        self.count_text.set(str(self.counts.get(self.method.get(), "-")))

    def on_he(self):
        if self.use_he.get():
            self.use_clahe.set(False)
        self.refresh()

    def on_clahe(self):
        if self.use_clahe.get():
            self.use_he.set(False)
        self.refresh()

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img = load_image(path, self.cfg["image"]["max_side"])
            self.refresh()
            self.status.set("Image loaded")

    def capture(self):
        backend = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
        cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status.set("Camera not available")
            return
        frame = None
        for _ in range(8):
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is None:
            self.status.set("Could not read a frame from the camera")
            return
        self.img = fit_max_side(frame, self.cfg["image"]["max_side"])
        self.refresh()
        self.status.set("Frame captured")

    def refresh(self):
        if self.img is None:
            return
        self.show("Original", self.img)
        self.show("Preprocessed", self.preprocess(self.img))
        self.clear_results()

    def detect(self):
        if self.img is None:
            self.status.set("Open an image or capture from the camera first")
            return
        try:
            self.apply_params()
        except (tk.TclError, ValueError):
            self.status.set("Invalid parameter value")
            return

        proc = self.preprocess(self.img)
        thick = max(2, proc.shape[1] // 150)
        ry = self.yolo.detect(proc)
        rh = self.haar.detect(proc)
        self.show("YOLO result", draw_boxes(proc, ry["boxes"], YOLO_COLOR, thick))
        self.show("Viola-Jones result",
                  draw_boxes(proc, rh["boxes"], HAAR_COLOR, thick))

        self.counts = {"yolo": ry["count"], "haar": rh["count"]}
        self.yolo_text.set(
            f"YOLO: {ry['count']} faces | {ry['time_ms']:.1f} ms (single run)")
        self.haar_text.set(
            f"Viola-Jones: {rh['count']} faces | {rh['time_ms']:.1f} ms (single run)")
        self.update_count()
        self.status.set("Detection finished")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()