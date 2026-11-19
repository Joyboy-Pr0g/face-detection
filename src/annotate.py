import os, glob, cv2

IMG_ROOT, LBL_ROOT = "dataset/images", "dataset/labels"
MAX_H = 900
boxes, drag, base, title = [], None, None, ""


def redraw():
    canvas = base.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if drag:
        cv2.rectangle(canvas, drag[0], drag[1], (0, 255, 255), 1)
    cv2.putText(canvas, f"{title} faces: {len(boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("annotate", canvas)


def on_mouse(event, x, y, flags, param):
    global drag
    if event == cv2.EVENT_LBUTTONDOWN:
        drag = [(x, y), (x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drag:
        drag[1] = (x, y)
        redraw()
    elif event == cv2.EVENT_LBUTTONUP and drag:
        (x1, y1), (x2, y2) = drag
        drag = None
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 >= 5 and y2 - y1 >= 5:
            boxes.append((x1, y1, x2, y2))
        redraw()


def label_path(img_path):
    rel = os.path.relpath(img_path, IMG_ROOT)
    return os.path.join(LBL_ROOT, os.path.splitext(rel)[0] + ".txt")


def load(path, w, h):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) == 5:
                    _, xc, yc, bw, bh = map(float, p)
                    out.append((int((xc - bw / 2) * w), int((yc - bh / 2) * h),
                                int((xc + bw / 2) * w), int((yc + bh / 2) * h)))
    return out


def save(path, w, h):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for x1, y1, x2, y2 in boxes:
            f.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")


paths = sorted(glob.glob(os.path.join(IMG_ROOT, "*", "*.jpg")),
               key=lambda p: (os.path.dirname(p),
                              int(os.path.splitext(os.path.basename(p))[0])))
cv2.namedWindow("annotate")
cv2.setMouseCallback("annotate", on_mouse)

i = 0
while 0 <= i < len(paths):
    img = cv2.imread(paths[i])
    s = min(1.0, MAX_H / img.shape[0])
    base = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = base.shape[:2]
    boxes = load(label_path(paths[i]), w, h)
    title = os.path.relpath(paths[i], IMG_ROOT)
    redraw()

    step = None
    while step is None:
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("n"), 32):
            step = 1
        elif k == ord("p"):
            step = -1
        elif k == ord("s"):
            step = 1
            save_changes = False
        elif k == ord("u") and boxes:
            boxes.pop()
            redraw()
        elif k == ord("q"):
            step = 10 ** 6
    save(label_path(paths[i]), w, h)
    i += step

cv2.destroyAllWindows()