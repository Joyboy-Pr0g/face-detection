import hashlib
import os
import urllib.request

URL = ("https://github.com/akanametov/yolo-face/releases/"
       "download/1.0.0/yolov8n-face.pt")
DEST = os.path.join("models", "yolov8n-face.pt")

os.makedirs("models", exist_ok=True)
if not os.path.exists(DEST):
    urllib.request.urlretrieve(URL, DEST)

with open(DEST, "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()

print("file:", DEST)
print("size MB:", round(os.path.getsize(DEST) / 1e6, 2))
print("sha256:", digest)