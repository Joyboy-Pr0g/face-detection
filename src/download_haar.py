import cv2, shutil, os
os.makedirs("models", exist_ok=True)
src = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
shutil.copy(src, "models/haarcascade_frontalface_default.xml")
print("copied from", src, "| OpenCV", cv2.__version__)