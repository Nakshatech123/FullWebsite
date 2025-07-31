from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")
img = cv2.imread("C:/Users/NTIN10/Pictures/Screenshots/image.jpg")  # replace with a test image

results = model.predict(img)
results[0].show()  # shows image with boxes

