from ultralytics import YOLO
import cv2
import sys

MODEL_PATH = r"D:\ppt\aiml\New folder\project1\runs\detect\dog_detector_one_stage2\weights\best.pt"
CONF = 0.3
IOU = 0.5
IMGSZ = 640

model = YOLO(MODEL_PATH)

img_path =r"C:\Users\D.Teju\Downloads\WhatsApp Image 2026-01-15 at 12.53.58.jpeg"
img = cv2.imread(img_path)

results = model(
    img,
    conf=CONF,
    iou=IOU,
    imgsz=IMGSZ,
    device=0
)

annotated = results[0].plot()
cv2.imshow("Dog Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
