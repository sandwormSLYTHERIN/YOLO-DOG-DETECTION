import cv2
import time
import serial
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = r"D:\ppt\aiml\New folder\project1\runs\detect\dog_detector_one_stage2\weights\best.pt"
CONF = 0.4
IOU = 0.5
IMGSZ = 640
CONSEC_FRAMES = 4     # debounce
SERIAL_PORT = "/dev/ttyUSB0"
BAUD = 115200
# ---------------------------------------

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)

dog_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(
        frame,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device=0,
        verbose=False
    )[0]

    dog_detected = False

    if results.boxes is not None:
        if len(results.boxes) > 0:
            dog_detected = True

    # -------- Debounce logic --------
    if dog_detected:
        dog_counter += 1
    else:
        dog_counter = 0

    # -------- Trigger STM32 --------
    if dog_counter >= CONSEC_FRAMES:
        ser.write(b"DOG\n")     # signal
        dog_counter = 0         # avoid retrigger spam
        time.sleep(0.5)         # cooldown
    else:
        ser.write(b"NO\n")

cap.release()
ser.close()
