# ===================== IMPORTS =====================
import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import deque
from enum import Enum, auto

# ===================== CONFIG ======================
MODEL_PATH = r"D:\ppt\aiml\New folder\project1\models\model_fp16.onnx"
VIDEO_PATH = r"C:\Users\D.Teju\Downloads\WhatsApp Video 2026-01-20 at 15.34.30.mp4" # <-- your 13s video
IMG_SIZE = 320

CONF_THRESH = 0.6
NMS_THRESH  = 0.5

TRIGGER_WINDOW = 8
TRIGGER_MIN_HITS = 4

MAX_ON_TIME   = 3.0    # seconds
COOLDOWN_TIME = 15.0   # seconds

SHOW_VIDEO = True      # set False for pure logging
# ==================================================


# ===================== DETECTOR ====================
class DogDetector:
    def __init__(self):
        self.sess = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def _preprocess(self, frame):
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img[:, :, ::-1]
        img = (img.astype(np.float32) / 255.0).astype(np.float16)
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def detect(self, frame):
        h0, w0 = frame.shape[:2]
        inp = self._preprocess(frame)
        preds = self.sess.run([self.output_name], {self.input_name: inp})[0]
        pred = preds[0].T  # (N,5)

        boxes, scores = [], []

        for cx, cy, w, h, conf in pred:
            if conf < CONF_THRESH:
                continue
            boxes.append([cx - w/2, cy - h/2, w, h])
            scores.append(float(conf))

        detections = []
        if boxes:
            idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.0, NMS_THRESH)
            if len(idxs) > 0:
                for i in idxs.flatten():
                    x, y, w, h = boxes[i]
                    detections.append({
                        "bbox": (
                            int(x * w0 / IMG_SIZE),
                            int(y * h0 / IMG_SIZE),
                            int((x+w) * w0 / IMG_SIZE),
                            int((y+h) * h0 / IMG_SIZE)
                        ),
                        "confidence": scores[i]
                    })
        return detections


# ===================== TRIGGER =====================
class DogTrigger:
    def __init__(self):
        self.window = deque(maxlen=TRIGGER_WINDOW)
        self.armed = True

    def update(self, dog_present):
        self.window.append(1 if dog_present else 0)
        hits = sum(self.window)

        if hits >= TRIGGER_MIN_HITS and self.armed:
            self.armed = False
            return True

        if hits == 0:
            self.armed = True

        return False


# ===================== CONTROLLER ==================
class State(Enum):
    IDLE = auto()
    ACTIVE = auto()
    COOLDOWN = auto()

class UltrasoundController:
    def __init__(self):
        self.state = State.IDLE
        self.t0 = time.time()

    def update(self, trigger_event):
        now = time.time()

        if self.state == State.IDLE:
            if trigger_event:
                self.state = State.ACTIVE
                self.t0 = now
                return True
            return False

        if self.state == State.ACTIVE:
            if now - self.t0 >= MAX_ON_TIME:
                self.state = State.COOLDOWN
                self.t0 = now
                return False
            return True

        if self.state == State.COOLDOWN:
            if now - self.t0 >= COOLDOWN_TIME:
                self.state = State.IDLE
            return False

        return False
