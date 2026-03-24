
import cv2
import numpy as np
import onnxruntime as ort



from actuator import DogTrigger, UltrasoundController


trigger = DogTrigger()

controller = UltrasoundController()


class DogDetector:
    def __init__(
        self,
        model_path: str,
        img_size: int = 320,
        conf_thresh: float = 0.7,
        nms_thresh: float = 0.5,
        provider: str = "CPUExecutionProvider",
    ):
        """
        FP16 ONNX YOLO-style detector.

        Assumptions (LOCKED):
        - Model expects FP16 input
        - Output format: (1, 5, N)
        - Output order: [cx, cy, w, h, conf]
        - Coordinates are pixel-space (0–img_size)
        - Sigmoid already applied inside model
        """

        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.sess = ort.InferenceSession(
            model_path,
            providers=[provider]
        )

        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess BGR frame to FP16 tensor.
        """
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = img[:, :, ::-1]  # BGR → RGB
        img = (img.astype(np.float32) / 255.0).astype(np.float16)
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0)
        return img
     
    def detect(self, frame: np.ndarray):
        """
        Run detection on a single frame.

        Returns:
            detections: list of dicts
            [
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float
                },
                ...
            ]
        """
        h0, w0 = frame.shape[:2]

        inp = self._preprocess(frame)
        preds = self.sess.run(
            [self.output_name],
            {self.input_name: inp}
        )[0]

        # (1, 5, N) → (N, 5)
        pred = preds[0].T

        boxes = []
        scores = []

        for cx, cy, w, h, conf in pred:
            if conf < self.conf_thresh:
                continue

            # pixel-space decode
            x1 = cx - w / 2
            y1 = cy - h / 2
            bw = w
            bh = h

            boxes.append([x1, y1, bw, bh])
            scores.append(float(conf))

        detections = []

        if boxes:
            idxs = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                score_threshold=0.0,   # IMPORTANT: no double filtering
                nms_threshold=self.nms_thresh
            )

            if len(idxs) > 0:
                for i in idxs.flatten():
                    x, y, w, h = boxes[i]

                    # scale back to original frame
                    sx1 = int(x * w0 / self.img_size)
                    sy1 = int(y * h0 / self.img_size)
                    sx2 = int((x + w) * w0 / self.img_size)
                    sy2 = int((y + h) * h0 / self.img_size)

                    detections.append({
                        "bbox": (sx1, sy1, sx2, sy2),
                        "confidence": scores[i]
                    })

        return detections


detector = DogDetector(
    model_path=r"D:\ppt\aiml\New folder\project1\models\model_fp16.onnx",
    img_size=320,
    conf_thresh=0.6,
    nms_thresh=0.5
)


import os 

output_dir=r"D:\ppt\aiml\New folder\project1\results"

output_path = os.path.join(output_dir, "detected_output1.mp4")

cap = cv2.VideoCapture(r"C:\Users\D.Teju\Downloads\WhatsApp Video 2026-01-20 at 15.34.30.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    dog_present=len(detections)>0

    dog_confirmed=trigger.update(dog_present)
    ultrasound_on=controller.update(dog_confirmed)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            frame,
            f"DOG {conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )
    out.write(frame)
    cv2.imshow("Detector Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

#Detection → Evidence → Decision → Actuation → Cooldown
