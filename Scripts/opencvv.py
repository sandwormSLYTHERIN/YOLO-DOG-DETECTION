import cv2
import torch
from collections import deque
from ultralytics import YOLO
import time
import os

# 1️⃣ Load YOLO model
model_path = r"D:\ppt\aiml\New folder\project1\runs\detect\dog_detector_one_stage2\weights\best.pt"
model = YOLO(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).fuse()
torch.set_grad_enabled(False)
model.overrides["verbose"] = False

# 2️⃣ Video input
input_video = r"D:\Funny dog walk - Sion Hughes (720p, h264, youtube).mp4"
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot access video source")

# 3️⃣ Ensure output directory exists
output_dir = r"D:\ppt\aiml\New folder\project1\results"
os.makedirs(output_dir, exist_ok=True)

# 4️⃣ Setup video writer
output_path = os.path.join(output_dir, "detected_output.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# 5️⃣ Sliding window parameters
frame_buffer = deque(maxlen=5)
half = device == "cuda"
frame_count = 0
fps_start = time.time()

print("[INFO] Starting real-time sliding window inference...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_buffer.append(frame)
    frame_count += 1

    if len(frame_buffer) == 5:
        results = model(
            list(frame_buffer),
            imgsz=640,
            device=device,
            half=half,
            stream=False,
            verbose=False
        )

        annotated = results[-1].plot()
        out.write(annotated)  # ✅ Save frame
        cv2.imshow("YOLO Real-Time (Sliding 5-Frame Window)", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_buffer.popleft()

    if frame_count % 50 == 0:
        fps_now = frame_count / (time.time() - fps_start)
        print(f"[INFO] Effective FPS: {fps_now:.2f}")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] ✅ Stream ended. Video saved successfully at:\n{output_path}")
