import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

ENGINE_PATH = "yolov11n_fp16.trt"
VIDEO_PATH  = "video.mp4"
INPUT_W, INPUT_H = 640, 640
CONF_THRESH = 0.25
NMS_THRESH  = 0.45
NUM_CLASSES = 80  # change if your model has different number of classes

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# COCO class names - replace with your own if custom trained
CLASSES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet",
    "tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
    "oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float16)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))           # HWC → CHW
    return np.ascontiguousarray(img[np.newaxis])  # add batch dim

def postprocess(output, orig_w, orig_h):
    # output shape: [1, 84, 8400]
    preds = output.reshape(84, 8400).T  # → [8400, 84]

    boxes      = preds[:, :4]   # cx, cy, w, h
    class_prob = preds[:, 4:]   # class scores

    scores     = np.max(class_prob, axis=1)
    class_ids  = np.argmax(class_prob, axis=1)

    # filter by confidence
    mask = scores > CONF_THRESH
    boxes     = boxes[mask]
    scores    = scores[mask]
    class_ids = class_ids[mask]

    # convert cx,cy,w,h → x1,y1,x2,y2
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * orig_w / INPUT_W
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * orig_h / INPUT_H
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * orig_w / INPUT_W
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * orig_h / INPUT_H

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)

    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), scores.tolist(), CONF_THRESH, NMS_THRESH
    )

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "box":      boxes_xyxy[i],
                "score":    float(scores[i]),
                "class_id": int(class_ids[i])
            })
    return results

def draw(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls   = det["class_id"]
        score = det["score"]
        label = f"{CLASSES[cls]} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

# ── load engine ──────────────────────────────────────────────
engine  = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

input_binding  = engine.get_binding_index("images")
output_binding = engine.get_binding_index("output0")

h_input  = cuda.pagelocked_empty(
    trt.volume(engine.get_binding_shape(input_binding)),  np.float16)
h_output = cuda.pagelocked_empty(
    trt.volume(engine.get_binding_shape(output_binding)), np.float16)
d_input  = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream   = cuda.Stream()

# ── open video ───────────────────────────────────────────────
cap     = cv2.VideoCapture(VIDEO_PATH)
orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_src = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter('output.mp4', fourcc, fps_src, (orig_w, orig_h))

frame_count = 0
prev_time   = time.time()

print("Running inference...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    blob = preprocess(frame)
    np.copyto(h_input, blob.ravel())

    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(
        bindings=[int(d_input), int(d_output)],
        stream_handle=stream.handle
    )
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    detections = postprocess(h_output, orig_w, orig_h)
    frame      = draw(frame, detections)
    out.write(frame)

    frame_count += 1
    curr_time    = time.time()
    fps          = 1.0 / (curr_time - prev_time)
    prev_time    = curr_time
    print(f"Frame {frame_count:04d} | FPS: {fps:.1f} | Detections: {len(detections)}")

cap.release()
out.release()
print(f"\nDone! Saved to output.mp4 — total frames: {frame_count}")