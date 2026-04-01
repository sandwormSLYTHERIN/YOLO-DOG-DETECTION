from flask import Flask, Response
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

ENGINE_PATH = "yolov11n_fp16.trt"
VIDEO_PATH  = "video.mp4"
INPUT_W, INPUT_H = 640, 640
CONF_THRESH = 0.25
NMS_THRESH  = 0.45

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

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float16)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img[np.newaxis])

def postprocess(output, orig_w, orig_h):
    preds     = output.reshape(84, 8400).T
    boxes     = preds[:, :4]
    class_prob = preds[:, 4:]
    scores    = np.max(class_prob, axis=1)
    class_ids = np.argmax(class_prob, axis=1)

    mask      = scores > CONF_THRESH
    boxes     = boxes[mask]
    scores    = scores[mask]
    class_ids = class_ids[mask]

    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * orig_w / INPUT_W
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * orig_h / INPUT_H
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * orig_w / INPUT_W
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * orig_h / INPUT_H
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)

    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), scores.tolist(), CONF_THRESH, NMS_THRESH
    )
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "box": boxes_xyxy[i],
                "score": float(scores[i]),
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

# ── flask stream ─────────────────────────────────────────────
app = Flask(__name__)

def generate():
    cap    = cv2.VideoCapture(VIDEO_PATH)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return '''
    <html>
    <head><title>YOLOv11n Live</title></head>
    <body style="background:#111;display:flex;flex-direction:column;
                 align-items:center;justify-content:center;height:100vh;margin:0">
      <h2 style="color:#fff;font-family:sans-serif">YOLOv11n TensorRT Stream</h2>
      <img src="/video" style="max-width:100%;border:2px solid #0f0"/>
    </body>
    </html>
    '''

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Stream started → open http://192.168.55.1:5000 on your laptop browser")
    app.run(host='0.0.0.0', port=5000, threaded=False)