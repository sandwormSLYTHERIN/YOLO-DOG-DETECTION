#!/bin/bash
# Run this first before infer_video.py or stream.py

echo "Converting ONNX to TensorRT engine..."

/usr/src/tensorrt/bin/trtexec \
  --onnx=fp16.onnx \
  --saveEngine=yolov11n_fp16.trt \
  --fp16 \
  --workspace=2048 \
  --verbose

echo "Conversion done! Engine saved as yolov11n_fp16.trt"