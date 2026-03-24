from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat
)
import onnxruntime as ort
from calib_reader import ImageCalibrationReader

model_fp32 = "model_simplified.onnx"
model_int8 = "model_int8.onnx"

# Get input name
sess = ort.InferenceSession(model_fp32)
input_name = sess.get_inputs()[0].name

calib_reader = ImageCalibrationReader(
    image_dir="calib_images",
    input_name=input_name
)

quantize_static(
    model_input=model_fp32,
    model_output=model_int8,
    calibration_data_reader=calib_reader,
    quant_format=QuantFormat.QDQ,   # ← CRITICAL
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)

print("INT8 ONNX model created (QDQ)")
