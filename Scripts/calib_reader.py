import os
import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader

class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, image_dir, input_name):
        self.image_dir = image_dir
        self.input_name = input_name
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        img = cv2.imread(self.image_paths[self.index])
        self.index += 1

        img = cv2.resize(img, (320, 320))
        img = img[:, :, ::-1]              # BGR → RGB
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return {self.input_name: img}
