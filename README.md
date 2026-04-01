# 🐶 YOLO Dog Detection

A deep learning project for **real-time dog detection** using YOLO-based object detection models.
This repository focuses on training, evaluating, and running inference on dog detection datasets.

---

## 🚀 Features

* ⚡ Real-time object detection using YOLO
* 🐕 Focused single-class detection (dog)
* 📊 Training and evaluation pipeline
* 📓 Experimentation via Jupyter notebooks
* 🧠 Supports custom dataset training

---

## 📂 Project Structure

```
YOLO-DOG-DETECTION/
│
├── Scripts/          # Training, inference, utility scripts
├── notebooks/        # Experimentation and analysis notebooks
├── results/          # (ignored) output images/videos
├── dataset/          # (ignored) training dataset
├── .gitignore
└── README.md
```

---

## 🧠 Model

This project uses **YOLO (You Only Look Once)** architecture for object detection.

Typical workflow:

1. Dataset preparation (YOLO format)
2. Model training
3. Evaluation
4. Inference on images/videos

YOLO enables:

* Fast inference
* High accuracy
* Real-time detection capability

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/sandwormSLYTHERIN/YOLO-DOG-DETECTION.git
cd YOLO-DOG-DETECTION
```

### 2. Create environment

```bash
conda create -n dogdet python=3.10
conda activate dogdet
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training

Run training script:

```bash
python Scripts/train.py
```

Make sure dataset is in YOLO format:

```
images/
labels/
```

---

## 🔍 Inference

Run detection:

```bash
python Scripts/inference.py --source path/to/image_or_video
```

Output:

* Bounding boxes
* Confidence scores
* Detected objects

---

## 📊 Results

* Real-time detection on images/videos
* Bounding box predictions
* Model performance depends on dataset quality

---

## 📁 Dataset

You can use:

* Custom dataset
* Roboflow datasets
* Open-source dog datasets

⚠️ Dataset is not included in repo due to size constraints.

---

## ⚠️ Notes

* Large files (datasets, videos, weights) are excluded via `.gitignore`
* Use GPU for faster training (CUDA recommended)
* Notebook outputs should be cleared before committing

---

## 🔮 Future Improvements

* Model optimization (FP16 / ONNX / TensorRT)
* Deployment (Flask / FastAPI / Streamlit)
* Multi-class detection (breed classification)
* Real-time webcam integration

---

## 🤝 Contributing

Feel free to:

* Open issues
* Submit pull requests
* Suggest improvements

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Teju**

---
