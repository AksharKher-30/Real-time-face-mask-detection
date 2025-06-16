# Face Mask Detection Using YOLOv8 😷

A real-time face mask detection system using Ultralytics' YOLOv8 object detection models. This project detects whether a person is wearing a mask or not, leveraging annotated datasets and custom-trained models.

---

## 🌐 Project Overview

This repository contains all the code and assets to:

- Convert XML annotations to YOLO format
- Train YOLOv8 (both `yolov8n` and `yolov8s` models)
- Run real-time mask detection via webcam
- Visualize training logs with TensorBoard

---

## 📁 Project Structure

```bash
face-mask-detection/
├── data/                        # Contains dataset and annotation scripts
│   ├── images            
│   └── labels                   # XML annotations
├── dataset.yaml                 # YOLO dataset configuration
├── notebook/
│   └── your_notebook.ipynb      # Training and experimentation notebook
├── runs/                        # YOLOv8 training output (weights, logs, plots)
│   ├── detect/                  #runs/train/face_mask_detector contain logs for YOLOv8n
│   ├── train/                   #runs/s_runs/s_train/face_mask_detector_s contaons logs for YOLOv8s
│   └── ...
├── yolov8n.pt                   # YOLOv8n pre-trained model (used initially)
├── yolov8s.pt                   # YOLOv8s pre-trained model (improved results)
├── realtime_mask_detection.py   # Script to run real-time detection
├── log_to_tensorboard.py        # TensorBoard logging (optional)
├── xml_to_yolo.py               # Converts Pascal VOC XML to YOLO format
├── test1.jpg, test2.jpg         # Sample images for testing
└── README.md                    # This file
```

---

## 🔧 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```
> Make sure you have Python 3.8+ and pip installed on your system.

--- 

## 📁 Data Preparation
- Annotation are provided in **Pascal VOC (XML)** format.
- Convert to **YOLO format** using the following command:
```bash
python xml_to_yolo.py
```
This script will:
- Parse all **.xml** files
- Convert annotations to **YOLO format**
- Save converted **.txt** files in the corresponding **labels/ directories**

Dataset format expected by YOLO:
```bash
data/
├── test/
│   ├── annotation/
│   ├── images/
│   └── labels/
├── train/
│   ├── annotation/
│   ├── images/
│   └── labesl/
```
> Update dataset.yaml accordingly with the correct paths and class names:
```bash
train: data/images/train
val: data/images/test
nc: 2
names: ['Mask', 'No Mask']
```
The trained weights will be saved under:
```bash
runs/s_runs/s_train/face_mask_detector_s/weights/
├── best.pt
└── last.pt
```
> **best.pt** is used for inference.

---
## 🎥 Real-Time Detection
To run live webcam detection using the trained model:
```bash
python realtime_mask_detection.py
```
> Make sure to set the correct path to your trained model weights, e.g. **best.pt**, inside the script.

---

## TensorBoard Logging (Optional)
To visualize training metrics with TensorBoard:
```bash
tensorboard --logdir runs/
```
Then open http://localhost:6006 in your browser.

--- 

## 📊 Performance

### 📉 Live Face Mask Detection Accuracy

| Model    | Mask Accuracy | No Mask Accuracy |
|----------|----------------|------------------|
| YOLOv8n  | 66.2%          | 49.5%            |
| YOLOv8s  | 84.5%          | 81.9%            |

> ✅ **YOLOv8s clearly outperformed YOLOv8n in real-time detection.**

---

## 📊 Results Example

### Here is a snapshot of the real-time detection in action:
> This is for **YOLOv8s** pre-trained model
<p align="center">
  <img src="assets/mask.png" alt="with mask" width="45%" />
  <img src="assets/no mask.png" alt="without mask" width="45%" />
</p>
