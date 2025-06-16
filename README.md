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
│   ├── detect/                  
│   ├── train/                   
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
Make sure you have Python 3.8+ and pip installed on your system.

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
Update dataset.yaml accordingly with the correct paths and class names:
```bash
train: data/images/train
val: data/images/test
nc: 2
names: ['Mask', 'No Mask']
```
