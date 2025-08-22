#  Artificial Intelligence Project  
This repository contains implementations of two AI tasks using Python and deep learning libraries:

1. **Real-Time Object Detection using YOLO**  
2. **Facial Emotion Recognition using CNN (FER-2013 dataset)** 

##  Task 3: Real-Time Object Detection using YOLO

# Description
Implements a **real-time object detection system** using the YOLO (You Only Look Once) algorithm.  
The model detects and labels multiple objects (person, car, dog, etc.) in **live video feed** or images.  

# Requirements
*  Python 3.8+
*  OpenCV
* PyTorch
* Ultralytics YOLOv8
# Features
 *Works on images and live video.
* Displays bounding boxes with labels and confidence scores.
* Supports multiple object categories.
  # Task 4: Facial Emotion Recognition (FER-2013)
# Description
Builds a facial emotion recognition system using a CNN model trained on the FER-2013 dataset.
Detects emotions such as:

* Angry 😡
* Disgust 🤢
* Fear 😨
* Happy 😀
* Sad 😢
* Surprise 😲
* Neutral 😐

# Requirements
* TensorFlow / Keras
* OpenCV
* Numpy
*FER-2013 Dataset (download: Kaggle FER-2013)
# Training Model
* Trains CNN on FER-2013 dataset
* Saves model as emotion_model.h5
# Real-Time Emotion Detection
* Opens webcam
* Detects face using Haar Cascade
* Classifies emotion using trained CNN
* Displays bounding box + emotion label
# Results
# YOLO Object Detection:
* Detects multiple objects with high speed and accuracy.
# Facial Emotion Recognition:
* Achieved ~65–70% accuracy on FER-2013 dataset.
* Works in real time on webcam.
