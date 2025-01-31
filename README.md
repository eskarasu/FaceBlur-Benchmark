# FaceBlur-Benchmark

A comprehensive benchmarking project comparing different face detection and blurring algorithms for privacy protection applications.

## Overview

This project implements and compares six different face detection methods combined with Gaussian blurring:
1. OpenCV DNN Face Detector (TensorFlow model)
2. Caffe Model (DNN Deploy Prototxt)
3. MTCNN (Multi-task Cascaded CNN)
4. Dlib HOG Face Detector
5. MediaPipe Face Detection
6. Haar Cascade Classifier

## Requirements

- Python 3.x
- OpenCV
- TensorFlow
- Dlib
- MediaPipe
- MTCNN
- Matplotlib
- NumPy

## Installation

```bash
pip install opencv-python tensorflow dlib mediapipe mtcnn matplotlib numpy
```

## Model Files Required

- `opencv_face_detector_uint8.pb` and `opencv_face_detector.pbtxt` for OpenCV DNN
- `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` for Caffe model
- Haar Cascade classifier (included in OpenCV)

## Usage

The project provides different functions for face blurring:

```python
blur_faces_dnn_opencv_fd(image_path)      # OpenCV DNN
blur_faces_dnn_deploy_prototxt(image_path) # Caffe Model
blur_faces_mtcnn(image_path)              # MTCNN
blur_faces_dlib(image_path)               # Dlib HOG
blur_faces_mediapipe(image_path)          # MediaPipe
blur_faces_haar(image_path)               # Haar Cascade
```

## Implementation Details

### 1. OpenCV DNN Face Detector
- Uses TensorFlow model
- Confidence threshold: 0.5
- Input size: 300x300
- Gaussian blur: (99,99) kernel, σ=30

### 2. Caffe Model
- Uses SSD framework
- Confidence threshold: 0.5
- Input size: 300x300
- Pre-trained on wider face dataset

### 3. MTCNN
- Multi-stage cascaded architecture
- Provides additional facial landmarks
- Good performance on various face orientations

### 4. Dlib HOG
- Uses Histogram of Oriented Gradients
- CPU-based detection
- Good balance of speed and accuracy

### 5. MediaPipe
- Real-time face detection
- Minimum detection confidence: 0.5
- Works well with different face orientations

### 6. Haar Cascade
- Traditional cascade classifier
- Fast but less accurate
- Works best with frontal faces
- Scale factor: 1.1, minNeighbors: 5

## Privacy Notice

This project is designed for privacy protection applications. Always ensure you have the right to process and modify images containing personal data.

## License

[MIT License](LICENSE)

## Face Detection Methods

| Method                   | Model Type            | Speed          | Accuracy      | Dependencies        | Suitable For            | Pros                                  | Cons                                      |
|--------------------------|-----------------------|----------------|---------------|----------------------|-------------------------|----------------------------------------|-------------------------------------------|
| OpenCV DNN Face Detector | Deep Learning (Caffe) | Fast            | High          | OpenCV, Caffe         | Real-time applications   | Fast and accurate                      | Requires pre-trained Caffe model          |
| MTCNN                    | Deep Learning (CNN)   | Moderate        | Very High     | TensorFlow, Keras     | Complex scenarios        | High accuracy with multi-scale detection| Slower on CPU, complex implementation     |
| Dlib HOG Face Detector    | Machine Learning (HOG)| Moderate        | Moderate      | Dlib                   | Real-time, low-resource  | Lightweight, works without GPU          | Less accurate for complex backgrounds     |
| MediaPipe Face Detection  | Deep Learning         | Very Fast       | High          | MediaPipe, OpenCV     | Mobile, web applications | Optimized for real-time, lightweight    | May struggle with extreme angles          |
| Haar Cascade Classifier   | Traditional ML        | Fast            | Low           | OpenCV                 | Simple applications      | Very fast, minimal dependencies         | High false-positive rate, low accuracy    |
