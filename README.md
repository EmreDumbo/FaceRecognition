# Face Recognition and Detection Project

This repository contains a comprehensive exploration of face recognition and detection methods using various techniques and frameworks. Below, you'll find details on each component and the tasks accomplished today.

---

## **Project Overview**

The goal of this project is to explore different face detection and recognition techniques, compare their performance, and integrate them into a cohesive system. The following approaches were implemented:

1. **Haar Cascade**: A classical approach to face detection using OpenCV.
2. **Dlib**: Used for both face detection and facial landmark extraction (68 landmarks).
3. **FaceNet**: A state-of-the-art neural network-based system for face recognition.

---

## **Technologies and Methods**

### 1. **Haar Cascade**
- **Purpose**: Detect faces in real-time.
- **Technology Used**: OpenCV.
- **Implementation**:
  - Haar Cascade classifiers were used to detect faces and eyes.
  - FPS (frames per second) was calculated for real-time performance evaluation.
- **Advantages**:
  - Lightweight and fast.
  - Works well with controlled environments.
- **Limitations**:
  - Struggles with complex backgrounds or poor lighting conditions.

### 2. **Dlib**
- **Purpose**: Perform face detection and extract 68 facial landmarks.
- **Technology Used**: Dlib library.
- **Implementation**:
  - Dlib's HOG-based face detection model was used.
  - Extracted 68 facial landmarks for facial feature analysis.
  - FPS was measured to compare performance.
- **Advantages**:
  - Accurate detection of landmarks.
  - Robust against varying angles and expressions.
- **Limitations**:
  - Slower compared to lightweight models.

### 3. **FaceNet**
- **Purpose**: Face recognition using deep learning.
- **Technology Used**: PyTorch, MTCNN, and InceptionResNetV1.
- **Implementation**:
  - **MTCNN** (Multi-Task Cascaded Convolutional Networks) was used for face detection and alignment.
  - **InceptionResNetV1** was used to generate embeddings for recognized faces.
  - **Triplet Loss**: Ensures that embeddings of the same person are closer than embeddings of different people.
  - Real-time detection and recognition with embeddings were implemented.
- **Advantages**:
  - High accuracy for recognition tasks.
  - Works well across various datasets.
- **Limitations**:
  - Requires more computational power compared to Haar Cascade or Dlib.

---

## **Steps Performed Today**

### 1. **Face Detection with Haar Cascade**
- Implemented face detection.
- Measured FPS for performance evaluation.
- Added eye detection using Haar Cascade.

### 2. **Facial Landmarks with Dlib**
- Extracted 68 facial landmarks.
- Calculated FPS for performance comparison.
- Integrated landmark detection into the overall pipeline.

### 3. **Face Recognition with FaceNet**
- Loaded pre-trained FaceNet model:
  - **MTCNN** for face detection and alignment.
  - **InceptionResNetV1** for embedding generation.
- Implemented the triplet loss mechanism for effective face embedding separation.
- Built a real-time recognition system:
  - Detected faces and calculated embeddings in real-time.
  - Compared embeddings with pre-saved embeddings to identify faces.

### 4. **FPS Comparison**
- Measured and compared FPS for:
  - Haar Cascade.
  - Dlib (face detection and landmarks).
- Observed that FaceNet offers higher accuracy but is slower compared to Haar Cascade and Dlib.

---

## **Future Improvements**
- Optimize FaceNet for faster real-time performance.
- Experiment with other models like YOLO or Mediapipe for face detection.
- Enhance Haar Cascade by training custom cascades.
- Integrate GPU acceleration to improve performance.

---

## **How to Run**

### Prerequisites
Ensure the following libraries are installed:
- OpenCV
- Dlib
- Facenet-PyTorch
- PyTorch
- tqdm
- PIL
- numpy
- imutils

Install dependencies using pip:
```bash
pip install opencv-python dlib facenet-pytorch torch tqdm
```

### Execution
1. **Run Haar Cascade**:
   ```bash
   python haarCam.py
   python haarPhoto.py
   ```
2. **Run Dlib**:
   ```bash
   python dlibCam.py
   python dlibPhoto.py
   python 68landmark.py 
   ```
3. **Run FaceNet**:
   ```bash
   python facenet.py
   python facenetPhoto.py
   ```

---

## **Acknowledgements**
- **Haar Cascade**: OpenCV documentation.
- **Dlib**: Davis King's Dlib library.
- **FaceNet**: "FaceNet: A Unified Embedding for Face Recognition and Clustering" by Google.
