# CS-GY 6613 — Assignment 3  
## Drone Detection and Kalman Filter Tracking

---

## Overview

This project implements an end-to-end pipeline for detecting and tracking drones in video sequences under challenging conditions, including small object size, background clutter, and intermittent detection failures.

The system consists of:

1. Drone object detection using YOLOv8
  - The `detections/` folder is included to provide full visual outputs of the detection pipeline for inspection and validation.
2. Kalman filter-based tracking for temporal consistency  

Key challenges addressed (mainly in video 2):
- Small object detection (drones occupy very few pixels)
- False positives from visually similar patterns (clouds, trees)
- Missing detections across consecutive frames  

Final outputs:
- A Parquet dataset of detection frames: https://huggingface.co/datasets/ck1894/cs-gy-6613-assignment-3/tree/main
- Tracking videos with bounding boxes and trajectory overlays
  - Video 1: https://youtu.be/Ez4mgopDXWg
  - Video 2: https://youtu.be/03haInyME9Q

---

## Dataset Choice and Detector Configuration

### Dataset Choice

To improve robustness and explicitly address observed failure modes, three datasets were used:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")

# Drone dataset (primary)
drone_project = rf.workspace("project-986i8").project("drone-uskpc")
drone_dataset = drone_project.version(1).download("yolov8")

# Trees dataset
trees_project = rf.workspace("dl-le4jj").project("trees-g6nlr")
trees_dataset = trees_project.version(1).download("yolov8")

# Clouds dataset
clouds_project = rf.workspace("arnab-dhara-rrr0g").project("clouds-zwvmh")
clouds_dataset = clouds_project.version(1).download("yolov8")
```

### Rationale

- Drone dataset: provides labeled bounding boxes for the target class  
- Trees and clouds datasets: used to reduce false positives by exposing the model to hard negative examples  

---

### Detector Configuration

The detector was tuned to balance recall (detecting small drones) and precision (reducing false positives).

- Model: YOLOv8 (Ultralytics)

- Confidence Threshold:
  - Default: 0.5  
  - Video 2: 0.4  

- Image Resolution:
  - Default: 1280  
  - Video 2: 1600  

- Frame Sampling:
  - Detection pipeline was run at full frame rate (~30 FPS)
  - For dataset publishing (Hugging Face), frames were sampled at 5 FPS to reduce dataset size

- Post-processing:
  - Large bounding boxes removed to reduce cloud misclassification  

---

## Detection Output

Detection results are stored in:

```
detections.parquet
```

Each row corresponds to a frame containing at least one detection:

- image  
- video_title  
- frame_name  
- frame_index  
- timestamp_sec  

Only frames with detections are saved.

---

## Kalman Filter Design

### State Representation

```
state = [x, y, vx, vy]
```

- (x, y): center of bounding box  
- (vx, vy): velocity  

---

### Motion Model

```
x_t = x_{t-1} + v_{t-1}
v_t = v_{t-1}
```

---

### Noise Parameters

- Process noise (Q): accounts for motion uncertainty  
- Measurement noise (R): accounts for detection noise  

---

### Tracking Pipeline

For each frame:

1. Predict next state  
2. Update with detection if available  
3. Continue prediction when detections are missing  

---

## Failure Cases and Handling

### Small Drone Size
- Issue: low visibility due to few pixels  
- Solution: higher resolution and lower threshold  

### False Positives
- Issue: clouds and trees resemble drones  
- Solution: additional datasets and bounding box filtering  

### Consecutive False Positives
- Limitation: Kalman filter does not model appearance  

### Missed Detections
- Kalman filter maintains trajectory through prediction  

---

## Tools Used

- Python  
- OpenCV  
- YOLOv8 (Ultralytics)  
- filterpy  
- pandas
- Roboflow
- Hugging Face
