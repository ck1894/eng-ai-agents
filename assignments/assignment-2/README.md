# CS-GY 6613 – Assignment 2  
## Image-to-Video Semantic Retrieval via Object Detection

This project implements a visual retrieval system that detects car exterior components in a video and retrieves temporally coherent video segments based on query images.

The system uses a fine-tuned YOLOv8 segmentation model (`best_50ep.pt`) trained on the Ultralytics CarParts dataset. Video frames are sampled at 1 FPS and stored in a frame-level Parquet file, which serves as the interface between detection and retrieval.

---

## Detection

**Video ID:** YcvECxtXoxQ  
**Frame Sampling Rate:** 1 FPS  
**Model:** YOLOv8 segmentation (50 epochs)  
**Confidence Threshold:** 0.30  

An additional COCO-based person filtering step was applied to reduce false positive wheel detections caused by overlap with human legs.

Frame-level detections are stored in:

`outputs/detections_50ep_framelevel.parquet`

Each row represents one frame and contains:

- `video_id`
- `frame_index`
- `timestamp_sec`
- `frame_file`
- `detections_json` (list of detection dictionaries)

Each detection includes:

- `class_label`
- `confidence_score`
- `bounding_box`

This file serves as the sole interface between detection and retrieval.

---

## Retrieval

For each query image: YOLO inference is performed, top-K labels are selected, frames are matched based on label overlap, and matching frames are grouped into contiguous temporal segments.

**Matching rule:**  
`overlap_count ≥ need`, where `need = max(min_need, ceil(min_fraction × num_query_labels))`

**Final retrieval parameters:** `--topk 3 --min-need 2 --min-fraction 0.6 --min-segment-sec 3`

---

## Scripts

Core pipeline scripts are located in:

`assignment-2/scripts/`

Main scripts:

- `extract_detections.py` – runs YOLO inference on video frames  
- `filter_detections_by_car_and_person.py` – applies COCO-based filtering  
- `jsonl_to_parquet.py` – converts frame detections to Parquet  
- `retrieve_segments.py` – performs query-to-video segment matching  

Additional debug and visualization scripts are included for analysis and qualitative inspection.

---

## Report

The full methodology, training experiments, qualitative results, filtering logic, and evaluation discussion are documented in:

`report.pdf`

---

## Hugging Face Dataset

The detection and retrieval Parquet outputs are also hosted on Hugging Face:

**Dataset:** https://huggingface.co/datasets/ck1894/cs-gy-6613-assignment-2

This repository contains:

- `detections_50ep_framelevel.parquet`
- `retrieval_segments.parquet`
- Dataset card documentation