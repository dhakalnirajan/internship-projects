# Pothole Detection Model Training & Evaluation Report

## 1. Executive Summary
This report summarizes the training, validation, benchmarking, and inference execution of the YOLOv8 model driven by `config.yaml` on the Pothole Detection dataset using Google Colab Free Tier (NVIDIA T4 GPU).

## 2. Configuration & Training Metrics
* Model Architecture: `yolov8s.pt` 
* Epochs Trained: 25
* Batch Size: 16
* Image Size: 640x640
* Compute Cost: $0.00 (Free Tier)

### Generated Plots & Artifacts
* Loss & Metric Curves: `runs/detect/train/results.png` 
* Confusion Matrix: `runs/detect/train/confusion_matrix.png` 
* Precision-Recall Curves: `runs/detect/train/PR_curve.png` 

## 3. Validation Performance
* Box mAP50: 0.7225
* Box mAP50-95: 0.5550
* Precision: 0.9744
* Recall: 0.7213

## 4. Latency & Benchmark Results
* Average Inference Latency: 71.68 ms / frame
* Approximate Throughput: 13.95 FPS
* Weights Path: `runs/detect/train/weights/best.pt` 

## 5. Sample Inferences
Inference visualizations with confidence filters (`conf=0.6`) are saved under `runs/detect/predict/`.
