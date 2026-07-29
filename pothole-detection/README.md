# Pothole Detection

A computer vision project for detecting potholes using YOLOv8. This project directory includes scripts for dataset preparation, model training, evaluation, image inference, real-time detection, and model export.

The project is built around the [Pothole Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/pothole-detection/data) and is intended as a complete workflow for training and testing a pothole detection model.

> **Note:** Real-time GPS integration is included in the codebase, but it has not been tested with physical GPS hardware. The functionality is experimental and may require modifications depending on your hardware and environment.

## Features

- Convert the PASCAL VOC dataset to YOLO format.
- Perform dataset augmentation and train/validation/test splitting.
- Train a YOLOv8 model on the prepared dataset.
- Evaluate model performance using standard detection metrics.
- Run inference on individual images or directories of images.
- Perform real-time detection from a webcam or video source.
- Optionally attach GPS coordinates to detections when a compatible serial GPS device is available.
- Export trained models to deployment formats such as ONNX, TensorRT, and TFLite (subject to the export formats supported by the installed YOLO version).

## Project Workflow

1. Prepare the dataset.
2. Train a YOLOv8 model.
3. Evaluate the trained model.
4. Run inference on images or live video.
5. Export the trained model if deployment is required.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download the dataset from Kaggle:

<https://www.kaggle.com/datasets/andrewmvd/pothole-detection/data>

Place the dataset files in the following structure:

```text
data/
└── raw/
    ├── images/            # all .png files
    └── annotations/       # all .xml files   (PASCAL VOC)
```

### 3. Configure the project

Update `config.yaml` with the appropriate dataset paths and, if required, the serial port for your GPS receiver.

GPS serial port (e.g., `/dev/ttyUSB0` on Linux or `COM3` on Windows) – only if you have a GPS receiver

### 4. Prepare the dataset

```bash
python src/data_prep.py
```

### 5. Train the model

```bash
python src/train.py
```

### 6. Evaluate the model

```bash
python src/evaluate.py
```

### 7. Run image inference

```bash
python src/detect_images.py \
    --input data/processed/images/test \
    --output runs/detect/images
```

### 8. Run real-time detection

```bash
python src/detect_realtime.py --source 0 --save
```

### 9. Export the model

```bash
python src/export.py
```

## GPS Support

The project includes optional support for serial GPS receivers that output NMEA data.

If a compatible GPS device is connected and configured in `config.yaml`, detected potholes can be geotagged with their corresponding coordinates.

If no GPS device is available, the detection pipeline continues to run without location information.

**Note:** This functionality has not been validated with physical GPS hardware and should be considered experimental.

### Tips

- Dataset size: With only 665 original images, augmentation helps, but you may want to add more real‑world data for production use.

- Performance: For the best speed, export to TensorRT or TFLite and run on an edge accelerator.

- Contributions: Feel free to open issues or pull requests – we’re happy to improve this together!

## License

This repository contains only the project code.

The dataset used for training is provided separately through Kaggle:

<https://www.kaggle.com/datasets/andrewmvd/pothole-detection/data>

Please refer to the dataset page for its licensing terms.
