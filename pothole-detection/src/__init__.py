"""
Pothole Detection System

This package provides modules for:
- Data preparation (VOC → YOLO)
- Training & evaluation of YOLOv8 models
- Real-time detection with GPS and object tracking
- Batch inference on static images
- Model export for edge deployment
"""

__version__ = "0.1.0"
__author__ = "Nirajan Dhakal"

# Optionally expose key classes for easier import
from .gps_reader import GPSReader
from .tracker import CentroidTracker
from .utils import estimate_severity, geotag_frame, check_alert, draw_info

__all__ = [
    "GPSReader",
    "CentroidTracker",
    "estimate_severity",
    "geotag_frame",
    "check_alert",
    "draw_info",
]