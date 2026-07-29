import cv2
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def estimate_severity(bbox_area: float, frame_area: float,
                      thresholds: Tuple[float, float] = (0.05, 0.15)) -> str:
    """
    Classify pothole severity based on bounding box area relative to frame area.
    :param bbox_area: area of bounding box (pixels^2)
    :param frame_area: total frame area (pixels^2)
    :param thresholds: tuple (minor_threshold, moderate_threshold)
    :return: 'Minor', 'Moderate', or 'Severe'
    """
    ratio = bbox_area / frame_area if frame_area > 0 else 0
    if ratio < thresholds[0]:
        return "Minor"
    elif ratio < thresholds[1]:
        return "Moderate"
    else:
        return "Severe"

def geotag_frame(lat: Optional[float], lon: Optional[float], alt: Optional[float]) -> str:
    """Return a formatted geotag string or 'No GPS' if coordinates are None."""
    if lat is None or lon is None:
        return "No GPS"
    alt_str = f" Alt:{alt:.1f}m" if alt is not None else ""
    return f"Lat:{lat:.6f} Lon:{lon:.6f}{alt_str}"

def check_alert(distance_m: float, alert_threshold: float) -> bool:
    """Return True if distance is below alert threshold."""
    return distance_m < alert_threshold

def draw_info(frame, obj_id: int, severity: str, lat: Optional[float], lon: Optional[float],
              alt: Optional[float], alert: bool) -> cv2.Mat:
    """
    Overlay pothole info and geotag on the frame.
    :param frame: image (numpy array)
    :param obj_id: tracked object ID
    :param severity: 'Minor', 'Moderate', 'Severe'
    :param lat, lon, alt: GPS coordinates (may be None)
    :param alert: whether to show alert
    :return: annotated frame
    """
    geotag = geotag_frame(lat, lon, alt)
    info = f"ID:{obj_id} Sev:{severity} {geotag}"
    if alert:
        info += " ALERT!"
    color = (0, 0, 255) if alert else (0, 255, 0)
    # Position at top-left, offset by object ID to avoid overlap
    y_offset = 30 + 20 * (obj_id % 5)
    cv2.putText(frame, info, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)
    return frame