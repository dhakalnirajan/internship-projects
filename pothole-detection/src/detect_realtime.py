import argparse
import logging
from pathlib import Path
import cv2
import yaml
from ultralytics import YOLO
from src.gps_reader import GPSReader
from src.tracker import CentroidTracker
from src.utils import check_alert, draw_info, estimate_severity

logger = logging.getLogger(__name__)


class RealtimeDetector:
    """Real-time pothole detection from video source with GPS and tracking."""

    def __init__(self, config: dict, model_path: str = "models/best.pt"):
        self.config = config
        self.model = YOLO(model_path)
        self.conf = config["inference"]["conf_threshold"]
        self.iou = config["inference"]["iou_threshold"]
        self.frame_skip = config["inference"]["frame_skip"]
        self.max_age = config["inference"]["max_age"]
        self.min_hits = config["inference"]["min_hits"]
        self.alert_dist = config["inference"]["alert_distance_meters"]
        self.gps = None

    def _init_gps(self):
        """Lazy initialization of GPS reader - only if configured."""
        if self.gps is not None:
            return
        gps_cfg = self.config.get("gps")
        if gps_cfg is None:
            logger.info("GPS not configured in config.yaml - continuing without location.")
            return
        try:
            self.gps = GPSReader(
                port=gps_cfg["port"],
                baudrate=gps_cfg.get("baudrate", 9600),
                timeout=gps_cfg.get("timeout", 1),
            )
            logger.info("GPS initialized successfully.")
        except Exception as e:
            logger.error(f"GPS initialization failed: {e}")
            self.gps = None

    def process(self, source: str, output_dir: Path = None, save_video: bool = False) -> None:
        """Process video stream from webcam or file."""
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(str(source))
            
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        out = None
        if save_video:
            if output_dir is None:
                output_dir = Path("runs/detect/video")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_dir / "output.mp4"), fourcc, fps, (width, height))

        tracker = CentroidTracker(max_age=self.max_age, min_hits=self.min_hits)
        frame_count = 0

        self._init_gps()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

            # Run YOLO inference
            results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        detections.append([x1, y1, x2, y2, confidence])

            tracked_objects = tracker.update(detections)

            # Fetch current GPS location if available
            lat = lon = alt = None
            if self.gps is not None:
                try:
                    lat, lon, alt = self.gps.get_location(max_attempts=5)
                except Exception as e:
                    logger.warning(f"GPS read error: {e}")

            # Draw information for each tracked pothole
            for obj_id, centroid, bbox in tracked_objects:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                
                severity = estimate_severity(area, frame.shape[0] * frame.shape[1])
                distance_m = 10.0 if area < 20000 else 3.0
                alert = check_alert(distance_m, self.alert_dist)
                
                frame = draw_info(frame, obj_id, severity, lat, lon, alt, alert)

            cv2.imshow("Pothole Detection", frame)
            if out is not None:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        if self.gps:
            self.gps.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", 
        type=str, 
        default="0", 
        help="Video source: 0 for webcam, or path to video file"
    )
    parser.add_argument("--output", type=str, default="runs/detect/video")
    parser.add_argument("--save", action="store_true", help="Save annotated video")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    detector = RealtimeDetector(cfg)
    detector.process(args.source, args.output, args.save)