import cv2
import yaml
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from src.utils import estimate_severity, geotag_frame
import logging

logger = logging.getLogger(__name__)

class ImageDetector:
    """Batch inference on static images with optional GPS from CSV."""

    def __init__(self, config: dict, model_path: str = 'models/best.pt'):
        self.config = config
        self.model = YOLO(model_path)
        self.conf = config['inference']['conf_threshold']
        self.iou = config['inference']['iou_threshold']

    def process(self, input_path: Path, output_dir: Path, gps_csv: Path = None) -> None:
        """
        Process all images in input_path.
        :param input_path: file or folder
        :param output_dir: where to save annotated images and CSV
        :param gps_csv: optional CSV with columns filename,lat,lon,alt
        """
        if input_path.is_file():
            image_paths = [input_path]
        else:
            image_paths = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        if not image_paths:
            raise ValueError(f"No images found in {input_path}")

        gps_data = {}
        if gps_csv and gps_csv.exists():
            df = pd.read_csv(gps_csv)
            for _, row in df.iterrows():
                gps_data[row['filename']] = (row['lat'], row['lon'], row.get('alt', 0.0))
        elif gps_csv:
            logger.warning(f"GPS CSV {gps_csv} not found, continuing without GPS.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_list = []

        for img_path in tqdm(image_paths, desc='Processing images'):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to read {img_path}, skipping.")
                continue
            h, w = img.shape[:2]

            results = self.model(img, conf=self.conf, iou=self.iou, verbose=False)
            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        detections.append((int(x1), int(y1), int(x2), int(y2), conf))

            # Annotate
            for (x1, y1, x2, y2, conf) in detections:
                area = (x2 - x1) * (y2 - y1)
                severity = estimate_severity(area, h*w)
                label = f"pothole {conf:.2f} [{severity}]"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            lat = lon = alt = None
            if img_path.name in gps_data:
                lat, lon, alt = gps_data[img_path.name]
                geotag = geotag_frame(lat, lon, alt)
                cv2.putText(img, geotag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), img)

            for (x1, y1, x2, y2, conf) in detections:
                results_list.append({
                    'filename': img_path.name,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': conf,
                    'severity': severity,
                    'lat': lat, 'lon': lon, 'alt': alt
                })

        if results_list:
            df = pd.DataFrame(results_list)
            df.to_csv(output_dir / 'detections.csv', index=False)
            logger.info(f"Detection report saved to {output_dir / 'detections.csv'}")
        logger.info(f"Annotated images saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/detect/images'))
    parser.add_argument('--gps_csv', type=Path, help='CSV with GPS per image')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    detector = ImageDetector(cfg)
    detector.process(args.input, args.output, args.gps_csv)