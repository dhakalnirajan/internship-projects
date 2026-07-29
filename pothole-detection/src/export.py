from ultralytics import YOLO
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Exporter:
    """Export trained model to various formats for edge deployment."""

    def __init__(self, config: dict, model_path: str = 'models/best.pt'):
        self.config = config
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.format = config['export']['format']
        self.half = config['export']['half']
        self.int8 = config['export']['int8']

    def export(self) -> None:
        """Export to specified format."""
        logger.info(f"Exporting model to {self.format} format...")
        try:
            if self.format == 'onnx':
                self.model.export(format='onnx', half=self.half, imgsz=640)
            elif self.format == 'tflite':
                self.model.export(format='tflite', int8=self.int8, imgsz=640)
            elif self.format == 'tensorrt':
                self.model.export(format='engine', half=self.half, imgsz=640)
            else:
                raise ValueError(f"Unsupported format: {self.format}")
            logger.info(f"Export successful. Check 'exports/' directory.")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO)
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    exporter = Exporter(cfg)
    exporter.export()