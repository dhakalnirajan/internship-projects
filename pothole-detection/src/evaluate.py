from ultralytics import YOLO
import yaml
import cv2
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluate trained model: mAP and FPS benchmark."""

    def __init__(self, config: dict, model_path: str = 'models/best.pt'):
        self.config = config
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def evaluate_map(self) -> dict:
        """Compute mAP on test set."""
        data_yaml = self.config['data']['dataset_yaml']
        metrics = self.model.val(data=data_yaml, split='test')
        logger.info(f"mAP@0.5: {metrics.box.map50:.4f}")
        logger.info(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        return {'map50': metrics.box.map50, 'map': metrics.box.map}

    def benchmark_fps(self, num_images: int = 50) -> float:
        """Measure inference FPS on sample test images."""
        test_img_dir = Path(self.config['data']['processed_images']) / 'test'
        img_files = list(test_img_dir.glob('*.jpg'))
        if len(img_files) < num_images:
            logger.warning(f"Only {len(img_files)} test images available, using all.")
            num_images = len(img_files)
        if num_images == 0:
            logger.error("No test images found for FPS benchmark.")
            return 0.0

        # Warm-up
        for _ in range(10):
            _ = self.model(img_files[0], verbose=False)

        times = []
        for img_path in img_files[:num_images]:
            start = time.perf_counter()
            _ = self.model(img_path, verbose=False)
            times.append(time.perf_counter() - start)
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        logger.info(f"Average inference FPS: {fps:.2f} on {len(times)} images")
        return fps

if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO)
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    evaluator = Evaluator(cfg)
    evaluator.evaluate_map()
    evaluator.benchmark_fps()