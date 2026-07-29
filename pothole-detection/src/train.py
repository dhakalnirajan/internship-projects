import logging
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO
import yaml

logger = logging.getLogger(__name__)


class Trainer:
    """Train YOLOv8 model on pothole dataset."""

    def __init__(self, config: dict):
        self.config = config
        self.model_size = config["train"]["model_size"]
        self.data_yaml = config["data"]["dataset_yaml"]
        self.epochs = config["train"]["epochs"]
        self.batch = config["train"]["batch_size"]
        self.imgsz = config["train"]["img_size"]
        self.lr0 = config["train"]["lr0"]
        self.lrf = config["train"]["lrf"]
        self.weight_decay = config["train"]["weight_decay"]
        self.augment = config["train"]["augment"]
        self.patience = config["train"]["patience"]
        self.device = 0 if torch.cuda.is_available() else "cpu"

    def train(self) -> None:
        """Execute training and save the best model weights."""
        if not Path(self.data_yaml).exists():
            raise FileNotFoundError(f"Dataset YAML {self.data_yaml} not found. Run data_prep first.")

        model = YOLO(self.model_size)
        logger.info(f"Starting training on device: {self.device}")

        # Run training using ultralytics YOLO
        model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            batch=self.batch,
            imgsz=self.imgsz,
            lr0=self.lr0,
            lrf=self.lrf,
            weight_decay=self.weight_decay,
            augment=self.augment,
            patience=self.patience,
            project="runs/train",
            name="pothole_detector",
            exist_ok=True,
            device=self.device,
            verbose=True,
        )

        # Copy the best model weights to the local models directory
        best_path = Path("runs/train/pothole_detector/weights/best.pt")
        if best_path.exists():
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            shutil.copy(str(best_path), str(models_dir / "best.pt"))
            logger.info(f"Best model saved to {models_dir / 'best.pt'}")
            
        logger.info("Training complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    trainer = Trainer(cfg)
    trainer.train()