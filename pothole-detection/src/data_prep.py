"""
Convert PASCAL VOC pothole dataset to YOLO format with augmentations.
"""

import os
import xml.etree.ElementTree as ET
import shutil
import yaml
import random
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from albumentations import (
    Compose, RandomRotate90, Flip, RandomBrightnessContrast,
    HueSaturationValue, GaussNoise, BBoxParams
)
import logging

logger = logging.getLogger(__name__)

class DataPreparer:
    """Convert PASCAL VOC dataset to YOLO format, augment, and split."""

    def __init__(self, config: dict):
        self.config = config
        self.raw_img_dir = Path(config['data']['raw_images'])
        self.raw_ann_dir = Path(config['data']['raw_annotations'])
        self.proc_img_dir = Path(config['data']['processed_images'])
        self.proc_lbl_dir = Path(config['data']['processed_labels'])
        self.dataset_yaml = Path(config['data']['dataset_yaml'])
        self.random_state = 42

    def _convert_voc_to_yolo(self, xml_path: Path, img_width: int, img_height: int) -> list:
        """Parse VOC XML and return YOLO format label lines."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name.lower() != 'pothole':
                continue
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            # YOLO: class_id x_center y_center width height (normalized)
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        return labels

    def _augment_image(self, image: np.ndarray, bboxes: list) -> tuple:
        """Apply heavy augmentations to a single image."""
        aug = Compose([
            RandomRotate90(p=0.5),
            Flip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
            GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ], bbox_params=BBoxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        augmented = aug(image=image, bboxes=bboxes, class_labels=[0]*len(bboxes))
        return augmented['image'], augmented['bboxes']

    def prepare(self) -> None:
        """Run the full data preparation pipeline."""
        # Clear previous processed data
        for d in [self.proc_img_dir, self.proc_lbl_dir]:
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

        # Gather XML files
        xml_files = list(self.raw_ann_dir.glob('*.xml'))
        if not xml_files:
            raise FileNotFoundError(f"No XML files found in {self.raw_ann_dir}")
        random.seed(self.random_state)
        random.shuffle(xml_files)

        # Split 70/20/10
        total = len(xml_files)
        train_end = int(0.7 * total)
        val_end = int(0.9 * total)
        splits = {
            'train': xml_files[:train_end],
            'val': xml_files[train_end:val_end],
            'test': xml_files[val_end:]
        }

        # Process each split
        for split_name, split_files in splits.items():
            for xml_path in tqdm(split_files, desc=f"Processing {split_name}"):
                img_path = self.raw_img_dir / (xml_path.stem + '.jpg')
                if not img_path.exists():
                    logger.warning(f"Image {img_path} not found, skipping.")
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Failed to read {img_path}, skipping.")
                    continue
                h, w = img.shape[:2]
                labels = self._convert_voc_to_yolo(xml_path, w, h)
                if not labels:
                    continue  # no pothole

                # Save original
                dest_img = self.proc_img_dir / split_name / img_path.name
                dest_lbl = self.proc_lbl_dir / split_name / (xml_path.stem + '.txt')
                dest_img.parent.mkdir(parents=True, exist_ok=True)
                dest_lbl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(img_path), str(dest_img))
                with open(dest_lbl, 'w') as f:
                    f.write('\n'.join(labels))

                # Augment training images
                if split_name == 'train':
                    bboxes = []
                    for lbl in labels:
                        _, xc, yc, wb, hb = map(float, lbl.split())
                        bboxes.append([xc, yc, wb, hb])
                    for aug_idx in range(2):  # create 2 augmented copies
                        aug_img, aug_bboxes = self._augment_image(img, bboxes)
                        aug_img_path = self.proc_img_dir / split_name / f"{img_path.stem}_aug{aug_idx}.jpg"
                        aug_lbl_path = self.proc_lbl_dir / split_name / f"{xml_path.stem}_aug{aug_idx}.txt"
                        cv2.imwrite(str(aug_img_path), aug_img)
                        with open(aug_lbl_path, 'w') as f:
                            for bbox in aug_bboxes:
                                f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        # Create dataset.yaml
        yaml_content = {
            'path': str(self.proc_img_dir.parent.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,
            'names': ['pothole']
        }
        self.dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dataset_yaml, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        logger.info(f"Dataset YAML created at {self.dataset_yaml}")

if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO)
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    preparer = DataPreparer(cfg)
    preparer.prepare()