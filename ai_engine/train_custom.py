"""
train_custom.py — YOLOv8n Custom Model Training

Trains a YOLOv8n model on a YOLO-format dataset exported from image_tools.py.
Saves to ai_engine/models/custom_microscope.pt.

Usage:
    python train_custom.py --data /path/to/dataset/data.yaml --epochs 50
"""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
OUTPUT_MODEL = Path(__file__).parent / "models" / "custom_microscope.pt"


def train(data_yaml: str, epochs: int = 50, imgsz: int = 320,
          batch: int = 16, device: str = "cpu") -> None:
    """
    Train YOLOv8n on a custom microscope dataset.

    Args:
        data_yaml: Path to YOLO data.yaml file.
        epochs: Number of training epochs.
        imgsz: Input image size (square).
        batch: Batch size.
        device: 'cpu' or '0' (CUDA GPU index).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")  # start from pretrained weights
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(OUTPUT_MODEL.parent),
        name="microscope_run",
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
    )

    # Copy best weights to canonical output path
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        import shutil
        shutil.copy(best, OUTPUT_MODEL)
        logger.info("Best model saved to: %s", OUTPUT_MODEL)
    else:
        logger.warning("best.pt not found in %s", results.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n for microscope anomaly detection")
    parser.add_argument("--data",   required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=320)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args.data, args.epochs, args.imgsz, args.batch, args.device)