"""
anomaly_detector.py — Microscope Anomaly Detection

Runs YOLOv8n inference on live frames to detect biological anomalies.
Overlays bounding boxes on stream when anomalies are found.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = Path(__file__).parent / "models" / "custom_microscope.pt"
FALLBACK_MODEL = "yolov8n.pt"  # downloads from ultralytics hub if not present


class AnomalyDetector:
    """
    YOLOv8-based anomaly detector for microscope slide images.

    Args:
        model_path: Path to .pt model file. Falls back to yolov8n if not found.
        confidence_threshold: Minimum confidence to report a detection.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.45,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self._model = None
        self._model_path = model_path or str(DEFAULT_MODEL)
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            mp = Path(self._model_path)
            if mp.exists():
                self._model = YOLO(str(mp))
                logger.info("Loaded custom model: %s", mp)
            else:
                logger.warning("Custom model not found — using %s", FALLBACK_MODEL)
                self._model = YOLO(FALLBACK_MODEL)
        except ImportError:
            logger.error("ultralytics not installed — pip install ultralytics")
        except Exception as exc:
            logger.error("Model load failed: %s", exc)

    def detect(self, frame_np: np.ndarray) -> dict:
        """
        Run anomaly detection on a single frame.

        Args:
            frame_np: BGR numpy array from OpenCV.

        Returns:
            {
              "detected": bool,
              "confidence": float,   # highest confidence detection
              "objects": [{"label": str, "confidence": float, "bbox": [x1,y1,x2,y2]}]
            }
        """
        result = {"detected": False, "confidence": 0.0, "objects": []}
        if self._model is None:
            return result

        try:
            predictions = self._model(frame_np, verbose=False, conf=self.conf_threshold)
            for pred in predictions:
                for box in pred.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = pred.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    result["objects"].append({
                        "label": label,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2],
                    })
                    if conf > result["confidence"]:
                        result["confidence"] = round(conf, 3)

            result["detected"] = len(result["objects"]) > 0
        except Exception as exc:
            logger.error("Detection failed: %s", exc)

        return result

    def draw_detections(
        self, frame: np.ndarray, results: dict
    ) -> np.ndarray:
        """
        Overlay bounding boxes and labels on a frame.

        Args:
            frame: BGR numpy frame.
            results: Output dict from detect().

        Returns:
            Annotated frame copy.
        """
        out = frame.copy()
        for obj in results.get("objects", []):
            x1, y1, x2, y2 = obj["bbox"]
            label = f"{obj['label']} {obj['confidence']:.2f}"
            color = (0, 80, 255) if obj["confidence"] > 0.7 else (0, 200, 80)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if results["detected"]:
            cv2.putText(out, "⚠ ANOMALY DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
        return out