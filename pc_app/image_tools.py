"""
image_tools.py — Microscope Image Processing Utilities

Provides distance measurement, annotation, contrast enhancement,
panoramic stitching, and YOLO dataset export.
"""

import csv
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def measure_distance(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    calibration_um_per_px: float = 1.0,
) -> float:
    """
    Measure Euclidean distance between two points in micrometres.

    Args:
        img: Source image (used only for bounds checking).
        pt1: (x1, y1) start point in pixels.
        pt2: (x2, y2) end point in pixels.
        calibration_um_per_px: Conversion factor µm/pixel.

    Returns:
        Distance in micrometres.
    """
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist_px = float(np.sqrt(dx * dx + dy * dy))
    return dist_px * calibration_um_per_px


def annotate_image(
    img: np.ndarray,
    annotations: list[dict],
) -> np.ndarray:
    """
    Draw annotations on an image.

    Each annotation dict may contain:
        type: 'circle' | 'rect' | 'arrow' | 'text'
        color: (B, G, R) tuple, default (0, 255, 0)
        label: str
        pt1, pt2, center, radius as applicable.

    Returns:
        Annotated copy of the image.
    """
    out = img.copy()
    for ann in annotations:
        color = ann.get("color", (0, 255, 0))
        label = ann.get("label", "")
        ann_type = ann.get("type", "circle")

        if ann_type == "circle":
            center = ann.get("center", (0, 0))
            radius = ann.get("radius", 10)
            cv2.circle(out, center, radius, color, 2)
            if label:
                cv2.putText(out, label, (center[0]+5, center[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        elif ann_type == "rect":
            cv2.rectangle(out, ann["pt1"], ann["pt2"], color, 2)
            if label:
                cv2.putText(out, label, ann["pt1"],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        elif ann_type == "arrow":
            cv2.arrowedLine(out, ann["pt1"], ann["pt2"], color, 2)

        elif ann_type == "text":
            cv2.putText(out, label, ann.get("pt1", (10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return out


def enhance_contrast(
    img: np.ndarray,
    method: str = "clahe",
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """
    Enhance image contrast.

    Args:
        img: BGR or grayscale image.
        method: 'clahe' | 'histogram' | 'normalize'
        clip_limit: CLAHE clip limit.
        tile_size: CLAHE tile grid size.

    Returns:
        Contrast-enhanced image.
    """
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                  tileGridSize=(tile_size, tile_size))
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return clahe.apply(img)

    elif method == "histogram":
        if len(img.shape) == 3:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return cv2.equalizeHist(img)

    elif method == "normalize":
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    return img


def stitch_frames(frame_list: list[np.ndarray]) -> Optional[np.ndarray]:
    """
    Stitch a list of overlapping microscope frames into a panorama.

    Args:
        frame_list: List of OpenCV BGR frames.

    Returns:
        Stitched panorama image, or None on failure.
    """
    if len(frame_list) < 2:
        return frame_list[0] if frame_list else None

    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, result = stitcher.stitch(frame_list)
    if status == cv2.Stitcher_OK:
        return result
    logger.error("Stitching failed with status code: %d", status)
    return None


def export_dataset(
    image_folder: str,
    labels_csv: str,
    output_zip: Optional[str] = None,
) -> str:
    """
    Export annotated images as a YOLO-format dataset ZIP.

    CSV format: filename, class_id, x_center, y_center, width, height
    (all bbox values normalised 0–1)

    Args:
        image_folder: Folder containing image files.
        labels_csv: CSV file with YOLO annotations.
        output_zip: Output ZIP path. Defaults to image_folder + '_yolo.zip'

    Returns:
        Path to generated ZIP file.
    """
    img_folder = Path(image_folder)
    csv_path = Path(labels_csv)
    if output_zip is None:
        output_zip = str(img_folder.parent / (img_folder.name + "_yolo.zip"))

    tmp_dir = img_folder.parent / "_yolo_export_tmp"
    (tmp_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    # Parse CSV and write .txt label files
    label_map: dict[str, list[str]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            line = f"{row['class_id']} {row['x_center']} {row['y_center']} {row['width']} {row['height']}"
            label_map.setdefault(fname, []).append(line)

    for img_file in img_folder.glob("*.jpg"):
        shutil.copy(img_file, tmp_dir / "images" / "train" / img_file.name)
        label_lines = label_map.get(img_file.name, [])
        label_file = tmp_dir / "labels" / "train" / img_file.with_suffix(".txt").name
        label_file.write_text("\n".join(label_lines))

    # Write data.yaml
    (tmp_dir / "data.yaml").write_text(
        "train: images/train\nval: images/train\nnc: 1\nnames: ['specimen']\n"
    )

    # Zip
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in tmp_dir.rglob("*"):
            zf.write(file, file.relative_to(tmp_dir))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("Dataset exported: %s", output_zip)
    return output_zip