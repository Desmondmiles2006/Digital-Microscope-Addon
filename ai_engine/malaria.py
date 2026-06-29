"""
malaria.py — Malaria Detection AI Engine using VGG19 Transfer Learning (PyTorch)

Dataset  : D:/micro_scope/digital-microscope-addon/ai_engine/data/cell_images
           Subfolders: Parasitized/ and Uninfected/

Classes:
  0 = Parasitized (malaria infected)
  1 = Uninfected   (healthy)

Usage:
  Train  : python malaria.py --mode train
  Predict: python malaria.py --mode predict --image path/to/cell.png
  Eval   : python malaria.py --mode evaluate
"""

import os
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = Path("D:/micro_scope/digital-microscope-addon/ai_engine/data/cell_images")
MODEL_PATH = BASE_DIR / "models" / "malaria_vgg19.pth"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 50
CLASS_NAMES = {0: "Parasitized", 1: "Uninfected"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)


# ── Transforms ───────────────────────────────────────────────
def get_transforms():
    """
    Returns train and validation image transforms.
    Train: augmentation (flip, rotate, color jitter) + normalize.
    Val  : resize + normalize only.
    """
    train_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    val_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_tf, val_tf


# ── Model ─────────────────────────────────────────────────────
def build_model(num_classes: int = 2) -> nn.Module:
    """
    Load pretrained VGG19, freeze all layers except the final classifier.
    Replace the last FC layer with a new head for num_classes outputs.
    """
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

    # Freeze all feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace final classifier layer
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    model = model.to(DEVICE)
    logger.info("VGG19 model built — %d output classes — device: %s", num_classes, DEVICE)
    return model


# ── Dataset ───────────────────────────────────────────────────
def get_dataloaders(data_dir: Path):
    """
    Load dataset from data_dir using ImageFolder.
    Splits 90% train / 10% validation automatically.

    Returns:
        train_loader, val_loader, class_to_idx
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            "Dataset not found: " + str(data_dir) +
            "\nExpected subfolders: Parasitized/ and Uninfected/"
        )

    train_tf, val_tf = get_transforms()

    # Load full dataset with train transforms first (will split after)
    full_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_tf)

    # Split into train/val
    val_size   = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply val transforms to validation split
    val_dataset.dataset = datasets.ImageFolder(root=str(data_dir), transform=val_tf)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True
    )

    logger.info("Train samples     : %d", train_size)
    logger.info("Validation samples: %d", val_size)
    logger.info("Class mapping     : %s", full_dataset.class_to_idx)

    return train_loader, val_loader, full_dataset.class_to_idx


# ── Training ──────────────────────────────────────────────────
def train(
    data_dir: Path = DATA_DIR,
    model_save_path: Path = MODEL_PATH,
    epochs: int = EPOCHS
) -> None:
    """
    Train VGG19 model on malaria cell dataset and save weights.

    Args:
        data_dir       : Path to cell_images folder.
        model_save_path: Where to save trained model (.pth).
        epochs         : Number of training epochs.
    """
    train_loader, val_loader, class_to_idx = get_dataloaders(data_dir)
    num_classes = len(class_to_idx)

    model     = build_model(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[6].parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train phase ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()

        # ── Validation phase ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total
        t_loss    = train_loss    / train_total
        v_loss    = val_loss      / val_total

        logger.info(
            "Epoch %2d/%d | Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f",
            epoch, epochs, t_loss, train_acc, v_loss, val_acc
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":         epoch,
                "model_state":   model.state_dict(),
                "class_to_idx":  class_to_idx,
                "val_accuracy":  val_acc,
            }, str(model_save_path))
            logger.info("  Saved best model (val_acc=%.4f)", val_acc)

    logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)
    logger.info("Model saved: %s", model_save_path)


# ── Prediction ────────────────────────────────────────────────
def predict(
    image_path: str,
    model_path: Path = MODEL_PATH
) -> dict:
    """
    Predict whether a cell image shows malaria infection.

    Args:
        image_path: Path to cell image (.png or .jpg).
        model_path: Path to trained .pth model file.

    Returns:
        {
          "label":      "Parasitized" or "Uninfected",
          "class_id":   0 or 1,
          "confidence": float,
          "probabilities": {"Parasitized": float, "Uninfected": float}
        }
    """
    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found: " + str(model_path) +
            "\nTrain first: python malaria.py --mode train"
        )

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError("Image not found: " + str(img_path))

    # Load checkpoint to get class mapping
    checkpoint   = torch.load(str(model_path), map_location=DEVICE)
    class_to_idx = checkpoint.get("class_to_idx", {"Parasitized": 0, "Uninfected": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes  = len(class_to_idx)

    # Build and load model
    model = build_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Preprocess image
    _, val_tf = get_transforms()
    img       = Image.open(str(img_path)).convert("RGB")
    tensor    = val_tf(img).unsqueeze(0).to(DEVICE)   # shape (1, 3, 224, 224)

    # Inference
    with torch.no_grad():
        outputs     = model(tensor)
        probs       = torch.softmax(outputs, dim=1)[0]
        class_id    = int(probs.argmax().item())
        confidence  = float(probs[class_id].item())

    label = idx_to_class.get(class_id, "Unknown")

    result = {
        "label":        label,
        "class_id":     class_id,
        "confidence":   round(confidence, 4),
        "probabilities": {
            name: round(float(probs[idx].item()), 4)
            for name, idx in class_to_idx.items()
        }
    }
    logger.info("Prediction: %s (confidence: %.2f%%)", label, confidence * 100)
    return result


# ── Evaluation ────────────────────────────────────────────────
def evaluate_on_dataset(
    data_dir: Path = DATA_DIR,
    model_path: Path = MODEL_PATH
) -> dict:
    """
    Evaluate model accuracy on the full dataset.

    Returns:
        {"accuracy": float, "correct": int, "total": int}
    """
    if not model_path.exists():
        raise FileNotFoundError("Model not found: " + str(model_path))

    _, val_tf = get_transforms()
    dataset   = datasets.ImageFolder(root=str(data_dir), transform=val_tf)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    checkpoint  = torch.load(str(model_path), map_location=DEVICE)
    num_classes = len(checkpoint.get("class_to_idx", {"Parasitized": 0, "Uninfected": 1}))

    model = build_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    accuracy = correct / total
    logger.info("Evaluation — Accuracy: %.4f (%d/%d)", accuracy, correct, total)
    return {"accuracy": round(accuracy, 4), "correct": correct, "total": total}


# ── MalariaDetector (pc_app integration) ─────────────────────
class MalariaDetector:
    """
    Drop-in integration class for the pc_app AI Analysis tab.

    Usage in pc_app/main.py:
        from ai_engine.malaria import MalariaDetector
        detector = MalariaDetector()
        result   = detector.analyze_slide("path/to/cell.jpg")
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path or MODEL_PATH

    def analyze_slide(self, image_path: str) -> str:
        """
        Analyze a microscope cell image for malaria.
        Returns formatted analysis string compatible with ai_engine.py interface.
        """
        result     = predict(image_path, self.model_path)
        label      = result["label"]
        confidence = result["confidence"] * 100
        probs      = result["probabilities"]

        analysis = (
            "## Malaria Cell Detection (VGG19 — PyTorch)\n\n"
            "**1. Diagnosis**\n"
            "Cell classification: **{label}**\n"
            "Confidence: {conf:.1f}%\n\n"
            "**2. Probabilities**\n"
            "- Parasitized (infected): {para:.1f}%\n"
            "- Uninfected (healthy)  : {unin:.1f}%\n\n"
            "**3. Clinical Interpretation**\n"
            "{interp}\n\n"
            "**4. Recommended Follow-up**\n"
            "{followup}\n"
        ).format(
            label=label,
            conf=confidence,
            para=probs.get("Parasitized", 0.0) * 100,
            unin=probs.get("Uninfected",  0.0) * 100,
            interp=(
                "The cell shows morphological features consistent with "
                "Plasmodium parasite infection. Characteristic ring-form "
                "trophozoites or intraerythrocytic stages may be present."
                if label == "Parasitized" else
                "No evidence of Plasmodium parasite infection detected. "
                "The erythrocyte appears morphologically normal."
            ),
            followup=(
                "Confirm with Giemsa-stained thick and thin blood smear. "
                "Consider species identification (P. falciparum, P. vivax, etc.). "
                "Quantify parasite density against WBC count if positive."
                if label == "Parasitized" else
                "Continue routine screening if clinically indicated. "
                "Repeat test if symptoms persist or clinical signs suggest infection."
            )
        )
        return analysis


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Malaria Detection using VGG19 Transfer Learning (PyTorch)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "evaluate"],
        required=True,
        help="train | predict | evaluate"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to cell image (required for predict mode)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DATA_DIR),
        help="Path to cell_images folder"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(MODEL_PATH),
        help="Path to save/load model .pth file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Training epochs (default 50)"
    )

    args       = parser.parse_args()
    data_path  = Path(args.data_dir)
    model_path = Path(args.model_path)

    if args.mode == "train":
        train(data_dir=data_path, model_save_path=model_path, epochs=args.epochs)

    elif args.mode == "predict":
        if not args.image:
            parser.error("--image is required for predict mode")
        result = predict(args.image, model_path=model_path)
        print("\n=== Malaria Detection Result ===")
        print("Label      :", result["label"])
        print("Confidence :", f"{result['confidence']*100:.2f}%")
        print("Parasitized:", f"{result['probabilities'].get('Parasitized', 0)*100:.2f}%")
        print("Uninfected :", f"{result['probabilities'].get('Uninfected',  0)*100:.2f}%")

    elif args.mode == "evaluate":
        metrics = evaluate_on_dataset(data_dir=data_path, model_path=model_path)
        print("\n=== Evaluation Results ===")
        print("Accuracy:", f"{metrics['accuracy']*100:.2f}%")
        print("Correct :", metrics["correct"])
        print("Total   :", metrics["total"])