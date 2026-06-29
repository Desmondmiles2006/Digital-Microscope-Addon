"""
generate_confusion_matrix.py — VGG19 Malaria Model Confusion Matrix

Generates confusion matrix, classification report, and accuracy metrics
from the trained malaria_vgg19.pth model on the full dataset.

Usage:
    python generate_confusion_matrix.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "malaria_vgg19.pth"
DATA_DIR   = Path("D:/micro_scope/digital-microscope-addon/ai_engine/data/cell_images/cell_images")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

print(f"Using device: {DEVICE}")

# ── Load Model ────────────────────────────────────────────────
from malaria import build_model

checkpoint   = torch.load(str(MODEL_PATH), map_location=DEVICE)
class_to_idx = checkpoint.get("class_to_idx", {"Parasitized": 0, "Uninfected": 1})
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes  = len(class_to_idx)

model = build_model(num_classes=num_classes)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"Model loaded — {num_classes} classes: {class_to_idx}")

# ── Load Dataset ──────────────────────────────────────────────
val_tf = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(root=str(DATA_DIR), transform=val_tf)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Dataset loaded — {len(dataset)} images")

# ── Run Inference ─────────────────────────────────────────────
all_preds  = []
all_labels = []

with torch.no_grad():
    for i, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        outputs = model(images)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        print(f"  Batch {i+1}/{len(loader)} processed", end="\r")

print(f"\nInference complete — {len(all_preds)} images processed")

# ── Metrics ───────────────────────────────────────────────────
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

class_names = [idx_to_class[i] for i in range(num_classes)]

accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted")
recall    = recall_score(all_labels, all_preds, average="weighted")
f1        = f1_score(all_labels, all_preds, average="weighted")
cm        = confusion_matrix(all_labels, all_preds)

print("\n" + "="*50)
print("EVALUATION METRICS")
print("="*50)
print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
print("Confusion Matrix:")
print(cm)

# ── Plot Confusion Matrix ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)

ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names, fontsize=13)
ax.set_yticklabels(class_names, fontsize=13)

# Add counts and percentages inside cells
total = cm.sum()
for i in range(num_classes):
    for j in range(num_classes):
        count = cm[i, j]
        pct   = count / total * 100
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                color=color, fontsize=12, fontweight="bold")

ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
ax.set_ylabel("True Label", fontsize=13, fontweight="bold")
ax.set_title(
    f"VGG19 Malaria Detection — Confusion Matrix\n"
    f"Accuracy: {accuracy*100:.2f}%  |  F1: {f1*100:.2f}%  |  Dataset: {len(dataset)} images",
    fontsize=12, fontweight="bold", pad=15
)

plt.tight_layout()

# Save
cm_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(str(cm_path), dpi=150, bbox_inches="tight")
print(f"\nConfusion matrix saved: {cm_path}")

# ── Per-Class Breakdown ───────────────────────────────────────
print("\nPer-Class Breakdown:")
for i, name in enumerate(class_names):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = total - tp - fn - fp
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"  {name}:")
    print(f"    TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"    Sensitivity (Recall) : {sens*100:.2f}%")
    print(f"    Specificity          : {spec*100:.2f}%")
# ── Additional Visualizations (PyPlot) ─────────────────────────

# 1. Metrics Bar Graph
metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metrics_values = [accuracy, precision, recall, f1]

plt.figure()
plt.bar(metrics_names, metrics_values)
plt.ylim(0, 1)
plt.title("Model Performance Metrics")
plt.ylabel("Score")
for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
plt.tight_layout()

metrics_path = OUTPUT_DIR / "metrics_bar.png"
plt.savefig(metrics_path, dpi=150)
print(f"Metrics bar graph saved: {metrics_path}")


# 2. Prediction Distribution Pie Chart
unique, counts = np.unique(all_preds, return_counts=True)
labels = [idx_to_class[i] for i in unique]

plt.figure()
plt.pie(counts, labels=labels, autopct='%1.1f%%')
plt.title("Prediction Distribution")
plt.tight_layout()

pie_path = OUTPUT_DIR / "prediction_distribution.png"
plt.savefig(pie_path, dpi=150)
print(f"Prediction distribution saved: {pie_path}")


# 3. Per-Class Precision, Recall, F1 Bar Chart
from sklearn.metrics import precision_recall_fscore_support

prec, rec, f1_cls, _ = precision_recall_fscore_support(all_labels, all_preds)

x = np.arange(len(class_names))

plt.figure()
plt.bar(x - 0.2, prec, width=0.2, label="Precision")
plt.bar(x, rec, width=0.2, label="Recall")
plt.bar(x + 0.2, f1_cls, width=0.2, label="F1 Score")

plt.xticks(x, class_names)
plt.ylim(0, 1)
plt.title("Per-Class Performance")
plt.legend()
plt.tight_layout()

class_metrics_path = OUTPUT_DIR / "class_metrics.png"
plt.savefig(class_metrics_path, dpi=150)
print(f"Per-class metrics graph saved: {class_metrics_path}")