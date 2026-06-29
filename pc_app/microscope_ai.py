"""
ai_engine.py — Microscope Analysis using Local Malaria Detection Model

Uses the locally trained VGG19 PyTorch model for slide analysis.
No external API or Ollama required.
"""

import datetime
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to trained malaria model
MODEL_PATH = Path(__file__).parent.parent / "ai_engine" / "models" / "malaria_vgg19.pth"


class MicroscopeAI:
    """
    Uses locally trained VGG19 malaria detection model for slide analysis.
    No API key or internet connection required.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        # api_key kept for compatibility with main.py — not used
        self._detector = None
        self._load_detector()

    def _load_detector(self) -> None:
        """Load malaria detector on startup."""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from ai_engine.malaria import MalariaDetector
            self._detector = MalariaDetector(model_path=MODEL_PATH)
            logger.info("Malaria detector loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load malaria detector: %s", exc)

    def analyze_slide(self, image_path: str) -> str:
        """
        Analyze a microscope slide image using the local malaria model.

        Args:
            image_path: Path to JPEG or PNG image file.

        Returns:
            Analysis text string.
        """
        if self._detector is None:
            raise RuntimeError(
                "Malaria detector not loaded. "
                "Make sure malaria_vgg19.pth exists in ai_engine/models/"
            )

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError("Image not found: " + str(image_path))

        return self._detector.analyze_slide(image_path)

    def generate_report(
        self,
        image_path: str,
        analysis_text: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a PDF lab report with image and AI analysis.

        Args:
            image_path: Path to slide image.
            analysis_text: Analysis text from analyze_slide().
            output_path: Where to save PDF. Defaults to same folder as image.

        Returns:
            Path to generated PDF file.
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer,
                Image as RLImage, HRFlowable
            )
            from reportlab.lib import colors
        except ImportError:
            raise ImportError("reportlab is required: pip install reportlab")

        image_path = Path(image_path)
        if output_path is None:
            output_path = str(image_path.with_suffix(".pdf"))

        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm
        )

        styles      = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ReportTitle", parent=styles["Heading1"],
            textColor=colors.HexColor("#1A3F5C"),
            fontSize=18, spaceAfter=6
        )
        body_style  = ParagraphStyle(
            "ReportBody", parent=styles["Normal"],
            fontSize=11, leading=16
        )
        note_style  = ParagraphStyle(
            "ReportNotes", parent=styles["Normal"],
            fontSize=10, leading=14,
            textColor=colors.grey
        )

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        story = [
            Paragraph("Digital Microscope — AI Lab Report", title_style),
            Paragraph("Generated: " + timestamp, note_style),
            Paragraph("Model: Local VGG19 Malaria Detector", note_style),
            HRFlowable(
                width="100%", thickness=1,
                color=colors.HexColor("#1A6B8A")
            ),
            Spacer(1, 0.4*cm),
        ]

        if image_path.exists():
            img = RLImage(
                str(image_path),
                width=12*cm, height=9*cm,
                kind="proportional"
            )
            story += [img, Spacer(1, 0.4*cm)]

        story.append(Paragraph("AI Analysis", styles["Heading2"]))
        for line in analysis_text.split("\n"):
            if line.strip():
                clean = line.replace("**", "").replace("*", "").replace("#", "")
                story.append(Paragraph(clean, body_style))
        story.append(Spacer(1, 0.6*cm))

        story.append(Paragraph("Student Notes", styles["Heading2"]))
        for _ in range(8):
            story.append(
                HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey)
            )
            story.append(Spacer(1, 0.5*cm))

        doc.build(story)
        logger.info("Report saved: %s", output_path)
        return output_path