"""
ai_engine.py — Claude Vision API Integration for Microscope Analysis

Sends captured slide images to Claude claude-sonnet-4-5 (Vision) and returns
structured biomedical analysis. Also generates PDF lab reports via reportlab.
"""

import base64
import datetime
import logging
from pathlib import Path
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert biomedical microscopy assistant helping undergraduate biology students.
Analyze the microscope slide image provided and describe:
1. What biological structures or specimens are visible
2. Morphological features and their significance
3. Any anomalies or points of interest
4. Suggested follow-up observations

Be educational, precise, and suitable for undergraduate biology students.
Structure your response with clear headings for each point."""


class MicroscopeAI:
    """
    Wraps the Anthropic Claude Vision API for microscope slide analysis.

    Args:
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5"

    def analyze_slide(self, image_path: str) -> str:
        """
        Analyze a microscope slide image using Claude Vision.

        Args:
            image_path: Path to JPEG/PNG image file.

        Returns:
            Markdown-formatted analysis string.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        media_type = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        with open(path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Please analyze this microscope slide image.",
                            },
                        ],
                    }
                ],
            )
            return message.content[0].text
        except anthropic.APIError as exc:
            logger.error("Claude API error: %s", exc)
            raise

    def generate_report(self, image_path: str, analysis_text: str,
                        output_path: Optional[str] = None) -> str:
        """
        Generate a PDF lab report with image and AI analysis.

        Args:
            image_path: Path to slide image.
            analysis_text: AI analysis markdown text.
            output_path: Where to save PDF. Defaults to same folder as image.

        Returns:
            Path to generated PDF file.
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, HRFlowable
            )
            from reportlab.lib import colors
        except ImportError:
            raise ImportError("reportlab is required: pip install reportlab")

        image_path = Path(image_path)
        if output_path is None:
            output_path = image_path.with_suffix(".pdf")

        doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                     textColor=colors.HexColor("#1A3F5C"),
                                     fontSize=18, spaceAfter=6)
        body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                    fontSize=11, leading=16)
        note_style = ParagraphStyle("Notes", parent=styles["Normal"],
                                    fontSize=10, leading=14,
                                    textColor=colors.grey)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story = [
            Paragraph("Digital Microscope — AI Lab Report", title_style),
            Paragraph(f"Generated: {timestamp}", note_style),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1A6B8A")),
            Spacer(1, 0.4*cm),
        ]

        # Embed image
        if image_path.exists():
            img = RLImage(str(image_path), width=12*cm, height=9*cm,
                          kind="proportional")
            story += [img, Spacer(1, 0.4*cm)]

        # Analysis
        story.append(Paragraph("AI Analysis", styles["Heading2"]))
        for line in analysis_text.split("\n"):
            if line.strip():
                clean = line.replace("**", "<b>").replace("*", "")
                story.append(Paragraph(clean, body_style))
        story.append(Spacer(1, 0.6*cm))

        # Student notes field
        story.append(Paragraph("Student Notes", styles["Heading2"]))
        for _ in range(8):
            story.append(HRFlowable(width="100%", thickness=0.5,
                                     color=colors.lightgrey))
            story.append(Spacer(1, 0.5*cm))

        doc.build(story)
        logger.info("Report saved: %s", output_path)
        return str(output_path)