# modules/pdf_export.py

import re
import io
import logging
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, KeepTogether
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = A4
MARGIN = 50

PALETTE = {
    "primary":     colors.HexColor("#2C3E50"),
    "accent":      colors.HexColor("#2980B9"),
    "subheading":  colors.HexColor("#34495E"),
    "warning":     colors.HexColor("#E74C3C"),
    "muted":       colors.HexColor("#7F8C8D"),
    "rule":        colors.HexColor("#BDC3C7"),
    "bg_note":     colors.HexColor("#EAF4FB"),
}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Convert markdown inline syntax to ReportLab-safe XML.
    Handles bold, italic, inline-code, special chars, and unicode stripping.
    """
    # Escape ampersands FIRST (before injecting any XML tags)
    text = text.replace("&", "&amp;")

    # Bold (**text** or __text__)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.*?)__",     r"<b>\1</b>", text)

    # Italic (*text* or _text_) — skip if already inside a bold tag
    text = re.sub(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!_)_(?!_)(.*?)(?<!_)_(?!_)",       r"<i>\1</i>", text)

    # Inline code
    text = re.sub(r"`(.*?)`", r"<font name='Courier'>\1</font>", text)

    # Strip table pipes
    text = text.replace("|", " ")

    # Strip non-ASCII safely (keeps accented chars that ReportLab can render)
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Style sheet
# ---------------------------------------------------------------------------
def _build_styles() -> dict:
    base = getSampleStyleSheet()

    def ps(name, parent_name="Normal", **kwargs) -> ParagraphStyle:
        return ParagraphStyle(name, parent=base[parent_name], **kwargs)

    return {
        "doc_title": ps(
            "DocTitle", "Title",
            fontSize=20, textColor=PALETTE["primary"],
            spaceAfter=4, alignment=TA_CENTER, leading=24,
        ),
        "doc_subtitle": ps(
            "DocSubtitle",
            fontSize=9, textColor=PALETTE["muted"],
            spaceAfter=14, alignment=TA_CENTER,
        ),
        "section_title": ps(
            "SectionTitle", "Title",
            fontSize=13, textColor=PALETTE["accent"],
            spaceBefore=14, spaceAfter=6, alignment=TA_CENTER,
        ),
        "h1": ps(
            "H1", "Heading1",
            fontSize=14, textColor=PALETTE["primary"],
            spaceBefore=10, spaceAfter=6, leading=18,
        ),
        "h2": ps(
            "H2", "Heading2",
            fontSize=12, textColor=PALETTE["accent"],
            spaceBefore=8, spaceAfter=5, leading=16,
        ),
        "h3": ps(
            "H3", "Heading3",
            fontSize=11, textColor=PALETTE["subheading"],
            spaceBefore=6, spaceAfter=4, leading=15,
        ),
        "body": ps(
            "Body",
            fontSize=10, leading=15, spaceAfter=4,
            alignment=TA_JUSTIFY, textColor=PALETTE["primary"],
        ),
        "bullet": ps(
            "Bullet",
            fontSize=10, leading=14, spaceAfter=3,
            leftIndent=18, firstLineIndent=-10,
            textColor=PALETTE["primary"],
        ),
        "numbered": ps(
            "Numbered",
            fontSize=10, leading=14, spaceAfter=3,
            leftIndent=22, firstLineIndent=-14,
            textColor=PALETTE["primary"],
        ),
        "blockquote": ps(
            "Blockquote",
            fontSize=10, leading=14,
            leftIndent=24, rightIndent=12,
            textColor=PALETTE["subheading"],
            spaceBefore=4, spaceAfter=4,
            fontName="Helvetica-Oblique",
        ),
        "disclaimer": ps(
            "Disclaimer",
            fontSize=8, textColor=PALETTE["muted"],
            fontName="Helvetica-Oblique",
            spaceBefore=8, spaceAfter=4, alignment=TA_CENTER,
        ),
    }


# ---------------------------------------------------------------------------
# Line classifier + flowable builder
# ---------------------------------------------------------------------------
def _line_to_flowable(line: str, styles: dict):
    """
    Map a single markdown line to a ReportLab flowable.
    Returns None for lines that should be skipped.
    """
    stripped = line.strip()

    # Empty line → small vertical gap
    if not stripped:
        return Spacer(1, 0.08 * inch)

    # Pure horizontal-rule lines (---, ***, ___)
    if re.fullmatch(r"[-*_]{3,}", stripped):
        return HRFlowable(
            width="100%", thickness=0.5,
            color=PALETTE["rule"], spaceAfter=6, spaceBefore=6,
        )

    # Pure table separator rows like |---|---|
    if re.fullmatch(r"[\|\- ]+", stripped):
        return None

    c = clean_text(line)
    if not c:
        return None

    try:
        if line.startswith("# "):
            return Paragraph(clean_text(line[2:]), styles["h1"])
        elif line.startswith("## "):
            return Paragraph(clean_text(line[3:]), styles["h2"])
        elif line.startswith("### "):
            return Paragraph(clean_text(line[4:]), styles["h3"])
        elif stripped.startswith(("- ", "* ")):
            content = clean_text(stripped[2:])
            return Paragraph(f"\u2022&nbsp;&nbsp;{content}", styles["bullet"])
        elif re.match(r"^\d+\.\s", stripped):
            # Preserve the number for numbered lists
            num, rest = stripped.split(".", 1)
            return Paragraph(
                f"<b>{num}.</b>&nbsp;{clean_text(rest.strip())}",
                styles["numbered"],
            )
        elif stripped.startswith(">"):
            return Paragraph(clean_text(stripped.lstrip("> ")), styles["blockquote"])
        else:
            return Paragraph(c, styles["body"])
    except Exception:
        logger.warning("Skipped unparseable line: %r", line[:80])
        return None


def _text_to_flowables(text: str, styles: dict) -> list:
    """Convert a full markdown string to a list of ReportLab flowables."""
    flowables = []
    for line in text.split("\n"):
        flowable = _line_to_flowable(line, styles)
        if flowable is not None:
            flowables.append(flowable)
    return flowables


# ---------------------------------------------------------------------------
# Page numbering callback
# ---------------------------------------------------------------------------
def _add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(PALETTE["muted"])
    page_num = f"Page {doc.page}"
    canvas.drawRightString(PAGE_W - MARGIN, MARGIN * 0.5, page_num)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def markdown_to_pdf(
    markdown_text: str,
    recommendations: str = None,
    title: str = "EduAssist — Study Notes",
) -> bytes:
    """
    Convert markdown notes (and optional AI recommendations) to a PDF.

    Args:
        markdown_text:   Main content in markdown format.
        recommendations: Optional AI-generated recommendations (markdown).
        title:           Document title shown at the top of the PDF.

    Returns:
        PDF as bytes, or b"" on failure.
    """
    try:
        logger.info("Generating PDF: %r", title)
        styles = _build_styles()
        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=MARGIN,
            leftMargin=MARGIN,
            topMargin=MARGIN,
            bottomMargin=MARGIN,
            title=title,
            author="EduAssist",
        )

        story = []

        # ----- Header -----
        timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
        story.append(Paragraph(clean_text(title), styles["doc_title"]))
        story.append(Paragraph(f"Generated on {timestamp}", styles["doc_subtitle"]))
        story.append(HRFlowable(
            width="100%", thickness=1,
            color=PALETTE["accent"], spaceAfter=10,
        ))

        # ----- Notes section -----
        if markdown_text and markdown_text.strip():
            story.extend(_text_to_flowables(markdown_text, styles))

        # ----- Recommendations section -----
        if recommendations and recommendations.strip():
            story.append(Spacer(1, 0.3 * inch))
            story.append(HRFlowable(
                width="100%", thickness=0.5,
                color=PALETTE["rule"], spaceAfter=8,
            ))
            # Keep section title + first paragraph together (no orphan heading)
            story.append(KeepTogether([
                Paragraph("AI Recommendations", styles["section_title"]),
                Spacer(1, 0.1 * inch),
            ]))
            story.extend(_text_to_flowables(recommendations, styles))

            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(
                "⚠ All recommendations are AI-generated. "
                "Please verify with your textbook or professor before use.",
                styles["disclaimer"],
            ))

        doc.build(
            story,
            onFirstPage=_add_page_number,
            onLaterPages=_add_page_number,
        )

        pdf_bytes = buffer.getvalue()
        buffer.close()
        logger.info("PDF generated successfully (%d bytes)", len(pdf_bytes))
        return pdf_bytes

    except Exception as e:
        logger.exception("PDF generation failed: %s", e)
        return b""