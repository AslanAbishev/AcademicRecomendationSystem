"""Render the Assignment 5 Markdown reports as submission-ready PDFs."""

from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FONT_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("TimesNewRoman", str(FONT_DIR / "times.ttf")))
pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", str(FONT_DIR / "timesbd.ttf")))
pdfmetrics.registerFontFamily(
    "TimesNewRoman", normal="TimesNewRoman", bold="TimesNewRoman-Bold"
)


class Report(BaseDocTemplate):
    def __init__(self, path: Path) -> None:
        super().__init__(
            str(path),
            pagesize=(612, 792),
            leftMargin=62,
            rightMargin=62,
            topMargin=56,
            bottomMargin=55,
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            leftPadding=0,
            bottomPadding=0,
            rightPadding=0,
            topPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=self.footer))

    @staticmethod
    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("TimesNewRoman", 9)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawCentredString(306, 31, f"Assignment 5  |  {doc.page}")
        canvas.restoreState()


TITLE = ParagraphStyle(
    "title", fontName="TimesNewRoman-Bold", fontSize=16, leading=20,
    alignment=TA_CENTER, spaceAfter=17,
)
HEADING = ParagraphStyle(
    "heading", fontName="TimesNewRoman-Bold", fontSize=12.5, leading=19,
    alignment=TA_LEFT, spaceBefore=11, spaceAfter=5, keepWithNext=True,
)
BODY = ParagraphStyle(
    "body", fontName="TimesNewRoman", fontSize=12, leading=18,
    alignment=TA_JUSTIFY, spaceAfter=5,
)
REFERENCE = ParagraphStyle(
    "reference", parent=BODY, fontSize=11, leading=16, alignment=TA_LEFT,
)


def markup(line: str) -> str:
    line = escape(line.replace("`", ""))
    return re.sub(
        r"(https?://[^\s]+)",
        lambda match: f'<link href="{match.group(1)}" color="#003366">{match.group(1)}</link>',
        line,
    )


def build(source: Path, target: Path) -> None:
    story = []
    in_references = False
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# "):
            story.append(Paragraph(markup(line[2:]), TITLE))
        elif line.startswith("## "):
            in_references = line[3:] == "References"
            story.append(Paragraph(markup(line[3:]), HEADING))
        else:
            story.append(Paragraph(markup(line), REFERENCE if in_references else BODY))
    Report(target).build(story)


if __name__ == "__main__":
    build(DOCS / "assignment5-theory.md", DOCS / "Assignment_5_Theoretical_Report.pdf")
    build(DOCS / "assignment5-practical.md", DOCS / "Assignment_5_Practical_Report.pdf")
