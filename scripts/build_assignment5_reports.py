"""Build the two Assignment 5 reports from their reviewed Markdown sources."""

from pathlib import Path
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def build(source: Path, target: Path) -> None:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.78)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)

    for style_name, size, before, after in (
        ("Title", 16, 0, 18),
        ("Heading 1", 13, 12, 5),
    ):
        style = document.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.replace("`", "")
        if line.startswith("# "):
            paragraph = document.add_paragraph(line[2:], style="Title")
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith("## "):
            document.add_paragraph(line[3:], style="Heading 1")
        elif re.match(r"^\d+\. ", line):
            document.add_paragraph(line, style="Normal")
        else:
            paragraph = document.add_paragraph(line, style="Normal")
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Assignment 5")
    target.parent.mkdir(parents=True, exist_ok=True)
    document.save(target)


if __name__ == "__main__":
    build(DOCS / "assignment5-theory.md", DOCS / "Assignment_5_Theoretical_Report.docx")
    build(DOCS / "assignment5-practical.md", DOCS / "Assignment_5_Practical_Report.docx")
