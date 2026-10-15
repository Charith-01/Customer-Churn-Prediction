"""Reproducible seven-page business-report generation for Phase 9.

The generator reads locked Phase 1-8 outputs, validates their consistency,
and assembles a deterministic A4 PDF. It does not load, fit, or modify the
serialized model.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import pymupdf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PURPLE = colors.HexColor("#5B3AA8")
PURPLE_LIGHT = colors.HexColor("#EEE9FA")
BLUE = colors.HexColor("#3569D4")
BLUE_LIGHT = colors.HexColor("#EAF0FC")
INK = colors.HexColor("#252330")
MUTED = colors.HexColor("#625E6E")
LINE = colors.HexColor("#D8D3E4")
WHITE = colors.white

REPORT_TITLE = "Customer Churn Prediction and Retention Analysis"
REPORT_SUBTITLE = "End-to-End Machine Learning and Business Retention Insights"
DEFAULT_OUTPUT_NAME = Path("reports") / "04_business_report.pdf"

REQUIRED_FIGURES = (
    "01_churn_distribution.png",
    "report_contract_churn.png",
    "report_usage_quintiles.png",
    "report_model_cv_comparison.png",
    "phase6_confusion_matrix.png",
    "phase7_global_feature_importance.png",
    "phase8_cumulative_gains.png",
)

REQUIRED_SECTION_TITLES = (
    "Executive Summary",
    "Dataset and Data Preparation",
    "Exploratory Analysis and Feature Engineering",
    "Machine Learning Development and Evaluation",
    "Model Explainability and Customer Insights",
    "Retention Simulation and Business Impact",
    "Recommendations, Limitations and Conclusion",
)


class InvariantCanvas(canvas.Canvas):
    """ReportLab canvas with deterministic metadata and object identifiers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["invariant"] = 1
        super().__init__(*args, **kwargs)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a local artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_report_figures(root: str | Path) -> tuple[Path, ...]:
    """Create report-specific charts from locked, machine-readable results."""
    project = Path(root).resolve()
    cleaned = pd.read_csv(project / "data" / "processed" / "02_cleaned_dataset.csv")
    split = json.loads(
        (project / "data" / "processed" / "02_split_indices.json").read_text(
            encoding="utf-8"
        )
    )
    comparison = pd.read_csv(
        project / "reports" / "modeling" / "phase5_model_comparison.csv"
    )
    training = cleaned.iloc[split["train_indices"]].copy()
    figure_dir = project / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    chart_style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.edgecolor": "#D8D3E4",
        "axes.grid": True,
        "grid.color": "#E7E3EF",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    }
    png_metadata = {"Software": "Customer Churn Prediction reporting"}

    with plt.rc_context(chart_style):
        contract_order = ["Month-to-month", "One year", "Two year"]
        contract = (
            training.groupby("contract_type", observed=True)["churn"]
            .agg(["mean", "count"])
            .reindex(contract_order)
        )
        fig, axis = plt.subplots(figsize=(5.8, 3.65))
        bars = axis.bar(
            ["Month-to-month", "One year", "Two year"],
            contract["mean"],
            color=["#6C43C0", "#8464CC", "#A28ADE"],
            width=0.64,
        )
        axis.set_title("Observed Churn Rate by Contract Type", pad=10)
        axis.set_ylabel("Churn rate")
        axis.set_ylim(0, 1.04)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="x", visible=False)
        for bar, rate, count in zip(bars, contract["mean"], contract["count"]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 0.025,
                f"{rate:.1%}\nn={int(count)}",
                ha="center",
                va="bottom",
                fontsize=9.2,
                color="#252330",
            )
        fig.tight_layout()
        output = figure_dir / "report_contract_churn.png"
        fig.savefig(output, dpi=180, facecolor="white", metadata=png_metadata)
        plt.close(fig)
        outputs.append(output)

        usage_bins = pd.qcut(training["monthly_usage_gb"], 5, duplicates="drop")
        usage = training.groupby(usage_bins, observed=True)["churn"].agg(["mean", "count"])
        fig, axis = plt.subplots(figsize=(5.8, 3.65))
        labels = ["Q1\nLowest", "Q2", "Q3", "Q4", "Q5\nHighest"]
        bars = axis.bar(labels, usage["mean"], color="#3569D4", width=0.68)
        bars[0].set_color("#7898DF")
        bars[-1].set_color("#5B3AA8")
        axis.set_title("Observed Churn Rate by Usage Quintile", pad=10)
        axis.set_xlabel("Training-only monthly usage group")
        axis.set_ylabel("Churn rate")
        axis.set_ylim(0, 1.08)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="x", visible=False)
        for bar, rate in zip(bars, usage["mean"]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 0.025,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
                fontsize=9.2,
                color="#252330",
            )
        fig.tight_layout()
        output = figure_dir / "report_usage_quintiles.png"
        fig.savefig(output, dpi=180, facecolor="white", metadata=png_metadata)
        plt.close(fig)
        outputs.append(output)

        tuned = comparison.loc[comparison["stage"] == "Tuned"].set_index("model")
        model_order = [
            "Logistic Regression",
            "Random Forest",
            "Histogram Gradient Boosting",
        ]
        labels = ["Logistic Regression", "Random Forest", "Histogram GB"]
        y_positions = np.arange(len(model_order))[::-1]
        fig, axis = plt.subplots(figsize=(6.3, 3.65))
        for y, model, label in zip(y_positions, model_order, labels):
            ap = float(tuned.loc[model, "mean_pr_auc"])
            roc = float(tuned.loc[model, "mean_roc_auc"])
            axis.plot([roc, ap], [y, y], color="#D8D3E4", linewidth=2.2)
            axis.scatter(roc, y, color="#3569D4", s=58, label="ROC-AUC" if y == 2 else None)
            axis.scatter(ap, y, color="#6C43C0", s=58, label="Average Precision" if y == 2 else None)
            axis.text(ap + 0.004, y, f"{ap:.4f}", va="center", fontsize=9.2)
        axis.set_yticks(y_positions, labels)
        axis.set_xlim(0.84, 1.005)
        axis.set_xlabel("Mean five-fold cross-validation score")
        axis.set_title("Tuned Model Ranking Performance", pad=10)
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            frameon=False,
            fontsize=9,
        )
        axis.grid(axis="y", visible=False)
        fig.subplots_adjust(left=0.29, right=0.97, top=0.84, bottom=0.29)
        output = figure_dir / "report_model_cv_comparison.png"
        fig.savefig(output, dpi=180, facecolor="white", metadata=png_metadata)
        plt.close(fig)
        outputs.append(output)

    return tuple(outputs)


def load_markdown_sections(path: str | Path) -> dict[str, list[str]]:
    """Load narrative paragraphs grouped by level-two Markdown headings."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    paragraph_lines: list[str] = []

    def flush() -> None:
        nonlocal paragraph_lines
        if current and paragraph_lines:
            paragraph = " ".join(line.strip() for line in paragraph_lines).strip()
            if paragraph:
                sections[current].append(paragraph)
        paragraph_lines = []

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            flush()
            current = line[3:].strip()
            sections.setdefault(current, [])
        elif line.startswith("# "):
            flush()
            current = None
        elif current is None:
            continue
        elif not line.strip():
            flush()
        elif line.lstrip().startswith(("|", "- ", "1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. ")):
            flush()
        else:
            paragraph_lines.append(line)
    flush()
    return sections


def load_verified_results(root: str | Path) -> dict[str, Any]:
    """Load and reconcile authoritative analytical outputs for the report."""
    project = Path(root).resolve()
    paths = {
        "content": project / "reports" / "business_report_content.md",
        "cleaned": project / "data" / "processed" / "02_cleaned_dataset.csv",
        "split": project / "data" / "processed" / "02_split_indices.json",
        "metadata": project / "models" / "final_model_metadata.json",
        "model": project / "models" / "03_model_artifact.joblib",
        "model_comparison": project / "reports" / "modeling" / "phase5_model_comparison.csv",
        "evaluation": project / "reports" / "evaluation" / "phase6_final_test_metrics.json",
        "calibration": project / "reports" / "evaluation" / "phase6_calibration_comparison.csv",
        "importance": project / "reports" / "explainability" / "phase7_global_permutation_importance.csv",
        "customers": project / "reports" / "explainability" / "phase7_selected_customers.csv",
        "retention": project / "reports" / "retention_simulation_results.csv",
        "sensitivity": project / "reports" / "retention_sensitivity_analysis.csv",
        "phase7_integrity": project / "reports" / "explainability" / "phase7_integrity.json",
        "phase8_integrity": project / "reports" / "retention_simulation_integrity.json",
    }
    paths.update(
        {
            f"figure_{name}": project / "reports" / "figures" / name
            for name in REQUIRED_FIGURES
        }
    )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required report sources: {missing}")

    cleaned = pd.read_csv(paths["cleaned"])
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    evaluation = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
    phase7_integrity = json.loads(paths["phase7_integrity"].read_text(encoding="utf-8"))
    phase8_integrity = json.loads(paths["phase8_integrity"].read_text(encoding="utf-8"))
    model_comparison = pd.read_csv(paths["model_comparison"])
    calibration = pd.read_csv(paths["calibration"])
    importance = pd.read_csv(paths["importance"])
    customers = pd.read_csv(paths["customers"])
    retention_table = pd.read_csv(paths["retention"])
    sensitivity = pd.read_csv(paths["sensitivity"])
    retention = retention_table.set_index("metric")["value"].to_dict()

    if cleaned.shape != (1200, 20):
        raise ValueError(f"Unexpected cleaned-data shape: {cleaned.shape}")
    if len(split["train_indices"]) != 960 or len(split["test_indices"]) != 240:
        raise ValueError("Fixed split row counts do not match Phase 2.")
    if split["dataset_fingerprint"] != metadata["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint differs between split and metadata.")
    if file_sha256(paths["split"]) != metadata["split_artifact_sha256"]:
        raise ValueError("Split artifact hash does not match locked metadata.")
    if file_sha256(paths["model"]) != metadata["model_artifact_sha256"]:
        raise ValueError("Final model artifact hash does not match locked metadata.")
    if not phase7_integrity["model_artifact_unchanged"]:
        raise ValueError("Phase 7 did not preserve the final model artifact.")
    if not phase8_integrity["model_artifact_unchanged"]:
        raise ValueError("Phase 8 did not preserve the final model artifact.")
    if not phase8_integrity["predictions_unchanged"]:
        raise ValueError("Phase 8 prediction integrity check failed.")
    if int(retention["total_customers"]) != 240 or int(retention["selected_customers"]) != 48:
        raise ValueError("Retention population or capacity differs from the validated result.")
    if {"customer_id", "_row_index"} & set(customers.columns):
        raise ValueError("Public explanation table exposes an identifier.")
    if len(customers) < 3:
        raise ValueError("Three anonymized customer examples are required.")

    tuned = model_comparison.loc[model_comparison["stage"] == "Tuned"].copy()
    expected_models = {
        "Logistic Regression",
        "Random Forest",
        "Histogram Gradient Boosting",
    }
    if set(tuned["model"]) != expected_models:
        raise ValueError("Tuned model comparison is incomplete.")

    return {
        "root": project,
        "paths": paths,
        "content": load_markdown_sections(paths["content"]),
        "cleaned": cleaned,
        "split": split,
        "metadata": metadata,
        "evaluation": evaluation["selected_business_threshold"],
        "calibration": calibration,
        "model_comparison": tuned,
        "importance": importance,
        "customers": customers,
        "retention": retention,
        "retention_table": retention_table,
        "sensitivity": sensitivity,
        "phase8_integrity": phase8_integrity,
    }


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=24,
            textColor=PURPLE,
            alignment=TA_LEFT,
            spaceAfter=3 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=15,
            textColor=BLUE,
            spaceAfter=3.5 * mm,
        ),
        "page_title": ParagraphStyle(
            "PageTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=21,
            textColor=PURPLE,
            spaceAfter=3.5 * mm,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=BLUE,
            spaceBefore=2.5 * mm,
            spaceAfter=1.6 * mm,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12.3,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=2.4 * mm,
        ),
        "compact": ParagraphStyle(
            "CompactBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=11.2,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=1.8 * mm,
        ),
        "small": ParagraphStyle(
            "SmallBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=9.7,
            textColor=INK,
            spaceAfter=1.4 * mm,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.2,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=0.8 * mm,
            spaceAfter=1.8 * mm,
        ),
        "card_number": ParagraphStyle(
            "CardNumber",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=18,
            textColor=PURPLE,
            alignment=TA_CENTER,
        ),
        "card_label": ParagraphStyle(
            "CardLabel",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=9.8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "callout_title": ParagraphStyle(
            "CalloutTitle",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.2,
            leading=12.2,
            textColor=PURPLE,
            spaceAfter=1.2 * mm,
        ),
        "callout_body": ParagraphStyle(
            "CalloutBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.9,
            leading=11.3,
            textColor=INK,
        ),
    }


def _safe_text(text: Any) -> str:
    value = str(text)
    replacements = {
        "—": "-",
        "–": "-",
        "×": "x",
        "±": "+/-",
        "“": '"',
        "”": '"',
        "’": "'",
        "`": "",
        "≥": ">=",
        "≤": "<=",
    }
    for original, replacement in replacements.items():
        value = value.replace(original, replacement)
    return escape(value)


def _paragraphs(
    content: dict[str, list[str]],
    key: str,
    style: ParagraphStyle,
    *,
    limit: int | None = None,
) -> list[Paragraph]:
    values = content.get(key)
    if not values:
        raise KeyError(f"Missing report narrative section: {key}")
    if limit is not None:
        values = values[:limit]
    return [Paragraph(_safe_text(value), style) for value in values]


def _table(
    rows: list[list[Any]],
    col_widths: list[float],
    *,
    font_size: float = 7.2,
    right_align_from: int | None = None,
) -> Table:
    formatted: list[list[Any]] = []
    styles = _styles()
    body_style = ParagraphStyle(
        f"TableBody{font_size}",
        parent=styles["small"],
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 1.7,
        textColor=INK,
        spaceAfter=0,
    )
    header_style = ParagraphStyle(
        f"TableHeader{font_size}",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=WHITE,
    )
    for row_index, row in enumerate(rows):
        cell_style = header_style if row_index == 0 else body_style
        formatted.append(
            [Paragraph(_safe_text(value), cell_style) for value in row]
        )
    table = Table(formatted, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_index in range(1, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PURPLE_LIGHT))
    if right_align_from is not None:
        commands.append(("ALIGN", (right_align_from, 1), (-1, -1), "RIGHT"))
    table.setStyle(TableStyle(commands))
    return table


def _callout(
    title: str,
    text: str,
    styles: dict[str, ParagraphStyle],
    *,
    width: float = 504,
    background: colors.Color = BLUE_LIGHT,
) -> Table:
    """Return a compact, reusable business-insight callout."""
    box = Table(
        [
            [Paragraph(_safe_text(title), styles["callout_title"])],
            [Paragraph(_safe_text(text), styles["callout_body"])],
        ],
        colWidths=[width],
        hAlign="LEFT",
    )
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.7, BLUE),
                ("LINEBEFORE", (0, 0), (0, -1), 3, PURPLE),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, 0), 7),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return box


def _scaled_image(path: Path, width: float, max_height: float) -> Image:
    image_reader = ImageReader(str(path))
    source_width, source_height = image_reader.getSize()
    scale = min(width / source_width, max_height / source_height)
    return Image(str(path), width=source_width * scale, height=source_height * scale)


def _figure_pair(
    first_path: Path,
    first_caption: str,
    second_path: Path,
    second_caption: str,
    styles: dict[str, ParagraphStyle],
    *,
    width: float = 244,
    height: float = 155,
) -> KeepTogether:
    first = _scaled_image(first_path, width, height)
    second = _scaled_image(second_path, width, height)
    figures = Table([[first, second]], colWidths=[252, 252], hAlign="CENTER")
    figures.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    captions = Table(
        [[Paragraph(_safe_text(first_caption), styles["caption"]), Paragraph(_safe_text(second_caption), styles["caption"])]],
        colWidths=[252, 252],
        hAlign="CENTER",
    )
    captions.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return KeepTogether([figures, captions])


def _page_header(story: list[Any], title: str, styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph(_safe_text(title), styles["page_title"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=PURPLE, spaceAfter=3 * mm))


def _section_heading(story: list[Any], title: str, styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph(_safe_text(title), styles["heading"]))


def _draw_page_frame(pdf_canvas: canvas.Canvas, doc: SimpleDocTemplate) -> None:
    pdf_canvas.saveState()
    page = doc.page
    width, height = A4
    if page > 1:
        pdf_canvas.setFont("Helvetica", 7.2)
        pdf_canvas.setFillColor(MUTED)
        pdf_canvas.drawString(18 * mm, height - 10 * mm, REPORT_TITLE)
        pdf_canvas.setStrokeColor(LINE)
        pdf_canvas.line(18 * mm, height - 12 * mm, width - 18 * mm, height - 12 * mm)
    pdf_canvas.setStrokeColor(LINE)
    pdf_canvas.line(18 * mm, 12 * mm, width - 18 * mm, 12 * mm)
    pdf_canvas.setFont("Helvetica", 7.5)
    pdf_canvas.setFillColor(MUTED)
    pdf_canvas.drawString(18 * mm, 8 * mm, REPORT_TITLE)
    pdf_canvas.drawRightString(width - 18 * mm, 8 * mm, f"Page {page} of 7")
    pdf_canvas.restoreState()


def generate_business_report(
    root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Generate the canonical deterministic seven-page A4 business report."""
    project = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    output = Path(output_path).resolve() if output_path else project / DEFAULT_OUTPUT_NAME
    generate_report_figures(project)
    results = load_verified_results(project)
    content = results["content"]
    styles = _styles()
    paths = results["paths"]
    evaluation = results["evaluation"]
    retention = results["retention"]

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=17 * mm,
        title=REPORT_TITLE,
        author="Customer Churn Prediction Project",
        subject="Final business report",
        pageCompression=1,
    )
    story: list[Any] = []

    # Page 1: Title and executive summary.
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(REPORT_TITLE, styles["title"]))
    story.append(Paragraph(REPORT_SUBTITLE, styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2.2, color=PURPLE, spaceAfter=3.5 * mm))
    _section_heading(story, "Executive Summary", styles)
    story.extend(_paragraphs(content, "Executive Summary", styles["body"], limit=3))
    cards = [
        [
            Paragraph("1,200", styles["card_number"]),
            Paragraph("0.8994", styles["card_number"]),
            Paragraph("0.9819", styles["card_number"]),
            Paragraph("1.1594", styles["card_number"]),
        ],
        [
            Paragraph("customers", styles["card_label"]),
            Paragraph("held-out ROC-AUC", styles["card_label"]),
            Paragraph("held-out Average Precision", styles["card_label"]),
            Paragraph("lift at top 20%", styles["card_label"]),
        ],
    ]
    card_table = Table(cards, colWidths=[126, 126, 126, 126], rowHeights=[24, 23])
    card_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PURPLE_LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.6, PURPLE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, WHITE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(Spacer(1, 1.5 * mm))
    story.append(card_table)
    _section_heading(story, "Business Problem and Objectives", styles)
    story.extend(_paragraphs(content, "Business Problem and Objectives", styles["body"]))
    _section_heading(story, "End-to-End Analytical Workflow", styles)
    workflow_rows = [
        ["1  Prepare", "2  Develop", "3  Explain", "4  Prioritize"],
        [
            "Audit, clean, split, and prevent leakage",
            "Compare, tune, calibrate, and test once",
            "Assess global behavior and anonymized cases",
            "Rank risk, apply capacity, and test scenarios",
        ],
    ]
    story.append(_table(workflow_rows, [126] * 4, font_size=7.7))
    story.append(Spacer(1, 2.5 * mm))
    story.append(
        _callout(
            "Decision takeaway",
            content["Executive Summary"][3],
            styles,
        )
    )
    _section_heading(story, "Evidence Boundary", styles)
    boundary_rows = [
        ["Supported use", "Not established by this study"],
        [
            "Prioritize limited outreach using validated risk ranking and human review",
            "Causal churn reduction, realized profit, production readiness, or automated action",
        ],
    ]
    story.append(_table(boundary_rows, [252, 252], font_size=7.8))
    story.append(PageBreak())

    # Page 2: Dataset and preparation.
    _page_header(story, "Dataset and Data Preparation", styles)
    _section_heading(story, "Dataset Overview", styles)
    story.extend(_paragraphs(content, "Dataset Overview", styles["body"]))
    dataset_rows = [
        ["Measure", "Verified result", "Decision relevance"],
        ["Dataset", "1,200 rows x 20 columns", "18 predictors plus ID and target"],
        ["Churn distribution", "1,036 churn / 164 retained", "86.33% positive class"],
        ["Missingness", "138 cells across 132 rows", "Training-fold median imputation"],
        ["Duplicates", "0 rows; 0 repeated IDs", "No records removed"],
        ["Fixed split", "960 train / 240 test", "Stratified, random state 42"],
    ]
    story.append(_table(dataset_rows, [100, 140, 264], font_size=7.8))
    story.append(Spacer(1, 2.5 * mm))
    churn_image = _scaled_image(paths["figure_01_churn_distribution.png"], 390, 235)
    story.append(KeepTogether([churn_image, Paragraph("Figure 1. Training target distribution; the high churn prevalence makes accuracy alone misleading.", styles["caption"])]))
    _section_heading(story, "Preparation and Leakage Controls", styles)
    story.extend(_paragraphs(content, "Data Preparation and Leakage Prevention", styles["body"]))
    story.append(PageBreak())

    # Page 3: EDA and engineered features.
    _page_header(story, "Exploratory Analysis and Feature Engineering", styles)
    _section_heading(story, "Observed Churn Associations", styles)
    story.extend(_paragraphs(content, "Exploratory Churn Patterns", styles["body"]))
    story.append(
        _figure_pair(
            paths["figure_report_contract_churn.png"],
            "Figure 2. Training-only contract association; group counts are shown above each bar.",
            paths["figure_report_usage_quintiles.png"],
            "Figure 3. Training-only usage association; missing usage values are excluded from quintiles.",
            styles,
            width=244,
            height=162,
        )
    )
    _section_heading(story, "Engineered Features", styles)
    story.extend(_paragraphs(content, "Feature Engineering", styles["body"]))
    feature_rows = [
        ["Feature", "Purpose"],
        ["tenure_years", "Relationship duration in an interpretable scale"],
        ["complaints_per_active_month", "Complaint intensity relative to active exposure"],
        ["payment_risk_score", "Combined recent complaint and late-payment friction"],
        ["complaint_satisfaction_interaction", "Service issues occurring with lower satisfaction"],
        ["monthly_contract_tenure", "Contract-by-tenure interaction observed in EDA"],
        ["late_payment_charge_interaction", "Payment difficulty combined with current charge"],
        ["usage_per_charge", "Usage intensity relative to monthly charge"],
    ]
    story.append(_table(feature_rows, [176, 328], font_size=7.2))
    story.append(Paragraph("All associations and derived features are predictive representations, not causal treatment effects.", styles["caption"]))
    story.append(PageBreak())

    # Page 4: Development and final evaluation.
    _page_header(story, "Machine Learning Development and Evaluation", styles)
    _section_heading(story, "Training-Only Model Comparison", styles)
    story.extend(_paragraphs(content, "Model Development", styles["compact"], limit=3))
    comparison = results["model_comparison"].set_index("model")
    model_rows = [["Tuned model", "CV AP", "AP SD", "CV ROC-AUC", "Precision@20%"]]
    for model_name in ("Logistic Regression", "Random Forest", "Histogram Gradient Boosting"):
        row = comparison.loc[model_name]
        model_rows.append(
            [
                model_name,
                f"{row['mean_pr_auc']:.4f}",
                f"{row['std_pr_auc']:.4f}",
                f"{row['mean_roc_auc']:.4f}",
                f"{row['mean_precision_at_20']:.4f}",
            ]
        )
    story.append(_table(model_rows, [174, 75, 65, 90, 100], font_size=7.1, right_align_from=1))
    story.append(Spacer(1, 1.5 * mm))
    story.append(
        _figure_pair(
            paths["figure_report_model_cv_comparison.png"],
            "Figure 4. Tuned ranking scores are close; Random Forest has the highest mean AP.",
            paths["figure_phase6_confusion_matrix.png"],
            "Figure 5. Locked-threshold held-out outcomes: 98 TP, 32 TN, 1 FP, and 109 FN.",
            styles,
            width=242,
            height=148,
        )
    )
    _section_heading(story, "Locked Final Evaluation", styles)
    story.append(
        Paragraph(
            "Sigmoid calibration was selected using nested training-only folds. It reduced OOF Brier score "
            "from 0.1061 to 0.0840, log loss from 0.3328 to 0.2752, and ECE from 0.1181 to 0.0219 "
            "without material ranking loss. Held-out Brier score was 0.0825 and log loss was 0.2627; "
            "calibration is improved, not perfect.",
            styles["compact"],
        )
    )
    metric_rows = [
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Average Precision"],
        [
            f"{evaluation['accuracy']:.4f}",
            f"{evaluation['precision']:.4f}",
            f"{evaluation['recall']:.4f}",
            f"{evaluation['f1']:.4f}",
            f"{evaluation['roc_auc']:.4f}",
            f"{evaluation['average_precision']:.4f}",
        ],
    ]
    story.append(_table(metric_rows, [84] * 6, font_size=7.2))
    evaluation_notes = Table(
        [
            [
                Paragraph(
                    "<b>Why accuracy is low</b><br/>Accuracy 0.5417 is below the 0.8625 always-churn "
                    "baseline. The high cutoff produces precision 0.9899 but misses 109 churners, so "
                    "high precision does not imply high recall. ROC-AUC 0.8994 and AP 0.9819 instead "
                    "support useful cross-threshold ranking.",
                    styles["callout_body"],
                ),
                Paragraph(
                    "<b>Threshold is not capacity</b><br/>The locked threshold 0.970709 selected 99 of "
                    "240 customers (41.25%). A real 20% campaign must rank all scores and select exactly "
                    "48 using the declared tie-breaker. The threshold was not retuned after test access.",
                    styles["callout_body"],
                ),
            ]
        ],
        colWidths=[252, 252],
        hAlign="LEFT",
    )
    evaluation_notes.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), BLUE_LIGHT),
                ("BACKGROUND", (1, 0), (1, 0), PURPLE_LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, WHITE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(Spacer(1, 1.5 * mm))
    story.append(evaluation_notes)
    story.append(PageBreak())

    # Page 5: Explainability and anonymized customers.
    _page_header(story, "Model Explainability and Customer Insights", styles)
    _section_heading(story, "Global Model Behavior", styles)
    story.extend(_paragraphs(content, "Global Explainability", styles["compact"]))
    importance_image = _scaled_image(paths["figure_phase7_global_feature_importance.png"], 405, 275)
    story.append(KeepTogether([importance_image, Paragraph("Figure 6. Held-out original-feature permutation importance measured by Average Precision decrease.", styles["caption"])]))
    _section_heading(story, "Anonymized Local Explanations", styles)
    story.extend(_paragraphs(content, "Anonymized Customer Explanations", styles["compact"]))
    customers = results["customers"].set_index("customer_label")
    factor_text = {
        "Customer A": "Usage raised risk most; small charge increase; tenure/satisfaction slightly lowered risk.",
        "Customer B": "One-year contract and low usage lowered risk; age and late payments raised it.",
        "Customer C": "Usage raised risk; one-year contract and charge effects kept score just below cutoff.",
    }
    customer_rows = [["Case", "Probability", "Actual", "Predicted", "Result", "Main local factors"]]
    for label in ("Customer A", "Customer B", "Customer C"):
        row = customers.loc[label]
        customer_rows.append(
            [
                label,
                f"{row['predicted_probability']:.4f}",
                "Churn" if int(row["actual_churn"]) else "Retained",
                "Churn" if int(row["predicted_class"]) else "Retained",
                "Correct" if bool(row["correct_prediction"]) else "False negative",
                factor_text[label],
            ]
        )
    story.append(_table(customer_rows, [60, 62, 57, 62, 68, 195], font_size=7.0))
    story.append(Paragraph("Explanations describe fitted-model sensitivity and association; they do not establish causation.", styles["caption"]))
    story.append(PageBreak())

    # Page 6: Capacity-constrained targeting and scenario economics.
    _page_header(story, "Retention Simulation and Business Impact", styles)
    _section_heading(story, "Top-20% Capacity Policy", styles)
    story.extend(_paragraphs(content, "Top-20% Retention Simulation", styles["compact"], limit=1))
    policy_rows = [
        ["Locked classification rule", "Operational campaign rule"],
        [
            "Probability >= 0.970709; selected 99/240 (41.25%); retained for honest evaluation",
            "Rank probability descending; select exactly 48/240 (20%); stable original-order tie-break",
        ],
    ]
    story.append(_table(policy_rows, [252, 252], font_size=7.6))
    story.append(Spacer(1, 1.5 * mm))
    story.append(Paragraph(_safe_text(content["Top-20% Retention Simulation"][1]), styles["compact"]))
    ranking_rows = [
        ["Evaluated", "Selected", "Churners selected", "Precision@20%", "Recall@20%", "Lift", "Random capture"],
        [
            "240",
            "48 (20.00%)",
            "48",
            f"{retention['precision_at_20']:.4f}",
            f"{retention['recall_at_20']:.4f}",
            f"{retention['lift_at_20']:.4f}",
            f"{retention['random_expected_churners_captured']:.1f}",
        ],
    ]
    story.append(_table(ranking_rows, [60, 79, 78, 72, 70, 60, 85], font_size=7.0))
    story.append(Spacer(1, 2 * mm))
    gains_image = _scaled_image(paths["figure_phase8_cumulative_gains.png"], 390, 220)
    story.append(
        KeepTogether(
            [
                gains_image,
                Paragraph(
                    "Figure 7. Solid purple shows model-ranked capture; the dashed blue diagonal is random selection, and the vertical dotted line marks the 20% contact limit.",
                    styles["caption"],
                ),
            ]
        )
    )
    _section_heading(story, "Hypothetical Financial Scenario", styles)
    story.extend(_paragraphs(content, "Hypothetical Business Impact", styles["compact"]))
    finance_rows = [
        ["Assumptions", "Model-ranked estimate", "Random-selection estimate"],
        ["$5/contact; $150/retained; 10% success", "$240 cost; $720 gross", "$240 cost; $621 gross"],
        ["Expected net benefit", "$480; ROI 200.00%", "$381; ROI 158.75%"],
        ["Incremental model benefit", "$99 versus random", "-"],
    ]
    story.append(_table(finance_rows, [184, 160, 160], font_size=7.2))
    story.append(Paragraph("All financial values are illustrative scenario estimates, not realized company profit or measured causal uplift.", styles["caption"]))
    story.append(PageBreak())

    # Page 7: Recommendations, roadmap, limitations, conclusion.
    _page_header(story, "Recommendations, Limitations and Conclusion", styles)
    _section_heading(story, "Evidence-Linked Retention Actions", styles)
    story.extend(_paragraphs(content, "Evidence-Linked Retention Recommendations", styles["compact"]))
    recommendation_rows = [
        ["Observed support", "Proposed action", "Mechanism", "Main uncertainty"],
        ["Usage is the strongest signal", "Service-quality and plan-fit review", "Resolve reliability or mismatch", "Usage timing and non-causality"],
        ["Lower satisfaction and complaints", "Proactive support resolution", "Reduce service friction", "Agent cost; outreach can backfire"],
        ["Late-payment history", "Optional payment support", "Reduce involuntary loss", "Privacy, compliance, and fairness"],
        ["Month-to-month contract association", "Renewal assistance", "Offer suitable voluntary options", "Discount leakage and self-selection"],
    ]
    story.append(_table(recommendation_rows, [126, 128, 126, 124], font_size=7.0))
    _section_heading(story, "Implementation Roadmap", styles)
    roadmap_rows = [
        ["Prepare and score", "Pilot and learn"],
        ["1. Validate current, consented inputs", "5. Review eligibility, privacy, and fairness"],
        ["2. Generate probabilities with the saved model", "6. Assign approved interventions"],
        ["3. Rank customers by churn probability", "7. Run a controlled pilot and record outcomes"],
        ["4. Apply the strict top-20% contact limit", "8. Monitor drift, calibration, capture, cost, and subgroups"],
    ]
    story.append(_table(roadmap_rows, [252, 252], font_size=7.5))
    _section_heading(story, "Limitations", styles)
    limitation_paragraphs = content["Limitations"]
    limitation_table = Table(
        [[
            Paragraph(_safe_text(limitation_paragraphs[0]), styles["callout_body"]),
            Paragraph(_safe_text(limitation_paragraphs[1]), styles["callout_body"]),
        ]],
        colWidths=[252, 252],
        hAlign="LEFT",
    )
    limitation_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F5FB")),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, WHITE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(limitation_table)
    _section_heading(story, "Conclusion", styles)
    story.append(_callout("Recommended next decision", content["Conclusion"][0], styles))

    doc.build(
        story,
        onFirstPage=_draw_page_frame,
        onLaterPages=_draw_page_frame,
        canvasmaker=InvariantCanvas,
    )
    return output


def pdf_page_count(pdf_path: str | Path) -> int:
    """Return page count using PyMuPDF."""
    with pymupdf.open(pdf_path) as document:
        return document.page_count


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Extract normalized text from every PDF page."""
    with pymupdf.open(pdf_path) as document:
        return "\n".join(page.get_text("text") for page in document)


def render_pdf_pages(
    pdf_path: str | Path,
    output_directory: str | Path,
    *,
    zoom: float = 2.0,
) -> list[Path]:
    """Render every PDF page to PNG for visual review."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    with pymupdf.open(pdf_path) as document:
        matrix = pymupdf.Matrix(zoom, zoom)
        for page_number, page in enumerate(document, start=1):
            output = destination / f"page_{page_number:02d}.png"
            page.get_pixmap(matrix=matrix, alpha=False).save(output)
            rendered.append(output)
    return rendered


def create_contact_sheet(
    pdf_path: str | Path,
    output_path: str | Path,
    *,
    scale: float = 0.42,
    columns: int = 2,
    gap: int = 18,
) -> Path:
    """Render every report page into one compact visual-review sheet."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_path) as document:
        page_pixmaps = [
            page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            for page in document
        ]
    if not page_pixmaps:
        raise ValueError("Cannot create a contact sheet for an empty PDF.")
    cell_width = max(pixmap.width for pixmap in page_pixmaps)
    cell_height = max(pixmap.height for pixmap in page_pixmaps)
    rows = int(np.ceil(len(page_pixmaps) / columns))
    sheet_width = columns * cell_width + (columns + 1) * gap
    sheet_height = rows * cell_height + (rows + 1) * gap
    sheet = PILImage.new("RGB", (sheet_width, sheet_height), color=(245, 245, 247))
    for index, pixmap in enumerate(page_pixmaps):
        row, column = divmod(index, columns)
        x = gap + column * (cell_width + gap)
        y = gap + row * (cell_height + gap)
        page_image = PILImage.frombytes(
            "RGB",
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )
        sheet.paste(page_image, (x, y))
    sheet.save(destination, format="PNG", optimize=True)
    return destination


def validate_generated_report(pdf_path: str | Path, root: str | Path) -> dict[str, Any]:
    """Run programmatic content, privacy, and integrity checks on the PDF."""
    results = load_verified_results(root)
    path = Path(pdf_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError("Generated business report is missing or empty.")
    with pymupdf.open(path) as document:
        page_texts = [page.get_text("text").strip() for page in document]
        page_rects = [page.rect for page in document]
    page_count = len(page_texts)
    if not 5 <= page_count <= 8:
        raise AssertionError(f"Report must contain 5-8 pages; found {page_count}.")
    if any(len(text) < 250 for text in page_texts):
        raise AssertionError("At least one PDF page is unexpectedly blank or sparse.")
    if any(rect.width >= rect.height for rect in page_rects):
        raise AssertionError("Every report page must use portrait orientation.")
    full_text = "\n".join(page_texts)
    for section in REQUIRED_SECTION_TITLES:
        if section not in full_text:
            raise AssertionError(f"Missing required report section: {section}")
    required_values = (
        "1,200",
        "86.33%",
        "0.9782",
        "0.8994",
        "0.9819",
        "0.9899",
        "0.4734",
        "0.6405",
        "0.970709",
        "1.0000",
        "0.2319",
        "1.1594",
        "41.4",
        "$480",
        "$381",
        "$99",
    )
    missing_values = [value for value in required_values if value not in full_text]
    if missing_values:
        raise AssertionError(f"PDF is missing verified report values: {missing_values}")
    if re.search(r"\b(?:TBD|TODO|PLACEHOLDER)\b", full_text, flags=re.IGNORECASE):
        raise AssertionError("PDF contains an unresolved placeholder.")
    if re.search(r"\bCUST\d{6}\b", full_text, flags=re.IGNORECASE):
        raise AssertionError("PDF exposes a raw customer identifier.")
    required_interpretation = (
        "0.8625",
        "41.25%",
        "99 of 240",
        "select exactly 48",
        "Customer A",
        "Customer B",
        "Customer C",
        "hypothetical",
    )
    missing_interpretation = [
        phrase for phrase in required_interpretation if phrase not in full_text
    ]
    if missing_interpretation:
        raise AssertionError(
            f"PDF is missing required interpretation content: {missing_interpretation}"
        )
    if "Phase 9 |" in full_text:
        raise AssertionError("PDF contains an internal development-phase footer.")
    if file_sha256(results["paths"]["model"]) != results["metadata"]["model_artifact_sha256"]:
        raise AssertionError("Final model artifact changed during report generation.")
    return {
        "pdf_exists": True,
        "page_count": page_count,
        "all_pages_nonblank": True,
        "all_pages_portrait": True,
        "required_sections_present": True,
        "required_metrics_present": True,
        "placeholders_absent": True,
        "customer_identifiers_absent": True,
        "model_artifact_unchanged": True,
        "text_characters": len(full_text),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    generated = generate_business_report(project_root)
    preview_directory = project_root / "reports" / "business_report_preview"
    rendered_pages = render_pdf_pages(generated, preview_directory)
    contact_sheet = create_contact_sheet(
        generated,
        preview_directory / "contact_sheet.png",
    )
    print(f"Generated {generated}")
    print(f"Rendered {len(rendered_pages)} pages and {contact_sheet}")
    print(validate_generated_report(generated, project_root))

