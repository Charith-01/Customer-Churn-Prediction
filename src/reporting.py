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
DEFAULT_OUTPUT_NAME = "04_business_report.pdf"

REQUIRED_FIGURES = (
    "01_churn_distribution.png",
    "02_contract_churn.png",
    "08_usage_churn.png",
    "m5_02_pr_auc_comparison.png",
    "phase6_confusion_matrix.png",
    "phase7_global_feature_importance.png",
    "phase8_cumulative_gains.png",
    "phase8_sensitivity_analysis.png",
)

REQUIRED_SECTION_TITLES = (
    "Executive Summary",
    "Dataset and Data Preparation",
    "Exploratory Analysis and Feature Engineering",
    "Machine Learning Development and Evaluation",
    "Explainable AI and Customer Insights",
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
            fontSize=23,
            leading=27,
            textColor=PURPLE,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12.5,
            leading=16,
            textColor=BLUE,
            spaceAfter=5 * mm,
        ),
        "page_title": ParagraphStyle(
            "PageTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=PURPLE,
            spaceAfter=3.5 * mm,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=BLUE,
            spaceBefore=2.5 * mm,
            spaceAfter=1.3 * mm,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.6,
            leading=11.2,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=2.2 * mm,
        ),
        "small": ParagraphStyle(
            "SmallBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=9.2,
            textColor=INK,
            spaceAfter=1.4 * mm,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=6.9,
            leading=8.2,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=0.8 * mm,
            spaceAfter=1.8 * mm,
        ),
        "card_number": ParagraphStyle(
            "CardNumber",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=17,
            textColor=PURPLE,
            alignment=TA_CENTER,
        ),
        "card_label": ParagraphStyle(
            "CardLabel",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9,
            textColor=MUTED,
            alignment=TA_CENTER,
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
    for row_index, row in enumerate(rows):
        cell_style = styles["small"]
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
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in range(1, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PURPLE_LIGHT))
    if right_align_from is not None:
        commands.append(("ALIGN", (right_align_from, 1), (-1, -1), "RIGHT"))
    table.setStyle(TableStyle(commands))
    return table


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
        pdf_canvas.setFont("Helvetica", 6.8)
        pdf_canvas.setFillColor(MUTED)
        pdf_canvas.drawString(18 * mm, height - 10 * mm, REPORT_TITLE)
        pdf_canvas.setStrokeColor(LINE)
        pdf_canvas.line(18 * mm, height - 12 * mm, width - 18 * mm, height - 12 * mm)
    pdf_canvas.setStrokeColor(LINE)
    pdf_canvas.line(18 * mm, 12 * mm, width - 18 * mm, 12 * mm)
    pdf_canvas.setFont("Helvetica", 7)
    pdf_canvas.setFillColor(MUTED)
    pdf_canvas.drawString(18 * mm, 8 * mm, "Phase 9 | Verified Phase 1-8 evidence")
    pdf_canvas.drawRightString(width - 18 * mm, 8 * mm, f"Page {page} of 7")
    pdf_canvas.restoreState()


def generate_business_report(
    root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Generate the canonical deterministic seven-page A4 business report."""
    project = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    output = Path(output_path).resolve() if output_path else project / DEFAULT_OUTPUT_NAME
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
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=16 * mm,
        title=REPORT_TITLE,
        author="Customer Churn Prediction Project",
        subject="Phase 9 final business report",
        pageCompression=1,
    )
    story: list[Any] = []

    # Page 1: Title and executive summary.
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(REPORT_TITLE, styles["title"]))
    story.append(Paragraph(REPORT_SUBTITLE, styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2.2, color=PURPLE, spaceAfter=5 * mm))
    _section_heading(story, "Executive Summary", styles)
    story.extend(_paragraphs(content, "Executive Summary", styles["body"]))
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
    story.append(Spacer(1, 2 * mm))
    story.append(card_table)
    _section_heading(story, "Business Problem and Objectives", styles)
    story.extend(_paragraphs(content, "Business Problem and Objectives", styles["body"]))
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
    story.append(_table(dataset_rows, [100, 140, 264], font_size=7.2))
    story.append(Spacer(1, 2.5 * mm))
    churn_image = _scaled_image(paths["figure_01_churn_distribution.png"], 335, 205)
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
            paths["figure_02_contract_churn.png"],
            "Figure 2. Month-to-month contracts show the highest observed training churn rate.",
            paths["figure_08_usage_churn.png"],
            "Figure 3. Churn rises sharply across usage groups and then saturates.",
            styles,
            width=244,
            height=145,
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
    story.append(_table(feature_rows, [176, 328], font_size=6.9))
    story.append(Paragraph("All associations and derived features are predictive representations, not causal treatment effects.", styles["caption"]))
    story.append(PageBreak())

    # Page 4: Development and final evaluation.
    _page_header(story, "Machine Learning Development and Evaluation", styles)
    _section_heading(story, "Training-Only Model Comparison", styles)
    story.extend(_paragraphs(content, "Model Development", styles["body"]))
    comparison = results["model_comparison"].set_index("model")
    model_rows = [["Tuned model", "CV AP", "CV ROC-AUC", "Precision@20%"]]
    for model_name in ("Logistic Regression", "Random Forest", "Histogram Gradient Boosting"):
        row = comparison.loc[model_name]
        model_rows.append(
            [
                model_name,
                f"{row['mean_pr_auc']:.4f}",
                f"{row['mean_roc_auc']:.4f}",
                f"{row['mean_precision_at_20']:.4f}",
            ]
        )
    story.append(_table(model_rows, [205, 100, 100, 99], font_size=7.2, right_align_from=1))
    _section_heading(story, "Locked Final Evaluation", styles)
    story.extend(_paragraphs(content, "Final Evaluation and Calibration", styles["body"]))
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
    story.append(_table(metric_rows, [84] * 6, font_size=6.9))
    story.append(Spacer(1, 2 * mm))
    story.append(
        _figure_pair(
            paths["figure_m5_02_pr_auc_comparison.png"],
            "Figure 4. Cross-validation Average Precision used for training-only model comparison.",
            paths["figure_phase6_confusion_matrix.png"],
            "Figure 5. Locked-threshold held-out confusion matrix: 98 TP, 32 TN, 1 FP, 109 FN.",
            styles,
            width=236,
            height=148,
        )
    )
    story.append(PageBreak())

    # Page 5: Explainability and anonymized customers.
    _page_header(story, "Explainable AI and Customer Insights", styles)
    _section_heading(story, "Global Model Behavior", styles)
    story.extend(_paragraphs(content, "Global Explainability", styles["body"]))
    importance_image = _scaled_image(paths["figure_phase7_global_feature_importance.png"], 360, 250)
    story.append(KeepTogether([importance_image, Paragraph("Figure 6. Held-out original-feature permutation importance measured by Average Precision decrease.", styles["caption"])]))
    _section_heading(story, "Anonymized Local Explanations", styles)
    story.extend(_paragraphs(content, "Anonymized Customer Explanations", styles["body"]))
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
    story.append(_table(customer_rows, [60, 62, 57, 62, 68, 195], font_size=6.5))
    story.append(Paragraph("Explanations describe fitted-model sensitivity and association; they do not establish causation.", styles["caption"]))
    story.append(PageBreak())

    # Page 6: Capacity-constrained targeting and scenario economics.
    _page_header(story, "Retention Simulation and Business Impact", styles)
    _section_heading(story, "Top-20% Capacity Policy", styles)
    story.extend(_paragraphs(content, "Top-20% Retention Simulation", styles["body"]))
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
    story.append(_table(ranking_rows, [60, 79, 78, 72, 70, 60, 85], font_size=6.4))
    story.append(Spacer(1, 2 * mm))
    story.append(
        _figure_pair(
            paths["figure_phase8_cumulative_gains.png"],
            "Figure 7. Ranked churner capture stays above the equal-budget random reference.",
            paths["figure_phase8_sensitivity_analysis.png"],
            "Figure 8. Hypothetical net benefit varies materially with success rate and retained value.",
            styles,
            width=238,
            height=174,
        )
    )
    _section_heading(story, "Hypothetical Financial Scenario", styles)
    story.extend(_paragraphs(content, "Hypothetical Business Impact", styles["body"]))
    finance_rows = [
        ["Assumptions", "Model-ranked estimate", "Random-selection estimate"],
        ["$5/contact; $150/retained; 10% success", "$240 cost; $720 gross", "$240 cost; $621 gross"],
        ["Expected net benefit", "$480; ROI 200.00%", "$381; ROI 158.75%"],
        ["Incremental model benefit", "$99 versus random", "-"],
    ]
    story.append(_table(finance_rows, [184, 160, 160], font_size=6.7))
    story.append(Paragraph("All financial values are illustrative scenario estimates, not realized company profit or measured causal uplift.", styles["caption"]))
    story.append(PageBreak())

    # Page 7: Recommendations, roadmap, limitations, conclusion.
    _page_header(story, "Recommendations, Limitations and Conclusion", styles)
    _section_heading(story, "Evidence-Linked Retention Actions", styles)
    story.extend(_paragraphs(content, "Evidence-Linked Retention Recommendations", styles["body"]))
    recommendation_rows = [
        ["Observed support", "Proposed action", "Mechanism", "Main uncertainty"],
        ["Usage is the strongest signal", "Service-quality and plan-fit review", "Resolve reliability or mismatch", "Usage timing and non-causality"],
        ["Lower satisfaction and complaints", "Proactive support resolution", "Reduce service friction", "Agent cost; outreach can backfire"],
        ["Late-payment history", "Optional payment support", "Reduce involuntary loss", "Privacy, compliance, and fairness"],
        ["Month-to-month contract association", "Renewal assistance", "Offer suitable voluntary options", "Discount leakage and self-selection"],
    ]
    story.append(_table(recommendation_rows, [126, 128, 126, 124], font_size=6.3))
    _section_heading(story, "Implementation Roadmap", styles)
    story.extend(_paragraphs(content, "Implementation Roadmap", styles["body"]))
    _section_heading(story, "Limitations", styles)
    story.extend(_paragraphs(content, "Limitations", styles["body"]))
    _section_heading(story, "Conclusion", styles)
    story.extend(_paragraphs(content, "Conclusion", styles["body"]))

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
    zoom: float = 1.5,
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


def validate_generated_report(pdf_path: str | Path, root: str | Path) -> dict[str, Any]:
    """Run programmatic content, privacy, and integrity checks on the PDF."""
    results = load_verified_results(root)
    path = Path(pdf_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError("Generated business report is missing or empty.")
    with pymupdf.open(path) as document:
        page_texts = [page.get_text("text").strip() for page in document]
    page_count = len(page_texts)
    if not 5 <= page_count <= 8:
        raise AssertionError(f"Report must contain 5-8 pages; found {page_count}.")
    if any(len(text) < 250 for text in page_texts):
        raise AssertionError("At least one PDF page is unexpectedly blank or sparse.")
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
    if file_sha256(results["paths"]["model"]) != results["metadata"]["model_artifact_sha256"]:
        raise AssertionError("Final model artifact changed during report generation.")
    return {
        "pdf_exists": True,
        "page_count": page_count,
        "all_pages_nonblank": True,
        "required_sections_present": True,
        "required_metrics_present": True,
        "placeholders_absent": True,
        "customer_identifiers_absent": True,
        "model_artifact_unchanged": True,
        "text_characters": len(full_text),
    }


if __name__ == "__main__":
    generated = generate_business_report()
    print(f"Generated {generated}")
    print(validate_generated_report(generated, generated.parent))

