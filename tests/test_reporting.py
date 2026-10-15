"""Validation tests for the Phase 9 business-report artifact."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pytest

from src.reporting import (
    REQUIRED_FIGURES,
    REQUIRED_SECTION_TITLES,
    extract_pdf_text,
    file_sha256,
    generate_business_report,
    generate_report_figures,
    load_verified_results,
    pdf_page_count,
    validate_generated_report,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPORT = ROOT / "reports" / "04_business_report.pdf"


@pytest.fixture(scope="module")
def generated_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate one isolated report used by the content tests."""
    destination = tmp_path_factory.mktemp("business-report") / "report.pdf"
    return generate_business_report(ROOT, destination)


def test_report_generation_succeeds(generated_report: Path) -> None:
    assert generated_report.is_file()
    assert generated_report.stat().st_size > 0
    validation = validate_generated_report(generated_report, ROOT)
    assert validation["pdf_exists"] is True


def test_canonical_report_exists() -> None:
    assert CANONICAL_REPORT.is_file()
    assert CANONICAL_REPORT.stat().st_size > 0


def test_report_has_five_to_eight_nonblank_pages(generated_report: Path) -> None:
    count = pdf_page_count(generated_report)
    assert 5 <= count <= 8
    with pymupdf.open(generated_report) as document:
        page_text = [page.get_text("text").strip() for page in document]
    assert len(page_text) == count
    assert all(len(text) >= 250 for text in page_text)


def test_required_sections_are_present(generated_report: Path) -> None:
    report_text = extract_pdf_text(generated_report)
    for section in REQUIRED_SECTION_TITLES:
        assert section in report_text


def test_report_metrics_match_verified_sources(generated_report: Path) -> None:
    sources = load_verified_results(ROOT)
    evaluation = sources["evaluation"]
    retention = sources["retention"]
    report_text = extract_pdf_text(generated_report)

    expected_values = {
        f'{len(sources["cleaned"]):,}',
        f'{sources["cleaned"]["churn"].mean():.2%}',
        f'{sources["model_comparison"]["mean_pr_auc"].max():.4f}',
        f'{evaluation["roc_auc"]:.4f}',
        f'{evaluation["average_precision"]:.4f}',
        f'{evaluation["precision"]:.4f}',
        f'{evaluation["recall"]:.4f}',
        f'{evaluation["f1"]:.4f}',
        f'{evaluation["threshold"]:.6f}',
        f'{float(retention["precision_at_20"]):.4f}',
        f'{float(retention["recall_at_20"]):.4f}',
        f'{float(retention["lift_at_20"]):.4f}',
        f'{float(retention["random_expected_churners_captured"]):.1f}',
        f'${float(retention["model_expected_net_benefit"]):,.0f}',
        f'${float(retention["random_expected_net_benefit"]):,.0f}',
        f'${float(retention["incremental_expected_net_benefit"]):,.0f}',
    }
    for value in expected_values:
        assert value in report_text


def test_all_referenced_images_exist() -> None:
    for filename in REQUIRED_FIGURES:
        figure = ROOT / "reports" / "figures" / filename
        assert figure.is_file(), filename
        assert figure.stat().st_size > 0


def test_pdf_text_extraction_works(generated_report: Path) -> None:
    report_text = extract_pdf_text(generated_report)
    assert len(report_text) > 10_000
    assert "Customer Churn Prediction and Retention Analysis" in report_text


def test_report_contains_no_unresolved_placeholders(generated_report: Path) -> None:
    report_text = extract_pdf_text(generated_report)
    assert re.search(r"\b(?:TBD|TODO|PLACEHOLDER)\b", report_text, re.IGNORECASE) is None


def test_report_contains_no_raw_customer_identifiers(generated_report: Path) -> None:
    report_text = extract_pdf_text(generated_report)
    assert re.search(r"\bCUST\d{6}\b", report_text, re.IGNORECASE) is None


def test_required_business_interpretation_is_explicit(generated_report: Path) -> None:
    report_text = extract_pdf_text(generated_report)
    normalized_text = " ".join(report_text.split())
    required_phrases = {
        "0.8625 always-churn baseline",
        "selected 99 of 240 customers (41.25%)",
        "select exactly 48",
        "Customer A",
        "Customer B",
        "Customer C",
        "hypothetical",
    }
    for phrase in required_phrases:
        assert phrase in normalized_text
    assert "Phase 9 |" not in report_text


def test_rendered_pages_and_contact_sheet_exist() -> None:
    preview_directory = ROOT / "reports" / "business_report_preview"
    page_previews = sorted(preview_directory.glob("page_*.png"))
    assert len(page_previews) == 7
    assert all(path.stat().st_size > 0 for path in page_previews)
    contact_sheet = preview_directory / "contact_sheet.png"
    assert contact_sheet.is_file()
    assert contact_sheet.stat().st_size > 0


def test_generation_does_not_modify_model_artifact(tmp_path: Path) -> None:
    sources = load_verified_results(ROOT)
    model_path = sources["paths"]["model"]
    before = file_sha256(model_path)
    generate_business_report(ROOT, tmp_path / "integrity-check.pdf")
    after = file_sha256(model_path)
    assert before == after == sources["metadata"]["model_artifact_sha256"]


def test_generation_does_not_modify_evaluation_results(tmp_path: Path) -> None:
    sources = load_verified_results(ROOT)
    evaluation_path = sources["paths"]["evaluation"]
    before = file_sha256(evaluation_path)
    generate_business_report(ROOT, tmp_path / "evaluation-integrity-check.pdf")
    assert file_sha256(evaluation_path) == before


def test_generation_does_not_modify_original_or_cleaned_data(tmp_path: Path) -> None:
    raw_path = ROOT / "data" / "raw" / "customer_churn_advanced_dataset.csv"
    cleaned_path = ROOT / "data" / "processed" / "02_cleaned_dataset.csv"
    before = (file_sha256(raw_path), file_sha256(cleaned_path))
    generate_business_report(ROOT, tmp_path / "data-integrity-check.pdf")
    assert (file_sha256(raw_path), file_sha256(cleaned_path)) == before


def test_report_specific_figures_are_reproducible() -> None:
    figure_paths = generate_report_figures(ROOT)
    before = {path.name: file_sha256(path) for path in figure_paths}
    regenerated = generate_report_figures(ROOT)
    after = {path.name: file_sha256(path) for path in regenerated}
    assert after == before


def test_report_generation_dependencies_are_documented() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for dependency in ("reportlab", "pymupdf", "pillow"):
        assert dependency in requirements


def test_report_generation_is_byte_reproducible(tmp_path: Path) -> None:
    first = generate_business_report(ROOT, tmp_path / "first.pdf")
    second = generate_business_report(ROOT, tmp_path / "second.pdf")
    assert file_sha256(first) == file_sha256(second)
    assert first.read_bytes() == second.read_bytes()
