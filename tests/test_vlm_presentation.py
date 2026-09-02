"""
Unit tests for VLM NLP Presentation Layer in SatQuery AI.

Verifies:
1. Dense VLM response is transformed into clean, readable structured sections:
   - Short headings (### Analysis, ### Key Observations, ### Interpretation, ### Confidence)
   - Bullet points for multiple observations
   - Separate sections with proper line breaks
   - Bold important findings
2. Factual integrity: numbers, features, percentages are strictly preserved.
3. No hallucinated or invented information.
4. Clean presentation across VLM VQA, captioning, change detection, and optical+SAR fusion.
"""
import pytest
from ai.synthesis.formatter import format_vlm_presentation


def test_dense_vlm_raw_output_structured():
    raw_vlm = (
        "This satellite image shows a landscape dominated by dense urban sprawl in the lower and upper sections, "
        "separated by a winding river near the top and a large, dark central area of forest or dense vegetation "
        "surrounded by agricultural fields and reddish-brown terrain."
    )
    formatted = format_vlm_presentation(raw_vlm, query="What is visible in this satellite scene?")

    # 1. Short headings
    assert "### Analysis" in formatted
    assert "### Key Observations" in formatted
    assert "### Interpretation" in formatted
    assert "### Confidence" in formatted

    # 2. Bullet points for multiple observations
    assert "- " in formatted
    bullet_lines = [l for l in formatted.splitlines() if l.strip().startswith("- ")]
    assert len(bullet_lines) >= 3

    # 3. Bold important findings
    assert "**" in formatted
    assert any("**Urban" in b or "**Hydrology" in b or "**Forestry" in b or "**Agriculture" in b for b in bullet_lines)

    # 4. Factual observations preserved without invention
    low = formatted.lower()
    assert "urban sprawl" in low
    assert "river" in low
    assert "forest" in low
    assert "agricultural" in low or "fields" in low

    # 5. Short paragraphs and proper line breaks
    sections = [s for s in formatted.split("### ") if s.strip()]
    assert len(sections) == 4


def test_spectral_distribution_vlm_output():
    raw_spectral = (
        "Scene analysis of 'opt_0611.png': dominant cover is Arable land (40.4% of scene), "
        "Broad-leaved forest (14.0% of scene), Industrial or commercial units (11.6% of scene), "
        "Inland waters (7.6% of scene)."
    )
    formatted = format_vlm_presentation(raw_spectral, query="What land cover is present?")

    assert "### Analysis" in formatted
    assert "### Key Observations" in formatted
    assert "### Interpretation" in formatted
    assert "### Confidence" in formatted

    bullet_lines = [l for l in formatted.splitlines() if l.strip().startswith("- ")]
    assert len(bullet_lines) == 5
    assert any("**Arable land**" in b and "40.4%" in b for b in bullet_lines)
    assert any("**Broad-leaved forest**" in b and "14.0%" in b for b in bullet_lines)
    assert any("**Inland waters**" in b and "7.6%" in b for b in bullet_lines)


def test_change_detection_structured_output():
    raw_change = (
        "Bi-temporal change detection confirms surface modifications between 2020 and 2024. "
        "Quantitatively, the change mask encompasses approximately 14.25 hectares distributed across 4 distinct cluster regions. "
        "Severity level is confirmed high."
    )
    formatted = format_vlm_presentation(raw_change, query="What changed?")

    assert "### Analysis" in formatted
    assert "### Key Observations" in formatted
    assert "### Interpretation" in formatted
    assert "### Confidence" in formatted

    assert "14.25 hectares" in formatted
    assert "4 distinct cluster regions" in formatted
    bullet_lines = [l for l in formatted.splitlines() if l.strip().startswith("- ")]
    assert len(bullet_lines) >= 2


def test_cross_modal_fusion_structured_output():
    raw_fusion = (
        "Optical and SAR joint analysis provides robust surface classification. "
        "Water bodies comprise approximately 14.9% of the scene with low radar backscatter. "
        "Built-up structures account for 13.5% of scene confirmed by double-bounce reflection. "
        "Vegetation cover spans 44.1% of scene footprint."
    )
    formatted = format_vlm_presentation(raw_fusion, query="Analyze optical and SAR data.")

    assert "### Analysis" in formatted
    assert "### Key Observations" in formatted
    assert "### Interpretation" in formatted
    assert "### Confidence" in formatted

    bullet_lines = [l for l in formatted.splitlines() if l.strip().startswith("- ")]
    assert len(bullet_lines) >= 3
    assert "14.9%" in formatted
    assert "13.5%" in formatted
    assert "44.1%" in formatted


def test_already_structured_text_normalized():
    text = (
        "### Analysis\n"
        "Direct observation of the scene.\n\n"
        "### Key Observations\n"
        "- **Bridges**: Two major crossings.\n"
        "- **Vegetation**: Dense canopy.\n\n"
        "### Interpretation\n"
        "Transportation corridor.\n\n"
        "### Confidence\n"
        "- **Status**: Verified"
    )
    formatted = format_vlm_presentation(text)
    assert formatted.strip() == text.strip()
