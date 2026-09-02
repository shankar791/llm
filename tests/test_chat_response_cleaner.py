"""
Unit tests for chat response cleaning, anti-hallucination, and reasoning-stripper logic.
Verifies requirement 7:
1. Normal answer remains intact.
2. "<think>" reasoning is removed.
3. Plain-text "Here's a thinking process" reasoning is removed.
4. Numbered reasoning blocks are removed.
5. Legitimate Markdown answer is preserved (e.g. ### Analysis, bullet points).
6. Response is not accidentally truncated.
7. Empty/malformed provider response is handled safely.
"""
from __future__ import annotations

import pytest
from backend.chat import clean_chat_response


# ==============================================================================
# TEST 1: Normal answer remains intact
# ==============================================================================
def test_normal_answer_remains_intact():
    text = "Nice to meet you, Shankar! 👋\nHow can I help you with satellite imagery, remote sensing, or GIS?"
    cleaned = clean_chat_response(text)
    assert cleaned == text


# ==============================================================================
# TEST 2: <think> reasoning is removed
# ==============================================================================
def test_think_tag_reasoning_removed():
    text = (
        "<think>\n"
        "User says: 'my name is shankar'.\n"
        "Plan: greet the user warmly and introduce capabilities.\n"
        "</think>\n"
        "Nice to meet you, Shankar! 👋\n"
        "How can I help you with satellite imagery, remote sensing, or GIS?"
    )
    cleaned = clean_chat_response(text)
    assert "<think>" not in cleaned
    assert "User says:" not in cleaned
    assert "Nice to meet you, Shankar! 👋" in cleaned
    assert "How can I help you with satellite imagery, remote sensing, or GIS?" in cleaned


# ==============================================================================
# TEST 3: Plain-text "Here's a thinking process:" reasoning is removed
# ==============================================================================
def test_plaintext_thinking_process_removed():
    text = (
        "Here's a thinking process:\n\n"
        "1. Analyze User Input:\n"
        "- The user states their name: 'shankar'\n"
        "- Intent: casual introduction / greeting\n\n"
        "2. Identify Role/Persona:\n"
        "- SatQuery AI assistant\n\n"
        "3. Determine Appropriate Response:\n"
        "- Acknowledge the user's name\n"
        "- Offer assistance in Earth observation\n\n"
        "4. Draft Response:\n"
        "Nice to meet you, Shankar! 👋\n"
        "How can I help you with satellite imagery, remote sensing, or GIS today?"
    )
    cleaned = clean_chat_response(text)
    assert "thinking process" not in cleaned.lower()
    assert "Analyze User Input" not in cleaned
    assert "Identify Role/Persona" not in cleaned
    assert "Draft Response" not in cleaned
    assert "Nice to meet you, Shankar! 👋" in cleaned
    assert "How can I help you with satellite imagery, remote sensing, or GIS today?" in cleaned


# ==============================================================================
# TEST 4: Numbered reasoning blocks are removed
# ==============================================================================
def test_numbered_reasoning_blocks_removed():
    text = (
        "1. Analyze User Input: User query is 'What is Sentinel-2?'.\n"
        "2. Retrieve Knowledge: ESA Copernicus constellation, 10m-60m resolution.\n"
        "3. Final Answer:\n"
        "**Sentinel-2** is an Earth observation constellation developed by ESA providing multispectral optical imagery."
    )
    cleaned = clean_chat_response(text)
    assert "Analyze User Input" not in cleaned
    assert "Retrieve Knowledge" not in cleaned
    assert "Final Answer" not in cleaned
    assert "Sentinel-2" in cleaned
    assert "Earth observation constellation developed by ESA" in cleaned


# ==============================================================================
# TEST 5: Legitimate Markdown answer is preserved
# ==============================================================================
def test_legitimate_markdown_answer_preserved():
    text = (
        "### Analysis\n"
        "Satellite scene analysis reveals a multi-temporal surface transformation.\n\n"
        "### Key Observations\n"
        "- **Vegetation Area**: Pasture covers 50.18% of the surface.\n"
        "- **Forest Cover**: Broad-leaved forest covers 35.30%.\n\n"
        "### Interpretation\n"
        "Spatial continuity reflects preserved vegetative buffers adjacent to urban fringes."
    )
    cleaned = clean_chat_response(text)
    # The word "Analysis" in legitimate heading must NOT be deleted
    assert "### Analysis" in cleaned
    assert "### Key Observations" in cleaned
    assert "### Interpretation" in cleaned
    assert "50.18%" in cleaned
    assert "35.30%" in cleaned


# ==============================================================================
# TEST 6: Response is not accidentally truncated
# ==============================================================================
def test_response_not_accidentally_truncated():
    long_answer = (
        "Hello Shankar! 👋 Welcome to SatQuery AI.\n\n"
        "Here are three core capabilities you can explore:\n"
        "1. **Surface Change Detection**: Compare multi-temporal optical or SAR image pairs to quantify changes in hectares.\n"
        "2. **Spatial Feature Grounding**: Locate, isolate, and generate bounding boxes for maritime vessels, storage tanks, or water bodies.\n"
        "3. **Cross-Modal Fusion**: Jointly analyze optical and synthetic aperture radar (SAR) imagery for all-weather monitoring."
    )
    cleaned = clean_chat_response(long_answer)
    assert cleaned == long_answer
    assert "Cross-Modal Fusion" in cleaned


# ==============================================================================
# TEST 7: Empty / malformed provider response is handled safely
# ==============================================================================
def test_empty_or_malformed_response_handled_safely():
    assert clean_chat_response(None) == ""
    assert clean_chat_response("") == ""
    assert clean_chat_response("   ") == ""
    # JSON envelope
    assert clean_chat_response('{"answer": "Hello World"}') == "Hello World"
    # Unclosed think tag
    assert "Hello!" in clean_chat_response("<think>incomplete thought cut off...")
