"""
Unit tests for text cleaner and contact info extraction.
"""

from src.preprocessing.text_cleaner import clean_text, extract_contact_info

def test_clean_text_basic():
    raw = "Senior PYTHON Developer! Check: https://example.com/cv and email me at test@dev.com"
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "test@dev.com" not in cleaned
    assert "python" in cleaned
    assert "developer" in cleaned

def test_clean_text_preserves_programming_tokens():
    raw = "Skills include C++, C#, and .NET frameworks."
    cleaned = clean_text(raw)
    assert "c++" in cleaned
    assert "c#" in cleaned
    assert ".net" in cleaned

def test_clean_text_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""

def test_extract_contact_info():
    text = "Candidate Name | Email: candidate@company.io | Phone: +1-415-555-0199 | Portfolio: https://github.com/candidate"
    info = extract_contact_info(text)
    assert info["email"] == "candidate@company.io"
    assert info["phone"] is not None
    assert len(info["links"]) >= 1
