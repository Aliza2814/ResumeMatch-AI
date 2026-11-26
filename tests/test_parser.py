"""
Unit tests for PDF, DOCX, and error handling in parser.
"""

from src.resume_parser.parser import parse_resume

def test_parse_valid_pdf(sample_pdf_path):
    result = parse_resume(sample_pdf_path)
    assert result["success"] is True
    assert "johnathan.doe@example.com" in result["raw_text"]
    assert "python" in result["skills"]["technical_skills"]
    assert result["experience_years"] == 5.0

def test_parse_valid_docx(sample_docx_path):
    result = parse_resume(sample_docx_path)
    assert result["success"] is True
    assert "react" in result["skills"]["technical_skills"]
    assert "node.js" in result["skills"]["technical_skills"]
    assert result["experience_years"] == 4.0

def test_parse_empty_file(empty_file_path):
    result = parse_resume(empty_file_path)
    assert result["success"] is False
    assert "empty" in result["error"].lower()

def test_parse_corrupt_bytes():
    result = parse_resume(b"Invalid fake corrupted byte content", filename="corrupt.pdf")
    assert result["success"] is False
    assert result["error"] is not None
