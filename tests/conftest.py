"""
Pytest configuration and test fixtures for ResumeMatch AI.
Generates genuine test PDF, DOCX, and text files.
"""

import sys
from pathlib import Path
import pytest
import fitz
import docx

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

@pytest.fixture(scope="session")
def sample_pdf_path():
    pdf_path = FIXTURES_DIR / "sample_resume.pdf"
    if not pdf_path.exists():
        doc = fitz.open()
        page = doc.new_page()
        text = """
        JOHNATHAN DOE
        Email: johnathan.doe@example.com | Phone: +1-555-0199
        Senior Python Engineer with 5 years experience in Machine Learning, Docker, SQL, and AWS.
        Education: B.Tech in Computer Science.
        Skills: Python, Scikit-learn, PyTorch, Docker, Git, PostgreSQL, Problem Solving.
        """
        page.insert_text((50, 50), text)
        doc.save(str(pdf_path))
        doc.close()
    return pdf_path

@pytest.fixture(scope="session")
def sample_docx_path():
    docx_path = FIXTURES_DIR / "sample_resume.docx"
    if not docx_path.exists():
        doc = docx.Document()
        doc.add_heading("JANE SMITH", level=1)
        doc.add_paragraph("Email: jane.smith@example.org | Phone: +1-555-0188")
        doc.add_paragraph("Full Stack Developer with 4 years experience in React, Node.js, TypeScript, and MongoDB.")
        doc.add_paragraph("Education: M.Tech in Software Engineering.")
        doc.add_paragraph("Skills: React, Node.js, JavaScript, TypeScript, MongoDB, Git, Communication.")
        doc.save(str(docx_path))
    return docx_path

@pytest.fixture(scope="session")
def empty_file_path():
    empty_path = FIXTURES_DIR / "empty.txt"
    empty_path.write_text("")
    return empty_path
