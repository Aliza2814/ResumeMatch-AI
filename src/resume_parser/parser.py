"""
Resume document parser module for ResumeMatch AI.
Extracts raw text, contact information, education, skills, and experience from PDF, DOCX, and TXT files.
Gracefully handles corrupted, empty, or unreadable files without crashing.
"""

import io
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
import fitz  # PyMuPDF
import docx

from src.preprocessing.text_cleaner import clean_text, extract_contact_info
from src.preprocessing.skill_extractor import extract_skills_from_text, extract_experience_years

def extract_text_from_pdf(file_bytes_or_path: Union[bytes, str, Path, io.BytesIO]) -> str:
    """
    Extracts text content from a PDF document using PyMuPDF.
    """
    text_chunks = []
    try:
        if isinstance(file_bytes_or_path, (bytes, bytearray)):
            doc = fitz.open(stream=file_bytes_or_path, filetype="pdf")
        elif isinstance(file_bytes_or_path, io.BytesIO):
            doc = fitz.open(stream=file_bytes_or_path.getvalue(), filetype="pdf")
        else:
            doc = fitz.open(file_bytes_or_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text:
                text_chunks.append(text)
        doc.close()
    except Exception as e:
        raise ValueError(f"Failed to parse PDF document: {e}")

    return "\n".join(text_chunks).strip()

def extract_text_from_docx(file_bytes_or_path: Union[bytes, str, Path, io.BytesIO]) -> str:
    """
    Extracts text content from a DOCX document using python-docx.
    """
    text_chunks = []
    try:
        if isinstance(file_bytes_or_path, (bytes, bytearray)):
            stream = io.BytesIO(file_bytes_or_path)
            doc = docx.Document(stream)
        elif isinstance(file_bytes_or_path, io.BytesIO):
            doc = docx.Document(file_bytes_or_path)
        else:
            doc = docx.Document(file_bytes_or_path)

        for para in doc.paragraphs:
            if para.text:
                text_chunks.append(para.text)

        # Also extract text inside tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text:
                        text_chunks.append(cell.text)
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX document: {e}")

    return "\n".join(text_chunks).strip()

def extract_candidate_name(raw_text: str) -> Optional[str]:
    """
    Attempts to identify the candidate's name from the top lines of the resume.
    Avoids false positives like 'Resume', 'Curriculum Vitae', email headers, etc.
    """
    if not raw_text:
        return None

    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    ignore_words = {
        "resume", "curriculum vitae", "cv", "profile", "contact", "email",
        "phone", "address", "education", "experience", "skills", "summary"
    }

    for line in lines[:8]:
        # Clean line
        candidate = re.sub(r"[^a-zA-Z\s.]", "", line).strip()
        words = candidate.split()
        if 2 <= len(words) <= 4:
            if not any(w.lower() in ignore_words for w in words):
                # Verify that words look like names (mostly title-cased or uppercase)
                if all(w[0].isupper() for w in words if w):
                    return " ".join(words)
    return None

def extract_sections(raw_text: str) -> Dict[str, str]:
    """
    Segments resume text into common sections: Experience, Education, Projects, Skills, Certifications.
    """
    sections = {
        "experience": "",
        "education": "",
        "projects": "",
        "skills": "",
        "certifications": "",
        "summary": ""
    }
    if not raw_text:
        return sections

    section_headers = {
        "experience": r"(?:work\s+experience|professional\s+experience|employment\s+history|experience)",
        "education": r"(?:education|academic\s+background|qualifications)",
        "projects": r"(?:projects|personal\s+projects|academic\s+projects)",
        "skills": r"(?:skills|technical\s+skills|core\s+competencies|technologies)",
        "certifications": r"(?:certifications|certificates|licenses)",
        "summary": r"(?:professional\s+summary|summary|profile|about\s+me)"
    }

    # Build regex to detect header lines
    header_pattern = r"(?im)^\s*(" + "|".join(section_headers.values()) + r")\s*[:\-]?\s*$"
    splits = re.split(header_pattern, raw_text)

    if len(splits) > 1:
        current_section = "summary"
        for i in range(1, len(splits), 2):
            header = splits[i].strip().lower()
            content = splits[i+1].strip() if i+1 < len(splits) else ""

            for sec_name, sec_regex in section_headers.items():
                if re.match(r"^" + sec_regex + r"$", header, re.IGNORECASE):
                    current_section = sec_name
                    break
            sections[current_section] = (sections[current_section] + "\n" + content).strip()
    else:
        sections["summary"] = raw_text

    return sections

def parse_resume(file_input: Union[bytes, str, Path, io.BytesIO], filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Master parse function that handles PDF, DOCX, and TXT files.
    Returns structured dictionary with extracted information.
    """
    raw_text = ""
    # Determine format
    ext = ""
    if filename:
        ext = Path(filename).suffix.lower()
    elif isinstance(file_input, (str, Path)):
        ext = Path(file_input).suffix.lower()

    try:
        if ext == ".pdf":
            raw_text = extract_text_from_pdf(file_input)
        elif ext in [".docx", ".doc"]:
            raw_text = extract_text_from_docx(file_input)
        elif ext == ".txt":
            if isinstance(file_input, (bytes, bytearray)):
                raw_text = file_input.decode("utf-8", errors="ignore")
            elif isinstance(file_input, io.BytesIO):
                raw_text = file_input.getvalue().decode("utf-8", errors="ignore")
            else:
                raw_text = Path(file_input).read_text(encoding="utf-8", errors="ignore")
        else:
            # Try PDF first, then DOCX, then TXT
            try:
                raw_text = extract_text_from_pdf(file_input)
            except Exception:
                try:
                    raw_text = extract_text_from_docx(file_input)
                except Exception:
                    if isinstance(file_input, (bytes, bytearray)):
                        raw_text = file_input.decode("utf-8", errors="ignore")
                    elif isinstance(file_input, io.BytesIO):
                        raw_text = file_input.getvalue().decode("utf-8", errors="ignore")
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "raw_text": "",
            "cleaned_text": "",
            "candidate_name": None,
            "contact_info": {},
            "skills": {},
            "experience_years": 0.0,
            "sections": {}
        }

    if not raw_text.strip():
        return {
            "success": False,
            "error": "The uploaded file is empty or contains no readable text.",
            "raw_text": "",
            "cleaned_text": "",
            "candidate_name": None,
            "contact_info": {},
            "skills": {},
            "experience_years": 0.0,
            "sections": {}
        }

    # Extract all components
    cleaned_text = clean_text(raw_text)
    candidate_name = extract_candidate_name(raw_text)
    contact_info = extract_contact_info(raw_text)
    skills = extract_skills_from_text(raw_text)
    experience_years = extract_experience_years(raw_text)
    sections = extract_sections(raw_text)

    return {
        "success": True,
        "error": None,
        "filename": filename or "resume",
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "candidate_name": candidate_name,
        "contact_info": contact_info,
        "skills": skills,
        "experience_years": experience_years,
        "sections": sections
    }
