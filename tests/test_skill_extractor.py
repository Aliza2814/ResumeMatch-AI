"""
Unit tests for skill extraction, education, and experience parsing.
"""

from src.preprocessing.skill_extractor import extract_skills_from_text, extract_experience_years

def test_extract_skills_technical_and_soft():
    text = """
    Software engineer skilled in Python, Docker, PostgreSQL, React, and Machine Learning.
    Demonstrated strong communication, teamwork, and problem solving skills.
    Holds a B.Tech in Information Technology and AWS Certified Solutions Architect.
    """
    res = extract_skills_from_text(text)
    
    # Check technical
    tech = res["technical_skills"]
    assert "python" in tech
    assert "docker" in tech
    assert "postgresql" in tech
    assert "react" in tech
    assert "machine learning" in tech

    # Check soft
    soft = res["soft_skills"]
    assert "communication" in soft
    assert "teamwork" in soft
    assert "problem solving" in soft

    # Check education & certifications
    assert "Bachelor of Technology" in res["education"]
    assert "aws certified" in res["certifications"]

def test_extract_experience_years():
    text1 = "Senior Engineer with 5+ years of experience in data analytics."
    assert extract_experience_years(text1) == 5.0

    text2 = "Over 3.5 yrs experience building cloud infrastructure."
    assert extract_experience_years(text2) == 3.5

    text3 = "Fresh graduate with no professional history."
    assert extract_experience_years(text3) == 0.0
