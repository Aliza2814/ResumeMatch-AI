"""
SQLAlchemy ORM models for ResumeMatch AI database.
Defines Resumes, ExtractedSkills, Jobs, JobSkills, AnalysisResults, and Recommendations.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class ResumeRecord(Base):
    """
    Stores candidate resume metadata, raw text, and extracted profile info.
    Includes option for candidate data deletion (privacy compliance).
    """
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    candidate_name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True, index=True)
    phone = Column(String(64), nullable=True)
    experience_years = Column(Float, default=0.0)
    raw_text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    skills = relationship("ExtractedSkill", back_populates="resume", cascade="all, delete-orphan")
    analysis_results = relationship("AnalysisResult", back_populates="resume", cascade="all, delete-orphan")
    recommendations = relationship("RecommendationRecord", back_populates="resume", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ResumeRecord(id={self.id}, name='{self.candidate_name}', email='{self.email}')>"

class ExtractedSkill(Base):
    """
    Stores individual skills parsed from candidate resumes with categorization.
    """
    __tablename__ = "extracted_skills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    resume_id = Column(Integer, ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False, index=True)
    skill_name = Column(String(128), nullable=False, index=True)
    skill_category = Column(String(64), default="technical")  # technical, soft, certification, education
    created_at = Column(DateTime, default=datetime.utcnow)

    resume = relationship("ResumeRecord", back_populates="skills")

    def __repr__(self):
        return f"<ExtractedSkill(id={self.id}, skill='{self.skill_name}', category='{self.skill_category}')>"

class JobRecord(Base):
    """
    Job catalog holding real postings and occupational requirements.
    """
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_title = Column(String(255), nullable=False, index=True)
    company = Column(String(255), nullable=False)
    location = Column(String(255), default="Remote / Hybrid")
    raw_description = Column(Text, nullable=False)
    cleaned_description = Column(Text, nullable=False)
    required_skills = Column(Text, nullable=True)  # Comma-separated list
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    skills = relationship("JobSkill", back_populates="job", cascade="all, delete-orphan")
    recommendations = relationship("RecommendationRecord", back_populates="job", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<JobRecord(id={self.id}, title='{self.job_title}', company='{self.company}')>"

class JobSkill(Base):
    """
    Normalized skills required per job listing.
    """
    __tablename__ = "job_skills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    skill_name = Column(String(128), nullable=False, index=True)
    is_mandatory = Column(Boolean, default=True)

    job = relationship("JobRecord", back_populates="skills")

class AnalysisResult(Base):
    """
    Stores classification results, predicted role categories, and confidence levels.
    """
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    resume_id = Column(Integer, ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False, index=True)
    predicted_category = Column(String(128), nullable=False)
    model_name = Column(String(128), nullable=False)
    confidence_score = Column(Float, default=0.0)
    total_skills_count = Column(Integer, default=0)
    top_strengths = Column(Text, nullable=True)
    identified_gaps = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    resume = relationship("ResumeRecord", back_populates="analysis_results")

class RecommendationRecord(Base):
    """
    Stores recommendation matches, compatibility scores, and explanations.
    """
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    resume_id = Column(Integer, ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    compatibility_score = Column(Float, nullable=False)
    skill_match_ratio = Column(Float, nullable=False)
    text_cosine_sim = Column(Float, nullable=False)
    matching_skills = Column(Text, nullable=True)
    missing_skills = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    resume = relationship("ResumeRecord", back_populates="recommendations")
    job = relationship("JobRecord", back_populates="recommendations")
