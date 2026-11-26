"""
Job Recommendation and Resume-Job Compatibility Engine for ResumeMatch AI.
Combines TF-IDF semantic similarity, skill overlap analysis, and transparent explainability.
Features:
- Content-based filtering using TF-IDF and Cosine Similarity
- Skill set intersection and difference (Skill Gap Analysis)
- Transparent compatibility scoring (clearly separating heuristic compatibility from ML predictions)
- Human-understandable recommendation explanations
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.feature_engineering.vectorizer import TextVectorizer, compute_cosine_similarity
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.skill_extractor import extract_skills_from_text

JOBS_PATH = BASE_DIR / "data" / "processed" / "cleaned_jobs.csv"
MODELS_DIR = BASE_DIR / "models"

class RecommendationEngine:
    def __init__(self, jobs_df: Optional[pd.DataFrame] = None):
        if jobs_df is not None:
            self.jobs_df = jobs_df.copy()
        elif JOBS_PATH.exists():
            self.jobs_df = pd.read_csv(JOBS_PATH)
        else:
            self.jobs_df = pd.DataFrame()

        self.vectorizer = TextVectorizer(max_features=2500, min_df=1)
        self._init_vectorizer()

    def _init_vectorizer(self):
        """Precomputes TF-IDF matrix for all available job descriptions."""
        if not self.jobs_df.empty and "cleaned_description" in self.jobs_df.columns:
            descriptions = self.jobs_df["cleaned_description"].fillna("").tolist()
            self.job_tfidf_matrix = self.vectorizer.fit_transform(descriptions)
        else:
            self.job_tfidf_matrix = None

    def calculate_compatibility(self, resume_data: Dict[str, Any], job_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates transparent compatibility between candidate resume and a specific job posting.
        Score composition:
        - 60% Skill Match Ratio (Exact extracted technical & soft skill overlap)
        - 40% TF-IDF Text Cosine Similarity (Domain and contextual alignment)
        """
        candidate_skills = set(resume_data.get("skills", {}).get("all_skills", []))
        
        # Parse job required skills
        raw_req_skills = job_record.get("required_skills", "")
        if isinstance(raw_req_skills, str):
            job_skills = set([s.strip().lower() for s in raw_req_skills.split(",") if s.strip()])
        elif isinstance(raw_req_skills, (list, set)):
            job_skills = set([s.lower() for s in raw_req_skills])
        else:
            job_skills = set()

        # Skill overlap
        matching_skills = sorted(list(candidate_skills.intersection(job_skills)))
        missing_skills = sorted(list(job_skills.difference(candidate_skills)))

        # Skill match percentage
        if job_skills:
            skill_score = len(matching_skills) / len(job_skills)
        else:
            skill_score = 0.5  # Neutral default if no specific skills recorded

        # Text Cosine Similarity
        cleaned_resume = resume_data.get("cleaned_text", "")
        cleaned_job = job_record.get("cleaned_description", "") or clean_text(str(job_record.get("raw_description", "")))

        if cleaned_resume and cleaned_job and self.vectorizer.is_fitted:
            resume_vec = self.vectorizer.transform([cleaned_resume])
            job_vec = self.vectorizer.transform([cleaned_job])
            text_sim = compute_cosine_similarity(resume_vec, job_vec)
        else:
            text_sim = 0.0

        # Overall transparent compatibility score (0 to 100%)
        # Explicit note: This is a mathematically weighted transparent index, not a black-box ML model.
        overall_score = round((0.60 * skill_score + 0.40 * text_sim) * 100, 1)
        overall_score = max(0.0, min(100.0, overall_score))

        # Generate Explainability Rationale
        explanation_points = []
        if matching_skills:
            explanation_points.append(f"Matched {len(matching_skills)} required skills: {', '.join(matching_skills[:5])}" + ("..." if len(matching_skills) > 5 else ""))
        if text_sim > 0.3:
            explanation_points.append(f"Strong contextual terminology alignment (Cosine similarity: {text_sim:.2f})")
        elif text_sim > 0.15:
            explanation_points.append(f"Moderate contextual alignment with role requirements ({text_sim:.2f})")
        else:
            explanation_points.append("Low general vocabulary overlap with job description")

        if missing_skills:
            explanation_points.append(f"Skill gap identified in {len(missing_skills)} areas: {', '.join(missing_skills[:4])}")

        return {
            "overall_score": overall_score,
            "skill_match_ratio": round(skill_score * 100, 1),
            "text_cosine_similarity": round(text_sim * 100, 1),
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "job_title": job_record.get("job_title", "Unknown Position"),
            "company": job_record.get("company", "Company"),
            "explanation": " | ".join(explanation_points),
            "is_ml_prediction": False
        }

    def recommend_jobs(self, resume_data: Dict[str, Any], top_n: int = 5, min_score: float = 15.0) -> List[Dict[str, Any]]:
        """
        Ranks all jobs in the catalog against candidate resume and returns top recommendations with full explanations.
        """
        if self.jobs_df.empty:
            return []

        recommendations = []
        for _, row in self.jobs_df.iterrows():
            compat = self.calculate_compatibility(resume_data, row.to_dict())
            if compat["overall_score"] >= min_score:
                compat["job_id"] = row.get("job_id", 0)
                compat["location"] = row.get("location", "Remote / Hybrid")
                compat["raw_description"] = row.get("raw_description", "")
                compat["source"] = row.get("source", "Live Job")
                compat["job_url"] = row.get("job_url", "")
                recommendations.append(compat)

        # Sort descending by overall compatibility score
        recommendations.sort(key=lambda x: x["overall_score"], reverse=True)
        return recommendations[:top_n]

    def aggregate_skill_gaps(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates missing skills across all top recommended roles to identify the highest ROI skills to learn.
        """
        gap_frequency = {}
        for rec in recommendations:
            for skill in rec["missing_skills"]:
                gap_frequency[skill] = gap_frequency.get(skill, 0) + 1

        # Sort by frequency
        sorted_gaps = sorted(gap_frequency.items(), key=lambda item: item[1], reverse=True)
        return {
            "high_priority_skills": [s for s, count in sorted_gaps if count >= 2],
            "all_missing_skills": sorted_gaps,
            "total_gap_count": len(sorted_gaps)
        }
