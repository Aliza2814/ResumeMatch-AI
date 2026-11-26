"""
Unit tests for recommendation engine, compatibility scoring, and skill gaps.
"""

from src.recommendation.engine import RecommendationEngine

def test_recommendation_and_compatibility():
    engine = RecommendationEngine()
    assert not engine.jobs_df.empty, "Jobs catalog should not be empty"

    mock_resume = {
        "cleaned_text": "software engineer with python, react, docker, and postgresql skills",
        "skills": {
            "all_skills": ["python", "react", "docker", "postgresql", "git"],
            "technical_skills": ["python", "react", "docker", "postgresql", "git"],
            "soft_skills": ["teamwork"]
        }
    }

    recs = engine.recommend_jobs(mock_resume, top_n=3)
    assert len(recs) > 0
    top = recs[0]
    assert "overall_score" in top
    assert 0 <= top["overall_score"] <= 100
    assert "matching_skills" in top
    assert "explanation" in top
    assert top["is_ml_prediction"] is False  # Transparent distinction

def test_skill_gap_aggregation():
    engine = RecommendationEngine()
    mock_recs = [
        {
            "job_title": "ML Engineer",
            "missing_skills": ["pytorch", "kubernetes", "ci/cd"]
        },
        {
            "job_title": "Cloud Architect",
            "missing_skills": ["kubernetes", "aws", "terraform"]
        }
    ]
    gaps = engine.aggregate_skill_gaps(mock_recs)
    assert "kubernetes" in gaps["high_priority_skills"]
    assert gaps["total_gap_count"] == 5
