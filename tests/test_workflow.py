"""
End-to-End User Workflow Test:
Upload resume -> Parse resume -> Analyse skills -> Predict job roles -> Recommend jobs -> Show skill gaps.
"""

from pathlib import Path
import joblib
from src.resume_parser.parser import parse_resume
from src.feature_engineering.vectorizer import TextVectorizer
from src.recommendation.engine import RecommendationEngine

BASE_DIR = Path(__file__).resolve().parents[1]

def test_end_to_end_workflow(sample_pdf_path):
    # Step 1: Upload and Parse Resume
    parsed = parse_resume(sample_pdf_path)
    assert parsed["success"] is True
    assert len(parsed["cleaned_text"]) > 20

    # Step 2: Skill Analysis
    skills = parsed["skills"]
    assert len(skills["all_skills"]) >= 3
    assert "python" in skills["technical_skills"]

    # Step 3: Predict Job Category
    classifier = joblib.load(BASE_DIR / "models" / "best_classifier.joblib")
    vectorizer = TextVectorizer.load(BASE_DIR / "models" / "tfidf_vectorizer.joblib")
    label_encoder = joblib.load(BASE_DIR / "models" / "label_encoder.joblib")

    vec = vectorizer.transform([parsed["cleaned_text"]])
    pred = classifier.predict(vec)[0]
    category = label_encoder.inverse_transform([pred])[0]
    assert isinstance(category, str)
    assert len(category) > 0

    # Step 4: Recommend Jobs
    engine = RecommendationEngine()
    recs = engine.recommend_jobs(parsed, top_n=5)
    assert len(recs) > 0
    assert recs[0]["overall_score"] > 0

    # Step 5: Aggregate Skill Gaps
    gaps = engine.aggregate_skill_gaps(recs)
    assert "all_missing_skills" in gaps
