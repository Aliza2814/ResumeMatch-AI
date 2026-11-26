"""
Unit tests for model loading and job role prediction.
"""

from pathlib import Path
import joblib
from src.feature_engineering.vectorizer import TextVectorizer

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

def test_models_exist_and_load():
    classifier_path = MODELS_DIR / "best_classifier.joblib"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    encoder_path = MODELS_DIR / "label_encoder.joblib"

    assert classifier_path.exists(), "Trained best_classifier.joblib must exist"
    assert vectorizer_path.exists(), "Fitted tfidf_vectorizer.joblib must exist"
    assert encoder_path.exists(), "Fitted label_encoder.joblib must exist"

    classifier = joblib.load(classifier_path)
    vectorizer = TextVectorizer.load(vectorizer_path)
    label_encoder = joblib.load(encoder_path)

    assert hasattr(classifier, "predict")
    assert vectorizer.is_fitted is True
    assert len(label_encoder.classes_) == 25

def test_predict_data_science_resume():
    classifier = joblib.load(MODELS_DIR / "best_classifier.joblib")
    vectorizer = TextVectorizer.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    sample_ds = "Data scientist experienced in Python, Machine Learning, Deep Learning, Pandas, Scikit-learn, Neural Networks."
    vec = vectorizer.transform([sample_ds])
    pred = classifier.predict(vec)[0]
    cat = label_encoder.inverse_transform([pred])[0]
    assert cat == "Data Science"
