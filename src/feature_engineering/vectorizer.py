"""
Feature extraction and vectorization module for ResumeMatch AI.
Provides TF-IDF vectorization, feature selection, and similarity metrics.
"""

import joblib
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TextVectorizer:
    """
    TF-IDF Vectorizer wrapper with sublinear term-frequency scaling and feature selection.
    """
    def __init__(self, max_features: int = 3000, ngram_range: Tuple[int, int] = (1, 2), min_df: int = 2):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
            stop_words="english"
        )
        self.feature_selector: Optional[SelectKBest] = None
        self.is_fitted = False

    def fit_transform(self, texts: List[str], y: Optional[List] = None, select_k: Optional[int] = None):
        """
        Fits TF-IDF vectorizer and optionally applies Chi-squared feature selection.
        """
        X_tfidf = self.vectorizer.fit_transform(texts)
        self.is_fitted = True

        if select_k and y is not None:
            k = min(select_k, X_tfidf.shape[1])
            self.feature_selector = SelectKBest(chi2, k=k)
            X_selected = self.feature_selector.fit_transform(X_tfidf, y)
            return X_selected

        return X_tfidf

    def transform(self, texts: List[str]):
        """
        Transforms new text documents using the fitted vectorizer.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transforming.")
        X_tfidf = self.vectorizer.transform(texts)
        if self.feature_selector:
            return self.feature_selector.transform(X_tfidf)
        return X_tfidf

    def get_feature_names(self) -> List[str]:
        """
        Returns feature terms.
        """
        if not self.is_fitted:
            return []
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        if self.feature_selector:
            return list(feature_names[self.feature_selector.get_support()])
        return list(feature_names)

    def save(self, filepath: Path):
        """
        Saves fitted vectorizer and selector to disk.
        """
        joblib.dump({"vectorizer": self.vectorizer, "selector": self.feature_selector, "is_fitted": self.is_fitted}, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "TextVectorizer":
        """
        Loads fitted vectorizer from disk. Handles both dict payloads and serialized TextVectorizer instances.
        """
        data = joblib.load(filepath)
        if isinstance(data, cls):
            return data
        if hasattr(data, "transform"):
            # Plain vectorizer
            obj = cls()
            obj.vectorizer = data
            obj.is_fitted = True
            return obj
        obj = cls()
        obj.vectorizer = data["vectorizer"]
        obj.feature_selector = data.get("selector")
        obj.is_fitted = data.get("is_fitted", True)
        return obj

def compute_cosine_similarity(vec1, vec2) -> float:
    """
    Computes cosine similarity between two feature vectors.
    """
    sim = cosine_similarity(vec1, vec2)
    return float(sim[0][0])
