"""
Classification pipeline and evaluation for ResumeMatch AI (Units 6 & 7).
Trains 6 classification algorithms on resume dataset:
1. Logistic Regression
2. K-Nearest Neighbours (KNN)
3. Decision Tree
4. Random Forest
5. Naive Bayes (MultinomialNB)
6. Support Vector Machine (LinearSVC)

Computes Accuracy, Precision, Recall, F1-Score (macro & weighted), Confusion Matrix, and 5-Fold Cross-Validation.
Selects and saves the best model for the production application.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from src.feature_engineering.vectorizer import TextVectorizer

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_resumes.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate_classifiers():
    print("=== Running Units 6 & 7: Supervised Classification & Model Evaluation ===")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    texts = df["Cleaned_Resume"].tolist()
    labels = df["Category"].tolist()

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    classes = list(label_encoder.classes_)
    print(f"  Dataset contains {len(texts)} samples across {len(classes)} job categories.")

    # Train-test split (80-20 stratified to preserve category distribution and prevent data leakage)
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorization (TF-IDF with n-grams (1, 2) and sublinear scaling)
    print("  Fitting TF-IDF Vectorizer...")
    vectorizer = TextVectorizer(max_features=2500, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)
    print(f"  Vocabulary size: {X_train.shape[1]} features.")

    # Define the 6 required algorithms
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "Decision Tree": DecisionTreeClassifier(criterion="gini", max_depth=25, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Support Vector Machine": LinearSVC(C=1.0, random_state=42, max_iter=2000)
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_fitted_model = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"  Training {name}...")
        # Fit model
        model.fit(X_train, y_train)

        # Training vs Test performance (to diagnose Overfitting vs Underfitting / Bias-Variance)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision_macro = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
        precision_weighted = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
        recall_macro = recall_score(y_test, y_test_pred, average="macro", zero_division=0)
        recall_weighted = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

        # 5-Fold Cross Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        results[name] = {
            "train_accuracy": round(float(train_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "precision_macro": round(float(precision_macro), 4),
            "precision_weighted": round(float(precision_weighted), 4),
            "recall_macro": round(float(recall_macro), 4),
            "recall_weighted": round(float(recall_weighted), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "cv_mean": round(cv_mean, 4),
            "cv_std": round(cv_std, 4),
            "bias_variance_diagnosis": "Slight Overfitting" if (train_acc - test_acc) > 0.08 else "Well-balanced (Good Fit)"
        }

        print(f"     -> Test Acc: {test_acc:.4f} | F1 Weighted: {f1_weighted:.4f} | 5-Fold CV: {cv_mean:.4f} (+/- {cv_std:.4f})")

        if f1_weighted > best_f1:
            best_f1 = f1_weighted
            best_model_name = name
            best_fitted_model = model

    print(f"\n  [BEST MODEL SELECTED]: {best_model_name} with F1-Score of {best_f1:.4f}")

    # Generate Confusion Matrix for the best model
    best_preds = best_fitted_model.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)

    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - Best Classifier ({best_model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Category", fontsize=12)
    plt.ylabel("True Category", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "classification_confusion_matrix.png", dpi=150)
    plt.close()

    # Generate Model Comparison Chart
    df_compare = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    plt.figure(figsize=(12, 6))
    x_indices = np.arange(len(df_compare))
    bar_width = 0.35

    plt.bar(x_indices - bar_width/2, df_compare["test_accuracy"], width=bar_width, label="Test Accuracy", color="#3B82F6")
    plt.bar(x_indices + bar_width/2, df_compare["cv_mean"], width=bar_width, label="5-Fold CV Accuracy", color="#10B981")
    plt.xticks(x_indices, df_compare["Model"], rotation=20, ha="right", fontsize=10)
    plt.ylabel("Accuracy Score", fontsize=11)
    plt.title("Unit 7: Classification Model Comparison (Accuracy vs 5-Fold CV)", fontsize=13, fontweight="bold")
    plt.ylim(0.5, 1.05)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_accuracy.png", dpi=150)
    plt.close()

    # Save artifacts for production deployment
    joblib.dump(best_fitted_model, MODELS_DIR / "best_classifier.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    vectorizer.save(MODELS_DIR / "tfidf_vectorizer.joblib")

    # Save complete evaluation metrics
    with open(REPORTS_DIR / "classification_metrics.json", "w") as f:
        json.dump({
            "models": results,
            "best_model": best_model_name,
            "classes": classes,
            "total_samples": len(texts),
            "train_samples": len(texts_train),
            "test_samples": len(texts_test)
        }, f, indent=4)

    print("  [OK] Saved best model, vectorizer, label encoder, and evaluation reports.")
    return results

if __name__ == "__main__":
    train_and_evaluate_classifiers()
