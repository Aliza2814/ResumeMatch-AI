"""
Master model training and evaluation runner for ResumeMatch AI.
Executes all AI/ML syllabus pipelines:
- Unit 5: Regression (Linear, Multiple, Polynomial)
- Units 6 & 7: Classification (6 algorithms, 5-Fold CV, Confusion Matrix)
- Unit 8: Clustering (K-Means, Hierarchical, DBSCAN, Silhouette)
- Unit 9: Association Rules (Apriori, FP-Growth, PCA)
- Unit 10: Reinforcement Learning (Q-Learning Agent)
"""

import sys
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.regression import run_regression_experiments
from src.models.classifier import train_and_evaluate_classifiers
from src.models.clustering import run_clustering_experiments
from src.models.association_rules import run_association_and_pca
from src.models.reinforcement_learning import run_reinforcement_learning

def main():
    print("==================================================================")
    print("  RESUMEMATCH AI - MASTER EXPERIMENTAL TRAINING PIPELINE")
    print("  Executing Units 5 through 10 of Academic Syllabus")
    print("==================================================================")
    start_time = time.time()

    # 1. Regression
    run_regression_experiments()
    print("-" * 60)

    # 2. Classification
    train_and_evaluate_classifiers()
    print("-" * 60)

    # 3. Clustering
    run_clustering_experiments()
    print("-" * 60)

    # 4. Association Rules & PCA
    run_association_and_pca()
    print("-" * 60)

    # 5. Reinforcement Learning
    run_reinforcement_learning()
    print("-" * 60)

    elapsed = time.time() - start_time
    print(f"=== ALL EXPERIMENTS SUCCESSFULLY COMPLETED IN {elapsed:.2f}s ===")

if __name__ == "__main__":
    main()
