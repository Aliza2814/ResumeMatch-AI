"""
Clustering experiments for ResumeMatch AI (Unit 8).
Implements K-Means, Hierarchical (Agglomerative), and DBSCAN clustering on candidate skill profiles.
Computes Silhouette Scores and visualizes clusters using PCA 2D projections.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.feature_engineering.vectorizer import TextVectorizer

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_resumes.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def run_clustering_experiments():
    print("=== Running Unit 8: Unsupervised Learning - Clustering Experiments ===")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    texts = df["Cleaned_Resume"].tolist()

    # Vectorize texts
    vectorizer = TextVectorizer(max_features=1000, min_df=3)
    X = vectorizer.fit_transform(texts).toarray()
    print(f"  Feature matrix for clustering: {X.shape[0]} profiles x {X.shape[1]} features")

    # PCA 2D projection for visualization and dense representation
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca2d = pca_2d.fit_transform(X)

    results = {}
    k_clusters = 5  # Grouping into 5 major job clusters: Data/AI, Software Dev, Web/Design, Operations/HR, Engineering

    # 1. K-Means Clustering
    print("  1. Implementing K-Means Clustering...")
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    sil_kmeans = silhouette_score(X, kmeans_labels)
    results["K-Means"] = {
        "n_clusters": k_clusters,
        "silhouette_score": round(float(sil_kmeans), 4),
        "cluster_distribution": {int(k): int(v) for k, v in pd.Series(kmeans_labels).value_counts().items()}
    }

    # 2. Hierarchical (Agglomerative) Clustering
    print("  2. Implementing Hierarchical Clustering (Ward Linkage)...")
    hierarchical = AgglomerativeClustering(n_clusters=k_clusters, linkage="ward")
    hier_labels = hierarchical.fit_predict(X)
    sil_hier = silhouette_score(X, hier_labels)
    results["Hierarchical"] = {
        "n_clusters": k_clusters,
        "linkage": "ward",
        "silhouette_score": round(float(sil_hier), 4),
        "cluster_distribution": {int(k): int(v) for k, v in pd.Series(hier_labels).value_counts().items()}
    }

    # 3. DBSCAN (Density-Based Spatial Clustering)
    print("  3. Implementing DBSCAN Clustering...")
    # Use PCA 2D for DBSCAN density clustering to avoid high-dimensional sparsity
    dbscan = DBSCAN(eps=0.25, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_pca2d)
    
    unique_db_labels = set(dbscan_labels)
    n_db_clusters = len(unique_db_labels) - (1 if -1 in unique_db_labels else 0)
    
    # Silhouette valid only if > 1 cluster and < n_samples
    if n_db_clusters > 1:
        sil_dbscan = silhouette_score(X_pca2d, dbscan_labels)
    else:
        sil_dbscan = -1.0

    results["DBSCAN"] = {
        "eps": 0.25,
        "min_samples": 10,
        "n_clusters_found": n_db_clusters,
        "noise_points": int(sum(dbscan_labels == -1)),
        "silhouette_score": round(float(sil_dbscan), 4) if sil_dbscan != -1.0 else "N/A (< 2 clusters)",
        "cluster_distribution": {int(k): int(v) for k, v in pd.Series(dbscan_labels).value_counts().items()}
    }

    # Save models
    joblib.dump(kmeans, MODELS_DIR / "kmeans_clusterer.joblib")
    joblib.dump(pca_2d, MODELS_DIR / "pca_2d.joblib")

    # Generate Clustering Comparison Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # K-Means Plot
    scatter1 = axes[0].scatter(X_pca2d[:, 0], X_pca2d[:, 1], c=kmeans_labels, cmap="tab10", s=25, alpha=0.7)
    axes[0].set_title(f"K-Means (k={k_clusters}, Silhouette={sil_kmeans:.3f})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("PCA Component 1")
    axes[0].set_ylabel("PCA Component 2")
    axes[0].grid(True, alpha=0.3)

    # Hierarchical Plot
    scatter2 = axes[1].scatter(X_pca2d[:, 0], X_pca2d[:, 1], c=hier_labels, cmap="tab10", s=25, alpha=0.7)
    axes[1].set_title(f"Hierarchical Ward (k={k_clusters}, Silhouette={sil_hier:.3f})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("PCA Component 1")
    axes[1].grid(True, alpha=0.3)

    # DBSCAN Plot
    scatter3 = axes[2].scatter(X_pca2d[:, 0], X_pca2d[:, 1], c=dbscan_labels, cmap="plasma", s=25, alpha=0.7)
    axes[2].set_title(f"DBSCAN (eps=0.25, Clusters={n_db_clusters})", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("PCA Component 1")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Unit 8: Candidate Skill Profile Clustering Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clustering_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save metrics JSON
    with open(REPORTS_DIR / "clustering_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("  [OK] Clustering experiments completed and saved:")
    for k, v in results.items():
        print(f"     {k}: Silhouette={v.get('silhouette_score')}")
    return results

if __name__ == "__main__":
    run_clustering_experiments()
