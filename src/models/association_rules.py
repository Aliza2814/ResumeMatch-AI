"""
Association Rule Learning and Principal Component Analysis for ResumeMatch AI (Unit 9).
Implements Apriori and FP-Growth algorithms on candidate and job skill transactions.
Calculates Support, Confidence, and Lift for skill co-occurrences.
Performs PCA for Dimensionality Reduction and scree plot analysis.
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

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.decomposition import PCA

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_resumes.csv"
JOBS_PATH = BASE_DIR / "data" / "processed" / "cleaned_jobs.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def run_association_and_pca():
    print("=== Running Unit 9: Association Rule Learning, Recommendation Systems & PCA ===")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    df_resumes = pd.read_csv(DATA_PATH)
    
    # 1. Market Basket Analysis on Skills
    print("  1. Preparing skill transactions from resumes and job postings...")
    transactions = []
    for s_str in df_resumes["All_Skills"].dropna():
        skills = [s.strip() for s in s_str.split(",") if s.strip()]
        if len(skills) >= 2:
            transactions.append(skills)

    if JOBS_PATH.exists():
        df_jobs = pd.read_csv(JOBS_PATH)
        for s_str in df_jobs["required_skills"].dropna():
            skills = [s.strip() for s in s_str.split(",") if s.strip()]
            if len(skills) >= 2:
                transactions.append(skills)

    print(f"  Total skill transaction baskets: {len(transactions)}")

    # Encode transactions into one-hot boolean dataframe
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    # Apriori Algorithm (Frequent Itemset Mining)
    print("  2. Running Apriori Algorithm (min_support=0.08)...")
    apriori_frequent = apriori(df_trans, min_support=0.08, use_colnames=True)
    print(f"     Found {len(apriori_frequent)} frequent itemsets with Apriori.")

    # FP-Growth Algorithm (Frequent Pattern Growth)
    print("  3. Running FP-Growth Algorithm (min_support=0.08)...")
    fpgrowth_frequent = fpgrowth(df_trans, min_support=0.08, use_colnames=True)
    print(f"     Found {len(fpgrowth_frequent)} frequent itemsets with FP-Growth.")

    # Generate Association Rules (Confidence >= 0.4)
    print("  4. Generating Association Rules (min_threshold=0.4, metric='confidence')...")
    rules = association_rules(fpgrowth_frequent, metric="confidence", min_threshold=0.4)
    rules["lift"] = rules["lift"].round(3)
    rules["confidence"] = rules["confidence"].round(3)
    rules["support"] = rules["support"].round(3)

    # Sort rules by Lift (strength of association)
    rules_sorted = rules.sort_values(by="lift", ascending=False)

    top_rules_list = []
    for _, row in rules_sorted.head(15).iterrows():
        top_rules_list.append({
            "antecedents": list(row["antecedents"]),
            "consequents": list(row["consequents"]),
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"])
        })

    # Save top rules JSON
    with open(REPORTS_DIR / "association_rules.json", "w") as f:
        json.dump({
            "total_transactions": len(transactions),
            "apriori_itemsets_count": len(apriori_frequent),
            "fpgrowth_itemsets_count": len(fpgrowth_frequent),
            "rules_count": len(rules),
            "top_rules": top_rules_list
        }, f, indent=4)

    # 2. Dimensionality Reduction using PCA
    print("  5. Implementing PCA Dimensionality Reduction...")
    # Sample top 50 frequent skills matrix
    top_cols = df_trans.sum().sort_values(ascending=False).head(50).index
    df_top_skills = df_trans[top_cols].astype(int)

    pca = PCA(n_components=10, random_state=42)
    pca.fit(df_top_skills)
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    # Scree Plot
    plt.figure(figsize=(9, 4.5))
    plt.bar(range(1, 11), var_ratio * 100, alpha=0.7, color="#3B82F6", label="Individual Explained Variance %")
    plt.step(range(1, 11), cum_var * 100, where="mid", color="#EF4444", linewidth=2, label="Cumulative Explained Variance %")
    plt.xlabel("Principal Component Index", fontsize=11)
    plt.ylabel("Explained Variance Ratio (%)", fontsize=11)
    plt.title("Unit 9: PCA Scree Plot - Skill Dimension Reduction", fontsize=13, fontweight="bold")
    plt.xticks(range(1, 11))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_scree_plot.png", dpi=150)
    plt.close()

    joblib.dump({"transaction_encoder": te, "rules": rules_sorted}, MODELS_DIR / "association_rules.joblib")
    joblib.dump(pca, MODELS_DIR / "pca_model.joblib")

    print(f"  [OK] Extracted {len(rules)} association rules and generated PCA scree plot.")
    return top_rules_list

if __name__ == "__main__":
    run_association_and_pca()
