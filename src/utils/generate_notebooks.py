"""
Generates clean, academic Jupyter Notebooks for Units 1 through 10.
Includes markdown documentation, theoretical explanations, and executable code blocks.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def create_notebook(filename: str, cells: list):
    nb_data = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.14.0"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    path = NOTEBOOKS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=2)
    print(f"Generated {filename}")

def md_cell(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code_cell(code: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code]}

def generate_all_notebooks():
    # 01_data_exploration.ipynb
    create_notebook("01_data_exploration.ipynb", [
        md_cell("# 01. Data Exploration & Discovery (ResumeMatch AI)\n\nThis notebook inspects the raw datasets for ResumeMatch AI:\n- `resumes.csv` (Updated Resume Dataset)\n- `placement_salary.csv` (Placement & Salary Benchmark)\n- `job_postings.csv` (Job Postings Catalog)"),
        code_cell("""import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("../data/raw")
df_resumes = pd.read_csv(data_dir / "resumes.csv")
print("Resume Dataset Shape:", df_resumes.shape)
df_resumes.head()"""),
        code_cell("""# Category Distribution in Resumes
plt.figure(figsize=(12, 6))
df_resumes['Category'].value_counts().plot(kind='bar', color='#3B82F6')
plt.title("Distribution of Resume Categories (25 Domains)")
plt.xlabel("Domain Category")
plt.ylabel("Number of Resumes")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()"""),
        code_cell("""# Inspect Placement & Salary Dataset
df_placement = pd.read_csv(data_dir / "placement_salary.csv")
print("Placement Data Shape:", df_placement.shape)
df_placement.describe()""")
    ])

    # 02_preprocessing.ipynb
    create_notebook("02_preprocessing.ipynb", [
        md_cell("# 02. Data Preprocessing & Feature Engineering (Unit 4)\n\nDemonstrates data cleaning, regex-based token normalization, stopword filtering, and domain skill extraction."),
        code_cell("""import sys
sys.path.append("..")
import pandas as pd
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.skill_extractor import extract_skills_from_text

sample_text = \"\"\"
Senior Data Scientist with 5+ years of experience in Python, PyTorch, Scikit-learn, and AWS.
Contact: john.doe@ai-lab.org | Phone: +1-555-0199
Education: M.Tech in Computer Science
Skills: Machine Learning, NLP, Docker, SQL, Leadership.
\"\"\"

print("Original Text:\\n", sample_text)
print("\\nCleaned Text:\\n", clean_text(sample_text))
print("\\nExtracted Skills & Metadata:\\n", extract_skills_from_text(sample_text))""")
    ])

    # 03_regression.ipynb
    create_notebook("03_regression.ipynb", [
        md_cell("# 03. Supervised Learning - Regression (Unit 5)\n\nComparison of Linear Regression, Multiple Linear Regression, and Polynomial Regression for salary prediction."),
        code_cell("""import sys
sys.path.append("..")
from src.models.regression import run_regression_experiments
results = run_regression_experiments()
import pandas as pd
pd.DataFrame(results).T"""),
        code_cell("""from IPython.display import Image
Image(filename="../reports/figures/regression_predictions.png")""")
    ])

    # 04_classification.ipynb
    create_notebook("04_classification.ipynb", [
        md_cell("# 04. Supervised Learning - Classification (Unit 6)\n\nTrains 6 classification algorithms on resume dataset:\n1. Logistic Regression\n2. K-Nearest Neighbours\n3. Decision Tree\n4. Random Forest\n5. Naive Bayes\n6. Support Vector Machine"),
        code_cell("""import sys
sys.path.append("..")
from src.models.classifier import train_and_evaluate_classifiers
results = train_and_evaluate_classifiers()
import pandas as pd
pd.DataFrame(results).T""")
    ])

    # 05_model_evaluation.ipynb
    create_notebook("05_model_evaluation.ipynb", [
        md_cell("# 05. Model Evaluation & Validation (Unit 7)\n\nExamines Confusion Matrix, 5-Fold Stratified Cross-Validation, and Bias-Variance trade-offs."),
        code_cell("""from IPython.display import Image, display
display(Image(filename="../reports/figures/classification_confusion_matrix.png"))
display(Image(filename="../reports/figures/model_comparison_accuracy.png"))""")
    ])

    # 06_clustering.ipynb
    create_notebook("06_clustering.ipynb", [
        md_cell("# 06. Unsupervised Learning - Clustering (Unit 8)\n\nImplements K-Means, Hierarchical (Ward), and DBSCAN clustering on candidate profiles with Silhouette Score analysis."),
        code_cell("""import sys
sys.path.append("..")
from src.models.clustering import run_clustering_experiments
results = run_clustering_experiments()
import pandas as pd
pd.DataFrame(results).T"""),
        code_cell("""from IPython.display import Image
Image(filename="../reports/figures/clustering_comparison.png")""")
    ])

    # 07_association_rules.ipynb
    create_notebook("07_association_rules.ipynb", [
        md_cell("# 07. Association Rule Learning & PCA (Unit 9)\n\nMarket Basket Analysis of skill co-occurrences using Apriori and FP-Growth, alongside PCA dimensionality reduction."),
        code_cell("""import sys
sys.path.append("..")
from src.models.association_rules import run_association_and_pca
top_rules = run_association_and_pca()
import pandas as pd
pd.DataFrame(top_rules)"""),
        code_cell("""from IPython.display import Image
Image(filename="../reports/figures/pca_scree_plot.png")""")
    ])

    # 08_recommendation_system.ipynb
    create_notebook("08_recommendation_system.ipynb", [
        md_cell("# 08. Content-Based Recommendation System & Skill Gap Analysis\n\nCalculates transparent resume-job compatibility combining TF-IDF cosine similarity and skill overlap."),
        code_cell("""import sys
sys.path.append("..")
from src.recommendation.engine import RecommendationEngine

engine = RecommendationEngine()
mock_resume = {
    "cleaned_text": "data scientist with python machine learning deep learning sql pandas",
    "skills": {"all_skills": ["python", "machine learning", "deep learning", "sql", "pandas"]}
}
recs = engine.recommend_jobs(mock_resume, top_n=5)
for r in recs:
    print(f"{r['job_title']} at {r['company']}: {r['overall_score']}% Match")
    print(f"   Why: {r['explanation']}\\n")""")
    ])

    # 09_reinforcement_learning.ipynb
    create_notebook("09_reinforcement_learning.ipynb", [
        md_cell("# 09. Reinforcement Learning - Career Optimization MDP (Unit 10)\n\nDemonstrates Tabular Q-Learning in a simulated Job Search Environment."),
        code_cell("""import sys
sys.path.append("..")
from src.models.reinforcement_learning import run_reinforcement_learning
results = run_reinforcement_learning()
print("Episodes Trained:", results["episodes_trained"])
print("Final Average Reward:", results["final_20_ep_reward_avg"])"""),
        code_cell("""from IPython.display import Image
Image(filename="../reports/figures/rl_learning_curve.png")""")
    ])

if __name__ == "__main__":
    generate_all_notebooks()
