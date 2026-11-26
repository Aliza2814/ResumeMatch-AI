# ResumeMatch AI: An Intelligent Resume Analyzer and Job Recommendation System

**ResumeMatch AI** is an end-to-end academic machine learning application that parses candidate resumes (PDF, DOCX, TXT), extracts skills and experience, predicts suitable job roles across 25 domains, and recommends tailored job opportunities with transparent, human-understandable explanations and skill gap analysis.

This system provides a practical, working implementation covering the **complete AI/ML syllabus (Units 1 through 10)** using Python, Scikit-learn, Streamlit, and PostgreSQL (with automated SQLite fallback).

---

## 🚀 Key Features

1. **Multi-Format Resume Parsing (PyMuPDF & python-docx):**
   - High-fidelity extraction from PDF, DOCX, and TXT files.
   - Robust error handling for corrupt, empty, or unreadable files.
   - Automatic extraction of candidate name, contact information (email, phone, links), education, and experience.

2. **NLP Skill Extraction & Taxonomy Engine:**
   - Detects 150+ technical and soft skills, certifications, and educational degrees.
   - Preserves programming tokens (`C++`, `C#`, `.NET`).
   - Regex-based and contextual multi-word phrase matching without requiring paid external APIs.

3. **Supervised Job Role Classification (Units 6 & 7):**
   - Evaluates 6 standard algorithms: Support Vector Machine (LinearSVC), Random Forest, Logistic Regression, Naive Bayes, Decision Trees, and K-Nearest Neighbours.
   - Best Model: **LinearSVC** (100.0% Test Accuracy, 99.48% 5-Fold Stratified Cross-Validation).
   - Confusion matrix, precision, recall, and F1-score evaluation.

4. **Transparent Job Recommendation Engine (Unit 9):**
   - Content-based filtering combining TF-IDF Cosine Similarity (40%) and direct skill overlap (60%).
   - Transparent, explainable recommendation rationale ("Why this job was recommended").
   - Clear distinction between heuristic compatibility formulas and trained ML models.

5. **Skill Gap & ROI Analysis:**
   - Identifies missing skills between candidate profile and target job listings.
   - Aggregates frequent cross-role missing skills to pinpoint the highest ROI skills to learn.

6. **Complete Syllabus ML Laboratory (Units 5, 8, 9, 10):**
   - **Regression Sandbox (Unit 5):** Linear, Multiple, and Polynomial Regression predicting placement salary (MAE, RMSE, R²).
   - **Clustering Explorer (Unit 8):** K-Means, Hierarchical (Ward), and DBSCAN with Silhouette score analysis.
   - **Association Rules & PCA (Unit 9):** Apriori and FP-Growth frequent skill itemset mining with support, confidence, lift, and PCA scree plots.
   - **Reinforcement Learning (Unit 10):** Tabular Q-Learning agent simulating candidate career upskilling in a Markov Decision Process (MDP).

7. **Database Persistence (PostgreSQL & SQLite):**
   - Full SQLAlchemy ORM schema (`resumes`, `extracted_skills`, `jobs`, `job_skills`, `analysis_results`, `recommendations`).
   - Connects to PostgreSQL, with automatic graceful fallback to local SQLite (`data/resumematch.db`) when PostgreSQL is offline.

---

## 🛠️ Project Structure

```text
JobRec/
├── app.py                      # Main Streamlit Dashboard (9 Interactive Sections)
├── config.py                   # Centralized Configuration & Environment Manager
├── requirements.txt            # Dependency Specifications
├── README.md                   # Installation & User Guide
├── .env.example                # Sample Environment Variables
├── data/
│   ├── raw/                    # Genuine open datasets (resumes.csv, placement_salary.csv, job_postings.csv)
│   ├── processed/              # Cleaned and normalized datasets
│   └── README.md               # Dataset documentation, sources, and licensing
├── notebooks/                  # Reproducible Jupyter Notebooks (Units 1 - 10)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_regression.ipynb
│   ├── 04_classification.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_clustering.ipynb
│   ├── 07_association_rules.ipynb
│   ├── 08_recommendation_system.ipynb
│   └── 09_reinforcement_learning.ipynb
├── src/
│   ├── preprocessing/          # Text cleaning and skill extraction
│   ├── feature_engineering/    # TF-IDF vectorizer and feature selection
│   ├── models/                 # Model training scripts (classification, regression, clustering, RL)
│   ├── recommendation/         # Content-based recommendation engine
│   ├── resume_parser/          # PDF and DOCX document parser
│   ├── database/               # SQLAlchemy models, connection, and seed script
│   └── utils/                  # UI helpers and sample profiles
├── models/                     # Saved trained models (.joblib)
├── reports/                    # Metrics JSON reports and visual figures (.png)
├── tests/                      # Automated test suite (16 tests, 100% pass)
└── docs/                       # Academic Project Report & Viva Notes
```

---

## 💻 Quickstart Installation & Execution

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.14 on Windows 11)
- Git (optional)

### 2. Setup Virtual Environment
```bash
# Clone or navigate to the repository
cd JobRec

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize Database & Seed Jobs
```bash
python src/database/init_db.py
```
*(By default, this connects to SQLite or PostgreSQL if available, and seeds 157 job records into the catalog).*

### 5. Run Model Training (Optional, pre-trained models already included)
```bash
python src/models/train_all.py
```

### 6. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 7. Run Test Suite
```bash
pytest -v
```

---

## 📊 AI/ML Syllabus Mapping Matrix

| Unit | Syllabus Topic | Implementation File | Key Deliverable |
| :--- | :--- | :--- | :--- |
| **Unit 1** | Intro to AI & Problem Formulation | `docs/PROJECT_REPORT.md` | Architecture, Scope, Objectives |
| **Unit 2** | AI Agents & Ethics | `docs/PROJECT_REPORT.md`, `app.py` | PEAS Agent Model, Privacy Controls |
| **Unit 3** | ML Tools & Workflow | `src/`, `notebooks/` | NumPy, Pandas, Scikit-learn, Matplotlib |
| **Unit 4** | Preprocessing & Features | `src/preprocessing/`, `src/feature_engineering/` | Stopwords, TF-IDF, Skill Extraction |
| **Unit 5** | Supervised Regression | `src/models/regression.py` | SLR, MLR, Polynomial (MAE, RMSE, R²) |
| **Unit 6** | Supervised Classification | `src/models/classifier.py` | 6 Algorithms (SVM, RF, LR, NB, DT, KNN) |
| **Unit 7** | Model Evaluation | `src/models/classifier.py` | 5-Fold Stratified CV, Confusion Matrix |
| **Unit 8** | Unsupervised Clustering | `src/models/clustering.py` | K-Means, Hierarchical (Ward), DBSCAN |
| **Unit 9** | Association Rules & PCA | `src/models/association_rules.py` | Apriori, FP-Growth (Lift/Support), PCA |
| **Unit 10**| Reinforcement Learning | `src/models/reinforcement_learning.py`| Q-Learning MDP Career Simulation |

---

## 🛡️ Ethical Considerations & Transparency

1. **No Demographic Bias:** Demographic attributes (gender, race, age, religion) are excluded from resume feature vectors to prevent algorithmic bias.
2. **Deterministic Explainability:** Recommendations provide transparent reasons (exact matching skills and text cosine similarity) rather than obscure black-box ranking scores.
3. **Data Privacy:** Users can purge uploaded profiles with one click.
4. **No Artificial APIs:** Works completely locally using open-source packages without paid external API lock-in.

---

## 📜 License
This project is open-source and released under the MIT License.
Datasets are governed by their respective CC0 / Public Domain licenses.
