# ACADEMIC PROJECT REPORT: RESUMEMATCH AI
## An Intelligent Resume Analyzer and Job Recommendation System

**Academic Year:** 2026  
**Discipline:** Artificial Intelligence & Machine Learning  
**Project Repository:** `JobRec`  

---

### ABSTRACT
Modern recruitment processes suffer from substantial friction caused by high volumes of unstructured candidate resumes and poorly calibrated keyword search filters. This project designs and implements **ResumeMatch AI**, an intelligent, explainable career intelligence platform. ResumeMatch AI parses unstructured resumes (PDF, DOCX, TXT), extracts granular technical and soft skill taxonomies, classifies candidates into 25 professional domains using supervised learning, and recommends tailored job openings via transparent content-based filtering. The system rigorously implements and evaluates all concepts within the academic AI/ML syllabus (Units 1 through 10): Simple/Multiple/Polynomial Regression for placement compensation analysis, 6 classification algorithms (LinearSVC, Random Forest, Logistic Regression, Naive Bayes, Decision Trees, KNN), Unsupervised Clustering (K-Means, Hierarchical, DBSCAN), Association Rule Mining (Apriori and FP-Growth), Principal Component Analysis (PCA), and Tabular Q-Learning Reinforcement Learning.

---

### CHAPTER 1: INTRODUCTION & PROBLEM FORMULATION (UNIT 1)
#### 1.1 Problem Statement
Candidates frequently struggle to understand how their resumes align with target job market requirements, which skills they lack, and what career trajectories maximize their employability. Conversely, traditional Applicant Tracking Systems (ATS) rely on brittle keyword matching that rejects qualified applicants.

#### 1.2 Objectives
1. Build a local, reproducible NLP document parsing pipeline for PDF, DOCX, and TXT resumes.
2. Formulate a 150+ entity skill taxonomy across technical and soft proficiencies.
3. Train, cross-validate, and evaluate 6 supervised classifiers for job role prediction.
4. Deliver transparent, explainable recommendations combining skill overlap and semantic TF-IDF cosine similarity.
5. Provide actionable skill gap analysis showing high-ROI skills to learn.
6. Empirically evaluate Regression, Clustering, Association Rules, PCA, and Reinforcement Learning models.

#### 1.3 Scope and Limitations
- **Scope:** Technical, engineering, management, and corporate professional resumes across 25 industry domains.
- **Limitations:** Does not assess candidate interpersonal demeanor or live coding ability. Assumes resume claims are factual.

---

### CHAPTER 2: AI AGENTS, PEAS REPRESENTATION & ETHICS (UNIT 2)
#### 2.1 PEAS Specification
- **Performance Measure (P):** Recommendation accuracy, skill extraction recall, low latency (< 1s), transparent explainability.
- **Environment (E):** Unstructured multi-format resume documents, fluctuating job descriptions, skill taxonomies.
- **Actuators (A):** Streamlit interactive UI, skill gap diagnostic alerts, predicted domain badges, database persistence.
- **Sensors (S):** File byte stream parsers (PyMuPDF, python-docx), regex tokenizers, TF-IDF vectorizers.

#### 2.2 Ethical Considerations
- **Fairness & Non-Discrimination:** Resumes are stripped of demographic indicators (gender, age, marital status, nationality) during tokenization to prevent bias.
- **Privacy:** Candidate resumes can be cleared on-demand; no external proprietary APIs transmit candidate data over the public internet.

---

### CHAPTER 3: DATA PREPROCESSING & FEATURE ENGINEERING (UNITS 3 & 4)
#### 3.1 Datasets
1. **Updated Resume Dataset (`resumes.csv`):** 962 authentic resumes across 25 categories (Data Science, Java Developer, DevOps, HR, etc.).
2. **Placement & Salary Benchmark (`placement_salary.csv`):** 215 academic records with secondary, higher secondary, degree, employability test scores, and compensation.
3. **Job Postings Catalog (`cleaned_jobs.csv`):** 157 job descriptions with titles, companies, locations, and requirements.

#### 3.2 Preprocessing Pipeline
- Unicode normalization (NFKD) and case folding.
- Preservation of programming tokens: `C++`, `C#`, `.NET`.
- Removal of URLs, emails, phone numbers, and special symbols.
- NLTK English stopword filtration.
- TF-IDF Vectorization with unigrams and bigrams (`ngram_range=(1, 2)`), sublinear TF scaling, and Chi-squared feature selection.

---

### CHAPTER 4: SUPERVISED LEARNING - REGRESSION (UNIT 5)
Evaluated on candidate placement salary data (`cleaned_salary_regression.csv`):
- **Simple Linear Regression (Degree % -> Salary):** MAE: ₹66,852.57, RMSE: ₹94,690.89, R²: -0.0606
- **Multiple Linear Regression (All academic & test features):** MAE: ₹71,898.23, RMSE: ₹94,877.51, R²: -0.0647
- **Polynomial Regression (Degree 2):** MAE: ₹67,632.28, RMSE: ₹91,561.30, R²: +0.0084

*Observation:* Academic test percentages alone exhibit non-linear and weak correlation with starting corporate salary, as compensation is heavily influenced by company tier and negotiation rather than pure grades.

---

### CHAPTER 5: SUPERVISED LEARNING - CLASSIFICATION (UNITS 6 & 7)
Trained across 962 resumes over 25 job domains with 80-20 stratified train-test splitting and 5-Fold Stratified Cross-Validation:

| Model | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Weighted) | 5-Fold CV Accuracy | Bias-Variance Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Support Vector Machine (LinearSVC)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.9948 (+/- 0.0026)** | **Optimal Fit (Selected)** |
| **Random Forest** | 0.9948 | 0.9952 | 0.9948 | 0.9949 | 0.9935 (+/- 0.0041) | Optimal Fit |
| **Naive Bayes (MultinomialNB)** | 0.9896 | 0.9904 | 0.9896 | 0.9896 | 0.9935 (+/- 0.0041) | Optimal Fit |
| **Logistic Regression** | 0.9896 | 0.9904 | 0.9896 | 0.9897 | 0.9922 (+/- 0.0049) | Optimal Fit |
| **Decision Tree** | 0.9845 | 0.9856 | 0.9845 | 0.9847 | 0.9740 (+/- 0.0109) | Well-Balanced |
| **K-Nearest Neighbours** | 0.9896 | 0.9904 | 0.9896 | 0.9896 | 0.9688 (+/- 0.0048) | Well-Balanced |

---

### CHAPTER 6: UNSUPERVISED LEARNING - CLUSTERING (UNIT 8)
Clustering performed on TF-IDF skill profile matrix:
- **K-Means (k=5):** Silhouette Score = **0.0591**. Identifies major clusters: AI/Data, Web Development, Cloud/DevOps, Operations/HR, Engineering.
- **Hierarchical Agglomerative (Ward Linkage, k=5):** Silhouette Score = **0.0523**.
- **DBSCAN (eps=0.25, min_samples=10):** Separates core high-density candidate nodes from sparse outlier profiles.

---

### CHAPTER 7: ASSOCIATION RULE MINING & PCA (UNIT 9)
- **Frequent Itemset Mining:** Apriori and FP-Growth mined 85 frequent skill itemsets at `min_support=0.08`.
- **Rules Generated:** 122 rules with `confidence >= 0.4`. Strongest associations:
  - `{pandas, scikit-learn} -> {python}` (Lift: 2.14, Confidence: 0.95)
  - `{docker, kubernetes} -> {ci/cd}` (Lift: 2.85, Confidence: 0.88)
- **Principal Component Analysis (PCA):** First 10 principal components capture over 54% of total variance across skill co-occurrences.

---

### CHAPTER 8: REINFORCEMENT LEARNING SIMULATION (UNIT 10)
- **Environment:** Career MDP with 12 discrete candidate states and 4 actions (Upskill, Tailor Resume, Apply General, Apply Targeted).
- **Q-Learning Agent:** Trained over 500 episodes with epsilon decay.
- **Outcome:** Moving average cumulative reward increased from 14.2 to 58.25. The agent learned the optimal policy: prioritize upskilling to High Skill level, tailor resume to target role, and apply to targeted positions.

---

### CHAPTER 9: DATABASE DESIGN & ORM
PostgreSQL relational architecture utilizing SQLAlchemy ORM:
- `resumes`: Primary candidate records, contacts, and timestamps.
- `extracted_skills`: Normalized skill entities with categorization.
- `jobs`: Catalog of active job postings.
- `job_skills`: Normalized required competencies per job listing.
- `analysis_results`: Model classification predictions and confidence scores.
- `recommendations`: Compatibility scores, skill overlap ratios, and explanation strings.

---

### CHAPTER 10: CONCLUSION & FUTURE SCOPE
ResumeMatch AI demonstrates that combining deterministic NLP skill extraction with supervised classification and content-based recommendation yields high accuracy, reproducible performance, and explainable recommendations. Future enhancements include graph neural networks for organizational skill taxonomies and automated cover letter synthesis.
