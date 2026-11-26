# VIVA VOCE PREPARATION GUIDE: RESUMEMATCH AI
## Comprehensive Q&A for Academic Project Defense across Units 1–10

---

### UNIT 1: INTRODUCTION TO ARTIFICIAL INTELLIGENCE
**Q1: What is the core AI problem addressed by ResumeMatch AI?**  
*Answer:* Unstructured natural language resume text processing, entity/skill extraction, multi-class domain classification across 25 professional categories, and content-based recommendation.

**Q2: How does ResumeMatch AI differ from a simple keyword matching system?**  
*Answer:* Traditional systems look for literal string matches. ResumeMatch AI combines regex boundary tokenization, TF-IDF semantic vectorization (capturing term frequency and inverse document importance), multi-class Support Vector Machine classification, and cosine similarity.

---

### UNIT 2: AI AGENTS & ETHICAL CONSIDERATIONS
**Q3: Describe ResumeMatch AI in terms of the PEAS representation.**  
- **P (Performance):** Recommendation accuracy, skill extraction precision, low inference latency (< 1s), explainability.
- **E (Environment):** Unstructured PDF/DOCX files, job postings database, skill taxonomy.
- **A (Actuators):** Streamlit UI badges, recommendation ranking, skill gap alerts, database storage.
- **S (Sensors):** PyMuPDF/python-docx text readers, regex tokenizers, TF-IDF vectorizer.

**Q4: How does the system handle algorithmic bias and privacy?**  
*Answer:* Sensitive demographic attributes (gender, age, race, nationality) are omitted from feature extraction. In addition, candidates have full control to purge their stored data at any time.

---

### UNIT 3: MACHINE LEARNING PROCESS & TOOLS
**Q5: What role did NumPy, Pandas, Scikit-learn, and Matplotlib play?**  
- **NumPy & Pandas:** Data loading, manipulation, handling missing values, and transaction encoding.
- **Scikit-learn:** Vectorization, model training (regression, classification, clustering, PCA), and metrics evaluation.
- **Matplotlib & Seaborn:** Generating confusion matrices, learning curves, scree plots, and regression residuals.

---

### UNIT 4: PREPROCESSING & FEATURE ENGINEERING
**Q6: How are special programming tokens like C++, C#, and .NET preserved?**  
*Answer:* Before non-alphanumeric punctuation stripping, regex replaces `C++`, `C#`, and `.NET` with temporary alphanumeric tokens (`token_cplusplus`, `token_csharp`, `token_dotnet`), which are restored back during token assembly.

**Q7: Explain the purpose of TF-IDF and sublinear term frequency scaling.**  
*Answer:* TF-IDF balances common terms against rare, highly informative terms. Sublinear TF scaling replaces term frequency $tf$ with $1 + \log(tf)$, dampening the effect of words that appear repeatedly in a single document.

---

### UNIT 5: SUPERVISED LEARNING - REGRESSION
**Q8: Why did you evaluate Simple, Multiple, and Polynomial Regression?**  
*Answer:* To test linear vs non-linear relationships between candidate academic/employability metrics and starting salary. Polynomial regression of degree 2 achieved slightly better RMSE than linear models, though overall correlation was weak because real-world corporate compensation depends heavily on company tier and negotiation rather than pure grades.

---

### UNIT 6 & 7: CLASSIFICATION & EVALUATION
**Q9: Which 6 classification algorithms were implemented? Which performed best?**  
*Answer:* Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, Naive Bayes, and Linear Support Vector Machine (LinearSVC). LinearSVC performed best with 100.0% test accuracy and 99.48% 5-Fold Stratified Cross-Validation.

**Q10: Why did Support Vector Machines perform so well on text classification?**  
*Answer:* TF-IDF feature spaces are high-dimensional (2,500 features) and sparse. Linear SVMs find the maximum-margin hyperplane in high dimensions with low susceptibility to overfitting.

**Q11: How did you diagnose Overfitting vs Underfitting (Bias-Variance trade-off)?**  
*Answer:* By comparing training accuracy against test accuracy and cross-validation accuracy. A difference greater than 8% signals overfitting; models with close training and test scores demonstrate a well-balanced fit.

---

### UNIT 8: UNSUPERVISED LEARNING - CLUSTERING
**Q12: How did you evaluate K-Means, Hierarchical, and DBSCAN?**  
*Answer:* Using the Silhouette Score on the feature space and visualizing 2D PCA projections. K-Means achieved a Silhouette Score of 0.0591, grouping candidate profiles into 5 distinct career clusters.

---

### UNIT 9: ASSOCIATION RULE MINING & RECOMMENDATIONS
**Q13: What is the difference between Apriori and FP-Growth?**  
*Answer:* Apriori repeatedly generates candidate itemsets and scans the database multiple times ($O(2^k)$). FP-Growth compresses the dataset into an FP-Tree (Frequent Pattern Tree) and mines itemsets without candidate generation, offering much faster execution.

**Q14: Explain Support, Confidence, and Lift.**  
- **Support:** Frequency of the itemset in the dataset: $P(A \cap B)$.
- **Confidence:** Probability of $B$ given $A$: $P(B|A) = \frac{P(A \cap B)}{P(A)}$.
- **Lift:** Ratio of observed joint probability to expected probability if independent: $\frac{P(A \cap B)}{P(A) \times P(B)}$. Lift > 1 indicates a positive association.

---

### UNIT 10: REINFORCEMENT LEARNING
**Q15: Why is Reinforcement Learning not used for the production recommendation engine?**  
*Answer:* Content-based filtering with TF-IDF cosine similarity is deterministic, explainable, and instantaneous. Reinforcement learning is suited for multi-step sequential decision problems over time (e.g. upskilling trajectories), demonstrated in our simulated career MDP.
