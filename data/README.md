# ResumeMatch AI Datasets

This directory stores raw and processed datasets utilized for training, evaluation, and recommendation pipelines.

## 1. Updated Resume Dataset (`data/raw/resumes.csv`)
- **Source**: Kaggle / Open GitHub Mirror (`AyushAggarwal1/Resume-Selector`)
- **License**: Open Data Commons Public Domain Dedication (CC0: Public Domain)
- **Description**: Contains 962 candidate resume text records categorized across 25 industry domains (Data Science, HR, Java Developer, DevOps Engineer, Python Developer, Web Designing, Advocate, Operations Manager, Mechanical Engineer, etc.).
- **Key Columns**:
  - `Category`: Target job category label.
  - `Resume`: Raw unformatted resume text.

## 2. Campus Placement & Salary Dataset (`data/raw/placement_salary.csv`)
- **Source**: Kaggle / GitHub (`GauravSahani1417/Campus-Placement-Dataset-Kaggle-`)
- **License**: CC0: Public Domain
- **Description**: Benchmark dataset with 215 records containing candidate academic performance, employability test scores, work experience, and resulting compensation (salary). Used for Unit 5 Supervised Regression experiments.
- **Key Columns**:
  - `ssc_p`: 10th grade percentage
  - `hsc_p`: 12th grade percentage
  - `degree_p`: Degree percentage
  - `etest_p`: Employability test score percentage
  - `workex`: Work experience status
  - `mba_p`: MBA percentage
  - `salary`: Numerical target (offered salary in INR)

## 3. Verified Live Job Catalog (`data/processed/cleaned_jobs.csv`)
- **Sources**: Real-time live public APIs requiring zero API keys and zero paid access:
  1. **Unstop API** (`https://unstop.com/api/public/opportunity/search-result?opportunity=jobs`) - Live campus and corporate vacancies in India & Remote.
  2. **Jobicy API** (`https://jobicy.com/api/v2/remote-jobs`) - Live tech, data, and management vacancies worldwide.
  3. **Remotive API** (`https://remotive.com/api/remote-jobs`) - Live software engineering and data science postings.
  4. **Himalayas API** (`https://himalayas.app/jobs/api`) - Verified live tech, DevOps, and developer roles.
  5. **Arbeitnow API** (`https://www.arbeitnow.com/api/job-board-api`) - Open global job postings API.
- **Content**: 100% active, live job postings enriched with our NLP skill extraction engine.

## Preprocessing and Storage Policies
- Large archives and sensitive private resumes are excluded via `.gitignore`.
- Preprocessed datasets are saved to `data/processed/`.
