"""
Dataset preprocessing and preparation pipeline for ResumeMatch AI.
Cleans raw resumes, processes job postings, extracts skill features, and saves reproducible sets to data/processed/.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.skill_extractor import extract_skills_from_text, extract_experience_years
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def process_resumes():
    raw_path = RAW_DIR / "resumes.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    print("Processing resume dataset...")
    df = pd.read_csv(raw_path)
    # Check nulls and duplicates
    initial_count = len(df)
    df = df.dropna(subset=["Resume", "Category"])
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Clean resume text
    print("  Cleaning resume text...")
    df["Cleaned_Resume"] = df["Resume"].apply(lambda t: clean_text(t, remove_stopwords=False))
    
    # Extract skills and entities
    print("  Extracting skill features...")
    skills_data = df["Resume"].apply(extract_skills_from_text)
    df["Technical_Skills"] = skills_data.apply(lambda d: ", ".join(d["technical_skills"]))
    df["Soft_Skills"] = skills_data.apply(lambda d: ", ".join(d["soft_skills"]))
    df["Education"] = skills_data.apply(lambda d: ", ".join(d["education"]))
    df["All_Skills"] = skills_data.apply(lambda d: ", ".join(d["all_skills"]))
    df["Skill_Count"] = skills_data.apply(lambda d: len(d["all_skills"]))
    df["Experience_Years"] = df["Resume"].apply(extract_experience_years)

    out_path = PROCESSED_DIR / "cleaned_resumes.csv"
    df.to_csv(out_path, index=False)
    print(f"  [OK] Saved {len(df)} cleaned resumes to {out_path.name}")
    print(f"  Categories count: {df['Category'].nunique()} unique categories")
    return df

def process_job_postings():
    raw_path = RAW_DIR / "job_postings.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    print("Processing job postings dataset...")
    try:
        df = pd.read_csv(raw_path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(raw_path, encoding="latin-1")

    # Inspect columns
    # In binoydutt dataset: position, company, description, etc.
    cols = {c.lower(): c for c in df.columns}
    
    title_col = cols.get("position") or cols.get("job_title") or cols.get("title") or df.columns[1]
    desc_col = cols.get("description") or cols.get("job_description") or cols.get("job description")
    company_col = cols.get("company") or cols.get("company_name", "Company")

    # If description is separate or concatenated
    if desc_col is None:
        # Check text columns
        text_cols = [c for c in df.columns if df[c].dtype == object and df[c].str.len().mean() > 50]
        desc_col = text_cols[0] if text_cols else title_col

    jobs_clean = pd.DataFrame()
    jobs_clean["job_id"] = range(1, len(df) + 1)
    jobs_clean["job_title"] = df[title_col].fillna("Specialist")
    jobs_clean["company"] = df[company_col].fillna("Tech Enterprise") if company_col in df.columns else "Tech Enterprise"
    jobs_clean["raw_description"] = df[desc_col].fillna(df[title_col])
    jobs_clean["cleaned_description"] = jobs_clean["raw_description"].apply(lambda t: clean_text(str(t)))
    
    # Extract required skills
    print("  Extracting required skills from job postings...")
    skills_data = jobs_clean["raw_description"].apply(extract_skills_from_text)
    jobs_clean["required_skills"] = skills_data.apply(lambda d: ", ".join(d["all_skills"]))
    jobs_clean["technical_skills"] = skills_data.apply(lambda d: ", ".join(d["technical_skills"]))
    jobs_clean["soft_skills"] = skills_data.apply(lambda d: ", ".join(d["soft_skills"]))
    jobs_clean["skill_count"] = skills_data.apply(lambda d: len(d["all_skills"]))
    
    out_path = PROCESSED_DIR / "cleaned_jobs.csv"
    jobs_clean.to_csv(out_path, index=False)
    print(f"  [OK] Saved {len(jobs_clean)} cleaned jobs to {out_path.name}")
    return jobs_clean

def process_placement_salary():
    raw_path = RAW_DIR / "placement_salary.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    print("Processing placement salary dataset for regression...")
    df = pd.read_csv(raw_path)
    
    # Fill missing salary for unplaced students with 0 or drop for salary-regression subset
    placed_df = df[df["status"] == "Placed"].copy()
    placed_df["salary"] = pd.to_numeric(placed_df["salary"], errors="coerce")
    placed_df = placed_df.dropna(subset=["salary"])
    
    # Binary encode workex
    placed_df["workex_num"] = (placed_df["workex"].str.lower() == "yes").astype(int)

    out_path = PROCESSED_DIR / "cleaned_salary_regression.csv"
    placed_df.to_csv(out_path, index=False)
    print(f"  [OK] Saved {len(placed_df)} placed records with salary to {out_path.name}")
    return placed_df

def main():
    print("=== ResumeMatch AI Data Preprocessing Pipeline ===")
    resumes_df = process_resumes()
    jobs_df = process_job_postings()
    salary_df = process_placement_salary()
    print("=== Data Preprocessing Successfully Completed ===")

if __name__ == "__main__":
    main()
