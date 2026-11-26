"""
Dataset acquisition and verification script for ResumeMatch AI.
Downloads genuine public datasets for resume classification, job recommendations, and regression.
"""

import os
import sys
import urllib.request
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Verified public dataset sources
DATASET_URLS = {
    # 1. Updated Resume Dataset (962 real resumes across 25 job categories)
    "resumes.csv": "https://raw.githubusercontent.com/AyushAggarwal1/Resume-Selector/master/UpdatedResumeDataSet.csv",

    # 2. Campus Placement / Candidate Features & Salary (Real benchmark dataset for Regression)
    "placement_salary.csv": "https://raw.githubusercontent.com/GauravSahani1417/Campus-Placement-Dataset-Kaggle-/master/Placement_Data_Full_Class.csv",

    # 3. Real Job Postings Dataset with Skills and Roles
    "job_postings.csv": "https://raw.githubusercontent.com/binoydutt/Resume-Job-Description-Matching/master/data.csv",
    
    # 4. Career Explorer Jobs
    "career_jobs.csv": "https://raw.githubusercontent.com/bunsenmcdubbs/career-explorer/master/job-data.csv"
}

def download_file(url: str, dest_path: Path) -> bool:
    print(f"Downloading from {url} to {dest_path.name}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        print(f"[OK] Downloaded {dest_path.name} ({len(data)} bytes)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return False

def verify_datasets():
    print("\n--- Verifying downloaded datasets ---")
    summary = {}
    for filename in DATASET_URLS.keys():
        filepath = RAW_DIR / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
                print(f"[VERIFIED] {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"   Columns: {list(df.columns[:6])}")
                summary[filename] = {"rows": df.shape[0], "cols": df.shape[1], "columns": list(df.columns)}
            except Exception as e:
                try:
                    df = pd.read_csv(filepath, encoding="latin-1", on_bad_lines="skip")
                    print(f"[VERIFIED (latin-1)] {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
                    print(f"   Columns: {list(df.columns[:6])}")
                    summary[filename] = {"rows": df.shape[0], "cols": df.shape[1], "columns": list(df.columns)}
                except Exception as ex:
                    print(f"[WARNING] Could not parse {filename}: {ex}")
        else:
            print(f"[MISSING] {filename} does not exist.")
    return summary

def main():
    print("=== ResumeMatch AI Dataset Acquisition ===")
    for filename, url in DATASET_URLS.items():
        dest = RAW_DIR / filename
        if not dest.exists():
            download_file(url, dest)
        else:
            print(f"[EXISTS] {filename} already downloaded.")
            
    summary = verify_datasets()
    return summary

if __name__ == "__main__":
    main()
