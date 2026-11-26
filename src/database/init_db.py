"""
Database initialization and seeding script for ResumeMatch AI.
Creates tables and seeds job catalog with real public job postings from data/processed/cleaned_jobs.csv.
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.database.connection import get_engine, get_session, init_schema, ACTIVE_DB_TYPE
from src.database.models import Base, JobRecord, JobSkill

JOBS_PATH = BASE_DIR / "data" / "processed" / "cleaned_jobs.csv"

def seed_database():
    print("=== Initializing ResumeMatch AI Database Schema ===")
    init_schema()
    session = get_session()

    # Check existing jobs
    existing_count = session.query(JobRecord).count()
    if existing_count > 0:
        print(f"Database already contains {existing_count} jobs.")
        session.close()
        return

    print("Seeding job postings catalog...")
    if not JOBS_PATH.exists():
        print(f"[WARNING] {JOBS_PATH} not found. Seeding with standard sample roles...")
        sample_jobs = [
            {
                "job_title": "Machine Learning Engineer",
                "company": "DeepTech Innovations",
                "location": "Remote",
                "raw_description": "Looking for an ML Engineer with expertise in Python, PyTorch, Scikit-learn, Docker, and CI/CD.",
                "cleaned_description": "looking for an ml engineer with expertise in python pytorch scikit-learn docker and ci/cd",
                "required_skills": "python, pytorch, scikit-learn, docker, ci/cd, machine learning, deep learning"
            },
            {
                "job_title": "Full Stack Developer",
                "company": "CloudScale Systems",
                "location": "Bangalore / Hybrid",
                "raw_description": "Seeking Full Stack Engineer with React, Node.js, TypeScript, PostgreSQL, and AWS.",
                "cleaned_description": "seeking full stack engineer with react node.js typescript postgresql and aws",
                "required_skills": "react, node.js, typescript, postgresql, aws, javascript, rest api"
            },
            {
                "job_title": "Data Analyst",
                "company": "Insight Analytics",
                "location": "Hyderabad",
                "raw_description": "Data Analyst proficient in SQL, Python, Tableau, Excel, and statistical modeling.",
                "cleaned_description": "data analyst proficient in sql python tableau excel and statistical modeling",
                "required_skills": "sql, python, tableau, excel, statistics, data analysis, pandas"
            }
        ]
        df_jobs = pd.DataFrame(sample_jobs)
    else:
        df_jobs = pd.read_csv(JOBS_PATH)

    jobs_added = 0
    for _, row in df_jobs.iterrows():
        job = JobRecord(
            job_title=str(row.get("job_title", "Software Engineer")),
            company=str(row.get("company", "Tech Enterprise")),
            location=str(row.get("location", "Remote / Hybrid")),
            raw_description=str(row.get("raw_description", "")),
            cleaned_description=str(row.get("cleaned_description", "")),
            required_skills=str(row.get("required_skills", ""))
        )
        session.add(job)
        session.flush()

        # Seed normalized skills
        skills_str = str(row.get("required_skills", ""))
        for s in skills_str.split(","):
            s_clean = s.strip()
            if s_clean:
                js = JobSkill(job_id=job.id, skill_name=s_clean, is_mandatory=True)
                session.add(js)

        jobs_added += 1

    session.commit()
    session.close()
    print(f"[OK] Successfully initialized schema and seeded {jobs_added} job records into database.")

if __name__ == "__main__":
    seed_database()
