"""
Live job fetcher for ResumeMatch AI.
Fetches real-time, live job listings exclusively from verified 100% free public APIs:
1. Unstop (Public Campus & Corporate Opportunities API)
2. Jobicy (Public Remote & Tech Jobs API)
3. Remotive (Public Tech & Software Jobs API)
4. Himalayas (Public Engineering & Data Jobs API)
5. Arbeitnow (Public Global Job Board API)

Requires zero API keys, zero paid services, and zero login credentials.
"""

import sys
import urllib.request
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.skill_extractor import extract_skills_from_text
from src.database.connection import get_session
from src.database.models import JobRecord, JobSkill

JOBS_PATH = BASE_DIR / "data" / "processed" / "cleaned_jobs.csv"

def make_request(url: str, timeout: int = 10) -> Optional[Any]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"[API ERROR] {url}: {e}")
        return None

def fetch_unstop_live_jobs(limit: int = 30) -> List[Dict[str, Any]]:
    """
    Fetches live opportunities from Unstop's public API.
    """
    url = f"https://unstop.com/api/public/opportunity/search-result?opportunity=jobs&per_page={limit}"
    payload = make_request(url)
    jobs = []
    if payload and "data" in payload:
        items = payload.get("data", {}).get("data", [])
        for it in items:
            title = it.get("title") or "Corporate Role"
            org = it.get("organisation", {}).get("name") or "Unstop Hiring Partner"
            job_detail = it.get("jobDetail", {})
            locations = job_detail.get("locations", [])
            loc_str = ", ".join(locations) if locations else "India / Remote"
            public_url = f"https://unstop.com/{it.get('public_url')}" if it.get("public_url") else "https://unstop.com/jobs"

            desc_parts = [
                f"Position: {title} at {org}.",
                f"Location: {loc_str}.",
                f"Type: {job_detail.get('type', 'Full-time')}.",
                f"Category: {it.get('category', {}).get('name', 'Technology & Business')}.",
                f"Apply link: {public_url}"
            ]
            raw_desc = " ".join(desc_parts)

            jobs.append({
                "job_title": title,
                "company": org,
                "location": loc_str,
                "raw_description": raw_desc,
                "job_url": public_url,
                "source": "Unstop"
            })
    print(f"[UNSTOP] Fetched {len(jobs)} live jobs.")
    return jobs

def fetch_jobicy_live_jobs(limit: int = 30) -> List[Dict[str, Any]]:
    """
    Fetches live tech and corporate jobs from Jobicy's free public API.
    """
    url = f"https://jobicy.com/api/v2/remote-jobs?count={limit}"
    payload = make_request(url)
    jobs = []
    if payload and "jobs" in payload:
        for it in payload["jobs"]:
            title = it.get("jobTitle") or "Software Specialist"
            company = it.get("companyName") or "Global Tech"
            location = it.get("jobGeo") or "Remote / Anywhere"
            raw_html = it.get("jobDescription", "")
            clean_desc = re.sub(r"<[^>]+>", " ", raw_html)
            url_link = it.get("url") or "https://jobicy.com"

            jobs.append({
                "job_title": title,
                "company": company,
                "location": location,
                "raw_description": f"{title} at {company}. {clean_desc[:2500]}",
                "job_url": url_link,
                "source": "Jobicy"
            })
    print(f"[JOBICY] Fetched {len(jobs)} live jobs.")
    return jobs

def fetch_remotive_live_jobs(limit: int = 25) -> List[Dict[str, Any]]:
    """
    Fetches live software, data science, and engineering jobs from Remotive's free public API.
    """
    url = f"https://remotive.com/api/remote-jobs?limit={limit}"
    payload = make_request(url)
    jobs = []
    if payload and "jobs" in payload:
        for it in payload["jobs"][:limit]:
            title = it.get("title") or "Software Engineer"
            company = it.get("company_name") or "Remote Enterprise"
            location = it.get("candidate_required_location") or "Worldwide / Remote"
            raw_html = it.get("description", "")
            clean_desc = re.sub(r"<[^>]+>", " ", raw_html)
            url_link = it.get("url") or "https://remotive.com"

            jobs.append({
                "job_title": title,
                "company": company,
                "location": location,
                "raw_description": f"{title} at {company}. {clean_desc[:2500]}",
                "job_url": url_link,
                "source": "Remotive"
            })
    print(f"[REMOTIVE] Fetched {len(jobs)} live jobs.")
    return jobs

def fetch_himalayas_live_jobs(limit: int = 25) -> List[Dict[str, Any]]:
    """
    Fetches live engineering and tech jobs from Himalayas' free public API.
    """
    url = f"https://himalayas.app/jobs/api?limit={limit}"
    payload = make_request(url)
    jobs = []
    if payload and "jobs" in payload:
        for it in payload["jobs"][:limit]:
            title = it.get("title") or "Developer"
            company = it.get("companyName") or "Tech Startup"
            location = "Remote"
            raw_html = it.get("description", "")
            clean_desc = re.sub(r"<[^>]+>", " ", raw_html)
            slug = it.get("slug", "")
            url_link = f"https://himalayas.app/jobs/{slug}" if slug else "https://himalayas.app"

            jobs.append({
                "job_title": title,
                "company": company,
                "location": location,
                "raw_description": f"{title} at {company}. {clean_desc[:2500]}",
                "job_url": url_link,
                "source": "Himalayas"
            })
    print(f"[HIMALAYAS] Fetched {len(jobs)} live jobs.")
    return jobs

def fetch_arbeitnow_live_jobs(limit: int = 30) -> List[Dict[str, Any]]:
    """
    Fetches live jobs from Arbeitnow's free public API.
    """
    url = "https://www.arbeitnow.com/api/job-board-api"
    payload = make_request(url)
    jobs = []
    if payload and "data" in payload:
        for it in payload["data"][:limit]:
            title = it.get("title") or "Technical Specialist"
            company = it.get("company_name") or "Enterprise"
            location = it.get("location") or "Remote / Hybrid"
            raw_html = it.get("description", "")
            clean_desc = re.sub(r"<[^>]+>", " ", raw_html)
            url_link = it.get("url") or "https://www.arbeitnow.com"

            jobs.append({
                "job_title": title,
                "company": company,
                "location": location,
                "raw_description": f"{title} at {company}. {clean_desc[:2500]}",
                "job_url": url_link,
                "source": "Arbeitnow"
            })
    print(f"[ARBEITNOW] Fetched {len(jobs)} live jobs.")
    return jobs

def fetch_all_live_jobs(replace_existing: bool = True) -> int:
    """
    Fetches live jobs from Unstop, Jobicy, Remotive, Himalayas, and Arbeitnow.
    Enriches with skills taxonomy and populates the database and cleaned_jobs.csv.
    If replace_existing=True, cleans out static offline records so ONLY live jobs exist!
    """
    print("=== Fetching Real-Time Live Jobs from Free Public APIs ===")
    all_live = []
    all_live.extend(fetch_unstop_live_jobs(30))
    all_live.extend(fetch_jobicy_live_jobs(30))
    all_live.extend(fetch_remotive_live_jobs(25))
    all_live.extend(fetch_himalayas_live_jobs(25))
    all_live.extend(fetch_arbeitnow_live_jobs(30))

    if not all_live:
        print("[WARNING] Could not fetch live jobs.")
        return 0

    print(f"Total raw live postings fetched: {len(all_live)}")

    # Deduplicate by (title + company)
    seen = set()
    unique_live = []
    for j in all_live:
        key = j["job_title"].strip().lower() + "||" + j["company"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique_live.append(j)

    # Check existing jobs in database to avoid duplicate inserts
    session = get_session()
    if replace_existing:
        print("Clearing out old catalog to maintain ONLY verified live vacancies...")
        try:
            session.query(JobSkill).delete()
            session.query(JobRecord).delete()
            session.commit()
            existing_db_keys = set()
        except Exception as e:
            session.rollback()
            existing_db_keys = set()
    else:
        existing_records = session.query(JobRecord.job_title, JobRecord.company).all()
        existing_db_keys = {f"{r[0].strip().lower()}||{r[1].strip().lower()}" for r in existing_records}

    processed_rows = []
    new_jobs_added = 0

    for j in unique_live:
        key = j["job_title"].strip().lower() + "||" + j["company"].strip().lower()
        if not replace_existing and key in existing_db_keys:
            continue

        cleaned = clean_text(j["raw_description"])
        extracted = extract_skills_from_text(j["raw_description"])
        req_skills = ", ".join(extracted["all_skills"])

        # Create record without manual primary key; let DB autoincrement assign unique ID
        job_rec = JobRecord(
            job_title=j["job_title"],
            company=j["company"],
            location=j.get("location", "Remote"),
            raw_description=j["raw_description"],
            cleaned_description=cleaned,
            required_skills=req_skills
        )
        session.add(job_rec)
        session.flush()  # Populates job_rec.id automatically from database

        # Add normalized skills
        for s in extracted["all_skills"]:
            js = JobSkill(job_id=job_rec.id, skill_name=s, is_mandatory=True)
            session.add(js)

        row = {
            "job_id": job_rec.id,
            "job_title": j["job_title"],
            "company": j["company"],
            "location": j.get("location", "Remote"),
            "job_url": j.get("job_url", ""),
            "source": j.get("source", "Live API"),
            "raw_description": j["raw_description"],
            "cleaned_description": cleaned,
            "required_skills": req_skills,
            "technical_skills": ", ".join(extracted["technical_skills"]),
            "soft_skills": ", ".join(extracted["soft_skills"]),
            "skill_count": len(extracted["all_skills"])
        }
        processed_rows.append(row)
        new_jobs_added += 1

    session.commit()

    # Query all active database jobs to export to cleaned_jobs.csv
    all_db_jobs = session.query(JobRecord).all()
    csv_rows = []
    # Build lookup for source and url from processed rows and existing file
    meta_lookup = {}
    if JOBS_PATH.exists():
        try:
            old_df = pd.read_csv(JOBS_PATH)
            for _, r in old_df.iterrows():
                k = f"{str(r.get('job_title')).strip().lower()}||{str(r.get('company')).strip().lower()}"
                meta_lookup[k] = {"job_url": r.get("job_url", ""), "source": r.get("source", "Live API")}
        except Exception:
            pass

    for r in processed_rows:
        k = f"{r['job_title'].strip().lower()}||{r['company'].strip().lower()}"
        meta_lookup[k] = {"job_url": r["job_url"], "source": r["source"]}

    for db_j in all_db_jobs:
        k = f"{db_j.job_title.strip().lower()}||{db_j.company.strip().lower()}"
        meta = meta_lookup.get(k, {"job_url": "", "source": "Live API"})
        extracted_skills = [s.strip() for s in (db_j.required_skills or "").split(",") if s.strip()]
        csv_rows.append({
            "job_id": db_j.id,
            "job_title": db_j.job_title,
            "company": db_j.company,
            "location": db_j.location,
            "job_url": meta["job_url"],
            "source": meta["source"],
            "raw_description": db_j.raw_description,
            "cleaned_description": db_j.cleaned_description,
            "required_skills": db_j.required_skills,
            "technical_skills": db_j.required_skills,
            "soft_skills": "",
            "skill_count": len(extracted_skills)
        })

    session.close()

    # Save to cleaned_jobs.csv
    df_live = pd.DataFrame(csv_rows)
    df_live.to_csv(JOBS_PATH, index=False)
    print(f"[OK] Successfully updated catalog: {new_jobs_added} new jobs added. Total active: {len(df_live)}")
    return new_jobs_added

def sync_live_jobs_to_catalog(replace_existing: bool = False) -> int:
    """Alias for fetching and syncing all live jobs into the catalog."""
    return fetch_all_live_jobs(replace_existing=replace_existing)

if __name__ == "__main__":
    fetch_all_live_jobs(replace_existing=True)
