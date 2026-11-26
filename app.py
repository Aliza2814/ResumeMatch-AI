"""
ResumeMatch AI: An Intelligent Resume Analyzer and Job Recommendation System
Core Design System: Modern Neo-Brutalism
Features:
- Bold typography with strong visual hierarchy (Space Grotesk & Plus Jakarta Sans).
- Thick, visible solid borders (2-3px #000000).
- High-contrast hard offset shadows (box-shadow: 4px 4px 0px #000000, zero blur).
- Vibrant colors: Cobalt Blue (#2563EB), Electric Yellow (#FFE600), Mint (#22C55E), Coral (#FF6B6B).
- Boxy cards with slightly rounded corners (8-14px).
- High-contrast buttons with tactile hover and pressed states.
- Minimal gradients. Zero glassmorphism. Zero neumorphism.
"""

import sys
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Setup base directory and path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.resume_parser.parser import parse_resume
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.skill_extractor import extract_skills_from_text
from src.feature_engineering.vectorizer import TextVectorizer
from src.recommendation.engine import RecommendationEngine
from src.recommendation.live_jobs_fetcher import sync_live_jobs_to_catalog
from src.database.connection import get_session, ACTIVE_DB_TYPE, init_schema
from src.database.models import ResumeRecord, ExtractedSkill, AnalysisResult, RecommendationRecord, JobRecord
from src.utils.sample_resumes import SAMPLE_RESUMES
from src.utils.ui_helpers import (
    render_header, render_badge, render_skill_badges, inject_custom_css, 
    render_hero_banner, render_stepper, render_candidate_card, render_job_card_header,
    render_html, render_sidebar_html,
    load_classification_metrics, load_regression_metrics,
    load_clustering_metrics, load_association_rules, load_rl_metrics
)

# Initialize database schema if running on clean cloud environment
try:
    init_schema()
except Exception:
    pass

# Page configuration
st.set_page_config(
    page_title="ResumeMatch AI | Career Intelligence Platform",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Modern Neo-Brutalism master stylesheet
inject_custom_css()

# Initialize Session State
if "parsed_resume" not in st.session_state:
    st.session_state["parsed_resume"] = None
if "predicted_role" not in st.session_state:
    st.session_state["predicted_role"] = None
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = None

@st.cache_resource
def load_ml_models():
    models_dir = BASE_DIR / "models"
    try:
        classifier = joblib.load(models_dir / "best_classifier.joblib")
        vectorizer = TextVectorizer.load(models_dir / "tfidf_vectorizer.joblib")
        label_encoder = joblib.load(models_dir / "label_encoder.joblib")
        return classifier, vectorizer, label_encoder
    except Exception as e:
        return None, None, None

@st.cache_resource
def load_recommendation_engine():
    return RecommendationEngine()

# -------------------------------------------------------------
# SIDEBAR NAVIGATION (Modern Neo-Brutalist Layout)
# -------------------------------------------------------------
render_sidebar_html(
    """
    <div style="padding: 0.6rem 0 1.2rem 0; text-align: left;">
        <div style="display: inline-flex; align-items: center; gap: 0.45rem; background: #FFE600; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; border-radius: 6px; padding: 0.25rem 0.75rem; margin-bottom: 0.75rem;">
            <span style="color: #000000; font-size: 0.75rem; font-weight: 900;">★</span>
            <span style="color: #000000; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;">AI Career Platform</span>
        </div>
        <h2 style="margin: 0; color: #000000; font-weight: 900; font-size: 1.7rem; letter-spacing: -0.04em; line-height: 1.15; font-family: 'Space Grotesk', sans-serif;">
            ResumeMatch <span style="background: #2563EB; color: #FFFFFF; padding: 0.05rem 0.45rem; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; border-radius: 4px; font-size: 1.35rem;">AI</span>
        </h2>
        <p style="color: #475569; font-size: 0.84rem; margin: 0.35rem 0 0 0; font-weight: 700;">Career Intelligence & Job Matcher</p>
    </div>
    """
)

navigation = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Home / Overview",
        "2. Resume Upload",
        "3. Resume Analysis",
        "4. Job Role Prediction",
        "5. Job Recommendations",
        "6. Skill Gap Analysis",
        "7. Model Performance",
        "8. ML Laboratory",
        "9. About Project"
    ]
)

st.sidebar.markdown("---")

# Active candidate profile status card
if st.session_state["parsed_resume"]:
    candidate = st.session_state["parsed_resume"].get("candidate_name") or "Loaded Candidate"
    exp = st.session_state["parsed_resume"].get("experience_years", 0.0)
    skills_n = len(st.session_state["parsed_resume"].get("skills", {}).get("all_skills", []))
    render_sidebar_html(
        f"""
        <div style="background: #FFFFFF; border: 2.5px solid #000000; box-shadow: 4px 4px 0px #000000; border-radius: 10px; padding: 0.95rem 1rem; margin-bottom: 0.9rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem;">
                <span style="color: #000000; font-size: 0.74rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">● CANDIDATE ACTIVE</span>
                <span style="background: #22C55E; color: #FFFFFF; border: 1.5px solid #000000; font-size: 0.68rem; padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: 800;">ACTIVE</span>
            </div>
            <div style="color: #000000; font-size: 1.05rem; font-weight: 900; font-family: 'Space Grotesk', sans-serif; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{candidate}</div>
            <div style="color: #475569; font-size: 0.8rem; font-weight: 700; margin-top: 0.25rem;">{exp} yrs exp &bull; {skills_n} skills</div>
        </div>
        """
    )
    if st.sidebar.button("✕ Clear Active Profile", use_container_width=True):
        st.session_state["parsed_resume"] = None
        st.session_state["predicted_role"] = None
        st.session_state["recommendations"] = None
        st.rerun()

render_sidebar_html(
    f"""
    <div style="display: flex; align-items: center; justify-content: space-between; background: #FFFFFF; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; border-radius: 8px; padding: 0.5rem 0.85rem; margin-top: 0.8rem;">
        <span style="color: #475569; font-size: 0.74rem; font-weight: 800;">STORAGE ENGINE</span>
        <span style="background: #FFE600; color: #000000; border: 1.5px solid #000000; padding: 0.1rem 0.45rem; border-radius: 4px; font-size: 0.72rem; font-weight: 900; font-family: 'JetBrains Mono', monospace;">{ACTIVE_DB_TYPE.upper()}</span>
    </div>
    """
)

# -------------------------------------------------------------
# 1. HOME / OVERVIEW
# -------------------------------------------------------------
if navigation == "1. Home / Overview":
    render_hero_banner()

    # Neo-Brutalist Telemetry HUD Bar
    render_html(
        """
        <div style="display: flex; gap: 1rem; align-items: center; justify-content: space-between; background: #FFFFFF; border: 2.5px solid #000000; box-shadow: 4px 4px 0px #000000; border-radius: 12px; padding: 0.85rem 1.4rem; margin-bottom: 2rem; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="width: 10px; height: 10px; background: #22C55E; border: 1.5px solid #000000; display: inline-block;"></span>
                <span style="color: #000000; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; font-weight: 800;">TENSOR CORE: ONLINE</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #2563EB; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; font-weight: 800;">CLASSIFIER: LINEARSVC (100% ACC)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #000000; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; font-weight: 800;">API GATEWAY: 5 LIVE PIPELINES</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #D97706; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; font-weight: 800;">TAXONOMY: 150+ SKILL NODES</span>
            </div>
        </div>
        """
    )

    # Bento Stats Grid (Boxy Neo-Brutalist Metric Cards)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Trained Resumes", "962 Profiles", "25 Professional Domains")
    with col2:
        st.metric("Classifier Precision", "100.0%", "99.48% 5-Fold Cross-Val")
    with col3:
        st.metric("Live Vacancies", "144 Active", "5 Free Public APIs")
    with col4:
        st.metric("Syllabus Coverage", "Units 1 - 10", "100% Practical AI/ML")

    st.markdown("<br>", unsafe_allow_html=True)

    # Neo-Brutalist Architecture & Pipeline Showcase
    render_html(
        """
        <div style="background: #FFFFFF; border: 3px solid #000000; box-shadow: 6px 6px 0px #000000; border-radius: 14px; padding: 2rem 2.2rem; margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.4rem; flex-wrap: wrap; gap: 0.8rem;">
                <div>
                    <h3 style="color: #000000; font-size: 1.45rem; font-weight: 900; margin: 0; font-family: 'Space Grotesk', sans-serif;">Autonomous AI Processing Architecture</h3>
                    <p style="color: #475569; font-size: 0.92rem; font-weight: 600; margin: 0.25rem 0 0 0;">End-to-end data transformation pipeline from document bytes to transparent recommendations</p>
                </div>
                <div>
                    <span style="background: #FFE600; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; color: #000000; font-size: 0.78rem; font-weight: 900; padding: 0.35rem 0.85rem; border-radius: 6px;">
                        GOAL-BASED AGENT
                    </span>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;">
                <div style="background: #F8FAFC; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 10px; padding: 1.2rem;">
                    <div style="display: inline-flex; width: 34px; height: 34px; border-radius: 6px; background: #2563EB; color: #FFFFFF; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; font-weight: 900; font-size: 0.95rem; align-items: center; justify-content: center; margin-bottom: 0.6rem; font-family: 'Space Grotesk', sans-serif;">01</div>
                    <div style="color: #000000; font-weight: 800; font-size: 0.98rem;">Document Parsing</div>
                    <div style="color: #475569; font-size: 0.84rem; font-weight: 500; margin-top: 0.35rem; line-height: 1.45;">PyMuPDF & python-docx extract raw byte streams from PDF/Word.</div>
                </div>
                <div style="background: #F8FAFC; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 10px; padding: 1.2rem;">
                    <div style="display: inline-flex; width: 34px; height: 34px; border-radius: 6px; background: #FFE600; color: #000000; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; font-weight: 900; font-size: 0.95rem; align-items: center; justify-content: center; margin-bottom: 0.6rem; font-family: 'Space Grotesk', sans-serif;">02</div>
                    <div style="color: #000000; font-weight: 800; font-size: 0.98rem;">Token Shielding</div>
                    <div style="color: #475569; font-size: 0.84rem; font-weight: 500; margin-top: 0.35rem; line-height: 1.45;">Safeguards C++, C#, and .NET tokens during regex cleaning.</div>
                </div>
                <div style="background: #F8FAFC; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 10px; padding: 1.2rem;">
                    <div style="display: inline-flex; width: 34px; height: 34px; border-radius: 6px; background: #22C55E; color: #FFFFFF; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; font-weight: 900; font-size: 0.95rem; align-items: center; justify-content: center; margin-bottom: 0.6rem; font-family: 'Space Grotesk', sans-serif;">03</div>
                    <div style="color: #000000; font-weight: 800; font-size: 0.98rem;">Sublinear TF-IDF</div>
                    <div style="color: #475569; font-size: 0.84rem; font-weight: 500; margin-top: 0.35rem; line-height: 1.45;">Unigram & bigram vectors with Chi-square (2,500 terms).</div>
                </div>
                <div style="background: #F8FAFC; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 10px; padding: 1.2rem;">
                    <div style="display: inline-flex; width: 34px; height: 34px; border-radius: 6px; background: #FF6B6B; color: #FFFFFF; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; font-weight: 900; font-size: 0.95rem; align-items: center; justify-content: center; margin-bottom: 0.6rem; font-family: 'Space Grotesk', sans-serif;">04</div>
                    <div style="color: #000000; font-weight: 800; font-size: 0.98rem;">LinearSVC Prediction</div>
                    <div style="color: #475569; font-size: 0.84rem; font-weight: 500; margin-top: 0.35rem; line-height: 1.45;">Linear SVM classifies candidate domain into 25 career paths.</div>
                </div>
                <div style="background: #F8FAFC; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 10px; padding: 1.2rem;">
                    <div style="display: inline-flex; width: 34px; height: 34px; border-radius: 6px; background: #A855F7; color: #FFFFFF; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; font-weight: 900; font-size: 0.95rem; align-items: center; justify-content: center; margin-bottom: 0.6rem; font-family: 'Space Grotesk', sans-serif;">05</div>
                    <div style="color: #000000; font-weight: 800; font-size: 0.98rem;">Hybrid Match Engine</div>
                    <div style="color: #475569; font-size: 0.84rem; font-weight: 500; margin-top: 0.35rem; line-height: 1.45;">60% Skill Overlap + 40% TF-IDF Cosine on real live APIs.</div>
                </div>
            </div>
        </div>
        """
    )

    # Core Capabilities Neo-Brutalist Bento Cards
    st.markdown("### Core Platform Capabilities")
    c1, c2 = st.columns(2)
    with c1:
        render_html(
            """
            <div style="background: #FFFFFF; border: 2.5px solid #000000; box-shadow: 5px 5px 0px #000000; border-radius: 12px; padding: 1.6rem; height: 100%;">
                <h4 style="color: #000000; margin: 0 0 0.9rem 0; font-size: 1.25rem; font-weight: 900; font-family: 'Space Grotesk', sans-serif;">🧠 Natural Language Understanding</h4>
                <ul style="color: #334155; font-size: 0.95rem; line-height: 1.7; margin: 0; padding-left: 1.2rem; font-weight: 500;">
                    <li><b style="color: #000000;">Entity Taxonomy Engine:</b> Identifies 150+ technical and soft skills grouped across 7 key engineering categories.</li>
                    <li><b style="color: #000000;">Zero Proprietary API Dependency:</b> Runs 100% locally with zero external paid APIs, ensuring privacy and speed.</li>
                    <li><b style="color: #000000;">Supervised Classification Suite:</b> Compares 6 standard algorithms (SVM, Random Forest, Naive Bayes, LR, KNN, DT).</li>
                </ul>
            </div>
            """
        )
    with c2:
        render_html(
            """
            <div style="background: #FFFFFF; border: 2.5px solid #000000; box-shadow: 5px 5px 0px #000000; border-radius: 12px; padding: 1.6rem; height: 100%;">
                <h4 style="color: #000000; margin: 0 0 0.9rem 0; font-size: 1.25rem; font-weight: 900; font-family: 'Space Grotesk', sans-serif;">⚡ Live Real-Time Integration</h4>
                <ul style="color: #334155; font-size: 0.95rem; line-height: 1.7; margin: 0; padding-left: 1.2rem; font-weight: 500;">
                    <li><b style="color: #000000;">5 Free Public APIs:</b> Automatically ingests active vacancies from Unstop, Jobicy, Remotive, Himalayas, and Arbeitnow.</li>
                    <li><b style="color: #000000;">Transparent Compatibility Index:</b> Deterministic 60/40 math score with clear matched skills and skill gaps.</li>
                    <li><b style="color: #000000;">Full Syllabus ML Experiments:</b> Features Regression, Clustering (K-Means/DBSCAN), Association Rules, PCA, and RL.</li>
                </ul>
            </div>
            """
        )

# -------------------------------------------------------------
# 2. RESUME UPLOAD
# -------------------------------------------------------------
elif navigation == "2. Resume Upload":
    render_stepper(1)
    render_header("Resume Upload & Document Parsing", "Upload a candidate resume in PDF, DOCX, or TXT format or select an instant benchmark profile.", "Step 1 of 5: Document Ingestion")

    upload_tab, sample_tab = st.tabs(["📤 Upload Your Resume", "⚡ Instant Benchmark Profiles (Test Instantly)"])

    with upload_tab:
        uploaded_file = st.file_uploader("Select Resume File", type=["pdf", "docx", "txt"])
        save_to_db = st.checkbox("Persist candidate profile in local database record", value=True)

        if uploaded_file is not None:
            with st.spinner("Extracting and parsing document text with PyMuPDF..."):
                file_bytes = uploaded_file.read()
                result = parse_resume(file_bytes, filename=uploaded_file.name)

                if result["success"]:
                    st.session_state["parsed_resume"] = result
                    st.success(f"Successfully parsed `{uploaded_file.name}` ({len(result['raw_text'].split())} words extracted)")

                    if save_to_db:
                        try:
                            session = get_session()
                            rec = ResumeRecord(
                                filename=uploaded_file.name,
                                candidate_name=result.get("candidate_name") or "Anonymous Candidate",
                                email=result.get("contact_info", {}).get("email"),
                                phone=result.get("contact_info", {}).get("phone"),
                                experience_years=result.get("experience_years", 0.0),
                                raw_text=result.get("raw_text", ""),
                                cleaned_text=result.get("cleaned_text", "")
                            )
                            session.add(rec)
                            session.commit()
                            session.close()
                            st.info("Candidate profile saved to database record.")
                        except Exception as ex:
                            st.warning(f"Could not save to DB: {ex}")
                else:
                    st.error(f"Failed to parse resume: {result.get('error')}")

    with sample_tab:
        st.markdown("**Test the entire machine learning pipeline instantly with verified sample profiles:**")
        selected_sample = st.selectbox("Choose Candidate Profile Domain", list(SAMPLE_RESUMES.keys()))
        if st.button("⚡ Load Selected Profile", type="primary"):
            sample_text = SAMPLE_RESUMES[selected_sample]
            result = parse_resume(sample_text.encode("utf-8"), filename=f"{selected_sample}_Sample.txt")
            st.session_state["parsed_resume"] = result
            st.success(f"Loaded **{selected_sample}** sample profile!")
            st.rerun()

    # If a resume is currently loaded in session
    if st.session_state["parsed_resume"]:
        p = st.session_state["parsed_resume"]
        render_candidate_card(p)

        with st.expander("📄 View Normalized Cleaned Text (Token Shielded)"):
            st.text_area("Cleaned Text Stream", p.get("cleaned_text", "")[:2000] + "...", height=180)

# -------------------------------------------------------------
# 3. RESUME ANALYSIS
# -------------------------------------------------------------
elif navigation == "3. Resume Analysis":
    render_stepper(2)
    render_header("Comprehensive NLP Resume Analysis", "Detailed taxonomy extraction of technical competencies, frameworks, cloud tools, databases, and soft skills.", "Step 2 of 5: Entity Extraction")

    if not st.session_state["parsed_resume"]:
        st.warning("Please upload a resume or select a benchmark sample in 'Resume Upload' first.")
    else:
        p = st.session_state["parsed_resume"]
        skills = p.get("skills", {})
        tech = skills.get("technical_skills", [])
        soft = skills.get("soft_skills", [])
        edu = skills.get("education", [])
        certs = skills.get("certifications", [])

        render_candidate_card(p)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("#### 🛠️ Technical Languages & Core Competencies")
            render_html(render_skill_badges(tech, "technical"))

            st.markdown("<br>#### 🤝 Interpersonal & Professional Soft Skills", unsafe_allow_html=True)
            render_html(render_skill_badges(soft, "soft"))

            st.markdown("<br>#### 🎓 Academic Degrees & Certifications", unsafe_allow_html=True)
            all_edu = edu + certs
            render_html(render_skill_badges(all_edu, "education"))

        with col2:
            st.markdown("#### 📊 Candidate Profile Diagnostics")
            st.metric("Total Skills Identified", len(skills.get("all_skills", [])))
            st.metric("Technical / Soft Ratio", f"{len(tech)} / {len(soft)}")
            st.metric("Estimated Experience", f"{p.get('experience_years', 0.0)} Years")

            st.markdown("<br>#### 💡 Profile Assessment", unsafe_allow_html=True)
            if len(tech) >= 6:
                st.success("Strong technical depth: Broad coverage across programming, frameworks, and modern tools.")
            elif len(tech) >= 3:
                st.info("Solid foundational skill set with focused core proficiencies.")
            else:
                st.warning("Low technical density: Consider explicitly mentioning tools, libraries, and languages used.")

# -------------------------------------------------------------
# 4. JOB ROLE PREDICTION
# -------------------------------------------------------------
elif navigation == "4. Job Role Prediction":
    render_stepper(3)
    render_header("Job Role Prediction (Supervised Classification)", "Predicts candidate professional domain from 25 trained categories using our champion LinearSVC model.", "Step 3 of 5: Domain Classification")

    if not st.session_state["parsed_resume"]:
        st.warning("Please upload or load a resume in 'Resume Upload' first.")
    else:
        p = st.session_state["parsed_resume"]
        classifier, vectorizer, label_encoder = load_ml_models()

        if classifier is None:
            st.error("Model artifacts not found in models/. Please run training first.")
        else:
            cleaned_text = p.get("cleaned_text", "")
            X_input = vectorizer.transform([cleaned_text])
            
            pred_encoded = classifier.predict(X_input)[0]
            predicted_role = label_encoder.inverse_transform([pred_encoded])[0]
            st.session_state["predicted_role"] = predicted_role

            # Neo-Brutalist Focal Prediction Plate
            render_html(
                f"""
                <div style="background: #FFFFFF; border: 3px solid #000000; box-shadow: 7px 7px 0px #000000; border-radius: 16px; padding: 2.4rem 2.6rem; text-align: center; margin-bottom: 2rem;">
                    <div style="display: inline-flex; align-items: center; gap: 0.5rem; background: #FFE600; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; border-radius: 6px; padding: 0.3rem 0.9rem; margin-bottom: 0.9rem;">
                        <span style="color: #000000; font-size: 0.8rem; font-weight: 900;">★</span>
                        <span style="color: #000000; font-size: 0.78rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.06em;">Predicted Primary Domain</span>
                    </div>
                    <h1 style="color: #000000; margin: 0.2rem 0 0.8rem 0; font-size: 3rem; font-weight: 900; letter-spacing: -0.04em; font-family: 'Space Grotesk', sans-serif;">
                        {predicted_role}
                    </h1>
                    <div style="display: flex; justify-content: center; gap: 0.8rem; align-items: center; margin-top: 1.2rem; flex-wrap: wrap;">
                        <span style="background: #FFFFFF; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; padding: 0.4rem 0.95rem; border-radius: 6px; color: #000000; font-size: 0.88rem; font-weight: 800;">
                            Model: <b style="color: #2563EB;">Linear Support Vector Machine (LinearSVC)</b>
                        </span>
                        <span style="background: #DCFCE7; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; padding: 0.4rem 0.95rem; border-radius: 6px; color: #166534; font-size: 0.88rem; font-weight: 900;">
                            Test Accuracy: 100.0%
                        </span>
                        <span style="background: #FEF08A; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; padding: 0.4rem 0.95rem; border-radius: 6px; color: #854D0E; font-size: 0.88rem; font-weight: 900;">
                            5-Fold CV: 99.48%
                        </span>
                    </div>
                </div>
                """
            )

            # Top Category Alignment Scores with Neo-Brutalist Progress Bars
            st.markdown("#### Top Category Alignment Probabilities")
            try:
                decision_scores = classifier.decision_function(X_input)[0]
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                probs = exp_scores / exp_scores.sum()

                top_indices = np.argsort(probs)[::-1][:5]
                top_classes = label_encoder.inverse_transform(top_indices)
                top_probs = probs[top_indices]

                for cls_name, prob_val in zip(top_classes, top_probs):
                    c_name, c_prog = st.columns([2, 5])
                    with c_name:
                        st.markdown(f"**{cls_name}** ({prob_val * 100:.1f}%)")
                    with c_prog:
                        st.progress(float(prob_val))
            except Exception as ex:
                st.info(f"Predicted class: {predicted_role}")

# -------------------------------------------------------------
# 5. JOB RECOMMENDATIONS
# -------------------------------------------------------------
elif navigation == "5. Job Recommendations":
    render_stepper(4)
    render_header("Intelligent Job Recommendations", "Content-based matching (60% Skill Overlap + 40% TF-IDF Cosine Similarity) connected to 5 live public APIs.", "Step 4 of 5: Recommendation Engine")

    if not st.session_state["parsed_resume"]:
        st.warning("Please upload or load a resume in 'Resume Upload' first.")
    else:
        p = st.session_state["parsed_resume"]
        engine = load_recommendation_engine()

        rec_tab, custom_tab = st.tabs(["💼 Verified Live Vacancies (5 APIs)", "🎯 Custom Job Matcher (Naukri / LinkedIn / Unstop)"])

        with rec_tab:
            c_top, c_sync, c_src, c_filter = st.columns([2.5, 1.5, 1.2, 0.8])
            with c_top:
                st.markdown("All vacancies are fetched live from **Unstop**, **Jobicy**, **Remotive**, **Himalayas**, and **Arbeitnow**.")
            with c_sync:
                if st.button("🔄 Refresh Live APIs", help="Pulls latest live job postings from Unstop, Jobicy, Remotive, Himalayas, and Arbeitnow"):
                    with st.spinner("Fetching latest live jobs from free public APIs..."):
                        synced_count = sync_live_jobs_to_catalog()
                        if synced_count > 0:
                            st.success(f"Updated catalog with {synced_count} fresh vacancies!")
                            st.cache_resource.clear()
                            st.rerun()
                        else:
                            st.info("Live catalog is already up to date.")
            with c_src:
                source_filter = st.selectbox("API Source", ["All Sources", "Unstop", "Jobicy", "Remotive", "Himalayas", "Arbeitnow"])
            with c_filter:
                top_k = st.slider("Results", 3, 20, 6)

            all_recs = engine.recommend_jobs(p, top_n=50)
            if source_filter != "All Sources":
                recs = [r for r in all_recs if r.get("source", "").lower() == source_filter.lower()][:top_k]
            else:
                recs = all_recs[:top_k]
            st.session_state["recommendations"] = recs

            if not recs:
                st.info(f"No active postings matched this profile under source '{source_filter}'. Try selecting 'All Sources' or broadening your skill profile.")
            else:
                for i, rec in enumerate(recs, 1):
                    score = rec["overall_score"]
                    source_name = rec.get("source", "Live API")
                    job_link = rec.get("job_url", "")
                    company = rec.get("company", "Enterprise")

                    with st.expander(f"#{i} [{source_name.upper()}] {rec['job_title']} at {company} — {score}% Match", expanded=(i <= 2)):
                        # Neo-Brutalist Job Card Header
                        render_job_card_header(rec)

                        # Neo-Brutalist Metrics
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Overall Score", f"{score}%")
                        with c2:
                            st.metric("Direct Skill Overlap", f"{rec['skill_match_ratio']}%")
                        with c3:
                            st.metric("Semantic Cosine", f"{rec['text_cosine_similarity']}%")

                        render_html(
                            f"""
                            <div style="background: #FFFBEB; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 8px; padding: 0.9rem 1.2rem; color: #000000; font-size: 0.92rem; line-height: 1.55; margin: 1rem 0; font-weight: 600;">
                                <b>💡 Recommendation Rationale:</b> {rec['explanation']}
                            </div>
                            """
                        )

                        if job_link:
                            st.markdown(f"🔗 **Direct Application Link:** [{job_link}]({job_link})")

                        st.markdown("<br>**Matched Skills:**", unsafe_allow_html=True)
                        render_html(render_skill_badges(rec["matching_skills"], "technical"))

                        if rec["missing_skills"]:
                            st.markdown("<br>**Missing Job Requirements:**", unsafe_allow_html=True)
                            render_html(render_skill_badges(rec["missing_skills"], "missing"))

                        with st.expander("View Full Job Description"):
                            st.write(rec.get("raw_description", "No description available."))

        with custom_tab:
            st.markdown("#### Test Any Job Posting from Naukri.com, LinkedIn, or Unstop")
            st.caption("Copy and paste any real-world job posting text to evaluate your resume against it instantly.")

            col_j1, col_j2 = st.columns(2)
            with col_j1:
                custom_title = st.text_input("Job Title", "Senior Python / ML Engineer")
            with col_j2:
                custom_company = st.text_input("Company Name", "Tech Enterprise")

            custom_desc = st.text_area(
                "Paste Job Description / Requirements Text Here",
                """Looking for a Senior Python Developer with 3+ years experience. 
Key Requirements:
- Strong proficiency in Python, Django, Flask, and PostgreSQL.
- Experience with Docker, Kubernetes, AWS, and Git.
- Good knowledge of Machine Learning, Scikit-learn, and Pandas is an asset.
- Excellent communication and problem solving skills.""",
                height=170
            )

            if st.button("Analyze Compatibility with Pasted Job", type="primary"):
                with st.spinner("Calculating compatibility..."):
                    custom_record = {
                        "job_title": custom_title,
                        "company": custom_company,
                        "raw_description": custom_desc,
                        "cleaned_description": clean_text(custom_desc),
                        "required_skills": ", ".join(extract_skills_from_text(custom_desc)["all_skills"])
                    }
                    compat = engine.calculate_compatibility(p, custom_record)
                    c_score = compat["overall_score"]

                    st.markdown("---")
                    c_b1, c_b2, c_b3 = st.columns(3)
                    with c_b1:
                        st.metric("Overall Match", f"{c_score}%")
                    with c_b2:
                        st.metric("Direct Skill Overlap", f"{compat['skill_match_ratio']}%")
                    with c_b3:
                        st.metric("Semantic Cosine", f"{compat['text_cosine_similarity']}%")

                    render_html(
                        f"""
                        <div style="background: #FFFBEB; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 8px; padding: 0.9rem 1.2rem; color: #000000; font-size: 0.92rem; line-height: 1.55; margin: 1rem 0; font-weight: 600;">
                            <b>💡 Rationale:</b> {compat['explanation']}
                        </div>
                        """
                    )

                    st.markdown("#### Matched Skills")
                    render_html(render_skill_badges(compat["matching_skills"], "technical"))

                    if compat["missing_skills"]:
                        st.markdown("#### Missing Job Requirements (Skill Gap)")
                        render_html(render_skill_badges(compat["missing_skills"], "missing"))

# -------------------------------------------------------------
# 6. SKILL GAP ANALYSIS
# -------------------------------------------------------------
elif navigation == "6. Skill Gap Analysis":
    render_stepper(5)
    render_header("Skill Gap & Career Growth Analysis", "Actionable diagnostics identifying high-ROI skills missing from your profile across top matched vacancies.", "Step 5 of 5: Action Plan")

    if not st.session_state["parsed_resume"]:
        st.warning("Please upload or load a resume in 'Resume Upload' first.")
    else:
        p = st.session_state["parsed_resume"]
        engine = load_recommendation_engine()
        recs = st.session_state.get("recommendations") or engine.recommend_jobs(p, top_n=6)

        gaps_summary = engine.aggregate_skill_gaps(recs)
        high_priority = gaps_summary["high_priority_skills"]
        all_gaps = gaps_summary["all_missing_skills"]

        st.markdown("### 🎯 High-ROI Priority Skills to Learn")
        st.markdown("These skills are required across multiple top-recommended positions but are currently absent from your resume:")

        if high_priority:
            render_html(render_skill_badges(high_priority, "education"))
        else:
            st.success("Great job! Your profile covers the core requirements for your primary matched positions.")

        st.markdown("<br>### 📋 Complete Skill Demand Frequency Across Matched Roles", unsafe_allow_html=True)
        if all_gaps:
            df_gaps = pd.DataFrame(all_gaps, columns=["Missing Skill", "Job Demand Count"])
            c_table, c_chart = st.columns([2, 3])
            with c_table:
                st.dataframe(df_gaps, use_container_width=True, height=260)
            with c_chart:
                st.bar_chart(df_gaps.set_index("Missing Skill").head(8))
        else:
            st.info("No gaps identified.")

# -------------------------------------------------------------
# 7. MODEL PERFORMANCE
# -------------------------------------------------------------
elif navigation == "7. Model Performance":
    render_header("Academic Model Performance & Evaluation", "Empirical benchmark evaluation results across all supervised learning algorithms (Units 5, 6, 7).", "Units 5, 6, 7")

    tab_class, tab_reg = st.tabs(["📊 Classification Evaluation (Units 6 & 7)", "📈 Regression Evaluation (Unit 5)"])

    with tab_class:
        class_metrics = load_classification_metrics()
        if class_metrics:
            models_data = class_metrics.get("models", {})
            df_models = pd.DataFrame(models_data).T.reset_index().rename(columns={"index": "Algorithm"})
            df_models = df_models[[
                "Algorithm", "test_accuracy", "precision_macro", "recall_macro", "f1_weighted", "cv_mean", "bias_variance_diagnosis"
            ]].rename(columns={
                "test_accuracy": "Test Acc",
                "precision_macro": "Precision",
                "recall_macro": "Recall",
                "f1_weighted": "F1 Weighted",
                "cv_mean": "5-Fold CV Acc",
                "bias_variance_diagnosis": "Diagnosis"
            })

            st.markdown(f"**Best Performing Model:** `{class_metrics.get('best_model')}`")
            st.dataframe(df_models, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                cm_img = BASE_DIR / "reports" / "figures" / "classification_confusion_matrix.png"
                if cm_img.exists():
                    st.image(str(cm_img), caption="Figure 1: Confusion Matrix (Best Classifier: LinearSVC)", use_container_width=True)
            with col2:
                cmp_img = BASE_DIR / "reports" / "figures" / "model_comparison_accuracy.png"
                if cmp_img.exists():
                    st.image(str(cmp_img), caption="Figure 2: Test Accuracy vs 5-Fold Cross Validation", use_container_width=True)
        else:
            st.info("Run `python src/models/train_all.py` to generate classification metrics.")

    with tab_reg:
        reg_metrics = load_regression_metrics()
        if reg_metrics:
            df_reg = pd.DataFrame(reg_metrics).T.reset_index().rename(columns={"index": "Algorithm"})
            st.dataframe(df_reg[["Algorithm", "MAE", "RMSE", "R2", "features"]], use_container_width=True)

            reg_img = BASE_DIR / "reports" / "figures" / "regression_predictions.png"
            if reg_img.exists():
                st.image(str(reg_img), caption="Figure 3: Regression Actual vs Predicted Placement Salary", use_container_width=True)
        else:
            st.info("Run `python src/models/train_all.py` to generate regression metrics.")

# -------------------------------------------------------------
# 8. ML LABORATORY
# -------------------------------------------------------------
elif navigation == "8. ML Laboratory":
    render_header("Interactive Machine Learning Laboratory", "Hands-on academic laboratory demonstrating Units 5, 8, 9, and 10 in real time.", "Units 5, 8, 9, 10 Sandbox")

    lab_class, lab_reg, lab_clust, lab_assoc, lab_rl = st.tabs([
        "🏷️ Classification Sandbox",
        "📐 Regression Sandbox",
        "🌐 Clustering & PCA (Unit 8)",
        "🛒 Association Rules (Unit 9)",
        "🤖 Reinforcement Learning (Unit 10)"
    ])

    with lab_class:
        st.markdown("#### Test Any Custom Text Snippet")
        sample_input = st.text_area(
            "Input Resume / Experience Text",
            "Experienced Python developer with 3 years building web APIs with Django, Flask, and PostgreSQL. Familiar with Docker, Git, and Linux.",
            height=120
        )
        if st.button("Classify Text"):
            classifier, vectorizer, label_encoder = load_ml_models()
            vec = vectorizer.transform([sample_input])
            pred = classifier.predict(vec)[0]
            cat = label_encoder.inverse_transform([pred])[0]
            st.success(f"Predicted Role: **{cat}**")

    with lab_reg:
        st.markdown("#### Salary Prediction Simulation (Unit 5)")
        c1, c2, c3 = st.columns(3)
        with c1:
            deg_p = st.slider("Degree Percentage", 50.0, 95.0, 72.0)
            etest_p = st.slider("Employability Test %", 50.0, 99.0, 75.0)
        with c2:
            ssc_p = st.slider("10th Grade %", 50.0, 95.0, 70.0)
            hsc_p = st.slider("12th Grade %", 50.0, 95.0, 68.0)
        with c3:
            mba_p = st.slider("MBA %", 50.0, 90.0, 65.0)
            workex = st.selectbox("Work Experience", ["Yes", "No"])

        mlr_path = BASE_DIR / "models" / "regression_mlr.joblib"
        if mlr_path.exists():
            data = joblib.load(mlr_path)
            mlr = data["model"]
            scaler = data["scaler"]
            features = np.array([[ssc_p, hsc_p, deg_p, etest_p, mba_p, 1 if workex == "Yes" else 0]])
            pred_sal = mlr.predict(scaler.transform(features))[0]
            st.metric("Estimated Placement Salary", f"₹{pred_sal:,.2f}")

    with lab_clust:
        st.markdown("#### Unsupervised Clustering (Unit 8)")
        clust_data = load_clustering_metrics()
        st.json(clust_data)
        clust_img = BASE_DIR / "reports" / "figures" / "clustering_comparison.png"
        if clust_img.exists():
            st.image(str(clust_img), caption="Figure 4: K-Means vs Hierarchical vs DBSCAN (PCA 2D Projection)", use_container_width=True)

    with lab_assoc:
        st.markdown("#### Market Basket Analysis on Skills (Unit 9)")
        assoc_data = load_association_rules()
        st.markdown(f"**Total Transactions:** {assoc_data.get('total_transactions', 0)} | **Rules Mined:** {assoc_data.get('rules_count', 0)}")
        top_rules = assoc_data.get("top_rules", [])
        if top_rules:
            df_rules = pd.DataFrame(top_rules)
            df_rules["antecedents"] = df_rules["antecedents"].apply(lambda l: ", ".join(l))
            df_rules["consequents"] = df_rules["consequents"].apply(lambda l: ", ".join(l))
            st.dataframe(df_rules[["antecedents", "consequents", "support", "confidence", "lift"]], use_container_width=True)

        scree_img = BASE_DIR / "reports" / "figures" / "pca_scree_plot.png"
        if scree_img.exists():
            st.image(str(scree_img), caption="Figure 5: PCA Scree Plot (Explained Variance Ratio)", use_container_width=True)

    with lab_rl:
        st.markdown("#### Reinforcement Learning MDP Simulation (Unit 10)")
        rl_data = load_rl_metrics()
        st.markdown(f"**Episodes Trained:** {rl_data.get('episodes_trained')} | **Final 20-Ep Average Reward:** {rl_data.get('final_20_ep_reward_avg')}")
        rl_img = BASE_DIR / "reports" / "figures" / "rl_learning_curve.png"
        if rl_img.exists():
            st.image(str(rl_img), caption="Figure 6: Q-Learning Agent Learning Curve across Application Cycles", use_container_width=True)
        with st.expander("Learned Optimal Career Policy"):
            st.json(rl_data.get("optimal_policy", {}))

# -------------------------------------------------------------
# 9. ABOUT PROJECT
# -------------------------------------------------------------
elif navigation == "9. About Project":
    render_header("About ResumeMatch AI & Academic Syllabus Alignment", "System architecture, PEAS agent specification, and syllabus coverage matrix.", "Units 1 & 2 Foundations")

    st.markdown(
        """
        ### PEAS Representation (Unit 2: AI Agents)
        - **Performance Measure (P):** Recommendation accuracy, skill overlap precision, explainability score, low latency (<1.5s).
        - **Environment (E):** Unstructured PDF/DOCX resumes, competitive job market postings, catalog taxonomy.
        - **Actuators (A):** Streamlit UI recommendations, skill gap alerts, role classification badges, database storage.
        - **Sensors (S):** Document file stream parser, text tokenizers, feature extraction vectorizer.

        ### Syllabus Units Implementation Matrix
        | Syllabus Unit | Concept Covered | Practical Implementation |
        | :--- | :--- | :--- |
        | **Unit 1: Intro to AI** | AI Problem Formulation, Foundations | Architecture, Problem Statement, Scope |
        | **Unit 2: AI Approaches & Agents** | PEAS Representation, Ethics, Limits | PEAS Model, Explainability, Bias & Privacy controls |
        | **Unit 3: ML Tools** | NumPy, Pandas, Scikit-learn, Matplotlib | Complete data pipeline and reproducible experiments |
        | **Unit 4: Preprocessing & Features** | Text Cleaning, Stopwords, TF-IDF, Chi2 | `text_cleaner.py`, `skill_extractor.py`, `vectorizer.py` |
        | **Unit 5: Regression** | SLR, MLR, Polynomial, MAE/RMSE/R² | `regression.py` on real placement & salary dataset |
        | **Unit 6: Classification** | 6 Classifiers (SVM, RF, LR, NB, DT, KNN) | `classifier.py` trained on 25 job domains |
        | **Unit 7: Evaluation** | 5-Fold CV, Confusion Matrix, Bias-Variance | Full evaluation report & confusion matrix |
        | **Unit 8: Clustering** | K-Means, Hierarchical Ward, DBSCAN | `clustering.py`, Silhouette scores, 2D PCA plots |
        | **Unit 9: Association Rules & PCA** | Apriori, FP-Growth, PCA Dimensionality | `association_rules.py`, Scree plot, Lift metrics |
        | **Unit 10: Reinforcement Learning**| Agent, Environment, Reward, Q-Learning | `reinforcement_learning.py` Career MDP simulation |

        ### Ethical Considerations & Limitations
        1. **Fairness & Bias:** Resumes are stripped of demographic indicators (gender, race, age) to avoid discriminatory ranking.
        2. **Privacy:** Candidate data can be purged at any time; no raw files are committed to version control.
        3. **Explainability:** System explicitly presents deterministic skill gaps and TF-IDF similarity rather than unverified "black-box" scores.
        """
    )

    st.markdown("---")
    st.markdown("### 📥 Export Official Academic Project Report")
    c_rep1, c_rep2, c_rep3 = st.columns([1.5, 1.5, 2])
    docx_path = BASE_DIR / "reports" / "ResumeMatch_AI_Academic_Report.docx"
    html_path = BASE_DIR / "reports" / "ResumeMatch_AI_Academic_Report.html"

    with c_rep1:
        if docx_path.exists():
            with open(docx_path, "rb") as f:
                st.download_button(
                    label="📄 Download Report (.DOCX Word)",
                    data=f.read(),
                    file_name="ResumeMatch_AI_Academic_Report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type="primary"
                )
    with c_rep2:
        if html_path.exists():
            with open(html_path, "rb") as f:
                st.download_button(
                    label="🌐 Download Web Report (.HTML / PDF)",
                    data=f.read(),
                    file_name="ResumeMatch_AI_Academic_Report.html",
                    mime="text/html"
                )
    with c_rep3:
        st.caption("Tip: Open the HTML report in Chrome/Edge and press **Ctrl+P** -> 'Save as PDF' to produce a PDF report.")
