"""
UI helper functions and component renderers for ResumeMatch AI Streamlit application.
Core Design System: Modern Neo-Brutalism
Features:
- Bold typography with strong visual hierarchy (Space Grotesk & Plus Jakarta Sans).
- Thick, visible solid borders (2-3px #000000).
- High-contrast hard offset shadows (3px-7px 0px #000000, zero blur).
- Vibrant color accents: Cobalt Blue (#2563EB), Electric Yellow (#FFE600), Mint Green (#22C55E), Coral (#FF6B6B).
- Boxy cards with slightly rounded corners (8-14px).
- High-contrast buttons with tactile hover and pressed offset states.
- Zero glassmorphism, zero neumorphism, minimal subtle gradients.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

def render_html(html_str: str):
    """
    Safely renders HTML in Streamlit by stripping all leading whitespace per line.
    Prevents Python-Markdown from misinterpreting indented lines as code blocks.
    """
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)

def render_sidebar_html(html_str: str):
    """
    Safely renders HTML in Streamlit sidebar by stripping all leading whitespace per line.
    """
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines())
    st.sidebar.markdown(cleaned, unsafe_allow_html=True)

def inject_custom_css():
    """
    Injects the Modern Neo-Brutalism master stylesheet.
    Color tokens:
    - Canvas: #FDFBF7 (Clean Off-White)
    - Sidebar: #F5F0E6 (Warm Off-White Paper)
    - Ink: #000000 (Pure Black Borders & High-Contrast Typography)
    - Cobalt Blue: #2563EB
    - Electric Yellow: #FFE600
    - Mint Emerald: #22C55E
    - Coral / Rose: #FF6B6B
    - Card Surfaces: #FFFFFF with hard offset shadows (box-shadow: Xpx Xpx 0px #000000)
    """
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700;800&family=JetBrains+Mono:wght@500;700;800&display=swap');

    /* Global Canvas & Base Neo-Brutalist Layout */
    .stApp {
        background-color: #FDFBF7;
        color: #000000;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        letter-spacing: -0.015em;
    }

    /* Neo-Brutalist Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif !important;
        color: #000000 !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
    }

    p, span, label, div {
        color: #1E293B;
    }

    /* Neo-Brutalist Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F5F0E6 !important;
        border-right: 3px solid #000000 !important;
    }
    section[data-testid="stSidebar"] div.stRadio > label {
        display: none;
    }
    section[data-testid="stSidebar"] div.stRadio > div {
        gap: 8px;
    }
    section[data-testid="stSidebar"] div.stRadio > div > label {
        background: #FFFFFF;
        border: 2.5px solid #000000;
        box-shadow: 3px 3px 0px #000000;
        border-radius: 8px;
        padding: 9px 14px;
        color: #000000;
        font-weight: 700;
        font-size: 0.9rem;
        transition: all 0.12s ease;
        cursor: pointer;
    }
    section[data-testid="stSidebar"] div.stRadio > div > label:hover {
        background: #FFFBEB;
        transform: translate(-2px, -2px);
        box-shadow: 5px 5px 0px #000000;
        color: #000000;
    }
    section[data-testid="stSidebar"] div.stRadio > div > label[data-checked="true"],
    section[data-testid="stSidebar"] div.stRadio > div > label:has(input:checked) {
        background: #FFE600 !important;
        color: #000000 !important;
        border: 2.5px solid #000000 !important;
        box-shadow: 1px 1px 0px #000000 !important;
        transform: translate(2px, 2px) !important;
        font-weight: 800 !important;
    }

    /* Neo-Brutalist Metric Cards (Boxy with Hard Black Offset Shadow) */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 3px solid #000000;
        box-shadow: 5px 5px 0px #000000;
        border-radius: 12px;
        padding: 1.3rem 1.4rem;
        transition: all 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translate(-2px, -2px);
        box-shadow: 7px 7px 0px #000000;
    }
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-size: 0.84rem !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 2.25rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.04em;
        font-family: 'Space Grotesk', sans-serif;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.86rem !important;
        font-weight: 800;
        color: #2563EB !important;
    }

    /* Neo-Brutalist High-Contrast Buttons */
    .stButton>button[kind="primary"] {
        background: #2563EB;
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 0.95rem;
        border: 2.5px solid #000000;
        border-radius: 8px;
        padding: 0.65rem 1.8rem;
        box-shadow: 4px 4px 0px #000000;
        transition: all 0.12s ease;
    }
    .stButton>button[kind="primary"]:hover {
        background: #1D4ED8;
        color: #FFFFFF !important;
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0px #000000;
    }
    .stButton>button[kind="primary"]:active {
        transform: translate(2px, 2px);
        box-shadow: 1px 1px 0px #000000;
    }

    .stButton>button {
        background: #FFFFFF;
        color: #000000 !important;
        border-radius: 8px;
        font-weight: 800;
        font-size: 0.92rem;
        border: 2.5px solid #000000;
        padding: 0.6rem 1.4rem;
        box-shadow: 4px 4px 0px #000000;
        transition: all 0.12s ease;
    }
    .stButton>button:hover {
        background: #FFE600;
        color: #000000 !important;
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0px #000000;
    }
    .stButton>button:active {
        transform: translate(2px, 2px);
        box-shadow: 1px 1px 0px #000000;
    }

    /* Neo-Brutalist Input Fields & Textareas */
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 2.5px solid #000000 !important;
        box-shadow: 3px 3px 0px #000000 !important;
        border-radius: 8px !important;
        color: #000000 !important;
        font-weight: 600 !important;
        transition: all 0.12s ease;
    }
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        box-shadow: 5px 5px 0px #2563EB !important;
        border-color: #000000 !important;
    }
    textarea, input {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Neo-Brutalist File Uploader */
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 3px dashed #000000 !important;
        box-shadow: 5px 5px 0px #000000 !important;
        border-radius: 12px !important;
        padding: 2.2rem !important;
        transition: all 0.15s ease;
    }
    [data-testid="stFileUploader"] section:hover {
        background-color: #FFFDF8 !important;
        transform: translate(-2px, -2px);
        box-shadow: 7px 7px 0px #000000 !important;
    }

    /* Neo-Brutalist Sliders */
    div[data-testid="stSlider"] div[role="slider"] {
        background: #FFE600 !important;
        border: 2.5px solid #000000 !important;
        box-shadow: 2px 2px 0px #000000 !important;
        width: 20px !important;
        height: 20px !important;
        border-radius: 4px !important;
    }
    div[data-testid="stSlider"] > div > div > div {
        background-color: #E2E8F0 !important;
        border: 2px solid #000000 !important;
        border-radius: 6px !important;
        height: 10px !important;
    }

    /* Neo-Brutalist Checkbox */
    div[data-testid="stCheckbox"] label span[role="checkbox"] {
        background: #FFFFFF !important;
        border: 2.5px solid #000000 !important;
        box-shadow: 2px 2px 0px #000000 !important;
        border-radius: 4px !important;
    }
    div[data-testid="stCheckbox"] label span[role="checkbox"][aria-checked="true"] {
        background: #2563EB !important;
    }

    /* Neo-Brutalist Selectbox Dropdown Menu */
    div[data-baseweb="popover"], div[data-baseweb="menu"] {
        background: #FFFFFF !important;
        border: 2.5px solid #000000 !important;
        box-shadow: 5px 5px 0px #000000 !important;
        border-radius: 8px !important;
    }
    li[data-baseweb="menu-item"] {
        color: #000000 !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
        transition: all 0.1s ease;
    }
    li[data-baseweb="menu-item"]:hover {
        background: #FFE600 !important;
        color: #000000 !important;
    }

    /* Neo-Brutalist Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #FFFFFF;
        border: 2.5px solid #000000;
        box-shadow: 4px 4px 0px #000000;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #475569;
        font-weight: 800;
        font-size: 0.9rem;
        padding: 9px 20px;
        border: 2px solid transparent;
        background: transparent;
        transition: all 0.12s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #000000;
        background: #F1F5F9;
    }
    .stTabs [aria-selected="true"] {
        background: #FFE600 !important;
        border: 2px solid #000000 !important;
        box-shadow: 2px 2px 0px #000000 !important;
        color: #000000 !important;
        font-weight: 900 !important;
    }

    /* Neo-Brutalist Expanders */
    .streamlit-expanderHeader {
        background: #FFFFFF !important;
        border: 2.5px solid #000000 !important;
        box-shadow: 4px 4px 0px #000000 !important;
        border-radius: 10px !important;
        color: #000000 !important;
        font-weight: 800 !important;
        padding: 1rem 1.4rem !important;
        transition: all 0.12s ease;
    }
    .streamlit-expanderHeader:hover {
        background: #FFFBEB !important;
        transform: translate(-1px, -1px);
        box-shadow: 5px 5px 0px #000000 !important;
    }
    .streamlit-expanderContent {
        background: #FFFFFF !important;
        border: 2.5px solid #000000 !important;
        border-top: none !important;
        box-shadow: 4px 4px 0px #000000 !important;
        border-radius: 0 0 10px 10px !important;
        padding: 1.5rem !important;
    }

    /* Neo-Brutalist Progress Bars */
    .stProgress > div > div > div {
        background: #F1F5F9 !important;
        border: 2px solid #000000 !important;
        border-radius: 6px;
        height: 14px;
        box-shadow: 2px 2px 0px #000000;
    }
    .stProgress > div > div > div > div {
        background: #2563EB !important;
        border-radius: 3px;
    }

    /* Neo-Brutalist Tables & DataFrames */
    [data-testid="stDataFrame"] {
        background: #FFFFFF;
        border: 2.5px solid #000000;
        box-shadow: 5px 5px 0px #000000;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Neo-Brutalist Alerts */
    div[data-testid="stAlert"] {
        background: #FFFFFF !important;
        border: 2.5px solid #000000 !important;
        box-shadow: 4px 4px 0px #000000 !important;
        border-radius: 8px !important;
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Neo-Brutalist JSON Viewer */
    [data-testid="stJson"] {
        background: #FFFFFF !important;
        border: 2.5px solid #000000 !important;
        box-shadow: 4px 4px 0px #000000 !important;
        border-radius: 10px !important;
        padding: 1.2rem !important;
    }
    </style>
    """
    render_html(css)

def render_stepper(current_step: int = 1):
    """
    Renders an interactive Neo-Brutalist workflow stepper with chunky colorful stage blocks.
    """
    steps = [
        ("1", "Profile Ingest", "2. Resume Upload"),
        ("2", "NLP Extraction", "3. Resume Analysis"),
        ("3", "Role Prediction", "4. Job Role Prediction"),
        ("4", "Live Matches", "5. Job Recommendations"),
        ("5", "Skill Gap Plan", "6. Skill Gap Analysis"),
    ]

    html_parts = ['<div style="display: flex; align-items: center; justify-content: space-between; background: #FFFFFF; border: 2.5px solid #000000; box-shadow: 4px 4px 0px #000000; border-radius: 12px; padding: 0.85rem 1.4rem; margin-bottom: 2rem; overflow-x: auto;">']

    for i, (num, label, _) in enumerate(steps, 1):
        is_active = (i == current_step)
        is_done = (i < current_step)

        if is_active:
            box_bg = "#FFE600"
            box_color = "#000000"
            box_border = "border: 2.5px solid #000000; box-shadow: 2px 2px 0px #000000;"
            label_color = "#000000"
            label_weight = "900"
            badge = '<span style="display: inline-block; width: 7px; height: 7px; background: #000000; margin-right: 6px;"></span>'
        elif is_done:
            box_bg = "#DCFCE7"
            box_color = "#15803D"
            box_border = "border: 2px solid #000000; box-shadow: 1px 1px 0px #000000;"
            label_color = "#334155"
            label_weight = "700"
            badge = ""
        else:
            box_bg = "#F8FAFC"
            box_color = "#94A3B8"
            box_border = "border: 2px solid #CBD5E1;"
            label_color = "#94A3B8"
            label_weight = "600"
            badge = ""

        check_mark = "✓" if is_done else num
        step_html = f'''
        <div style="display: flex; align-items: center; gap: 0.65rem; white-space: nowrap;">
            <div style="width: 32px; height: 32px; border-radius: 6px; background: {box_bg}; {box_border} color: {box_color}; display: flex; align-items: center; justify-content: center; font-size: 0.88rem; font-weight: 900; font-family: 'Space Grotesk', sans-serif;">
                {check_mark}
            </div>
            <span style="color: {label_color}; font-size: 0.88rem; font-weight: {label_weight}; display: flex; align-items: center;">{badge}{label}</span>
        </div>
        '''
        html_parts.append(step_html)

        if i < len(steps):
            line_color = "#000000" if is_done else "#CBD5E1"
            html_parts.append(f'<div style="flex: 1; min-width: 24px; height: 2.5px; background: {line_color}; margin: 0 0.8rem;"></div>')

    html_parts.append('</div>')
    render_html("".join(html_parts))

def render_header(title: str, subtitle: str = "", pill_label: str = "AI Career Intelligence"):
    """
    Renders a bold Neo-Brutalist header with sticker pill and punchy typography.
    """
    html = f"""
    <div style="padding: 0.5rem 0 1.5rem 0; margin-bottom: 1.8rem; border-bottom: 2.5px solid #000000;">
        <div style="display: inline-flex; align-items: center; gap: 0.5rem; background: #FFE600; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; border-radius: 6px; padding: 0.25rem 0.85rem; margin-bottom: 0.75rem;">
            <span style="color: #000000; font-size: 0.8rem; font-weight: 900;">★</span>
            <span style="color: #000000; font-size: 0.76rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em;">{pill_label}</span>
        </div>
        <h1 style="color: #000000; margin: 0; font-size: 2.45rem; font-weight: 900; letter-spacing: -0.035em; line-height: 1.15; font-family: 'Space Grotesk', sans-serif;">
            {title}
        </h1>
        {f'<p style="color: #475569; margin-top: 0.45rem; font-size: 1.05rem; line-height: 1.5; font-weight: 500; max-width: 860px;">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    render_html(html)

def render_hero_banner():
    """
    Renders the Modern Neo-Brutalist AI Career Assistant Hero Panel.
    """
    html = """
    <div style="background: #FFFFFF; border: 3px solid #000000; box-shadow: 6px 6px 0px #000000; border-radius: 14px; padding: 2.4rem 2.6rem; margin-bottom: 2rem;">
        <div style="display: flex; gap: 0.75rem; align-items: center; margin-bottom: 1.3rem; flex-wrap: wrap;">
            <span style="background: #FFE600; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; color: #000000; font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em; padding: 0.35rem 0.85rem; border-radius: 6px;">
                ✦ AI-POWERED PLATFORM
            </span>
            <span style="background: #BBF7D0; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; color: #166534; font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em; padding: 0.35rem 0.85rem; border-radius: 6px;">
                ● 5 LIVE PUBLIC APIS
            </span>
            <span style="background: #FFDDD2; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; color: #9A3412; font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em; padding: 0.35rem 0.85rem; border-radius: 6px;">
                ⚡ UNITS 1 - 10 SYLLABUS COMPLETE
            </span>
        </div>

        <h1 style="color: #000000; font-size: 3rem; font-weight: 900; margin: 0 0 0.85rem 0; letter-spacing: -0.04em; line-height: 1.1; font-family: 'Space Grotesk', sans-serif;">
            ResumeMatch <span style="background: #2563EB; color: #FFFFFF; padding: 0.1rem 0.6rem; border: 2.5px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 6px;">AI</span>
        </h1>
        <p style="color: #334155; font-size: 1.18rem; max-width: 860px; line-height: 1.55; margin: 0 0 1.6rem 0; font-weight: 500;">
            A bold, intelligent career intelligence engine. Extracts 150+ competencies,
            predicts 25 professional domains with 100% test accuracy, and connects directly
            to live public vacancies with transparent, explainable recommendations.
        </p>

        <div style="display: flex; gap: 0.9rem; flex-wrap: wrap;">
            <div style="display: inline-flex; align-items: center; gap: 0.5rem; background: #FFFFFF; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 8px; padding: 0.55rem 1rem;">
                <span style="font-size: 1rem;">📄</span>
                <span style="color: #000000; font-size: 0.88rem; font-weight: 800;">Multi-Format Parser (PDF, Word, TXT)</span>
            </div>
            <div style="display: inline-flex; align-items: center; gap: 0.5rem; background: #FFFFFF; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 8px; padding: 0.55rem 1rem;">
                <span style="font-size: 1rem;">⚖️</span>
                <span style="color: #000000; font-size: 0.88rem; font-weight: 800;">60/40 Hybrid Cosine Engine</span>
            </div>
            <div style="display: inline-flex; align-items: center; gap: 0.5rem; background: #FFFFFF; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 8px; padding: 0.55rem 1rem;">
                <span style="font-size: 1rem;">🎯</span>
                <span style="color: #000000; font-size: 0.88rem; font-weight: 800;">High-ROI Skill Gap Analysis</span>
            </div>
        </div>
    </div>
    """
    render_html(html)

def render_badge(text: str, bg_color: str = "#FFE600", text_color: str = "#000000", border: str = None) -> str:
    """
    Renders a Neo-Brutalist sticker badge with thick border and hard offset shadow.
    """
    # Safe handling if a caller passes dot colors
    bg = bg_color if (bg_color and not bg_color.startswith("#818CF8")) else "#FFE600"
    txt = text_color if text_color else "#000000"

    return f"""<span style="background: {bg}; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; color: {txt}; padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 0.82rem; font-weight: 800; margin: 0.2rem 0.25rem; display: inline-flex; align-items: center; gap: 0.4rem; letter-spacing: 0.015em;">{text}</span>"""

def render_skill_badges(skills: List[str], category: str = "technical") -> str:
    """
    Renders high-contrast Neo-Brutalist sticker badges.
    """
    if not skills:
        return "<span style='color: #64748B; font-style: italic; font-size: 0.88rem;'>None identified</span>"

    if category == "technical":
        bg = "#DBEAFE"  # Soft Cobalt Tint
        txt = "#1E3A8A"
    elif category == "soft":
        bg = "#FEF08A"  # Electric Yellow Tint
        txt = "#854D0E"
    elif category == "education":
        bg = "#E0E7FF"  # Indigo Tint
        txt = "#3730A3"
    elif category == "missing":
        bg = "#FFE4E6"  # Coral Pink Tint
        txt = "#9F1239"
    else:
        bg = "#F1F5F9"
        txt = "#0F172A"

    badges = []
    for s in skills:
        badges.append(
            f"""<span style="background: {bg}; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; color: {txt}; padding: 0.32rem 0.85rem; border-radius: 6px; font-size: 0.84rem; font-weight: 800; margin: 0.22rem 0.28rem; display: inline-block;">{s}</span>"""
        )
    return " ".join(badges)

def render_candidate_card(p: Dict[str, Any]):
    """
    Renders a Neo-Brutalist Candidate Profile Card with chunky avatar and high-contrast stat boxes.
    """
    name = p.get("candidate_name") or "Candidate Profile"
    initials = "".join([part[0].upper() for part in name.split()[:2]]) if name else "CP"
    email = p.get("contact_info", {}).get("email") or "Not provided"
    phone = p.get("contact_info", {}).get("phone") or "Not provided"
    exp = p.get("experience_years", 0.0)
    skills_count = len(p.get("skills", {}).get("all_skills", []))
    edu = ", ".join(p.get("skills", {}).get("education", [])) or "Graduate / Technical Track"

    html = f"""
    <div style="background: #FFFFFF; border: 3px solid #000000; box-shadow: 6px 6px 0px #000000; border-radius: 14px; padding: 1.6rem 1.8rem; margin: 1.5rem 0;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.3rem; border-bottom: 2.5px solid #000000; padding-bottom: 1.2rem;">
            <div style="display: flex; align-items: center; gap: 1.1rem;">
                <div style="width: 56px; height: 56px; border-radius: 10px; background: #FFE600; border: 2.5px solid #000000; box-shadow: 3px 3px 0px #000000; color: #000000; font-size: 1.4rem; font-weight: 900; display: flex; align-items: center; justify-content: center; font-family: 'Space Grotesk', sans-serif;">
                    {initials}
                </div>
                <div>
                    <h3 style="color: #000000; margin: 0; font-size: 1.4rem; font-weight: 900; letter-spacing: -0.02em; font-family: 'Space Grotesk', sans-serif;">{name}</h3>
                    <p style="color: #475569; margin: 0.15rem 0 0 0; font-size: 0.88rem; font-weight: 600;">{email} &bull; {phone}</p>
                </div>
            </div>
            <div>
                <span style="background: #22C55E; color: #FFFFFF; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; padding: 0.38rem 0.95rem; border-radius: 6px; font-size: 0.82rem; font-weight: 800; display: inline-flex; align-items: center; gap: 0.45rem;">
                    ● ACTIVE IN MEMORY
                </span>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;">
            <div style="background: #F8FAFC; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 8px; padding: 1rem 1.2rem;">
                <div style="color: #475569; font-size: 0.78rem; font-weight: 800; text-transform: uppercase;">Experience</div>
                <div style="color: #000000; font-size: 1.35rem; font-weight: 900; margin-top: 0.2rem; font-family: 'Space Grotesk', sans-serif;">{exp} Years</div>
            </div>
            <div style="background: #F8FAFC; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 8px; padding: 1rem 1.2rem;">
                <div style="color: #475569; font-size: 0.78rem; font-weight: 800; text-transform: uppercase;">Skills Detected</div>
                <div style="color: #2563EB; font-size: 1.35rem; font-weight: 900; margin-top: 0.2rem; font-family: 'Space Grotesk', sans-serif;">{skills_count} Identified</div>
            </div>
            <div style="background: #F8FAFC; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 8px; padding: 1rem 1.2rem;">
                <div style="color: #475569; font-size: 0.78rem; font-weight: 800; text-transform: uppercase;">Education / Track</div>
                <div style="color: #000000; font-size: 0.95rem; font-weight: 800; margin-top: 0.2rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{edu}</div>
            </div>
        </div>
    </div>
    """
    render_html(html)

def render_job_card_header(rec: Dict[str, Any]):
    """
    Renders the Neo-Brutalist Job Card header with initial avatar, bold title, company,
    and hard offset match score sticker.
    """
    score = rec.get("overall_score", 0)
    source_name = rec.get("source", "Live API")
    company = rec.get("company", "Enterprise")
    comp_initial = company[0].upper() if company else "J"

    if score >= 40:
        badge_bg = "#22C55E"
        badge_text = "#FFFFFF"
        match_label = f"★ {score}% PRIME MATCH"
    elif score >= 25:
        badge_bg = "#FFE600"
        badge_text = "#000000"
        match_label = f"● {score}% MODERATE MATCH"
    else:
        badge_bg = "#F1F5F9"
        badge_text = "#000000"
        match_label = f"○ {score}% BASE MATCH"

    html = f"""
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.8rem; margin-bottom: 1.2rem;">
        <div style="display: flex; align-items: center; gap: 0.9rem;">
            <div style="width: 48px; height: 48px; border-radius: 8px; background: #2563EB; border: 2.5px solid #000000; box-shadow: 3px 3px 0px #000000; color: #FFFFFF; font-weight: 900; font-size: 1.3rem; display: flex; align-items: center; justify-content: center; font-family: 'Space Grotesk', sans-serif;">
                {comp_initial}
            </div>
            <div>
                <h4 style="color: #000000; margin: 0; font-size: 1.2rem; font-weight: 800; letter-spacing: -0.015em; font-family: 'Space Grotesk', sans-serif;">{rec.get('job_title', 'Job Vacancy')}</h4>
                <p style="color: #475569; margin: 0.15rem 0 0 0; font-size: 0.88rem; font-weight: 600;">{company} &bull; {rec.get('location', 'Remote / India')}</p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
            <span style="background: {badge_bg}; border: 2.5px solid #000000; box-shadow: 3px 3px 0px #000000; color: {badge_text}; padding: 0.38rem 0.95rem; border-radius: 6px; font-weight: 900; font-size: 0.84rem; display: inline-flex; align-items: center; gap: 0.45rem;">
                {match_label}
            </span>
            <span style="background: #FFFFFF; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; color: #000000; border-radius: 6px; font-size: 0.78rem; font-weight: 800; padding: 0.32rem 0.85rem; text-transform: uppercase;">
                {source_name}
            </span>
        </div>
    </div>
    """
    render_html(html)

def load_classification_metrics() -> Dict[str, Any]:
    metrics_file = REPORTS_DIR / "classification_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_regression_metrics() -> Dict[str, Any]:
    metrics_file = REPORTS_DIR / "regression_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_clustering_metrics() -> Dict[str, Any]:
    metrics_file = REPORTS_DIR / "clustering_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_association_rules() -> Dict[str, Any]:
    rules_file = REPORTS_DIR / "association_rules.json"
    if rules_file.exists():
        with open(rules_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_rl_metrics() -> Dict[str, Any]:
    rl_file = REPORTS_DIR / "rl_metrics.json"
    if rl_file.exists():
        with open(rl_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
