"""
Skill extraction and taxonomy engine for ResumeMatch AI.
Extracts technical skills, soft skills, educational degrees, and certifications.
"""

import re
from typing import Dict, List, Set

# Comprehensive, industry-aligned skill taxonomy
TECHNICAL_SKILLS = {
    # Programming Languages
    "python", "java", "c++", "c#", "c", "javascript", "typescript", "php", "ruby", "go",
    "golang", "rust", "scala", "r", "kotlin", "swift", "dart", "perl", "bash", "shell",
    "sql", "pl/sql", "html", "html5", "css", "css3", "sass", "matlab",

    # Machine Learning & AI
    "machine learning", "deep learning", "artificial intelligence", "nlp", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "opencv",
    "spacy", "nltk", "hugging face", "transformers", "llm", "bert", "gpt", "xgboost", "lightgbm",
    "reinforcement learning", "neural networks", "cnn", "rnn", "lstm", "supervised learning",
    "unsupervised learning", "clustering", "k-means", "pca", "random forest", "decision trees",

    # Data Science & Analytics
    "data science", "data analysis", "data analytics", "data mining", "pandas", "numpy",
    "scipy", "matplotlib", "seaborn", "plotly", "power bi", "tableau", "excel", "advanced excel",
    "statistics", "statistical modeling", "hypothesis testing", "a/b testing", "big data",
    "spark", "apache spark", "pyspark", "hadoop", "hive", "kafka",

    # Web & Software Development
    "react", "react.js", "reactjs", "angular", "angularjs", "vue", "vue.js", "node.js", "nodejs",
    "express", "express.js", "django", "flask", "fastapi", "spring", "spring boot", "asp.net",
    ".net", "dotnet", "rest api", "restful api", "graphql", "microservices", "redux", "next.js",
    "bootstrap", "tailwind", "jquery", "full stack", "frontend", "backend",

    # Cloud & DevOps
    "aws", "amazon web services", "azure", "google cloud", "gcp", "docker", "kubernetes", "k8s",
    "ci/cd", "jenkins", "git", "github", "gitlab", "bitbucket", "linux", "unix", "ansible",
    "terraform", "cloud computing", "serverless", "devops", "monitoring", "prometheus", "grafana",

    # Databases
    "postgresql", "postgres", "mysql", "mongodb", "sqlite", "oracle", "redis", "cassandra",
    "dynamodb", "elasticsearch", "neo4j", "mariadb", "snowflake", "database management",

    # Cybersecurity & Networks
    "cybersecurity", "network security", "information security", "penetration testing", "ethical hacking",
    "firewall", "cryptography", "tcp/ip", "dns", "vpn", "siem",

    # Engineering, Design & Others
    "autocad", "solidworks", "catia", "plc", "scada", "matlab simulink", "embedded systems",
    "iot", "robotics", "agile", "scrum", "jira", "sdlc", "qa", "software testing", "selenium",
    "unit testing", "pytest", "postman"
}

SOFT_SKILLS = {
    "communication", "verbal communication", "written communication", "teamwork", "collaboration",
    "leadership", "problem solving", "critical thinking", "adaptability", "flexibility",
    "time management", "work ethic", "creativity", "interpersonal skills", "negotiation",
    "conflict resolution", "emotional intelligence", "decision making", "presentation skills",
    "project management", "organization", "multitasking", "active listening", "analytical skills"
}

EDUCATION_DEGREES = {
    "b.tech": "Bachelor of Technology",
    "btech": "Bachelor of Technology",
    "b.e": "Bachelor of Engineering",
    "be": "Bachelor of Engineering",
    "m.tech": "Master of Technology",
    "mtech": "Master of Technology",
    "m.e": "Master of Engineering",
    "b.sc": "Bachelor of Science",
    "bsc": "Bachelor of Science",
    "m.sc": "Master of Science",
    "msc": "Master of Science",
    "b.s": "Bachelor of Science",
    "m.s": "Master of Science",
    "bca": "Bachelor of Computer Applications",
    "mca": "Master of Computer Applications",
    "bba": "Bachelor of Business Administration",
    "mba": "Master of Business Administration",
    "ph.d": "Doctor of Philosophy",
    "phd": "Doctor of Philosophy",
    "diploma": "Diploma",
    "associate degree": "Associate Degree",
    "high school": "High School"
}

CERTIFICATIONS = [
    "aws certified", "azure certified", "gcp certified", "pmp", "scrum master", "csm",
    "cissp", "ceh", "comptia", "ccna", "ccnp", "oracle certified", "itil", "six sigma",
    "google certified", "tensorflow developer", "microsoft certified"
]

def extract_skills_from_text(text: str) -> Dict[str, List[str]]:
    """
    Extracts technical and soft skills, education, and certifications using matching against taxonomy.
    """
    if not text:
        return {
            "technical_skills": [],
            "soft_skills": [],
            "education": [],
            "certifications": [],
            "all_skills": []
        }

    # Normalize text for matching
    lower_text = " " + text.lower() + " "
    # Replace common punctuation with spaces to allow clean boundary checking
    cleaned_lower = re.sub(r"[,/|;•\(\)\n\r\t]+", " ", lower_text)

    found_tech: Set[str] = set()
    found_soft: Set[str] = set()
    found_edu: Set[str] = set()
    found_certs: Set[str] = set()

    # Match Technical Skills (Multi-word phrases first, then single words with word boundaries)
    for skill in sorted(TECHNICAL_SKILLS, key=len, reverse=True):
        pattern = r"(?:\b|_)" + re.escape(skill) + r"(?:\b|_)"
        if re.search(pattern, cleaned_lower):
            found_tech.add(skill)

    # Match Soft Skills
    for skill in sorted(SOFT_SKILLS, key=len, reverse=True):
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, cleaned_lower):
            found_soft.add(skill)

    # Match Education Degrees
    for degree_key, degree_name in EDUCATION_DEGREES.items():
        pattern = r"\b" + re.escape(degree_key) + r"\b"
        if re.search(pattern, cleaned_lower):
            found_edu.add(degree_name)

    # Match Certifications
    for cert in CERTIFICATIONS:
        if cert in cleaned_lower:
            found_certs.add(cert)

    tech_list = sorted(list(found_tech))
    soft_list = sorted(list(found_soft))
    edu_list = sorted(list(found_edu))
    cert_list = sorted(list(found_certs))

    return {
        "technical_skills": tech_list,
        "soft_skills": soft_list,
        "education": edu_list,
        "certifications": cert_list,
        "all_skills": sorted(list(set(tech_list + soft_list)))
    }

def extract_experience_years(text: str) -> float:
    """
    Extracts mentioned years of experience from text (e.g., '5+ years of experience', '3 years experience').
    """
    if not text:
        return 0.0

    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)(?:\s*of)?\s*(?:experience|exp)",
        r"(?:experience|exp)\s*(?:of)?\s*(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)",
    ]

    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            try:
                years = [float(m) for m in matches if float(m) < 40]
                if years:
                    return max(years)
            except Exception:
                continue
    return 0.0
