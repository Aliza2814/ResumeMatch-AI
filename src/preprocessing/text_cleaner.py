"""
Text cleaning and normalization pipeline for ResumeMatch AI.
Removes noise, URLs, special characters, and formats text for NLP & ML.
"""

import re
import string
import unicodedata
from typing import List, Optional

try:
    import nltk
    from nltk.corpus import stopwords
    try:
        STOPWORDS = set(stopwords.words("english"))
    except Exception:
        # Fallback common stopwords if NLTK data isn't loaded yet
        STOPWORDS = {
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "which", "this", "that", "these", "those", "then", "just", "so", "than",
            "such", "both", "through", "about", "for", "is", "of", "while", "during",
            "to", "from", "in", "out", "on", "off", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
            "don", "should", "now"
        }
except Exception:
    STOPWORDS = set()

def clean_text(text: str, remove_stopwords: bool = False, keep_alphanumeric_only: bool = True) -> str:
    """
    Cleans raw resume or job description text.
    - Normalizes unicode characters
    - Removes URLs, hyperlinks, and emails
    - Removes phone numbers and mentions
    - Strips non-printable symbols while preserving key programming tokens (like C++, C#, .NET)
    - Optionally removes English stopwords
    - Normalizes whitespaces
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize unicode (NFKD)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove emails
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)

    # Protect key programming tokens before punctuation stripping
    text = re.sub(r"(?:^|(?<=\s|[^\w]))c\+\+(?=[^\w]|$)", " token_cplusplus ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:^|(?<=\s|[^\w]))c#(?=[^\w]|$)", " token_csharp ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:^|(?<=\s|[^\w]))\.net(?=[^\w]|$)", " token_dotnet ", text, flags=re.IGNORECASE)

    # Remove mentions/hashtags
    text = re.sub(r"[@#]\w+", " ", text)

    # Remove phone numbers
    text = re.sub(r"\+?\d[\d -]{8,}\d", " ", text)

    # Remove non-ascii and special punctuation
    if keep_alphanumeric_only:
        text = re.sub(r"[^a-zA-Z0-9_\s]", " ", text)
    else:
        text = re.sub(r"[\r\n\t]+", " ", text)

    # Normalize whitespaces
    tokens = text.split()

    if remove_stopwords and STOPWORDS:
        tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 1]

    # Re-normalize tokens (token_cplusplus -> c++, token_csharp -> c#, token_dotnet -> .net)
    cleaned_tokens = []
    for t in tokens:
        if t == "token_cplusplus":
            cleaned_tokens.append("c++")
        elif t == "token_csharp":
            cleaned_tokens.append("c#")
        elif t == "token_dotnet":
            cleaned_tokens.append(".net")
        else:
            cleaned_tokens.append(t)

    return " ".join(cleaned_tokens)

def extract_contact_info(text: str) -> dict:
    """
    Extracts phone numbers, emails, and links from raw resume text.
    """
    if not text:
        return {"email": None, "phone": None, "links": []}

    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    phone_pattern = r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?"
    url_pattern = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"

    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    links = re.findall(url_pattern, text)

    email = emails[0] if emails else None
    
    # Format phone if found
    phone = None
    if phones:
        raw_p = phones[0]
        phone = "".join([c for c in str(raw_p) if c.isdigit()])
        if len(phone) < 7:
            # Try simpler regex for 10-digit standard Indian/US formats
            simpler = re.findall(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text)
            phone = simpler[0] if simpler else None

    return {
        "email": email,
        "phone": phone,
        "links": links
    }
