# app.py (Streamlit)
# Role-focused resume analyzer with PDF preview, highlighting, good/poor points, score, and downloadable report.

import io
import re
import base64
from typing import List, Dict, Tuple

# import numpy as np
# import pandas as pd
import streamlit as st
# from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
    
try:
    import PyPDF2  # PDF text extraction
except Exception:
    PyPDF2 = None
    

st.set_page_config(page_title="AI Resume Analyzer — Role Focus", page_icon="🧠", layout="wide")
st.title("🧠 AI Resume Analyzer — Role Focus")
st.write("Select a target role, upload your resume PDF (or paste text), and get role-specific highlights, strengths/weaknesses, and a score.")

# ------------------------------
# Role templates (add/edit freely)
# ------------------------------
ROLE_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "Machine Learning": {
        "core": [
            "python", "pytorch", "tensorflow", "scikit-learn", "sklearn", "numpy", "pandas",
            "xgboost", "lightgbm", "transformers", "nlp", "computer vision", "opencv",
        ],
        "nice": [
            "mlops", "docker", "kubernetes", "aws", "gcp", "azure", "fastapi", "streamlit", "gradio",
            "experiment tracking", "wandb", "mlflow", "feature engineering", "hyperparameter tuning",
        ],
    },
    "Computer Vision": {
        "core": [
            "opencv", "pytorch", "yolo", "segmentation", "detection", "tracking", "slam", "feature extraction",
            "cnn", "resnet", "transformers",
        ],
        "nice": [
            "cuda", "tensorrt", "onnx", "augmentation", "multi-view", "stereo", "calibration",
        ],
    },
    "Firmware / Embedded": {
        "core": [
            "c", "c++", "embedded", "microcontroller", "stm32", "arm", "freertos", "rtos", "spi", "i2c", "uart",
            "can", "gpio", "dma", "interrupts", "jtag", "swd",
        ],
        "nice": [
            "zephyr", "keil", "iar", "platformio", "oscilloscope", "logic analyzer",
        ],
    },
    "Software (Backend)": {
        "core": [
            "python", "java", "go", "node", "typescript", "api", "rest", "grpc", "postgres", "mysql", "redis",
            "docker", "kubernetes", "aws", "gcp", "ci/cd",
        ],
        "nice": [
            "kafka", "rabbitmq", "microservices", "terraform", "prometheus", "grafana",
        ],
    },
    "Control Systems / Robotics": {
        "core": [
            "control", "pid", "state-space", "kalman", "ros", "ros2", "trajectory", "simulink", "matlab",
        ],
        "nice": [
            "gazebo", "moveit", "rtabmap", "cartographer", "hardware-in-the-loop",
        ],
    },
    "Mechanical": {
        "core": [
            "solidworks", "cad", "fea", "ansys", "gd&t", "manufacturing", "dfm", "3d printing", "cnc",
        ],
        "nice": [
            "fusion 360", "abaqus", "catia", "composites",
        ],
    },
}

ACTION_VERBS = [
    "built", "designed", "implemented", "optimized", "deployed", "automated",
    "benchmarked", "refactored", "evaluated", "integrated",
]

METRIC_REGEX = re.compile(r"(\b\d+\.?\d*%\b|\b\d+\.?\d*\s*(ms|s|x|kb|mb|gb|requests|qps|fps)\b)", re.I)
CLEAN_RE = re.compile(r"[^a-z0-9\s+\-/#]", re.I)

# ------------------------------
# Helpers
# ------------------------------

def clean_text(t: str) -> str:
    t = (t or "").lower()
    t = CLEAN_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    # Try PyPDF2 first
    if PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            if text.strip():
                return text
        except Exception as e:
            st.warning(f"PyPDF2 parse failed, falling back to PyMuPDF: {e}")
    # Fallback: PyMuPDF
    if fitz is not None:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            return "\n".join((page.get_text() or "") for page in doc)
        except Exception as e:
            st.error(f"Failed to parse PDF with PyMuPDF: {e}")
            return ""
    st.error("No PDF parser installed. Add `PyPDF2` or `pymupdf` to requirements.txt.")
    return ""



def preview_pdf_pages(pdf_bytes: bytes, max_pages: int = 3, zoom: float = 2.0):
    """
    Render up to `max_pages` pages of the PDF to images using PyMuPDF.
    Falls back to a download/open link if PyMuPDF isn't installed.
    """
    if fitz is None:
        st.info("Install PyMuPDF to see inline previews: `pip install pymupdf`")
        # Fallback: open/download link (may still be blocked inline by Chrome, but works as a download)
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        href = f'<a download="resume.pdf" href="data:application/pdf;base64,{b64}" target="_blank">Open / download resume.pdf</a>'
        st.markdown(href, unsafe_allow_html=True)
        return

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total = len(doc)
        pages_to_show = min(max_pages, total)
        for i in range(pages_to_show):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            st.image(pix.tobytes("png"), caption=f"Page {i+1} of {total}", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render PDF preview: {e}")



def split_sentences(text: str) -> List[str]:
    # Simple sentence splitter
    parts = re.split(r"(?<=[\.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def find_skills(text: str, skills: List[str]) -> List[str]:
    text_l = f" {clean_text(text)} "
    present = []
    for s in skills:
        s_norm = s.lower()
        pat = r"(?<![a-z0-9])" + re.escape(s_norm) + r"(?![a-z0-9])"
        if re.search(pat, text_l):
            present.append(s)
    # unique preserve order
    seen = set()
    uniq = []
    for x in present:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def highlight_terms_html(text: str, terms: List[str]) -> str:
    html = text
    # sort by length desc to avoid nested partial matches
    for term in sorted(set(terms), key=len, reverse=True):
        if not term:
            continue
        pattern = re.compile(rf"(?i)(?<![a-z0-9])({re.escape(term)})((?![a-z0-9]))")
        html = pattern.sub(r"<mark>\1</mark>", html)
    return f"<div style='line-height:1.6'>{html}</div>"


def score_resume(core_hits: List[str], nice_hits: List[str], text: str) -> float:
    core_score = len(core_hits)
    nice_score = len(nice_hits)
    metrics_present = 1 if METRIC_REGEX.search(text) else 0
    verbs_present = 1 if any(re.search(rf"\b{re.escape(v)}\b", text, re.I) for v in ACTION_VERBS) else 0

    # Normalization by template sizes
    core_norm = core_score / max(1, 8)  # scaled to ~8 core skills target
    nice_norm = nice_score / max(1, 10)
    signal = (metrics_present + verbs_present) / 2

    score = 0.6 * core_norm + 0.25 * nice_norm + 0.15 * signal
    return float(max(0.0, min(1.0, score)))


def bullet_suggestions(missing_core: List[str], role: str, n: int = 3) -> List[str]:
    out = []
    for i, skill in enumerate(missing_core[:n]):
        v = ACTION_VERBS[i % len(ACTION_VERBS)].title()
        metric = ["by 20%", "by 2x", "by 150ms", "by 500K requests/day"][i % 4]
        out.append(f"{v} a {skill} pipeline for {role.lower()} that improved key metric {metric}.")
    return out

# ------------------------------
# Sidebar — role selection & upload
# ------------------------------
with st.sidebar:
    st.header("Target Role")
    role = st.selectbox("Choose a role", list(ROLE_TEMPLATES.keys()))
    st.caption("Pick the role you want your resume reviewed against.")

colL, colR = st.columns([1,1])

with colL:
    st.subheader("Resume Upload")
    resume_pdf = st.file_uploader("Upload resume (PDF)", type=["pdf"], key="resume")
    resume_text = ""
    if resume_pdf is not None:
        resume_bytes = resume_pdf.read()
        with st.expander("📄 Preview Resume", expanded=True):
            preview_pdf_pages(resume_bytes, max_pages=3, zoom=2.0)
        resume_text = extract_text_from_pdf(resume_bytes)
    else:
        resume_text = st.text_area("…or paste resume text", height=300, placeholder="Paste your resume if no PDF available…")

with colR:
    st.subheader("Role Keywords")
    st.write("**Core skills** are weighted higher than **Nice-to-have** skills.")
    st.write("**Core:** ", ", ".join(ROLE_TEMPLATES[role]["core"]))
    st.write("**Nice-to-have:** ", ", ".join(ROLE_TEMPLATES[role]["nice"]))

st.divider()

# ------------------------------
# Analysis
# ------------------------------
if not resume_text:
    st.info("Upload a resume PDF or paste text to analyze.")
    st.stop()

clean_resume = clean_text(resume_text)
core = ROLE_TEMPLATES[role]["core"]
nice = ROLE_TEMPLATES[role]["nice"]

core_hits = find_skills(clean_resume, core)
nice_hits = find_skills(clean_resume, nice)
missing_core = [s for s in core if s not in core_hits]
missing_nice = [s for s in nice if s not in nice_hits]

score = score_resume(core_hits, nice_hits, clean_resume)

# KPIs
k1, k2, k3 = st.columns(3)
k1.metric("Match Score", f"{score*100:.1f}%")
k2.metric("Core Skills Present", f"{len(core_hits)} / {len(core)}")
k3.metric("Nice-to-have Present", f"{len(nice_hits)} / {len(nice)}")

st.markdown("### Strengths & Areas to Improve")
left, right = st.columns(2)
with left:
    st.markdown("**Strengths**")
    strengths = []
    if len(core_hits) >= max(2, len(core)//3):
        strengths.append(f"Solid coverage of core {role} skills: {', '.join(core_hits[:8])}…")
    if METRIC_REGEX.search(resume_text):
        strengths.append("Uses metrics/quantitative impact (e.g., %/x/ms).")
    if any(re.search(rf"\b{re.escape(v)}\b", resume_text, re.I) for v in ACTION_VERBS):
        strengths.append("Strong action verbs in bullet points.")
    if not strengths:
        strengths.append("Clear structure; consider adding quantified impact and role keywords.")
    for s in strengths:
        st.write("• ", s)

with right:
    st.markdown("**Areas to Improve**")
    weaknesses = []
    if missing_core:
        weaknesses.append(f"Missing core skills: {', '.join(missing_core[:8])}…")
    if len(core_hits) < max(3, len(core)//4):
        weaknesses.append("Add projects/experience that directly showcase the core toolchain for this role.")
    if not METRIC_REGEX.search(resume_text):
        weaknesses.append("Add measurable results (%, x, ms, throughput).")
    for w in weaknesses:
        st.write("• ", w)

st.markdown("### Role-focused Highlights")
# Extract sentences containing any matched skill; highlight terms
sentences = split_sentences(resume_text)
highlight_terms = list(set(core_hits + nice_hits))
matched_snippets = [s for s in sentences if any(re.search(rf"\b{re.escape(t)}\b", s, re.I) for t in highlight_terms)]

if matched_snippets:
    for i, snip in enumerate(matched_snippets[:12], 1):
        st.markdown(f"**{i}.** ")
        st.markdown(highlight_terms_html(snip, highlight_terms), unsafe_allow_html=True)
else:
    st.write("No strong matches found — consider adding role-specific keywords to your bullets.")

# Downloadable report
# Build sections first (no backslashes inside f-expressions)
# Build plain-text report
strengths_txt    = "\n".join(f"- {s}" for s in strengths) or "- (none)"
weaknesses_txt   = "\n".join(f"- {w}" for w in weaknesses) or "- (none)"
suggestions_list = bullet_suggestions(missing_core, role, n=3)
suggestions_txt  = "\n".join(f"- {b}" for b in suggestions_list) or "- (none)"

report_txt = (
    f"Role-focused Resume Review — {role}\n"
    f"{'=' * (28 + len(role))}\n\n"
    f"Match Score: {score*100:.1f}%\n\n"
    f"Core skills present ({len(core_hits)}/{len(core)}): {', '.join(core_hits) or '—'}\n"
    f"Nice-to-have present ({len(nice_hits)}/{len(nice)}): {', '.join(nice_hits) or '—'}\n"
    f"Missing core: {', '.join(missing_core) or '—'}\n"
    f"Missing nice-to-have: {', '.join(missing_nice) or '—'}\n\n"
    f"Strengths:\n{strengths_txt}\n\n"
    f"Areas to Improve:\n{weaknesses_txt}\n\n"
    f"Suggested bullets:\n{suggestions_txt}\n"
)

st.download_button(
    label="⬇️ Download Report (.txt)",
    data=report_txt.encode("utf-8"),
    file_name=f"resume_review_{role.lower().replace(' ', '_')}.txt",
    mime="text/plain",
)