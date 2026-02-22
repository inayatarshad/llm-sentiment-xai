"""
Streamlit UI — Explainable NLP Pipeline
Run: streamlit run app.py
"""

import streamlit as st
from nlp_pipeline import NLPPipeline

st.set_page_config(page_title="NLP//CORE", page_icon="⚡", layout="centered")

# ── Neon Cyberpunk CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

/* Background */
.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(0, 255, 140, 0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(180, 0, 255, 0.06) 0%, transparent 60%);
    font-family: 'Share Tech Mono', monospace;
}

/* Hide default streamlit elements */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 780px;}

/* Title */
h1 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 900 !important;
    font-size: 2.4rem !important;
    background: linear-gradient(90deg, #00ff8c, #b400ff, #00ff8c);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    letter-spacing: 0.05em;
    margin-bottom: 0 !important;
}

@keyframes shimmer {
    0% { background-position: 0% }
    100% { background-position: 200% }
}

/* Caption */
.stApp p, .stApp .stCaption {
    color: #555577 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Text area */
.stTextArea textarea {
    background: #0d0d18 !important;
    border: 1px solid #00ff8c33 !important;
    border-radius: 4px !important;
    color: #00ff8c !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9rem !important;
    caret-color: #00ff8c;
}
.stTextArea textarea:focus {
    border-color: #00ff8c !important;
    box-shadow: 0 0 12px rgba(0,255,140,0.2) !important;
}
.stTextArea textarea::placeholder { color: #333355 !important; }
.stTextArea label { color: #666688 !important; font-family: 'Share Tech Mono', monospace !important; }

/* Button */
.stButton > button {
    background: transparent !important;
    border: 1px solid #00ff8c !important;
    color: #00ff8c !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 2rem !important;
    border-radius: 2px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(0,255,140,0.08) !important;
    box-shadow: 0 0 20px rgba(0,255,140,0.3) !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #0d0d18 !important;
    border: 1px solid #b400ff44 !important;
    border-radius: 4px !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label {
    color: #b400ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e0e0ff !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.1rem !important;
}

/* Divider */
hr {
    border-color: #1a1a2e !important;
    margin: 1.5rem 0 !important;
}

/* Result boxes */
.result-box {
    background: #0d0d18;
    border: 1px solid #b400ff33;
    border-left: 3px solid #b400ff;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.88rem;
    color: #c0c0e0;
    line-height: 1.6;
}
.result-box .label {
    color: #b400ff;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    margin-bottom: 0.4rem;
    font-family: 'Orbitron', monospace;
}
.result-box.green {
    border-left-color: #00ff8c;
}
.result-box.green .label { color: #00ff8c; }

.theme-tag {
    display: inline-block;
    background: rgba(180,0,255,0.1);
    border: 1px solid #b400ff44;
    color: #b400ff;
    padding: 0.15rem 0.6rem;
    border-radius: 2px;
    font-size: 0.78rem;
    margin: 0.2rem 0.15rem;
    font-family: 'Share Tech Mono', monospace;
}

.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: #444466;
    margin: 1.5rem 0 0.8rem 0;
    border-bottom: 1px solid #1a1a2e;
    padding-bottom: 0.4rem;
}

/* Spinner */
.stSpinner > div { border-top-color: #00ff8c !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("NLP // CORE")
st.caption("▸ BERT SENTIMENT  ·  GROQ LLaMA  ·  LANGCHAIN")

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ── Load Pipeline ──────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    with st.spinner("INITIALIZING MODELS..."):
        st.session_state.pipeline = NLPPipeline()

# ── Input ──────────────────────────────────────────────────────────────────────
text = st.text_area(
    "[ INPUT TEXT ]",
    height=180,
    placeholder="paste text to analyze...",
    label_visibility="visible"
)

col_btn, col_space = st.columns([1, 3])
with col_btn:
    analyze = st.button("▶  ANALYZE", type="primary", use_container_width=True)

# ── Run & Display ──────────────────────────────────────────────────────────────
if analyze and text.strip():
    with st.spinner("PROCESSING..."):
        result = st.session_state.pipeline.run(text)

    s = result["sentiment"]
    m = result["summary"]

    # Sentiment metrics
    st.markdown("<div class='section-header'>// SENTIMENT ANALYSIS — BERT</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("CLASSIFICATION", s["sentiment"])
    col2.metric("POLARITY", s["polarity"].upper())
    col3.metric("CONFIDENCE", s["confidence"])

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Summary section
    st.markdown("<div class='section-header'>// EXPLAINABLE SUMMARY — GROQ LLaMA</div>", unsafe_allow_html=True)

    if m.get("summary"):
        st.markdown(f"""
        <div class='result-box green'>
            <div class='label'>SUMMARY</div>
            {m['summary']}
        </div>""", unsafe_allow_html=True)

    col_tone, col_exp = st.columns(2)
    with col_tone:
        if m.get("tone"):
            st.markdown(f"""
            <div class='result-box'>
                <div class='label'>TONE</div>
                {m['tone']}
            </div>""", unsafe_allow_html=True)
    with col_exp:
        if m.get("explanation"):
            st.markdown(f"""
            <div class='result-box'>
                <div class='label'>WHY</div>
                {m['explanation']}
            </div>""", unsafe_allow_html=True)

    if m.get("themes"):
        st.markdown("<div style='margin-top:0.8rem'>", unsafe_allow_html=True)
        tags = "".join(f"<span class='theme-tag'>{t}</span>" for t in m["themes"])
        st.markdown(f"""
        <div class='result-box'>
            <div class='label'>THEMES</div>
            <div style='margin-top:0.3rem'>{tags}</div>
        </div>""", unsafe_allow_html=True)

elif analyze and not text.strip():
    st.warning("⚠ No input detected.")