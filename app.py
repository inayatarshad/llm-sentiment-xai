"""
Streamlit UI — Explainable NLP Pipeline
Run: streamlit run app.py
"""

import streamlit as st
from nlp_pipeline import NLPPipeline

st.set_page_config(page_title="Explainable NLP", page_icon="🧠", layout="centered")

st.title("🧠 Explainable NLP Pipeline")
st.caption("BERT Sentiment · Gemini Summarization · LangChain")

if "pipeline" not in st.session_state:
    with st.spinner("Loading models..."):
        st.session_state.pipeline = NLPPipeline()

text = st.text_area("Paste your text here:", height=200, placeholder="Enter any text to analyze...")

if st.button("Analyze", type="primary") and text.strip():
    with st.spinner("Running pipeline..."):
        result = st.session_state.pipeline.run(text)

    s = result["sentiment"]
    m = result["summary"]

    # Sentiment
    col1, col2 = st.columns(2)
    color = {"positive": "green", "negative": "red", "neutral": "orange"}.get(s["polarity"], "blue")
    col1.metric("Sentiment", s["sentiment"])
    col2.metric("Confidence", s["confidence"])

    st.divider()

    # Summary
    st.subheader("📝 Explainable Summary")
    st.write(m.get("summary", ""))
    st.write(f"**Tone:** {m.get('tone', '')}")
    st.write(f"**Why:** {m.get('explanation', '')}")
    if m.get("themes"):
        st.write("**Themes:** " + " · ".join(f"`{t}`" for t in m["themes"]))