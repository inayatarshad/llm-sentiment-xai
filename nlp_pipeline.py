"""
Explainable NLP Pipeline
Sentiment Classification (BERT) + Text Summarization (LangChain + Groq/Gemini)
"""

import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
DEVICE          = 0 if torch.cuda.is_available() else -1   # GPU if available


# ── 1. Sentiment Classification (BERT) ───────────────────────────────────────
class SentimentAnalyzer:
    def __init__(self):
        print("🔄 Loading BERT sentiment model...")
        self.tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        self.model     = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
        self.pipeline  = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=DEVICE,
            truncation=True,
            max_length=512,
        )
        # Map star ratings → readable labels
        self.label_map = {
            "1 star":  "Very Negative",
            "2 stars": "Negative",
            "3 stars": "Neutral",
            "4 stars": "Positive",
            "5 stars": "Very Positive",
        }
        print("✅ BERT model ready.\n")

    def analyze(self, text: str) -> dict:
        result    = self.pipeline(text)[0]
        label     = self.label_map.get(result["label"], result["label"])
        score     = round(result["score"] * 100, 2)
        polarity  = "positive" if "Positive" in label else ("negative" if "Negative" in label else "neutral")
        return {
            "text":      text[:120] + "..." if len(text) > 120 else text,
            "sentiment": label,
            "polarity":  polarity,
            "confidence": f"{score}%",
            "raw_label": result["label"],
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        return [self.analyze(t) for t in texts]


# ── 2. Explainable Summarization (LangChain + Groq) ────────────────────────
class ExplainableSummarizer:
    SUMMARY_TEMPLATE = """You are an expert NLP assistant. Given the following text, provide:
1. A concise summary (2-3 sentences)
2. Key themes or topics (bullet points)
3. Overall tone assessment
4. Why you believe the text has this tone (brief explanation)

Text:
{text}

Respond in this exact format:
SUMMARY: <your summary>
THEMES: <comma-separated themes>
TONE: <tone>
EXPLANATION: <why this tone>"""

    def __init__(self):
        print("🔄 Loading Grok summarization model ...")
        self.llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
)
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=self.SUMMARY_TEMPLATE,
        )
        self.chain = self.prompt | self.llm | StrOutputParser()


    def summarize(self, text: str) -> dict:
        raw = self.chain.invoke({"text": text})
        parsed = {}
        for line in raw.strip().split("\n"):
            if line.startswith("SUMMARY:"):
                parsed["summary"] = line.replace("SUMMARY:", "").strip()
            elif line.startswith("THEMES:"):
                parsed["themes"] = [t.strip() for t in line.replace("THEMES:", "").split(",")]
            elif line.startswith("TONE:"):
                parsed["tone"] = line.replace("TONE:", "").strip()
            elif line.startswith("EXPLANATION:"):
                parsed["explanation"] = line.replace("EXPLANATION:", "").strip()
        return parsed


# ── 3. Full Pipeline ──────────────────────────────────────────────────────────
class NLPPipeline:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.summarizer         = ExplainableSummarizer()

    def run(self, text: str) -> dict:
        print("⚙️  Running full NLP pipeline...\n")
        sentiment = self.sentiment_analyzer.analyze(text)
        summary   = self.summarizer.summarize(text)
        return {
            "sentiment": sentiment,
            "summary":   summary,
        }

    def display(self, result: dict):
        s = result["sentiment"]
        m = result["summary"]

        print("=" * 60)
        print("📊 SENTIMENT ANALYSIS  (BERT)")
        print("=" * 60)
        print(f"  Sentiment   : {s['sentiment']}")
        print(f"  Polarity    : {s['polarity']}")
        print(f"  Confidence  : {s['confidence']}")
        print()
        print("=" * 60)
        print("📝 EXPLAINABLE SUMMARY  (Groq)")
        print("=" * 60)
        print(f"  Summary     : {m.get('summary', 'N/A')}")
        print(f"  Themes      : {', '.join(m.get('themes', []))}")
        print(f"  Tone        : {m.get('tone', 'N/A')}")
        print(f"  Explanation : {m.get('explanation', 'N/A')}")
        print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("   🧠 Explainable NLP Pipeline  |  BERT + Groq")
    print("=" * 60)

    pipeline_obj = NLPPipeline()

    print("\nPaste your text below (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "":
            if lines:
                break
        else:
            lines.append(line)
    text = " ".join(lines)

    result = pipeline_obj.run(text)
    pipeline_obj.display(result)


if __name__ == "__main__":
    main()