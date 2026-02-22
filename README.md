# 🧠 NLP // CORE
> Explainable NLP Pipeline — BERT Sentiment + Groq LLaMA Summarization

## What it does
NLP // CORE is an explainable NLP pipeline that combines the power of BERT and large language models to analyze any text input. It uses a fine-tuned BERT model from HuggingFace to classify sentiment into granular categories (Very Negative to Very Positive), then passes the text to Groq's LLaMA 3.3 70B via LangChain to generate a concise summary, identify key themes, assess the tone, and explain the reasoning behind it — all in real time. The entire pipeline is wrapped in a custom-styled Streamlit UI with a cyberpunk neon aesthetic, making it both a functional NLP tool and a visually distinct demo of how traditional deep learning models and modern LLMs can work together in a single production-ready pipeline.
SO,
Paste any text → get sentiment classification + an explainable summary with themes, tone, and reasoning.

## Stack
| Component | Tool |
|-----------|------|
| Sentiment | BERT (`nlptown/bert-base-multilingual-uncased-sentiment`) |
| Summarization | Groq LLaMA 3.3 70B via LangChain |
| UI | Streamlit |

## Setup

```bash
pip install transformers torch langchain-core langchain-groq streamlit python-dotenv langchain-huggingface
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

## Run

```bash
streamlit run app.py
```

## Files
- `app.py` — Streamlit UI
- `nlp_pipeline.py` — BERT + Groq pipeline logic
<img width="983" height="554" alt="image" src="https://github.com/user-attachments/assets/b69a40c9-a8ab-4435-bb1e-3724736ed726" />
<img width="1259" height="722" alt="image" src="https://github.com/user-attachments/assets/8b196456-8d7b-4cf1-a28c-0e9c33231384" />
<img width="1315" height="673" alt="image" src="https://github.com/user-attachments/assets/2a874ba5-e983-44c8-93ac-0749700a1132" />
<img width="1206" height="670" alt="image" src="https://github.com/user-attachments/assets/b06e279a-72b1-462e-8e79-cd47b1f1e97e" />
<img width="1295" height="673" alt="image" src="https://github.com/user-attachments/assets/47fd70b9-4427-4960-b65d-6421768d8c82" />
