# NLP // CORE

An explainable sentiment-analysis playground that combines a multilingual BERT
classifier with Groq-hosted GPT-OSS summarization in a polished Streamlit interface.

Paste text in any supported language and receive:

- a five-level sentiment label with confidence and polarity;
- a concise LLM-generated summary;
- key themes and an overall tone assessment;
- a short, human-readable explanation of the tone.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-not%20specified-lightgrey)

## Demo

[Open the live NLP // CORE demo](https://llm-sentiment-xai.vercel.app)

## How it works

```text
User text
   ├── nlptown multilingual BERT ──> sentiment + confidence
   └── Groq GPT-OSS 120B ──────────> summary + themes + tone + explanation
                                      │
                                      └── Streamlit results dashboard
```

The classifier is
[`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment).
Its one-to-five-star output is mapped to labels from **Very Negative** through
**Very Positive**. The second branch uses LangChain and Groq's
`openai/gpt-oss-120b` model to produce the explainable analysis.

## Quick start

### Prerequisites

- Python 3.11 recommended
- A free [Groq API key](https://console.groq.com/keys)
- Around 2 GB of free disk space for Python packages and the model cache

### Install and run

```bash
git clone https://github.com/inayatarshad/llm-sentiment-xai.git
cd llm-sentiment-xai

python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and configure the API key:

```bash
pip install -r requirements-local.txt
cp .env.example .env
```

On Windows PowerShell, use `Copy-Item .env.example .env` instead of `cp`. Open
`.env`, replace the placeholder value, and start the app:

```bash
streamlit run streamlit_app.py
```

The app will be available at <http://localhost:8501>. The BERT model downloads
from Hugging Face on the first run, so initial startup is slower than subsequent
runs.

## Docker

```bash
docker build -t nlp-core .
docker run --rm -p 8501:8501 -e GROQ_API_KEY=your_key_here nlp-core
```

Then open <http://localhost:8501>.

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Lightweight WSGI entrypoint for the Vercel-hosted edition |
| `streamlit_app.py` | Full local Streamlit interface |
| `nlp_pipeline.py` | BERT classification and Groq summarization pipeline |
| `requirements-local.txt` | Local/Streamlit runtime dependencies |
| `Dockerfile` | Reproducible container deployment |
| `*_chain.py`, `*_parser.py` | Small LangChain learning examples |

## Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes | API key used by the Groq summarization model |
| `GROQ_MODEL` | No | Groq model ID; defaults to `openai/gpt-oss-120b` |

Never commit `.env` or Streamlit secrets. Both are excluded by `.gitignore`.

## Deployment

This app needs a long-running Python process and downloads a large
Transformers/PyTorch model. A container host such as Render, Railway, Fly.io, or
Hugging Face Spaces is therefore a better fit than a size-limited serverless
function.

For a Docker-capable host:

1. Connect this GitHub repository.
2. Select Docker deployment.
3. Add `GROQ_API_KEY` as a secret environment variable.
4. Expose port `8501` (or let the platform detect it from the Dockerfile).

## Security and limitations

- Text is sent to Groq for summarization. Do not submit secrets or regulated data.
- Sentiment is inferred from a review-rating model; nuanced or domain-specific
  language may be misclassified.
- Confidence is model confidence, not a calibrated statement of truth.
- Input is capped at 12,000 characters; BERT uses its first 512 tokens.
- Model-generated output is HTML-escaped before it is rendered in the UI.

## Contributing

Issues and pull requests are welcome. For substantial changes, please open an
issue first so the approach can be discussed.
