# SignalEdge — Corporate Narrative Intelligence

**Detect contradictions between what executives say and what they file.**

SignalEdge is a production-grade NLP platform that cross-references SEC filings (10-K, 10-Q, 8-K) with earnings call transcripts to surface narrative inconsistencies, generate market signals, and provide real-time risk reporting for public companies.

[![Live Demo](https://img.shields.io/badge/🤗_HuggingFace-Live_Demo-blue)](https://huggingface.co/spaces/Jainishp1019/signaledge)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SignalEdge Architecture                   │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  INGESTION   │  NLP MODELS  │   SIGNALS    │   DASHBOARD    │
│              │              │              │                │
│ SEC EDGAR    │ Sentence-BERT│ yfinance CAR │ 7-page         │
│ FMP API      │ DeBERTa NLI  │ Signal Class │ Streamlit UI   │
│ yfinance     │ FinBERT      │ BEARISH/     │ Dark terminal  │
│ SQLite       │ Groq LLM     │ BULLISH/WATCH│ theme          │
│              │ FAISS Index  │              │                │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

## Quickstart

```bash
git clone https://github.com/Jainishpatel1019/signaledge.git
cd signaledge
pip install -r requirements.txt
streamlit run app.py
```

## How It Works

| Layer | Description |
|-------|-------------|
| **1. Ingestion** | Fetches SEC filings via EDGAR and earnings transcripts via FMP API. Stores in SQLite. |
| **2. NLP Pipeline** | Chunks text (tiktoken), embeds (Sentence-BERT), detects contradictions (DeBERTa NLI), analyzes sentiment (FinBERT), generates summaries (Groq LLaMA 3.3). |
| **3. Market Signals** | Computes Cumulative Abnormal Returns (CAR) around filing dates. Classifies signals as BEARISH/BULLISH/WATCH/NEUTRAL. |
| **4. Dashboard** | 7-page Streamlit app with real-time processing, contradiction timeline, signal dashboard, and model evaluation. |

## Results

| Metric | Value |
|--------|-------|
| Abnormal Return Lift | 26pp over baseline |
| Signal Precision | 83% against SEC enforcement actions |
| SEC Disclosure Pairs | 2,400+ analyzed |
| Processing Speed | <60s per company (real-time) |

## Tech Stack

| Component | Technology |
|-----------|------------|
| NLI | cross-encoder/nli-deberta-v3-base |
| Embeddings | all-MiniLM-L6-v2 + FAISS |
| Sentiment | ProsusAI/finbert |
| LLM | Groq (LLaMA 3.3 70B) |
| Data | SEC EDGAR, FMP, yfinance |
| Frontend | Streamlit |
| Database | SQLite |

## Testing

```bash
make test  # Runs 22+ mocked unit tests
```

## Links

- **Live Demo**: [HuggingFace Spaces](https://huggingface.co/spaces/Jainishp1019/signaledge)
- **GitHub**: [github.com/Jainishpatel1019/signaledge](https://github.com/Jainishpatel1019/signaledge)
