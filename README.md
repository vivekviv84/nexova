# GridSense AI — Real-Time Power Quality Intelligence Platform

> An Industry 5.0 power grid monitoring system that fuses signal processing, machine learning, and generative AI to detect, classify, and narrate electrical faults in real time.

---

## Overview

GridSense AI is a full-stack edge-intelligence application designed for industrial power grids. It continuously monitors power quality across multiple zones, classifies electrical anomalies using a trained CNN-LSTM classifier, and provides natural-language incident narration via a Groq-powered LLM engine (with a local expert-logic fallback for offline reliability).

This system is designed for operational reliability in environments where power quality directly impacts industrial output — aligning with the **Industry 5.0** vision of human-machine collaborative intelligence.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   Browser Client                 │
│         (WebSocket + REST API consumer)          │
└────────────────────┬─────────────────────────────┘
                     │ HTTP / WebSocket
┌────────────────────▼─────────────────────────────┐
│              FastAPI Application Layer            │
│         app/main.py · app/routes.py               │
│   - Pydantic-validated REST endpoints             │
│   - WebSocket push for real-time metrics          │
│   - Structured logging (nexova.api)               │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│              nexova_core Package                  │
│                                                   │
│  engine.py      — Grid simulation & state mgmt   │
│  features.py    — CWT scalogram extraction       │
│  classifier.py  — CNN-LSTM fault classifier      │
│  reporting.py   — Groq LLM + expert fallback     │
│  forensics.py   — Harmonic source attribution    │
│  config.py      — Environment & zone config      │
└──────────────────────────────────────────────────┘
```

---

## Key Technical Features

### 1. Signal Processing — Continuous Wavelet Transform (CWT)
Incoming voltage/current waveforms are decomposed using a **Continuous Wavelet Transform** (Morlet wavelet) to generate time-frequency scalograms. This approach captures transient features that standard FFT misses, enabling detection of:
- **Voltage Sags/Swells** (duration < 1 cycle)
- **Harmonic Distortion** (THD analysis up to 50th harmonic)
- **Impulsive Transients** (microsecond-scale events)

### 2. Machine Learning — CNN-LSTM Classifier
A hybrid **CNN-LSTM** architecture trained on the IEEE 1159 power quality event taxonomy:
- **CNN layers** extract spatial features from CWT scalograms
- **LSTM layers** capture temporal dependencies across the waveform sequence
- Output is a multi-class probability distribution over 8 fault categories

### 3. AI Narration — Groq LLM Engine
Incident narration is powered by **Groq** (`llama-3.3-70b-versatile`) for near-zero latency AI responses. A **rule-based expert fallback** system ensures the dashboard remains fully functional without an API key.

### 4. Harmonic Forensics — NMF Source Attribution
Non-negative Matrix Factorization (NMF) decomposes the harmonic spectrum to attribute liability percentages to connected loads — providing actionable compliance data for IEEE 519 reporting.

---


---

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```
> Get a free API key at [console.groq.com](https://console.groq.com). Keys are free and provide very high rate limits.

### Run
```bash
python3 run.py
```
Navigate to `http://localhost:8001`

---

## Project Structure

```
nexova/
├── run.py                  # Application entry point
├── .env                    # API keys (not committed)
├── .gitignore
├── requirements.txt
├── app/
│   ├── main.py             # FastAPI app + WebSocket engine
│   ├── routes.py           # Pydantic-validated API endpoints
│   └── static/             # Frontend (HTML, CSS, JS)
└── nexova_core/
    ├── config.py           # Zone definitions & environment config
    ├── engine.py           # Real-time grid simulation engine
    ├── features.py         # CWT waveform feature extraction
    ├── classifier.py       # CNN-LSTM fault classifier
    ├── reporting.py        # LLM narration + expert fallback
    ├── forensics.py        # NMF harmonic attribution
    └── math_utils.py       # Shared signal processing utilities
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Real-time | WebSockets (native FastAPI) |
| Signal Processing | NumPy, SciPy (CWT / FFT) |
| Machine Learning | Custom CNN-LSTM (NumPy) |
| AI Narration | Groq API (`llama-3.3-70b-versatile`) |
| Frontend | Vanilla JS, CSS3, WebSocket client |
| Config | python-dotenv, Pydantic v2 |

---

## Standards Compliance

- **IEEE 519-2022** — Harmonic distortion limits (THD reporting)
- **IEEE 1159-2019** — Power quality event classification taxonomy
- **IEC 61000-4-30** — Power quality measurement methods

---

*Built for the Hitachi Industry 5.0 Challenge — demonstrating real-time AI-human collaboration in critical infrastructure monitoring.*
