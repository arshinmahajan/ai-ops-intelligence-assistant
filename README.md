# 🚀 AI Operations Intelligence Assistant

> Eliminate manual reporting lag. Accelerate enterprise insight discovery.

## Executive Summary

Enterprise operations teams spend hours every week manually compiling KPI reports from raw data exports. This tool replaces that entirely — operations leaders simply ask plain-English questions and receive instant, data-backed answers in seconds.

**Built as a portfolio MVP to demonstrate:**
- AI-powered product thinking applied to a real B2B operations problem
- End-to-end ownership from data architecture to conversational UX
- LLM integration via LangChain connecting natural language to live datasets

---
## 📸 Live Demo

![Dashboard Overview](assets/AI%20Operations%20Assistant%20-%20Dashboard.png)

![Anomaly Detection](assets/AI%20operations%20assistant%20-%20anomaly.png)

![KPI Breakdown](assets/AI%20Operations%20Assistant%20-%20KPI.png)

## Core User Journeys

### Journey 1: KPI Summarisation
> *"What are total costs broken down by region?"*

The assistant instantly aggregates and compares regional performance — replacing a 2-hour manual pivot table exercise with a 5-second query.

### Journey 2: Anomaly Detection
> *"What happened to EMEA costs in November 2025?"*

The assistant identifies the root cause, quantifies the cost overrun, and frames the business consequence — reducing time-to-diagnose from days to seconds.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI / Frontend | Streamlit |
| AI Agent | LangChain + GPT-4o |
| Data Layer | Pandas |
| Language | Python 3.12 |
| LLM Provider | OpenAI API |

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/ai-ops-intelligence-assistant.git
cd ai-ops-intelligence-assistant

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI key
echo OPENAI_API_KEY=your-key-here > .env

# 5. Generate data
python generate_data.py

# 6. Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Sample Questions to Ask

- *What are total costs broken down by region?*
- *Which region had the highest SLA breach rate in November 2025?*
- *Summarise the top 3 anomalies in the dataset*
- *Compare EMEA vs NA average processing time*
- *What was the business impact of the November 2025 EMEA incident?*

---

## Project Context

Built as part of an AI Product & Platform portfolio to demonstrate practical experience designing and prototyping LLM-powered enterprise tools. Covers the full product lifecycle: problem framing, data architecture, AI agent design, and conversational UX.