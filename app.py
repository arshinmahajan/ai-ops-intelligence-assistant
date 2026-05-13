"""
app.py
------
AI Operations Intelligence Assistant — Streamlit MVP
Architecture:
  1. Load CSV → Pandas DataFrame (cached)
  2. Top section: 3 static KPI metric cards
  3. Bottom section: Conversational chat interface
  4. Backend: LangChain Pandas DataFrame Agent (GPT-4o)
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()  # loads OPENAI_API_KEY from .env if present

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Ops Intelligence Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Header bar */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3b82f6;
    }
    .app-header h1 { color: #f1f5f9; font-size: 1.6rem; margin: 0; }
    .app-header p  { color: #94a3b8; font-size: 0.85rem; margin: 0.3rem 0 0; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(160deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .kpi-label { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase;
                 letter-spacing: 0.06em; margin-bottom: 0.4rem; }
    .kpi-value { color: #f1f5f9; font-size: 2rem; font-weight: 700; }
    .kpi-sub   { color: #64748b; font-size: 0.75rem; margin-top: 0.3rem; }

    /* Section labels */
    .section-label {
        color: #475569; font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 0.08em; margin: 1.5rem 0 0.6rem; font-weight: 600;
    }

    /* Chat container */
    .stChatMessage { border-radius: 10px !important; }

    /* Anomaly badge */
    .anomaly-badge {
        display: inline-block;
        background: #ef444420;
        color: #ef4444;
        border: 1px solid #ef444440;
        border-radius: 6px;
        padding: 0.15rem 0.6rem;
        font-size: 0.72rem;
        font-weight: 600;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "enterprise_ops_data.csv") -> pd.DataFrame:
    import os
    if not os.path.exists(path):
        import numpy as np
        from datetime import datetime, timedelta
        import random
        random.seed(42)
        np.random.seed(42)
        END_DATE = datetime(2026, 5, 13)
        START_DATE = END_DATE - timedelta(days=182)
        REGIONS = ["NA", "EMEA", "APAC", "LATAM"]
        REGION_WEIGHTS = [0.35, 0.25, 0.25, 0.15]
        def random_date(start, end):
            delta = end - start
            return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))
        rows = []
        for i in range(1000):
            region = np.random.choice(REGIONS, p=REGION_WEIGHTS)
            date = random_date(START_DATE, END_DATE)
            base_costs = {"NA": 420, "EMEA": 510, "APAC": 380, "LATAM": 290}
            base_pt = {"NA": 280, "EMEA": 310, "APAC": 260, "LATAM": 240}
            processing_time = max(50, int(np.random.normal(base_pt[region], 60)))
            operational_cost = max(50, round(np.random.normal(base_costs[region], 80), 2))
            sla_breached = np.random.rand() < 0.08
            rows.append({
                "Transaction_ID": f"TXN-{i+1:05d}",
                "Region": region,
                "Date": date.strftime("%Y-%m-%d"),
                "Processing_Time_ms": processing_time,
                "SLA_Breached": sla_breached,
                "Operational_Cost": operational_cost,
            })
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        emea_nov_mask = (
            (df["Region"] == "EMEA") &
            (df["Date"].dt.year == 2025) &
            (df["Date"].dt.month == 11)
        )
        df.loc[emea_nov_mask, "Operational_Cost"] = (
            df.loc[emea_nov_mask, "Operational_Cost"] * 4.0
        ).round(2)
        df.loc[emea_nov_mask, "SLA_Breached"] = (
            np.random.rand(emea_nov_mask.sum()) < 0.72
        )
        df.loc[emea_nov_mask, "Processing_Time_ms"] = (
            df.loc[emea_nov_mask, "Processing_Time_ms"] * 2.5
        ).astype(int)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        df.to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


# ── LLM + Agent factory (cached by resource) ─────────────────────────────────
@st.cache_resource
def build_agent(_df: pd.DataFrame):
    api_key = os.getenv("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=api_key,
        max_tokens=1024,
    )

    system_prompt = """You are a **Senior Operations Intelligence Assistant** for an enterprise B2B analytics platform.

Your responsibilities:
- Analyse the provided operations DataFrame and answer questions from operations leaders with precision.
- Always lead with the **business impact** before diving into data specifics.
- Use **bullet points** for multi-part answers. Keep responses concise (under 200 words unless the user explicitly asks for detail).
- When you detect anomalies or outliers, clearly state: the region, the time period, the metric affected, and the probable business consequence.
- Format currency values with $ prefix and commas. Format percentages with one decimal place.
- If a question is ambiguous, make a reasonable assumption and state it.
- Never say "I cannot answer this." Instead, use the available tools to explore the dataframe and give a best-effort answer.

DataFrame columns available:
- Transaction_ID: unique transaction identifier
- Region: one of NA, EMEA, APAC, LATAM
- Date: transaction date (datetime)
- Processing_Time_ms: time to process transaction in milliseconds
- SLA_Breached: boolean — whether the SLA was violated
- Operational_Cost: cost of the transaction in USD
"""

    agent = create_pandas_dataframe_agent(
    llm=llm,
    df=_df,
    verbose=False,
    prefix=system_prompt,
    allow_dangerous_code=True,
    handle_parsing_errors=True,
    max_iterations=8,
    agent_executor_kwargs={"handle_parsing_errors": True},
)
    return agent


# ── Load Data ─────────────────────────────────────────────────────────────────
try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    st.error(
        "⚠️ **`enterprise_ops_data.csv` not found.** "
        "Run `python generate_data.py` first, then refresh this page."
    )
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🧠 AI Operations Intelligence Assistant</h1>
    <p>Enterprise KPI monitoring · Anomaly detection · Conversational analytics</p>
</div>
""", unsafe_allow_html=True)


# ── KPI Metrics ───────────────────────────────────────────────────────────────
total_cost      = df["Operational_Cost"].sum()
sla_breach_rate = df["SLA_Breached"].mean() * 100
avg_proc_time   = df["Processing_Time_ms"].mean()

st.markdown('<div class="section-label">📊 Live KPI Overview</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Operational Cost</div>
        <div class="kpi-value">${total_cost:,.0f}</div>
        <div class="kpi-sub">Across all regions · Last 6 months</div>
    </div>""", unsafe_allow_html=True)

with col2:
    badge = '<span class="anomaly-badge">⚠ Anomaly Detected</span>' if sla_breach_rate > 10 else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">SLA Breach Rate {badge}</div>
        <div class="kpi-value" style="color:{'#ef4444' if sla_breach_rate > 10 else '#f1f5f9'}">
            {sla_breach_rate:.1f}%
        </div>
        <div class="kpi-sub">% of transactions breaching SLA</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg Processing Time</div>
        <div class="kpi-value">{avg_proc_time:.0f}<span style="font-size:1rem;color:#64748b"> ms</span></div>
        <div class="kpi-sub">Mean transaction latency</div>
    </div>""", unsafe_allow_html=True)


# ── Build Agent ───────────────────────────────────────────────────────────────
agent = build_agent(df)

if agent is None:
    st.warning(
        "🔑 **OpenAI API key not found.** "
        "Set `OPENAI_API_KEY` in a `.env` file or as an environment variable to enable the AI chat."
    )


# ── Chat Interface ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">💬 Ask Your Data Anything</div>', unsafe_allow_html=True)

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your **Senior Operations Intelligence Assistant**.\n\n"
                "I have full access to your enterprise operations dataset. Here are some things you can ask me:\n\n"
                "- *What are the total costs broken down by region?*\n"
                "- *Which region had the highest SLA breach rate last November?*\n"
                "- *Summarise the top 3 anomalies in the dataset.*\n"
                "- *What was the average processing time for EMEA in Q4 2025?*\n\n"
                "What would you like to explore?"
            ),
        }
    ]

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your operations data..."):

    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        if agent is None:
            response = (
                "⚠️ AI chat is disabled — OpenAI API key missing. "
                "Please add `OPENAI_API_KEY` to your `.env` file and restart the app."
            )
            st.markdown(response)
        else:
            with st.spinner("Analysing your data…"):
                try:
                    result = agent.invoke({"input": prompt})
                    output = result.get("output", "")
                    if "Could not parse LLM output:" in output:
                        response = output.replace("Could not parse LLM output:`", "").replace("`", "").strip()
                    else:
                        response = output if output else "I wasn't able to generate an answer. Please rephrase your question."
                except Exception as e:
                    response = (
                        f"⚠️ **Analysis error:** {str(e)}\n\n"
                        "Please try rephrasing your question. "
                        "For example: *'Show me SLA breach rates by region'*."
                    )
                st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


# ── Sidebar: Suggested Prompts ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💡 Suggested Prompts")
    prompts = [
        "Summarise KPIs by region",
        "Which month had the highest SLA breach rate?",
        "Detect anomalies in operational cost",
        "Compare EMEA vs NA processing time",
        "What happened to EMEA costs in November 2025?",
        "Which region has the worst SLA performance?",
        "Show cost trend month over month",
    ]
    for p in prompts:
        if st.button(p, use_container_width=True):
            # Inject as a user message (triggers rerun)
            st.session_state.messages.append({"role": "user", "content": p})
            st.rerun()

    st.markdown("---")
    st.markdown("### 📁 Dataset Info")
    st.markdown(f"**Rows:** {len(df):,}")
    st.markdown(f"**Date range:** {df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}")
    st.markdown(f"**Regions:** {', '.join(df['Region'].dropna().astype(str).unique())}")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()