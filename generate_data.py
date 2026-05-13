"""
generate_data.py
----------------
Generates a realistic enterprise_ops_data.csv with 1,000 rows.
Hardcoded anomaly: EMEA region in November 2025 has a 300% spike
in Operational_Cost and a significantly elevated SLA_Breached rate.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Config ───────────────────────────────────────────────────────────────────
NUM_ROWS       = 1000
REGIONS        = ["NA", "EMEA", "APAC", "LATAM"]
REGION_WEIGHTS = [0.35, 0.25, 0.25, 0.15]   # realistic distribution

# Date range: last 6 months ending today (2026-05-13)
END_DATE   = datetime(2026, 5, 13)
START_DATE = END_DATE - timedelta(days=182)   # ~6 months


def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


# ── Base row generator ────────────────────────────────────────────────────────
def generate_row(txn_id: int) -> dict:
    region = np.random.choice(REGIONS, p=REGION_WEIGHTS)
    date   = random_date(START_DATE, END_DATE)

    # Baseline values — slightly vary by region
    base_costs = {"NA": 420, "EMEA": 510, "APAC": 380, "LATAM": 290}
    base_pt    = {"NA": 280, "EMEA": 310, "APAC": 260, "LATAM": 240}

    processing_time = max(50, int(np.random.normal(base_pt[region], 60)))
    operational_cost = max(50, round(np.random.normal(base_costs[region], 80), 2))

    # Baseline SLA breach: ~8% across the board
    sla_breached = np.random.rand() < 0.08

    return {
        "Transaction_ID":   f"TXN-{txn_id:05d}",
        "Region":           region,
        "Date":             date.strftime("%Y-%m-%d"),
        "Processing_Time_ms": processing_time,
        "SLA_Breached":     sla_breached,
        "Operational_Cost": operational_cost,
    }


# ── Generate base dataset ─────────────────────────────────────────────────────
rows = [generate_row(i + 1) for i in range(NUM_ROWS)]
df   = pd.DataFrame(rows)
df["Date"] = pd.to_datetime(df["Date"])

# ── Inject EMEA November 2025 Anomaly ────────────────────────────────────────
# Target: rows where Region == EMEA AND month == November 2025
emea_nov_mask = (
    (df["Region"] == "EMEA") &
    (df["Date"].dt.year == 2025) &
    (df["Date"].dt.month == 11)
)

n_affected = emea_nov_mask.sum()
print(f"[Anomaly Injection] Targeting {n_affected} EMEA rows in November 2025...")

if n_affected > 0:
    # 300% spike in Operational_Cost (multiply by 4 = original + 300%)
    df.loc[emea_nov_mask, "Operational_Cost"] = (
        df.loc[emea_nov_mask, "Operational_Cost"] * 4.0
    ).round(2)

    # Bump SLA breach rate from ~8% to ~72%
    df.loc[emea_nov_mask, "SLA_Breached"] = (
        np.random.rand(n_affected) < 0.72
    )

    # Also increase processing time by ~150% to reinforce the anomaly signal
    df.loc[emea_nov_mask, "Processing_Time_ms"] = (
        df.loc[emea_nov_mask, "Processing_Time_ms"] * 2.5
    ).astype(int)
else:
    # Edge case: if random sampling missed Nov 2025 EMEA, force-create 25 rows
    print("[WARNING] No EMEA rows fell in Nov 2025 — force-injecting 25 anomaly rows.")
    forced = []
    nov_start = datetime(2025, 11, 1)
    nov_end   = datetime(2025, 11, 30)
    for i in range(25):
        d = random_date(nov_start, nov_end)
        forced.append({
            "Transaction_ID":     f"TXN-A{i+1:04d}",
            "Region":             "EMEA",
            "Date":               d,
            "Processing_Time_ms": random.randint(600, 950),
            "SLA_Breached":       True,
            "Operational_Cost":   round(random.uniform(1800, 2400), 2),
        })
    df = pd.concat([df, pd.DataFrame(forced)], ignore_index=True)

# ── Finalise & Save ──────────────────────────────────────────────────────────
df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

output_path = "enterprise_ops_data.csv"
df.to_csv(output_path, index=False)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n✅ Dataset saved → {output_path}")
print(f"   Total rows  : {len(df)}")
print(f"   Date range  : {df['Date'].min()}  →  {df['Date'].max()}")
print(f"   Regions     : {df['Region'].value_counts().to_dict()}")
print(f"\n── Anomaly Check (EMEA Nov-2025) ──────────────────────────────")

df["Date"] = pd.to_datetime(df["Date"])
anomaly_check = df[
    (df["Region"] == "EMEA") &
    (df["Date"].dt.year == 2025) &
    (df["Date"].dt.month == 11)
]
normal_emea = df[
    (df["Region"] == "EMEA") &
    ~((df["Date"].dt.year == 2025) & (df["Date"].dt.month == 11))
]

if len(anomaly_check) > 0:
    print(f"   Anomaly rows        : {len(anomaly_check)}")
    print(f"   Avg Cost (Nov EMEA) : ${anomaly_check['Operational_Cost'].mean():.2f}")
    print(f"   Avg Cost (EMEA rest): ${normal_emea['Operational_Cost'].mean():.2f}")
    print(f"   SLA Breach (Nov)    : {anomaly_check['SLA_Breached'].mean()*100:.1f}%")
    print(f"   SLA Breach (rest)   : {normal_emea['SLA_Breached'].mean()*100:.1f}%")