"""Generate synthetic BankCo source-system extracts.

Deterministic (seeded). Produces a 20,000-customer pilot extract with a ~2% annual
mortgage churn rate whose drivers mirror what retail mortgage teams see in practice:
rate gap to market, proximity to fixed-term expiry, payoff-quote requests, competitor
comparison behaviour, complaints, low product depth, low LTV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ["North", "South", "East", "West", "Central"]
CALL_REASONS = ["payment_query", "rate_query", "payoff_quote", "complaint", "hardship", "general"]
MARKET_RATE = 5.60  # current market fixed rate, %


def generate(n_customers: int, seed: int, out_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    ids = np.arange(100001, 100001 + n_customers)

    # --- customers --------------------------------------------------------
    age = rng.integers(24, 72, n_customers)
    tenure_years = np.clip(rng.gamma(2.2, 3.0, n_customers), 0.2, 30).round(1)
    products_held = rng.choice([1, 2, 3, 4, 5], n_customers, p=[0.30, 0.34, 0.20, 0.11, 0.05])
    region = rng.choice(REGIONS, n_customers)
    income = np.clip(rng.lognormal(11.4, 0.4, n_customers), 40_000, 600_000).round(-2)
    customers = pd.DataFrame(
        {
            "customer_id": ids,
            "age": age,
            "tenure_years": tenure_years,
            "products_held": products_held,
            "region": region,
            "annual_income": income,
            "has_salary_account": rng.random(n_customers) < 0.55,
        }
    )

    # --- mortgages --------------------------------------------------------
    balance = np.clip(rng.lognormal(12.8, 0.45, n_customers), 60_000, 2_500_000).round(-3)
    ltv = np.clip(rng.normal(0.62, 0.17, n_customers), 0.10, 0.98).round(2)
    rate = np.clip(rng.normal(5.85, 0.55, n_customers), 4.2, 7.9).round(2)
    fixed = rng.random(n_customers) < 0.70
    months_to_fixed_expiry = np.where(fixed, rng.integers(0, 60, n_customers), -1)
    arrears_days = np.where(rng.random(n_customers) < 0.06, rng.integers(1, 120, n_customers), 0)
    remaining_term = rng.integers(36, 348, n_customers)
    mortgages = pd.DataFrame(
        {
            "customer_id": ids,
            "mortgage_id": [f"M{7_000_000 + i}" for i in range(n_customers)],
            "balance": balance,
            "interest_rate": rate,
            "is_fixed": fixed,
            "months_to_fixed_expiry": months_to_fixed_expiry,
            "ltv": ltv,
            "arrears_days": arrears_days,
            "remaining_term_months": remaining_term,
            "origination_channel": rng.choice(["branch", "broker", "digital"], n_customers, p=[0.45, 0.40, 0.15]),
        }
    )

    # --- latent churn propensity -----------------------------------------
    rate_gap = rate - MARKET_RATE
    near_expiry = ((months_to_fixed_expiry >= 0) & (months_to_fixed_expiry <= 6)).astype(float)
    intent = rng.normal(0, 1, n_customers)  # unobserved shopping-around intent
    logit = (
        -4.25
        + 1.6 * np.clip(rate_gap, -1, 2)
        + 1.3 * near_expiry
        - 0.35 * (products_held - 1)
        - 0.05 * tenure_years
        - 1.2 * (ltv - 0.6)
        + 0.9 * intent
        - 0.4 * customers["has_salary_account"].astype(float).to_numpy()
    )
    p = 1 / (1 + np.exp(-logit))
    churned = rng.random(n_customers) < p

    # --- digital interactions (12 monthly rows per customer) --------------
    months = pd.period_range("2025-01", periods=12, freq="M").astype(str)
    base_logins = rng.poisson(4, n_customers)
    rows = []
    for mi, month in enumerate(months):
        recency = (mi + 1) / 12  # signals intensify toward snapshot
        shop = np.clip(intent, 0, None) * recency
        rows.append(
            pd.DataFrame(
                {
                    "customer_id": ids,
                    "month": month,
                    "logins": rng.poisson(base_logins + 1 * shop),
                    "rate_page_views": rng.poisson(0.3 + 2.5 * shop + 1.5 * near_expiry * recency),
                    "payoff_quote_requests": rng.binomial(1, np.clip(0.01 + 0.25 * shop * churned, 0, 0.9)),
                    "competitor_compare_clicks": rng.poisson(0.1 + 1.8 * shop * (0.5 + churned)),
                }
            )
        )
    digital = pd.concat(rows, ignore_index=True)

    # --- call centre ------------------------------------------------------
    n_calls = rng.poisson(0.9 + 1.5 * np.clip(intent, 0, None) + 1.0 * (arrears_days > 0), n_customers)
    call_rows = []
    for cid, k, ch, arr in zip(ids, n_calls, churned, arrears_days):
        for _ in range(k):
            if ch and rng.random() < 0.35:
                reason = rng.choice(["rate_query", "payoff_quote", "complaint"], p=[0.45, 0.35, 0.20])
            elif arr > 0 and rng.random() < 0.5:
                reason = "hardship"
            else:
                reason = rng.choice(CALL_REASONS, p=[0.35, 0.20, 0.05, 0.10, 0.05, 0.25])
            sentiment = float(np.clip(rng.normal(-0.6 if reason in ("complaint", "payoff_quote") else 0.1, 0.5), -1, 1))
            call_rows.append(
                {
                    "customer_id": cid,
                    "call_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 365))),
                    "reason": reason,
                    "sentiment": round(sentiment, 2),
                    "complaint_flag": reason == "complaint",
                    "duration_sec": int(rng.integers(60, 1500)),
                }
            )
    calls = pd.DataFrame(call_rows).sort_values(["customer_id", "call_date"])

    labels = pd.DataFrame(
        {
            "customer_id": ids,
            "churned": churned.astype(int),
            "churn_month": np.where(churned, rng.choice(pd.period_range("2026-01", periods=12, freq="M").astype(str), n_customers), ""),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    customers.to_csv(out_dir / "customers.csv", index=False)
    mortgages.to_csv(out_dir / "mortgages.csv", index=False)
    digital.to_csv(out_dir / "digital_interactions.csv", index=False)
    calls.to_csv(out_dir / "call_center.csv", index=False)
    labels.to_csv(out_dir / "churn_labels.csv", index=False)
    print(f"customers={n_customers} churn_rate={churned.mean():.3%} calls={len(calls)} digital_rows={len(digital)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "raw")
    a = ap.parse_args()
    generate(a.n, a.seed, a.out)
