"""Build the customer-level feature table from the raw source-system extracts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn import PROCESSED_DIR
from churn.ingest import load_raw

MARKET_RATE = 5.60

FEATURE_COLUMNS = [
    "rate_gap",
    "months_to_fixed_expiry",
    "near_expiry",
    "is_fixed",
    "ltv",
    "balance",
    "arrears_days",
    "remaining_term_months",
    "age",
    "tenure_years",
    "products_held",
    "annual_income",
    "has_salary_account",
    "logins_12m",
    "rate_page_views_12m",
    "rate_page_views_3m",
    "payoff_quote_requests_12m",
    "competitor_compare_clicks_12m",
    "calls_12m",
    "rate_query_calls",
    "payoff_quote_calls",
    "complaints_12m",
    "avg_sentiment",
    "min_sentiment",
]

ID_COLUMNS = ["customer_id", "region"]


def _digital_features(digital: pd.DataFrame) -> pd.DataFrame:
    digital = digital.sort_values(["customer_id", "month"])
    last3 = digital.groupby("customer_id").tail(3)
    agg = digital.groupby("customer_id").agg(
        logins_12m=("logins", "sum"),
        rate_page_views_12m=("rate_page_views", "sum"),
        payoff_quote_requests_12m=("payoff_quote_requests", "sum"),
        competitor_compare_clicks_12m=("competitor_compare_clicks", "sum"),
    )
    agg["rate_page_views_3m"] = last3.groupby("customer_id")["rate_page_views"].sum()
    return agg


def _call_features(calls: pd.DataFrame) -> pd.DataFrame:
    agg = calls.groupby("customer_id").agg(
        calls_12m=("reason", "size"),
        rate_query_calls=("reason", lambda s: int((s == "rate_query").sum())),
        payoff_quote_calls=("reason", lambda s: int((s == "payoff_quote").sum())),
        complaints_12m=("complaint_flag", "sum"),
        avg_sentiment=("sentiment", "mean"),
        min_sentiment=("sentiment", "min"),
    )
    return agg


def build_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    customers = tables["customers"]
    mortgages = tables["mortgages"]
    df = customers.merge(mortgages, on="customer_id", how="inner")
    df["rate_gap"] = (df["interest_rate"] - MARKET_RATE).round(3)
    df["near_expiry"] = df["is_fixed"] & df["months_to_fixed_expiry"].between(0, 6)
    df = df.merge(_digital_features(tables["digital_interactions"]), on="customer_id", how="left")
    df = df.merge(_call_features(tables["call_center"]), on="customer_id", how="left")
    fill = {
        "logins_12m": 0,
        "rate_page_views_12m": 0,
        "rate_page_views_3m": 0,
        "payoff_quote_requests_12m": 0,
        "competitor_compare_clicks_12m": 0,
        "calls_12m": 0,
        "rate_query_calls": 0,
        "payoff_quote_calls": 0,
        "complaints_12m": 0,
        "avg_sentiment": 0.0,
        "min_sentiment": 0.0,
    }
    df = df.fillna(fill)
    for col in ("is_fixed", "near_expiry", "has_salary_account"):
        df[col] = df[col].astype(int)
    for col in ("complaints_12m", "calls_12m", "rate_query_calls", "payoff_quote_calls"):
        df[col] = df[col].astype(int)
    if "churn_labels" in tables:
        df = df.merge(tables["churn_labels"][["customer_id", "churned"]], on="customer_id", how="left")
        cols = ID_COLUMNS + FEATURE_COLUMNS + ["churned"]
    else:
        cols = ID_COLUMNS + FEATURE_COLUMNS
    return df[cols].sort_values("customer_id").reset_index(drop=True)


def main(out_dir: Path = PROCESSED_DIR) -> Path:
    tables = load_raw()
    features = build_features(tables)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "features.parquet"
    features.to_parquet(path, index=False)
    print(f"features: {len(features)} rows x {len(features.columns)} cols -> {path}")
    return path


if __name__ == "__main__":
    main()
