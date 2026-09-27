"""Explain a customer's churn propensity and map it to a next-best retention action."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from churn.features import FEATURE_COLUMNS

HIGH_RISK = 0.15
MEDIUM_RISK = 0.06

DRIVER_LABELS = {
    "rate_gap": "Paying above market rate",
    "near_expiry": "Fixed term expiring within 6 months",
    "months_to_fixed_expiry": "Fixed-term horizon",
    "payoff_quote_requests_12m": "Requested payoff quotes online",
    "payoff_quote_calls": "Requested payoff quote by phone",
    "competitor_compare_clicks_12m": "Comparing competitor rates",
    "rate_page_views_3m": "Recent rate-page browsing",
    "rate_page_views_12m": "Rate-page browsing",
    "rate_query_calls": "Called about rates",
    "complaints_12m": "Open complaints",
    "min_sentiment": "Negative call sentiment",
    "avg_sentiment": "Call sentiment",
    "products_held": "Few products held",
    "has_salary_account": "No salary account",
    "tenure_years": "Short tenure",
    "ltv": "Loan-to-value",
    "arrears_days": "Arrears",
}


@dataclass
class Driver:
    feature: str
    label: str
    value: float
    contribution: float


@dataclass
class RetentionAction:
    action: str
    owner: str
    rationale: str
    priority: str


def risk_band(p: float) -> str:
    if p >= HIGH_RISK:
        return "high"
    if p >= MEDIUM_RISK:
        return "medium"
    return "low"


def top_drivers(model, row: pd.Series, reference: pd.DataFrame, k: int = 3) -> list[Driver]:
    """Counterfactual attribution: prob(row) minus prob(row with feature set to the population median)."""
    x = row[FEATURE_COLUMNS].to_frame().T.astype(float)
    base = float(model.predict_proba(x)[0, 1])
    medians = reference[FEATURE_COLUMNS].median()
    contributions = []
    for f in FEATURE_COLUMNS:
        if x.iloc[0][f] == medians[f]:
            continue
        x_cf = x.copy()
        x_cf[f] = medians[f]
        contributions.append((f, base - float(model.predict_proba(x_cf)[0, 1])))
    contributions.sort(key=lambda t: -t[1])
    return [Driver(f, DRIVER_LABELS.get(f, f), float(row[f]), round(c, 4)) for f, c in contributions[:k] if c > 0]


def next_best_action(p: float, drivers: list[Driver], row: pd.Series) -> RetentionAction:
    band = risk_band(p)
    if band == "low":
        return RetentionAction("No action", "—", "Below the medium-risk threshold; monitor.", "none")
    names = {d.feature for d in drivers}
    priority = "P1" if band == "high" else "P2"
    if "complaints_12m" in names or "min_sentiment" in names:
        return RetentionAction(
            "Complaint resolution call within 48h",
            "Customer care lead",
            "Service failure is driving intent; fix the experience before discussing price.",
            priority,
        )
    if "near_expiry" in names or "months_to_fixed_expiry" in names:
        return RetentionAction(
            "Proactive refix offer before expiry",
            "Mortgage advisor",
            "Fixed term ends soon; lock the customer in with a competitive refix ahead of the market.",
            priority,
        )
    if "rate_gap" in names or "rate_query_calls" in names or "rate_page_views_3m" in names:
        gap = float(row["rate_gap"])
        return RetentionAction(
            f"Rate review: offer up to {min(gap, 0.75):.2f} pt reduction",
            "Mortgage advisor",
            "Customer is paying above market and shopping around; a retention rate beats losing $7.9K/yr.",
            priority,
        )
    if "payoff_quote_requests_12m" in names or "payoff_quote_calls" in names or "competitor_compare_clicks_12m" in names:
        return RetentionAction(
            "Retention specialist outbound call",
            "Retention desk",
            "Payoff-quote / competitor-comparison activity signals an active refinance decision.",
            priority,
        )
    if "products_held" in names or "has_salary_account" in names:
        return RetentionAction(
            "Bundle offer: salary account + offset",
            "Branch relationship manager",
            "Single-product customers churn most; deepen the relationship.",
            priority,
        )
    return RetentionAction("Advisor check-in call", "Mortgage advisor", "Elevated risk without a single dominant driver.", priority)


def explain(model, row: pd.Series, reference: pd.DataFrame) -> dict:
    p = float(model.predict_proba(row[FEATURE_COLUMNS].to_frame().T.astype(float))[0, 1])
    drivers = top_drivers(model, row, reference)
    action = next_best_action(p, drivers, row)
    return {
        "customer_id": int(row["customer_id"]),
        "region": str(row["region"]),
        "churn_propensity": round(p, 4),
        "risk_band": risk_band(p),
        "annual_value_at_risk": round(p * 7_900, 0),
        "top_drivers": [asdict(d) for d in drivers],
        "next_best_action": asdict(action),
    }


def score_frame(model, df: pd.DataFrame) -> pd.DataFrame:
    out = df[["customer_id", "region"]].copy()
    out["churn_propensity"] = model.predict_proba(df[FEATURE_COLUMNS].astype(float))[:, 1]
    out["risk_band"] = [risk_band(p) for p in out["churn_propensity"]]
    out["annual_value_at_risk"] = np.round(out["churn_propensity"] * 7_900, 0)
    return out
