"""Ingest raw source-system extracts with schema and data-quality gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from churn import RAW_DIR

SCHEMAS: dict[str, dict[str, str]] = {
    "customers": {
        "customer_id": "int",
        "age": "int",
        "tenure_years": "float",
        "products_held": "int",
        "region": "str",
        "annual_income": "float",
        "has_salary_account": "bool",
    },
    "mortgages": {
        "customer_id": "int",
        "mortgage_id": "str",
        "balance": "float",
        "interest_rate": "float",
        "is_fixed": "bool",
        "months_to_fixed_expiry": "int",
        "ltv": "float",
        "arrears_days": "int",
        "remaining_term_months": "int",
        "origination_channel": "str",
    },
    "digital_interactions": {
        "customer_id": "int",
        "month": "str",
        "logins": "int",
        "rate_page_views": "int",
        "payoff_quote_requests": "int",
        "competitor_compare_clicks": "int",
    },
    "call_center": {
        "customer_id": "int",
        "call_date": "str",
        "reason": "str",
        "sentiment": "float",
        "complaint_flag": "bool",
        "duration_sec": "int",
    },
    "churn_labels": {"customer_id": "int", "churned": "int", "churn_month": "str"},
}

NULLABLE = {("churn_labels", "churn_month")}

PRIMARY_KEYS = {
    "customers": ["customer_id"],
    "mortgages": ["mortgage_id"],
    "digital_interactions": ["customer_id", "month"],
    "call_center": ["customer_id", "call_date", "reason", "duration_sec"],
    "churn_labels": ["customer_id"],
}

RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "customers": {"age": (18, 100), "tenure_years": (0, 60), "products_held": (1, 10)},
    "mortgages": {
        "interest_rate": (0.5, 15.0),
        "ltv": (0.0, 1.5),
        "balance": (1, 5_000_000),
        "arrears_days": (0, 365),
    },
    "call_center": {"sentiment": (-1.0, 1.0), "duration_sec": (0, 14_400)},
    "churn_labels": {"churned": (0, 1)},
}


@dataclass
class DataQualityReport:
    issues: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues


class DataQualityError(ValueError):
    pass


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    return "str"


def _check_table(name: str, df: pd.DataFrame, report: DataQualityReport) -> None:
    report.row_counts[name] = len(df)
    expected = SCHEMAS[name]
    missing = set(expected) - set(df.columns)
    extra = set(df.columns) - set(expected)
    if missing:
        report.issues.append(f"{name}: missing columns {sorted(missing)}")
    if extra:
        report.issues.append(f"{name}: unexpected columns {sorted(extra)}")
    for col, kind in expected.items():
        if col in df.columns and _kind(df[col]) != kind:
            report.issues.append(f"{name}.{col}: type {_kind(df[col])} != {kind}")
    keys = PRIMARY_KEYS[name]
    if all(k in df.columns for k in keys):
        dupes = int(df.duplicated(subset=keys).sum())
        if dupes:
            report.issues.append(f"{name}: {dupes} duplicate keys on {keys}")
    for col, (lo, hi) in RANGES.get(name, {}).items():
        if col in df.columns:
            bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if bad:
                report.issues.append(f"{name}.{col}: {bad} values outside [{lo}, {hi}]")
    nulls = df.isna().sum()
    for col, n in nulls[nulls > 0].items():
        if (name, col) in NULLABLE:
            continue
        report.issues.append(f"{name}.{col}: {int(n)} nulls")


def validate(tables: dict[str, pd.DataFrame]) -> DataQualityReport:
    report = DataQualityReport()
    for name, df in tables.items():
        _check_table(name, df, report)
    if "mortgages" in tables and "customers" in tables:
        orphans = ~tables["mortgages"]["customer_id"].isin(tables["customers"]["customer_id"])
        if orphans.any():
            report.issues.append(f"mortgages: {int(orphans.sum())} rows with unknown customer_id")
    return report


def load_raw(raw_dir: Path = RAW_DIR, strict: bool = True) -> dict[str, pd.DataFrame]:
    tables = {name: pd.read_csv(raw_dir / f"{name}.csv") for name in SCHEMAS}
    report = validate(tables)
    if strict and not report.ok:
        raise DataQualityError("\n".join(report.issues))
    return tables
