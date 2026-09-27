"""BankCo retail-mortgage churn early-warning MVP."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = Path(os.environ.get("CHURN_PROCESSED_DIR", ROOT / "data" / "processed"))
MODELS_DIR = Path(os.environ.get("CHURN_MODELS_DIR", ROOT / "models"))

# Value case from the client brief (per-customer annual net profit, remaining life).
NET_PROFIT_PER_CUSTOMER = 7_900
REMAINING_LIFE_YEARS = 7
BASELINE_CHURN = 0.020
TARGET_CHURN = 0.018
ANNUAL_PROFIT_TARGET = 1_200_000
