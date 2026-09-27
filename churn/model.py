"""Train, evaluate and version the churn-propensity model."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from churn import MODELS_DIR, PROCESSED_DIR
from churn.features import FEATURE_COLUMNS

MODEL_NAME = "churn_propensity"
MIN_AUC_LIFT_OVER_BASELINE = 0.05
MIN_TOP_DECILE_LIFT = 2.0


def naive_baseline(X: pd.DataFrame) -> np.ndarray:
    """Rule a retention team would use today: rate gap plus fixed-term expiry."""
    score = X["rate_gap"].clip(-1, 2) + 1.5 * X["near_expiry"]
    return score.rank(pct=True).to_numpy()


def top_decile_lift(y_true: np.ndarray, scores: np.ndarray) -> float:
    n = max(1, len(scores) // 10)
    top = np.argsort(-scores)[:n]
    base = y_true.mean()
    return float(y_true[top].mean() / base) if base > 0 else 0.0


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "top_decile_lift": top_decile_lift(y_true, scores),
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - not in a git checkout
        return "unknown"


def train(features_path: Path = PROCESSED_DIR / "features.parquet", out_dir: Path = MODELS_DIR, seed: int = 42) -> dict:
    df = pd.read_parquet(features_path)
    X, y = df[FEATURE_COLUMNS], df["churned"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)

    metrics = evaluate(y_te, model.predict_proba(X_te)[:, 1])
    baseline = evaluate(y_te, naive_baseline(X_te))

    accepted = metrics["auc"] >= baseline["auc"] + MIN_AUC_LIFT_OVER_BASELINE and metrics["top_decile_lift"] >= MIN_TOP_DECILE_LIFT
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()[:12]
    version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    card = {
        "model": MODEL_NAME,
        "version": version,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "training_data_hash": data_hash,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "churn_rate": float(y.mean()),
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "baseline_metrics": baseline,
        "acceptance": {
            "min_auc_lift_over_baseline": MIN_AUC_LIFT_OVER_BASELINE,
            "min_top_decile_lift": MIN_TOP_DECILE_LIFT,
            "accepted": accepted,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / f"{MODEL_NAME}.joblib")
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps({"metrics": metrics, "baseline": baseline, "accepted": accepted}, indent=2))
    if not accepted:
        raise SystemExit("model did not meet acceptance criteria")
    return card


def load_model(models_dir: Path = MODELS_DIR):
    model = joblib.load(models_dir / f"{MODEL_NAME}.joblib")
    card = json.loads((models_dir / "model_card.json").read_text())
    return model, card


if __name__ == "__main__":
    train()
