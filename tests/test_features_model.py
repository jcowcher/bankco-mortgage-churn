import json

import pandas as pd

from churn.actions import Driver, explain, next_best_action, risk_band, score_frame
from churn.features import FEATURE_COLUMNS, build_features
from churn.model import load_model, naive_baseline, top_decile_lift


def test_feature_table_is_one_row_per_customer_with_no_nulls(features, tables):
    assert len(features) == len(tables["customers"])
    assert features["customer_id"].is_unique
    assert features[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_features_handle_customers_with_no_calls_or_digital_activity(tables):
    t = dict(tables)
    t["call_center"] = tables["call_center"].iloc[0:0]
    t["digital_interactions"] = tables["digital_interactions"].iloc[0:0]
    f = build_features(t)
    assert (f["calls_12m"] == 0).all() and (f["logins_12m"] == 0).all()


def test_model_beats_baseline_and_writes_versioned_card(trained):
    _, models_dir = trained
    model, card = load_model(models_dir)
    assert card["acceptance"]["accepted"]
    assert card["metrics"]["auc"] >= card["baseline_metrics"]["auc"] + 0.05
    assert card["metrics"]["top_decile_lift"] >= 2.0
    assert card["features"] == FEATURE_COLUMNS
    assert card["version"] and card["training_data_hash"]
    assert (models_dir / "model_card.json").exists()
    json.loads((models_dir / "model_card.json").read_text())


def test_top_decile_lift_math():
    import numpy as np

    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0] * 2)
    s = np.linspace(1, 0, 20)
    assert top_decile_lift(y, s) == 5.0


def test_naive_baseline_ranks_rate_gap_and_expiry(features):
    b = naive_baseline(features)
    assert len(b) == len(features) and 0 <= b.min() and b.max() <= 1


def test_scores_are_calibrated_to_base_rate(trained, features):
    _, models_dir = trained
    model, _ = load_model(models_dir)
    scored = score_frame(model, features)
    assert abs(scored["churn_propensity"].mean() - features["churned"].mean()) < 0.01
    assert set(scored["risk_band"]) <= {"low", "medium", "high"}


def test_explain_returns_drivers_and_action(trained, features):
    _, models_dir = trained
    model, _ = load_model(models_dir)
    scored = score_frame(model, features)
    top = scored.sort_values("churn_propensity").iloc[-1]["customer_id"]
    row = features[features["customer_id"] == top].iloc[0]
    out = explain(model, row, features)
    assert out["risk_band"] == "high"
    assert 1 <= len(out["top_drivers"]) <= 3
    assert out["top_drivers"][0]["contribution"] >= out["top_drivers"][-1]["contribution"]
    assert out["next_best_action"]["priority"] == "P1"


def test_next_best_action_routing():
    row = pd.Series({"rate_gap": 1.2})
    assert risk_band(0.01) == "low"
    assert next_best_action(0.01, [], row).action == "No action"
    assert "Complaint" in next_best_action(0.3, [Driver("complaints_12m", "", 2, 0.1)], row).action
    assert "refix" in next_best_action(0.3, [Driver("near_expiry", "", 1, 0.1)], row).action
    rate = next_best_action(0.08, [Driver("rate_gap", "", 1.2, 0.1)], row)
    assert rate.action.startswith("Rate review") and rate.priority == "P2"
