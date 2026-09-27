from pathlib import Path

import pandas as pd

from churn import NET_PROFIT_PER_CUSTOMER
from dashboard.steerco import value_captured


def test_value_captured_scales_pilot_to_book():
    pilot = pd.DataFrame(
        {
            "risk_band": ["high"] * 10 + ["low"] * 990,
            "churned": [1] * 4 + [0] * 6 + [0] * 990,
        }
    )
    v = value_captured(pilot, save_rate=0.5)
    assert v["contacted"] == 10
    assert v["saved_pilot"] == 2.0
    assert v["annual_profit_pilot"] == 2.0 * NET_PROFIT_PER_CUSTOMER
    assert v["annual_profit_book"] == 2.0 * NET_PROFIT_PER_CUSTOMER * 75
    assert v["churn_reduction_bps"] == 20.0


def test_dashboard_renders_without_exceptions(trained, monkeypatch):
    from streamlit.testing.v1 import AppTest

    fpath, models_dir = trained
    monkeypatch.setenv("CHURN_PROCESSED_DIR", str(fpath.parent))
    monkeypatch.setenv("CHURN_MODELS_DIR", str(models_dir))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "dashboard" / "steerco.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.metric) == 4
