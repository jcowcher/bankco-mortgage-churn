"""SteerCo dashboard: pilot-region predicted vs actual churn and value captured vs target.

Run: streamlit run dashboard/steerco.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from churn import ANNUAL_PROFIT_TARGET, BASELINE_CHURN, NET_PROFIT_PER_CUSTOMER, PROCESSED_DIR, TARGET_CHURN
from churn.actions import score_frame
from churn.model import load_model

BOOK_CUSTOMERS = 75_000
PILOT_REGION_DEFAULT = "North"
ASSUMED_SAVE_RATE = 0.25  # share of contacted high-risk customers retained by the action


@st.cache_data
def load_scored() -> tuple[pd.DataFrame, dict]:
    model, card = load_model()
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    scored = score_frame(model, features).merge(features[["customer_id", "churned", "balance"]], on="customer_id")
    return scored, card


def value_captured(pilot: pd.DataFrame, save_rate: float) -> dict[str, float]:
    high = pilot[pilot["risk_band"] == "high"]
    expected_churners = float(high["churned"].sum())
    saved = expected_churners * save_rate
    scale = BOOK_CUSTOMERS / max(1, len(pilot))
    return {
        "contacted": float(len(high)),
        "churners_in_contacted": expected_churners,
        "saved_pilot": saved,
        "annual_profit_pilot": saved * NET_PROFIT_PER_CUSTOMER,
        "annual_profit_book": saved * NET_PROFIT_PER_CUSTOMER * scale,
        "churn_reduction_bps": (saved / max(1, len(pilot))) * 10_000,
    }


def main() -> None:
    st.set_page_config(page_title="BankCo Churn SteerCo", layout="wide")
    scored, card = load_scored()
    st.title("BankCo Retail Mortgage Churn — SteerCo view")
    st.caption(
        f"Model {card['model']} v{card['version']} · held-out AUC {card['metrics']['auc']:.3f} "
        f"(baseline {card['baseline_metrics']['auc']:.3f}) · top-decile lift {card['metrics']['top_decile_lift']:.1f}x"
    )

    regions = sorted(scored["region"].unique())
    col1, col2 = st.columns([1, 1])
    region = col1.selectbox("Pilot region", regions, index=regions.index(PILOT_REGION_DEFAULT))
    save_rate = col2.slider("Assumed save rate on contacted high-risk customers", 0.05, 0.6, ASSUMED_SAVE_RATE, 0.05)
    pilot = scored[scored["region"] == region]

    v = value_captured(pilot, save_rate)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pilot customers", f"{len(pilot):,}")
    m2.metric("High-risk to contact", f"{int(v['contacted']):,}")
    m3.metric(
        "Churn reduction (bps)",
        f"{v['churn_reduction_bps']:.0f}",
        help=f"Target: {(BASELINE_CHURN - TARGET_CHURN) * 10_000:.0f} bps",
    )
    m4.metric(
        "Annual profit, scaled to book",
        f"${v['annual_profit_book'] / 1e6:.2f}M",
        delta=f"{v['annual_profit_book'] / ANNUAL_PROFIT_TARGET * 100:.0f}% of $1.2M target",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Predicted vs actual churn by risk decile")
        pilot = pilot.copy()
        pilot["decile"] = pd.qcut(pilot["churn_propensity"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
        by_dec = (
            pilot.groupby("decile", observed=True).agg(predicted=("churn_propensity", "mean"), actual=("churned", "mean")).reset_index()
        )
        fig = px.bar(
            by_dec.melt(id_vars="decile", var_name="series", value_name="churn"),
            x="decile",
            y="churn",
            color="series",
            barmode="group",
        )
        fig.update_layout(yaxis_tickformat=".1%", legend_title=None, height=380)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Where the risk sits")
        by_region = (
            scored.groupby("region")
            .agg(
                customers=("customer_id", "size"),
                high_risk=("risk_band", lambda s: int((s == "high").sum())),
                value_at_risk=("annual_value_at_risk", "sum"),
            )
            .reset_index()
        )
        fig2 = px.bar(
            by_region,
            x="region",
            y="value_at_risk",
            text="high_risk",
            labels={"value_at_risk": "Annual value at risk ($)"},
        )
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, width="stretch")

    st.subheader(f"High-risk customers in {region} — advisor worklist")
    worklist = pilot[pilot["risk_band"] == "high"].sort_values("churn_propensity", ascending=False)
    st.dataframe(
        worklist[["customer_id", "churn_propensity", "annual_value_at_risk", "balance"]].head(50),
        width="stretch",
        hide_index=True,
        column_config={"churn_propensity": st.column_config.NumberColumn(format="%.1%")},
    )


if __name__ == "__main__":
    main()
