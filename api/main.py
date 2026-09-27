"""Scoring API: churn propensity, drivers and next-best-action for advisors and CRM."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from churn import MODELS_DIR, PROCESSED_DIR
from churn.actions import explain, score_frame
from churn.model import load_model


class _State:
    model = None
    card: dict = {}
    features: pd.DataFrame = pd.DataFrame()


state = _State()


def load_state() -> None:
    state.model, state.card = load_model(MODELS_DIR)
    state.features = pd.read_parquet(PROCESSED_DIR / "features.parquet").set_index("customer_id", drop=False)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_state()
    yield


app = FastAPI(title="BankCo Mortgage Churn Early Warning", version="0.1.0", lifespan=lifespan)


class ScoreResponse(BaseModel):
    customer_id: int
    region: str
    churn_propensity: float
    risk_band: str
    annual_value_at_risk: float
    top_drivers: list[dict]
    next_best_action: dict
    model_version: str


class AtRiskCustomer(BaseModel):
    customer_id: int
    region: str
    churn_propensity: float
    risk_band: str
    annual_value_at_risk: float


class AtRiskResponse(BaseModel):
    region: str | None
    threshold: float
    count: int
    total_annual_value_at_risk: float
    model_version: str
    customers: list[AtRiskCustomer]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": state.card.get("model"), "version": state.card.get("version")}


@app.get("/model")
def model_card() -> dict:
    return state.card


@lru_cache(maxsize=4096)
def _explain_cached(customer_id: int, version: str) -> dict:
    row = state.features.loc[customer_id]
    return explain(state.model, row, state.features)


@app.get("/score/{customer_id}", response_model=ScoreResponse)
def score(customer_id: int) -> ScoreResponse:
    if customer_id not in state.features.index:
        raise HTTPException(status_code=404, detail=f"customer {customer_id} not found")
    result = _explain_cached(customer_id, state.card["version"])
    return ScoreResponse(**result, model_version=state.card["version"])


@app.get("/at-risk", response_model=AtRiskResponse)
def at_risk(
    region: str | None = Query(default=None),
    threshold: float = Query(default=0.15, ge=0.0, le=1.0),
    limit: int = Query(default=100, ge=1, le=5000),
) -> AtRiskResponse:
    df = state.features
    if region is not None:
        df = df[df["region"] == region]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"region {region} not found")
    scored = score_frame(state.model, df)
    scored = scored[scored["churn_propensity"] >= threshold].sort_values("churn_propensity", ascending=False)
    return AtRiskResponse(
        region=region,
        threshold=threshold,
        count=int(len(scored)),
        total_annual_value_at_risk=float(scored["annual_value_at_risk"].sum()),
        model_version=state.card["version"],
        customers=[AtRiskCustomer(**r) for r in scored.head(limit).to_dict("records")],
    )
