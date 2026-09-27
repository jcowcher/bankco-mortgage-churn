# BankCo — Retail Mortgage Customer Churn

**Client brief for the MVP build.** BankCo (fictional) is a retail bank with a ~$30B mortgage book
(~75,000 customers, ~$400K average balance) losing ~2% of mortgage customers per year to refinancing
with competitors. Net profit lost to churn is ~$12M per year; the lifetime value at risk is ~$80M.

The strategy recommendation: build a churn-propensity capability that identifies at-risk mortgage
customers early and triggers a retention action (rate review, advisor outreach) before they refinance.
**Reducing churn by 20 bps (2.0% → 1.8%) is worth ~$8M in LTV and ~$1.2M in annual profit.**

## What exists today (`data/raw/`)

Extracts from four BankCo source systems, one snapshot each, keyed on `customer_id`:

| File | System | Grain | Notes |
|---|---|---|---|
| `customers.csv` | Core banking | 1 row / customer | Demographics, tenure, products held, region |
| `mortgages.csv` | Mortgage servicing | 1 row / active mortgage | Balance, rate, fixed-term expiry, LTV, arrears |
| `digital_interactions.csv` | Digital analytics | 1 row / customer / month (12 months) | Logins, rate-page views, payoff-quote requests, competitor-compare clicks |
| `call_center.csv` | Contact centre | 1 row / call | Call reason, sentiment, complaint flag |
| `churn_labels.csv` | Core banking (12 months later) | 1 row / customer | `churned` = discharged mortgage to another lender within 12 months of snapshot |

Data is synthetic (see `data/generate_synthetic.py`) and contains no real customer information.

## MVP scope (12 weeks)

1. **Weeks 1–3 — Ingest & govern.** Land the four extracts into a governed feature table with data-quality checks and lineage.
2. **Weeks 4–8 — Model.** Train a churn-propensity model; store versioned model artefacts with metrics and metadata ("model management").
3. **Weeks 9–12 — Act & measure.** Score live customers, trigger retention actions for a pilot region, and compare predicted vs. actual churn on a dashboard for the SteerCo (CLO, VP Technology).

## Acceptance criteria for the MVP

- Reproducible pipeline: `make data && make train && make test` from a clean checkout.
- Data-quality gate fails the pipeline on schema drift, duplicate keys, or out-of-range values.
- Model beats a naive baseline on held-out AUC and reports lift in the top decile.
- Scoring API (`/score`, `/at-risk`) returning propensity + top drivers per customer.
- Retention-trigger rules mapping propensity + drivers to a next-best-action for advisors.
- SteerCo dashboard: predicted vs. actual churn for the pilot region, and value captured vs. the $1.2M/yr target.
- Unit tests and CI on every PR.

## Repository layout (target)

```
data/raw/            source-system extracts (input)
data/processed/      governed feature table (generated)
churn/               ingest, features, model, scoring, actions
api/                 FastAPI scoring service
dashboard/           SteerCo dashboard (Streamlit)
models/              versioned model artefacts + metadata
tests/               pytest
```
