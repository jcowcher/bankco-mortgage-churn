PY ?= python

.PHONY: install data features train test lint api dashboard all

install:
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) data/generate_synthetic.py

features:
	$(PY) -m churn.features

train: features
	$(PY) -m churn.model

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

api:
	$(PY) -m uvicorn api.main:app --reload --port 8000

dashboard:
	$(PY) -m streamlit run dashboard/steerco.py

all: data train test lint
