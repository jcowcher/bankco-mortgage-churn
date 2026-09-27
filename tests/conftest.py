from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from churn.features import build_features  # noqa: E402
from churn.ingest import load_raw  # noqa: E402
from churn.model import train  # noqa: E402
from data.generate_synthetic import generate  # noqa: E402


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("raw")
    generate(n_customers=3_000, seed=7, out_dir=out)
    return out


@pytest.fixture(scope="session")
def tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    return load_raw(raw_dir)


@pytest.fixture(scope="session")
def features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return build_features(tables)


@pytest.fixture(scope="session")
def trained(features: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    proc = tmp_path_factory.mktemp("processed")
    models = tmp_path_factory.mktemp("models")
    fpath = proc / "features.parquet"
    features.to_parquet(fpath, index=False)
    train(features_path=fpath, out_dir=models, seed=7)
    return fpath, models
