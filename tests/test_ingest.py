import pandas as pd
import pytest

from churn.ingest import DataQualityError, load_raw, validate


def test_clean_extracts_pass_quality_gates(tables):
    report = validate(tables)
    assert report.ok, report.issues
    assert report.row_counts["customers"] == 3_000


def test_schema_drift_is_detected(tables):
    broken = dict(tables)
    broken["customers"] = tables["customers"].rename(columns={"annual_income": "income"})
    report = validate(broken)
    assert any("missing columns ['annual_income']" in i for i in report.issues)
    assert any("unexpected columns ['income']" in i for i in report.issues)


def test_duplicate_keys_are_detected(tables):
    broken = dict(tables)
    broken["mortgages"] = pd.concat([tables["mortgages"], tables["mortgages"].head(5)])
    report = validate(broken)
    assert any("5 duplicate keys" in i for i in report.issues)


def test_out_of_range_values_are_detected(tables):
    broken = dict(tables)
    m = tables["mortgages"].copy()
    m.loc[m.index[:3], "ltv"] = 2.5
    broken["mortgages"] = m
    report = validate(broken)
    assert any("ltv: 3 values outside" in i for i in report.issues)


def test_orphan_mortgages_are_detected(tables):
    broken = dict(tables)
    m = tables["mortgages"].copy()
    m.loc[m.index[0], "customer_id"] = -1
    broken["mortgages"] = m
    assert any("unknown customer_id" in i for i in validate(broken).issues)


def test_strict_load_raises_on_bad_data(raw_dir, tmp_path):
    for f in raw_dir.glob("*.csv"):
        (tmp_path / f.name).write_text(f.read_text())
    c = pd.read_csv(tmp_path / "customers.csv")
    c.loc[0, "age"] = 5
    c.to_csv(tmp_path / "customers.csv", index=False)
    with pytest.raises(DataQualityError, match="age"):
        load_raw(tmp_path)
