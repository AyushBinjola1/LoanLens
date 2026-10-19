"""Checks for target derivation, units, leakage, schema errors, unseen
categories, artifact loading, and single/batch consistency.

Tiny hand-written frames below are *unit-test fixtures* that exercise parsing
logic only; they are never used for training, evaluation, or demos. Tests that
need real data/artifacts are skipped until the pipeline has been run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loanlens import features as F  # noqa: E402
from loanlens.build_dataset import SchemaError, validate_raw  # noqa: E402
from loanlens.common import path  # noqa: E402

DATA = path("processed_dir") / "loanlens_dataset.parquet"
MODEL = path("artifacts_dir") / "loanlens_model.joblib"
need_data = pytest.mark.skipif(not DATA.exists(), reason="processed data not built yet")
need_model = pytest.mark.skipif(not MODEL.exists(), reason="model not trained yet")


def test_target_mapping():
    y = F.derive_target(pd.Series(["1", "2", "3", "4", "6"]))
    assert y.tolist()[:3] == [1, 1, 0] and y.iloc[3:].isna().all()


def _fixture_row(**kw):
    base = {"income": "85", "loan_amount": "255000", "loan_type": "1", "loan_purpose": "1",
            "lien_status": "1", "occupancy_type": "1", "preapproval": "2",
            "derived_dwelling_category": "Single Family (1-4 Units):Site-Built",
            "open-end_line_of_credit": "2", "derived_msa-md": "20100",
            "co-applicant_credit_score_type": "10", "combined_loan_to_value_ratio": "80.0",
            "property_value": "325000", "loan_term": "360", "debt_to_income_ratio": "36"}
    base.update(kw)
    return pd.DataFrame([base])


def test_income_units_thousands():
    f = F.engineer(_fixture_row())
    assert f.income_k.iloc[0] == 85
    assert f.loan_to_income.iloc[0] == pytest.approx(255000 / 85000)
    assert F.to_dollars(85) == 85000


def test_zero_negative_and_na_income_are_safe():
    for v in ["0", "-5", "NA", "Exempt"]:
        f = F.engineer(_fixture_row(income=v))
        assert np.isnan(f.income_k.iloc[0]) and np.isnan(f.loan_to_income.iloc[0])


def test_metro_and_coapplicant_flags():
    f = F.engineer(_fixture_row(**{"derived_msa-md": "99999", "co-applicant_credit_score_type": "1111"}))
    assert f.metro_area.iloc[0] == "Non-metro" and f.has_coapplicant.iloc[0] == "Unknown"


def test_feature_lists_exclude_leakage_protected_and_ids():
    allowed = set(F.CORE_NUMERIC + F.CORE_CATEGORICAL + F.CANDIDATE_NUMERIC + F.CANDIDATE_CATEGORICAL)
    banned = set(F.LEAKAGE_COLUMNS) | set(F.ID_GEO_COLUMNS) | set(F.PROTECTED_COLUMNS) | set(F.PROXY_COLUMNS)
    assert not allowed & banned
    assert "applicant_credit_score_type" not in allowed  # model type, not a score


def test_schema_validation_rejects_bad_raw():
    with pytest.raises(SchemaError):
        validate_raw(pd.DataFrame({"activity_year": ["2023"]}), 2023)


@need_data
def test_processed_has_no_banned_columns_and_years_dont_overlap():
    d = pd.read_parquet(DATA)
    for c in list(F.LEAKAGE_COLUMNS) + F.PROTECTED_COLUMNS + list(F.PROXY_COLUMNS):
        assert c not in d.columns
    assert set(d[d.split == "test"].activity_year) == {"2024"}
    assert set(d[d.split != "test"].activity_year) == {"2023"}
    assert d.row_id.is_unique


@need_model
def test_artifact_loads_and_predicts_probabilities():
    from loanlens.predict import load
    model, schema, meta = load()
    d = pd.read_parquet(DATA)
    p = model.predict_proba(d[d.split == "val"][schema["order"]].head(50))[:, 1]
    assert ((p >= 0) & (p <= 1)).all() and 0 < meta["threshold"] < 1


@need_model
def test_single_equals_batch_and_unseen_category_is_handled():
    from loanlens.predict import load, prepare, score, unseen_categories
    model, schema, meta = load()
    d = pd.read_parquet(DATA)
    rows = d[d.split == "test"].head(20)[[c for c in schema["order"] if c != "loan_to_income"]]
    Xb, err = prepare(rows.reset_index(drop=True), schema)
    batch = score(model, meta, Xb)
    for i in range(3):
        Xs, _ = prepare(rows.reset_index(drop=True).iloc[[i]], schema)
        assert score(model, meta, Xs).prob_approved.iloc[0] == batch.prob_approved.iloc[i]
    odd = rows.head(1).reset_index(drop=True).copy()
    odd["loan_type"] = "99"
    Xo, _ = prepare(odd, schema)
    assert unseen_categories(Xo, schema) == {"loan_type": ["99"]}
    assert 0 <= score(model, meta, Xo).prob_approved.iloc[0] <= 1


@need_model
def test_batch_schema_errors_are_row_level():
    from loanlens.predict import load, prepare
    _, schema, _ = load()
    d = pd.read_parquet(DATA)
    rows = d[d.split == "test"].head(3)[[c for c in schema["order"] if c != "loan_to_income"]].astype(str)
    rows = rows.reset_index(drop=True)
    rows.loc[1, "loan_amount_usd"] = "abc"
    X, err = prepare(rows, schema)
    assert list(err.row.unique()) == [1] and len(X) == 2
    with pytest.raises(ValueError):
        prepare(rows.drop(columns=["loan_type"]), schema)
