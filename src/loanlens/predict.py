"""Load the saved pipeline and score single or batch inputs, with schema checks.
There is NO fallback: if the artifact is missing, callers get ModelNotTrained."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .common import path, read_json

INPUT_NUMERIC_USER = ["income_k", "loan_amount_usd"]   # loan_to_income is derived here


class ModelNotTrained(RuntimeError):
    pass


def load():
    art = path("artifacts_dir")
    fp = art / "loanlens_model.joblib"
    if not fp.exists():
        raise ModelNotTrained("No trained model. Run: python -m loanlens.train")
    return joblib.load(fp), read_json(art / "feature_schema.json"), read_json(art / "model_metadata.json")


def required_inputs(schema: dict) -> list[str]:
    return [c for c in schema["order"] if c != "loan_to_income"]


def prepare(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (clean rows ready for the model, row-level error table)."""
    errors = []
    if not df.columns.is_unique:
        raise ValueError("Input column names must be unique.")
    if not df.index.is_unique:
        raise ValueError("Input row indices must be unique.")
    missing_cols = [c for c in required_inputs(schema) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    X = df[required_inputs(schema)].copy()
    for c in schema["numeric"]:
        if c in X:
            raw = X[c]
            blank = raw.isna() | raw.astype(str).str.strip().isin(["", "nan", "NaN", "NA", "None", "Exempt"])
            X[c] = pd.to_numeric(raw.where(~blank), errors="coerce")
            bad = ~blank & X[c].isna()
            for i in X.index[bad]:
                errors.append({"row": i, "column": c, "problem": f"not a number: {raw[i]!r}"})
            for i in X.index[X[c].notna() & ~np.isfinite(X[c])]:
                errors.append({"row": i, "column": c, "problem": "must be a finite number"})
            for i in X.index[X[c] < 0]:
                errors.append({"row": i, "column": c, "problem": "negative value"})
    for i in X.index[X["loan_amount_usd"].isna() | (X["loan_amount_usd"] <= 0)]:
        errors.append({"row": i, "column": "loan_amount_usd", "problem": "required and must be > 0"})
    for c in schema["categorical"]:
        X[c] = X[c].astype(str).str.strip().replace({"nan": "NA", "": "NA", "None": "NA"})
    X["income_k"] = X["income_k"].where(X["income_k"] > 0)          # same rule as training
    X["loan_to_income"] = (X["loan_amount_usd"] / (X["income_k"] * 1000.0)).where(X["income_k"] > 0)
    err = pd.DataFrame(errors, columns=["row", "column", "problem"])
    ok = X.drop(index=err["row"].unique()) if len(err) else X
    return ok[schema["order"]], err


def unseen_categories(X: pd.DataFrame, schema: dict) -> dict:
    out = {}
    for c, info in schema["categorical"].items():
        new = sorted(set(X[c].astype(str)) - set(info["values_seen_in_train"]))
        if new:
            out[c] = new
    return out


def score(model, meta, X: pd.DataFrame) -> pd.DataFrame:
    p = model.predict_proba(X)[:, 1]
    t = meta["threshold"]
    return pd.DataFrame({"prob_approved": np.round(p, 4),
                         "prediction": np.where(p >= t, "Approved", "Rejected"),
                         "threshold": t}, index=X.index)


def batch_output(raw: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep input metadata, replacing any previous prediction columns on re-upload."""
    out = raw.loc[predictions.index].copy()
    for column in predictions:
        out[column] = predictions[column]
    return out
