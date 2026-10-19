"""Explanations.

Local: SHAP values computed for the *uncalibrated* base pipeline in its
transformed feature space, then one-hot columns are summed back to the
original feature. Calibration is a monotonic map, so the ranking of
applications and the direction of each push are preserved, but SHAP units
are the base model's margin (log-odds for LR/HistGB, probability for RF/DT),
not the calibrated probability shown to the user.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def base_pipeline(model):
    est = getattr(model, "estimator", model)           # CalibratedClassifierCV -> FrozenEstimator
    return getattr(est, "estimator", est)               # FrozenEstimator -> Pipeline


def _orig_map(prep, order: list[str]) -> list[str]:
    names = prep.get_feature_names_out()
    out = []
    for n in names:
        body = n.split("__", 1)[1]
        match = max((c for c in order if body == c or body.startswith(c + "_")), key=len, default=body)
        out.append(match)
    return out


def make_explainer(model, background: pd.DataFrame):
    import shap
    pipe = base_pipeline(model)
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    bg = prep.transform(background)
    kind = type(clf).__name__
    if kind == "LogisticRegression":
        ex = shap.LinearExplainer(clf, bg)
        unit = "log-odds"
    elif kind in ("RandomForestClassifier", "DecisionTreeClassifier", "HistGradientBoostingClassifier"):
        ex = shap.TreeExplainer(clf, bg if kind == "HistGradientBoostingClassifier" else None)
        unit = "log-odds" if kind == "HistGradientBoostingClassifier" else "probability"
    else:
        raise TypeError(f"No SHAP explainer configured for {kind}")
    return {"explainer": ex, "prep": prep, "unit": unit, "kind": kind}


def explain_rows(E: dict, X: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Rows = applications, columns = original features, values = SHAP contribution toward Approved."""
    Z = E["prep"].transform(X)
    sv = E["explainer"].shap_values(Z, check_additivity=False) if E["kind"] != "LogisticRegression" \
        else E["explainer"].shap_values(Z)
    sv = np.asarray(sv)
    if sv.ndim == 3:                                   # (n, features, classes) -> class 1
        sv = sv[..., 1]
    cols = _orig_map(E["prep"], order)
    df = pd.DataFrame(sv, columns=cols, index=X.index)
    return df.T.groupby(level=0).sum().T[order]
