"""Model selection with stratified CV on the TRAIN split only, then calibration
and threshold choice on the VALIDATION split. The test year is never read here.

    python -m loanlens.train
"""
from __future__ import annotations

import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import features as F
from .common import CFG, path, read_json, write_json
from .modeling import candidates, feature_lists, make_pipeline

SCORING = {"roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro",
           "average_precision": "average_precision"}


def expected_cost(y, p, t, c_fa, c_fr):
    pred = (p >= t).astype(int)
    false_approve = ((pred == 1) & (y == 0)).sum()
    false_reject = ((pred == 0) & (y == 1)).sum()
    return (c_fa * false_approve + c_fr * false_reject) / len(y)


def main() -> None:
    proc = path("processed_dir")
    audit = read_json(proc / "data_audit.json")
    data = pd.read_parquet(proc / "loanlens_dataset.parquet")
    num, cat = feature_lists(audit["rich_features_allowed"])
    cols = num + cat
    tr, va = data[data.split == "train"], data[data.split == "val"]
    Xtr, ytr, Xva, yva = tr[cols], tr[F.TARGET].values, va[cols], va[F.TARGET].values

    seed = CFG["split"]["random_state"]
    cv = StratifiedKFold(CFG["train"]["cv_folds"], shuffle=True, random_state=seed)
    rows, fitted = [], {}
    for name, (est, grid) in candidates(seed).items():
        t0 = time.perf_counter()
        gs = GridSearchCV(make_pipeline(est, num, cat), grid or {"clf__strategy": ["prior"]},
                          scoring=SCORING, refit="roc_auc", cv=cv, n_jobs=1)
        gs.fit(Xtr, ytr)
        secs = time.perf_counter() - t0
        i = gs.best_index_
        r = {"model": name, "best_params": gs.best_params_, "search_seconds": round(secs, 2)}
        for m in SCORING:
            r[f"cv_{m}_mean"] = gs.cv_results_[f"mean_test_{m}"][i]
            r[f"cv_{m}_std"] = gs.cv_results_[f"std_test_{m}"][i]
        rows.append(r)
        fitted[name] = gs.best_estimator_
        print(f"{name:30s} CV ROC-AUC {r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f}  ({secs:.1f}s)")
    cvres = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)

    best_name = cvres.iloc[0]["model"]
    base = fitted[best_name]

    # ---- calibration on the validation split (base model is frozen)
    calib = {}
    for method in ("sigmoid", "isotonic"):
        c = CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(Xva, yva)
        calib[method] = (c, brier_score_loss(yva, c.predict_proba(Xva)[:, 1]))
    raw_brier = brier_score_loss(yva, base.predict_proba(Xva)[:, 1])
    method = min(calib, key=lambda k: calib[k][1])
    model = calib[method][0]
    p_va = model.predict_proba(Xva)[:, 1]

    # ---- threshold from stated disagreement costs (NOT default losses)
    c_fa, c_fr = CFG["train"]["cost_false_approve"], CFG["train"]["cost_false_reject"]
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [expected_cost(yva, p_va, t, c_fa, c_fr) for t in grid]
    thr = float(grid[int(np.argmin(costs))])

    art = path("artifacts_dir")
    joblib.dump(model, art / "loanlens_model.joblib", compress=3)
    cvres.to_csv(path("reports_dir") / "cv_model_comparison.csv", index=False)

    schema = {"numeric": {c: {"label": F.HUMAN_LABELS[c], "train_p01": float(np.nanpercentile(tr[c], 1)),
                              "train_median": float(np.nanmedian(tr[c])), "train_p99": float(np.nanpercentile(tr[c], 99))}
                          for c in num},
              "categorical": {c: {"label": F.HUMAN_LABELS[c],
                                  "values_seen_in_train": sorted(tr[c].astype(str).unique().tolist()),
                                  "code_labels": F.CODE_LABELS.get(c, {})} for c in cat},
              "derived": {"loan_to_income": "loan_amount_usd / (income_k * 1000); computed by the app, not entered"},
              "order": cols}
    write_json(schema, art / "feature_schema.json")
    meta = {
        "model_version": time.strftime("loanlens-%Y%m%d-%H%M%S"),
        "selected_model": best_name, "best_params": cvres.iloc[0]["best_params"],
        "selection_rule": "highest mean 5-fold stratified CV ROC-AUC on the 2023 train split",
        "calibration": {"method": method, "fit_on": "validation split (2023)",
                        "val_brier_uncalibrated": raw_brier,
                        "val_brier_sigmoid": calib["sigmoid"][1], "val_brier_isotonic": calib["isotonic"][1]},
        "threshold": thr,
        "threshold_policy": {"cost_false_approve": c_fa, "cost_false_reject": c_fr,
                             "meaning": "costs of disagreeing with the historical decision; not observed losses",
                             "chosen_on": "validation split", "val_expected_cost": float(min(costs))},
        "val_roc_auc_calibrated": roc_auc_score(yva, p_va),
        "features": cols, "positive_class": "Approved",
        "rows": {"train": len(tr), "val": len(va)},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "model_file_bytes": (art / "loanlens_model.joblib").stat().st_size,
    }
    write_json(meta, art / "model_metadata.json")
    write_json({"thresholds": grid.tolist(), "val_cost": costs}, art / "threshold_curve.json")
    print(f"\nSelected: {best_name} | calibration={method} | threshold={thr}")


if __name__ == "__main__":
    main()
