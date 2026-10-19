"""Evaluate the FROZEN model on the later-year test split, once.

    python -m loanlens.evaluate
    python -m loanlens.evaluate --rerun-reason "fixed plotting bug"   # disclosed rerun

Also writes validation-only global importance, real error examples, an
evaluation-only subgroup table, and the figures used by the app/report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,  # noqa: E402
                             brier_score_loss, classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

from . import features as F  # noqa: E402
from .common import path, read_json, write_json  # noqa: E402
from .explain import explain_rows, make_explainer  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    rep = classification_report(y, pred, labels=[1, 0], target_names=["Approved", "Rejected"],
                                output_dict=True, zero_division=0)
    return {
        "threshold": t, "n": int(len(y)),
        "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": rep["macro avg"]["f1-score"],
        "per_class": {k: {m: rep[k][m] for m in ("precision", "recall", "f1-score", "support")}
                      for k in ("Approved", "Rejected")},
        "confusion_matrix": {"labels": ["Approved", "Rejected"],
                             "rows_actual_cols_predicted": confusion_matrix(y, pred, labels=[1, 0]).tolist()},
        "roc_auc": roc_auc_score(y, p),
        "pr_auc_approved_positive": average_precision_score(y, p),
        "pr_auc_rejected_positive": average_precision_score(1 - y, 1 - p),
        "brier": brier_score_loss(y, p),
        "base_rate_approved": float(y.mean()),
    }


def guard(art, meta, reason):
    fp = art / "test_evaluation_log.json"
    log = read_json(fp) if fp.exists() else {"runs": []}
    prior = [r for r in log["runs"] if r["model_version"] == meta["model_version"]]
    if prior and not reason:
        raise SystemExit("This model version was already evaluated on the test set. "
                         "Re-running requires --rerun-reason, which is recorded and must be disclosed.")
    log["runs"].append({"model_version": meta["model_version"], "at": dt.datetime.now().isoformat(timespec="seconds"),
                        "rerun_reason": reason, "run_number_for_version": len(prior) + 1})
    write_json(log, fp)
    return len(prior) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-reason", default=None)
    a = ap.parse_args()

    art, rep, proc = path("artifacts_dir"), path("reports_dir"), path("processed_dir")
    fig = rep / "figures"
    fig.mkdir(exist_ok=True)
    model = joblib.load(art / "loanlens_model.joblib")
    meta = read_json(art / "model_metadata.json")
    cols, thr = meta["features"], meta["threshold"]
    data = pd.read_parquet(proc / "loanlens_dataset.parquet")
    va, te = data[data.split == "val"], data[data.split == "test"]

    run_no = guard(art, meta, a.rerun_reason)

    # ---------------- validation (allowed to inspect; used for selection)
    p_va = model.predict_proba(va[cols])[:, 1]
    val_m = metrics_at(va[F.TARGET].values, p_va, thr)

    # ---------------- test: frozen model, single pass
    t0 = time.perf_counter()
    p_te = model.predict_proba(te[cols])[:, 1]
    infer_ms = (time.perf_counter() - t0) / len(te) * 1000
    y = te[F.TARGET].values
    test_m = metrics_at(y, p_te, thr)
    test_m["inference_ms_per_row_batch"] = infer_ms
    test_m["test_run_number_for_this_model_version"] = run_no
    write_json({"model_version": meta["model_version"], "validation": val_m, "test": test_m},
               rep / "metrics.json")

    # ---------------- figures
    fpr, tpr, _ = roc_curve(y, p_te)
    pr, rc, _ = precision_recall_curve(y, p_te)
    fp_, mp_ = calibration_curve(y, p_te, n_bins=10, strategy="quantile")
    curves = {"roc": {"fpr": fpr[::max(1, len(fpr) // 400)].tolist(), "tpr": tpr[::max(1, len(tpr) // 400)].tolist()},
              "pr": {"recall": rc[::max(1, len(rc) // 400)].tolist(), "precision": pr[::max(1, len(pr) // 400)].tolist()},
              "calibration": {"mean_predicted": mp_.tolist(), "fraction_approved": fp_.tolist()}}
    write_json(curves, rep / "test_curves.json")

    f, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", c="grey")
    ax[0].set(title=f"ROC (test {te.activity_year.iloc[0]}) AUC={test_m['roc_auc']:.3f}", xlabel="FPR", ylabel="TPR")
    ax[1].plot(rc, pr); ax[1].axhline(y.mean(), ls="--", c="grey")
    ax[1].set(title=f"PR, Approved = positive (AP={test_m['pr_auc_approved_positive']:.3f})",
              xlabel="Recall", ylabel="Precision")
    ax[2].plot(mp_, fp_, "o-"); ax[2].plot([0, 1], [0, 1], "--", c="grey")
    ax[2].set(title=f"Calibration (Brier={test_m['brier']:.3f})", xlabel="Mean predicted P(Approved)",
              ylabel="Observed approval rate")
    f.tight_layout(); f.savefig(fig / "test_roc_pr_calibration.png", dpi=130); plt.close(f)

    cm = np.array(test_m["confusion_matrix"]["rows_actual_cols_predicted"])
    f, ax_ = plt.subplots(figsize=(4.5, 4))
    ax_.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax_.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12)
    ax_.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Approved", "Rejected"], yticklabels=["Approved", "Rejected"],
            xlabel="Predicted", ylabel="Actual (historical)", title=f"Confusion matrix @ t={thr}")
    f.tight_layout(); f.savefig(fig / "test_confusion_matrix.png", dpi=130); plt.close(f)

    # ---------------- global importance (validation data, not test)
    pi = permutation_importance(model, va[cols], va[F.TARGET].values, scoring="roc_auc",
                                n_repeats=5, random_state=0, n_jobs=1)
    imp = pd.DataFrame({"feature": cols, "label": [F.HUMAN_LABELS[c] for c in cols],
                        "perm_importance_auc_drop": pi.importances_mean, "std": pi.importances_std})
    try:
        bg = va[cols].sample(min(300, len(va)), random_state=0)
        E = make_explainer(model, bg)
        sv = explain_rows(E, va[cols].sample(min(2000, len(va)), random_state=1), cols)
        imp["mean_abs_shap"] = sv.abs().mean().reindex(cols).values
        imp["shap_unit"] = E["unit"]
        shap_ok = True
    except Exception as e:  # recorded, not hidden
        imp["mean_abs_shap"] = np.nan
        shap_ok = f"SHAP failed: {e!r}"
    imp = imp.sort_values("perm_importance_auc_drop", ascending=False)
    imp.to_csv(rep / "global_importance_validation.csv", index=False)
    f, ax_ = plt.subplots(figsize=(7, 5))
    ax_.barh(imp.label[::-1], imp.perm_importance_auc_drop[::-1], xerr=imp["std"][::-1])
    ax_.set(xlabel="Drop in validation ROC-AUC when shuffled", title="Permutation importance (validation)")
    f.tight_layout(); f.savefig(fig / "global_importance.png", dpi=130); plt.close(f)

    # ---------------- real error examples + demo examples (held-out test rows)
    te_out = te[["row_id"] + cols + [F.TARGET]].copy()
    te_out["prob_approved"] = p_te.round(4)
    te_out["predicted"] = np.where(p_te >= thr, "Approved", "Rejected")
    te_out["actual"] = te_out[F.TARGET].map(F.CLASS_NAMES)
    fa = te_out[(te_out.predicted == "Approved") & (te_out.actual == "Rejected")].nlargest(5, "prob_approved")
    fr = te_out[(te_out.predicted == "Rejected") & (te_out.actual == "Approved")].nsmallest(5, "prob_approved")
    pd.concat([fa.assign(error="false approve (model Approved, history Rejected)"),
               fr.assign(error="false reject (model Rejected, history Approved)")]).to_csv(
        rep / "error_examples_test.csv", index=False)
    rng = np.random.default_rng(7)
    demo = []
    for (pred, act), g in te_out.groupby(["predicted", "actual"]):
        demo.append(g.iloc[rng.choice(len(g), size=min(3, len(g)), replace=False)])
    pd.concat(demo).drop(columns=[F.TARGET]).to_csv(art / "demo_examples_heldout.csv", index=False)

    # ---------------- evaluation-only subgroup audit
    prot = pd.read_parquet(proc / "audit_only_protected.parquet")
    sub = te_out.merge(prot, on="row_id", how="left")
    rows = []
    for attr in [c for c in F.PROTECTED_COLUMNS if c in sub.columns and c != "applicant_age_above_62"]:
        for g, d in sub.groupby(attr):
            n = len(d)
            rej, app = d[d.actual == "Rejected"], d[d.actual == "Approved"]
            fa_k = int((rej.predicted == "Approved").sum())
            fr_k = int((app.predicted == "Rejected").sum())
            rows.append({"attribute": attr, "group": g, "n": n,
                         "actual_approval_rate": round(d[F.TARGET].mean(), 4),
                         "predicted_approval_rate": round((d.predicted == "Approved").mean(), 4),
                         "false_approve_rate": round(fa_k / len(rej), 4) if len(rej) else np.nan,
                         "false_approve_95ci": wilson(fa_k, len(rej)), "n_rejected": len(rej),
                         "false_reject_rate": round(fr_k / len(app), 4) if len(app) else np.nan,
                         "false_reject_95ci": wilson(fr_k, len(app)), "n_approved": len(app),
                         "small_sample_warning": n < 100})
    pd.DataFrame(rows).to_csv(rep / "subgroup_audit_test.csv", index=False)

    write_json({"shap": shap_ok, "infer_ms_per_row": infer_ms}, rep / "evaluation_notes.json")
    print("VALIDATION:", {k: round(v, 4) for k, v in val_m.items() if isinstance(v, float)})
    print("TEST      :", {k: round(v, 4) for k, v in test_m.items() if isinstance(v, float)})
    print("per-class test:", test_m["per_class"])
    print("CM test (rows actual A/R, cols pred A/R):", test_m["confusion_matrix"]["rows_actual_cols_predicted"])
    print("SHAP:", shap_ok)


if __name__ == "__main__":
    main()
