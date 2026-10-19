"""LoanLens — Streamlit dashboard.  Run:  streamlit run app/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from loanlens import features as F  # noqa: E402
from loanlens.common import CFG, path, read_json  # noqa: E402
from loanlens.predict import ModelNotTrained, load, prepare, score, unseen_categories  # noqa: E402

st.set_page_config(page_title="LoanLens", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.6rem;max-width:1200px}
.badge{display:inline-block;padding:.15rem .55rem;border-radius:999px;font-size:.78rem;font-weight:600}
.real{background:#e3f2e9;color:#1b6b3a}.hypo{background:#fdf0dc;color:#8a5300}
.note{font-size:.85rem;opacity:.8}
</style>""", unsafe_allow_html=True)

ART, REP, PROC, RAW = path("artifacts_dir"), path("reports_dir"), path("processed_dir"), path("raw_dir")
DISCLAIMER = ("Educational tool. It predicts what **historical US lenders decided** on similar HMDA applications. "
              "It does not estimate repayment risk, is not affiliated with HDFC, is not validated for Indian "
              "lending, and every output needs human review.")


@st.cache_resource
def get_model():
    try:
        return load()
    except ModelNotTrained:
        return None


@st.cache_data
def get_data():
    fp = PROC / "loanlens_dataset.parquet"
    return pd.read_parquet(fp) if fp.exists() else None


@st.cache_resource
def get_explainer(_model, cols):
    from loanlens.explain import make_explainer
    d = get_data()
    bg = d[d.split == "train"][cols].sample(300, random_state=0)
    return make_explainer(_model, bg)


def jfile(p):
    return read_json(p) if p.exists() else None


def need_training():
    st.warning("**Training required.** No saved model artifact was found, so no predictions are shown.\n\n"
               "Run `python -m loanlens.build_dataset`, `python -m loanlens.train`, `python -m loanlens.evaluate`.")


bundle = get_model()
data = get_data()
meta = bundle[2] if bundle else None

st.sidebar.title("🔎 LoanLens")
page = st.sidebar.radio("Go to", ["Overview", "Data Explorer", "Single Prediction", "Batch Prediction",
                                  "Model Evaluation", "Provenance & Limitations"])
st.sidebar.caption(f"Model: `{meta['model_version']}`" if meta else "Model: not trained")

# ====================================================================== OVERVIEW
if page == "Overview":
    st.title("LoanLens: Explainable Loan Application Approval Prediction")
    st.info(DISCLAIMER)
    s = CFG["source"]
    c1, c2, c3, c4 = st.columns(4)
    if data is not None:
        c1.metric("Real applications used", f"{len(data):,}")
        c2.metric(f"Train/val ({s['train_year']})", f"{(data.split != 'test').sum():,}")
        c3.metric(f"Held-out test ({s['test_year']})", f"{(data.split == 'test').sum():,}")
    m = jfile(REP / "metrics.json")
    if m:
        c4.metric("Test balanced accuracy", f"{m['test']['balanced_accuracy']:.3f}")
    st.markdown(f"""
**Business context.** The assignment frames this as HDFC Bank improving loan processing. No HDFC data is public, so
the model is trained on **real US mortgage applications** from the CFPB/FFIEC Home Mortgage Disclosure Act (HMDA)
public records for **{s['state']}**.

**Task.** Binary classification at the moment an application is complete but undecided:
**Approved** (action_taken 1 or 2) vs **Rejected** (action_taken 3). Positive class = Approved.

**Model.** {meta['selected_model'] + f" with {meta['calibration']['method']} calibration, decision threshold {meta['threshold']}" if meta else "not trained yet"}.
""")

# ================================================================ DATA EXPLORER
elif page == "Data Explorer":
    st.title("Data Explorer")
    if data is None:
        st.warning("No processed dataset. Run `python -m loanlens.build_dataset`.")
        st.stop()
    f1, f2 = st.columns(2)
    splits = f1.multiselect("Split", ["train", "val", "test"], ["train", "val", "test"])
    purposes = f2.multiselect("Loan purpose", sorted(data.loan_purpose.unique()), sorted(data.loan_purpose.unique()),
                              format_func=lambda v: F.CODE_LABELS["loan_purpose"].get(v, v))
    d = data[data.split.isin(splits) & data.loan_purpose.isin(purposes)].copy()
    d["Outcome"] = d[F.TARGET].map(F.CLASS_NAMES)
    st.caption(f"{len(d):,} real records match the filters.")
    a, b = st.columns(2)
    cb = d.groupby(["split", "Outcome"]).size().reset_index(name="n")
    a.plotly_chart(px.bar(cb, x="split", y="n", color="Outcome", barmode="group", title="Class balance by split"),
                   use_container_width=True)
    feat = b.selectbox("Distribution of", F.CORE_NUMERIC + F.CANDIDATE_NUMERIC,
                       format_func=lambda c: F.HUMAN_LABELS[c])
    lo, hi = d[feat].quantile([0.01, 0.99])
    b.plotly_chart(px.histogram(d[d[feat].between(lo, hi)], x=feat, color="Outcome", nbins=60, barmode="overlay",
                                histnorm="probability density", labels={feat: F.HUMAN_LABELS[feat]},
                                title="Clipped to 1st–99th percentile"), use_container_width=True)
    catf = st.selectbox("Approval rate by", F.CORE_CATEGORICAL + F.CANDIDATE_CATEGORICAL,
                        format_func=lambda c: F.HUMAN_LABELS[c])
    g = d.groupby(catf)[F.TARGET].agg(["mean", "size"]).reset_index()
    g["label"] = g[catf].map(lambda v: F.CODE_LABELS.get(catf, {}).get(v, v))
    st.plotly_chart(px.bar(g, x="label", y="mean", hover_data=["size"], labels={"mean": "Historical approval rate",
                                                                                "label": F.HUMAN_LABELS[catf]}),
                    use_container_width=True)
    au = jfile(PROC / "data_audit.json")
    if au:
        st.subheader("Missingness audit (train split)")
        st.dataframe(pd.DataFrame(au["missingness_audit"]), use_container_width=True, hide_index=True)

# ============================================================= SINGLE PREDICTION
elif page == "Single Prediction":
    st.title("Single Prediction")
    if not bundle:
        need_training(); st.stop()
    model, schema, meta = bundle
    st.caption(DISCLAIMER)
    test_rows = data[data.split == "test"] if data is not None else None

    if st.button("📂 Load real held-out example", help=f"Random application from the {CFG['source']['test_year']} test year"):
        r = test_rows.sample(1).iloc[0]
        st.session_state["loaded"] = r.to_dict()
        for c in schema["order"]:
            if c != "loan_to_income":
                v = r[c]
                st.session_state[f"in_{c}"] = (None if pd.isna(v) else float(v)) if c in schema["numeric"] else str(v)
    loaded = st.session_state.get("loaded")
    for c, info in schema["numeric"].items():          # defaults before widgets are created
        if c != "loan_to_income":
            st.session_state.setdefault(f"in_{c}", float(info["train_median"]))
    for c, info in schema["categorical"].items():
        st.session_state.setdefault(f"in_{c}", info["values_seen_in_train"][0])

    with st.form("single"):
        cols = st.columns(3)
        vals = {}
        for i, c in enumerate([c for c in schema["order"] if c != "loan_to_income"]):
            box = cols[i % 3]
            if c in schema["numeric"]:
                info = schema["numeric"][c]
                vals[c] = box.number_input(info["label"] + " (blank = not reported)", min_value=0.0, key=f"in_{c}",
                                           help=f"Training 1st–99th pct: {info['train_p01']:,.2f} – {info['train_p99']:,.2f}")
            else:
                info = schema["categorical"][c]
                vals[c] = box.selectbox(info["label"], info["values_seen_in_train"], key=f"in_{c}",
                                        format_func=lambda v, c=c: f"{v} – {schema['categorical'][c]['code_labels'][v]}"
                                        if v in schema["categorical"][c]["code_labels"] else v)
        go_ = st.form_submit_button("Predict", type="primary")

    if go_:
        X, err = prepare(pd.DataFrame([vals]), schema)
        if len(err):
            st.error(err.to_string(index=False)); st.stop()
        def same(c):
            lv, iv = loaded[c], vals[c]
            if c in schema["categorical"]:
                return str(lv) == str(iv)
            if pd.isna(lv):
                return iv is None
            return iv is not None and abs(float(lv) - float(iv)) < 1e-6
        edited = loaded is None or not all(same(c) for c in vals)
        if edited:
            st.markdown('<span class="badge hypo">HYPOTHETICAL user input — not an observed applicant</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge real">REAL held-out record {loaded["row_id"]} · historical decision: '
                        f'{F.CLASS_NAMES[int(loaded[F.TARGET])]}</span>', unsafe_allow_html=True)
        res = score(model, meta, X).iloc[0]
        a, b, c = st.columns(3)
        a.metric("Predicted (historical-pattern) decision", res.prediction)
        b.metric("Calibrated P(Approved)", f"{res.prob_approved:.1%}")
        c.metric("Decision threshold", f"{res.threshold:.2f}")
        st.caption(f"Derived loan-to-income ratio: {X.loan_to_income.iloc[0]:.2f}" if pd.notna(X.loan_to_income.iloc[0])
                   else "Loan-to-income not computable (income missing or ≤ 0).")
        try:
            E = get_explainer(model, schema["order"])
            from loanlens.explain import explain_rows
            sv = explain_rows(E, X, schema["order"]).iloc[0].sort_values()
            fig = go.Figure(go.Bar(x=sv.values, y=[F.HUMAN_LABELS[k] for k in sv.index], orientation="h",
                                   marker_color=np.where(sv.values > 0, "#2e8b57", "#c0392b")))
            fig.update_layout(title=f"Local SHAP contributions toward Approved ({E['unit']} of the uncalibrated model)",
                              height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Green pushes toward Approved, red toward Rejected. These are associations learned from historical "
                       "decisions, not causes, and not advice on how to get approved.")
        except Exception as e:
            st.info(f"Local explanation unavailable: {e}")

# ============================================================== BATCH PREDICTION
elif page == "Batch Prediction":
    st.title("Batch Prediction")
    if not bundle:
        need_training(); st.stop()
    model, schema, meta = bundle
    req = [c for c in schema["order"] if c != "loan_to_income"]
    st.markdown("Upload a CSV with these columns (codes as in HMDA; `income_k` in **thousand USD**, "
                "`loan_amount_usd` in USD). `loan_to_income` is computed for you.")
    st.code(",".join(req))
    st.download_button("Download empty template", ",".join(req) + "\n", "loanlens_template.csv")
    demo_fp = ART / "demo_examples_heldout.csv"
    if demo_fp.exists():
        st.download_button("Download real held-out sample rows", demo_fp.read_bytes(), "heldout_sample.csv")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        raw = pd.read_csv(up, dtype=str, keep_default_na=False)
        try:
            X, err = prepare(raw, schema)
        except ValueError as e:
            st.error(str(e)); st.stop()
        if len(err):
            st.error(f"{err.row.nunique()} row(s) rejected"); st.dataframe(err, hide_index=True)
        unseen = unseen_categories(X, schema)
        if unseen:
            st.warning(f"Categories not seen in training (treated as 'infrequent'/ignored by the encoder): {unseen}")
        if len(X):
            out = raw.loc[X.index].join(score(model, meta, X))
            st.success(f"Scored {len(out):,} rows with the saved pipeline.")
            st.dataframe(out.head(200), use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index_label="input_row"), "loanlens_predictions.csv")

# ============================================================== MODEL EVALUATION
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    m = jfile(REP / "metrics.json")
    if not m:
        need_training(); st.stop()
    t, v = m["test"], m["validation"]
    st.caption(f"Test = {CFG['source']['test_year']} applications, evaluated once with the frozen model "
               f"(run #{t['test_run_number_for_this_model_version']} for this version). Positive class = Approved.")
    cols = st.columns(5)
    for col, (k, lab) in zip(cols, [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced acc."),
                                    ("macro_f1", "Macro-F1"), ("roc_auc", "ROC-AUC"), ("brier", "Brier")]):
        col.metric(lab, f"{t[k]:.3f}", delta=f"{t[k] - v[k]:+.3f} vs val", delta_color="off")
    st.subheader("Per-class metrics (test)")
    st.dataframe(pd.DataFrame(t["per_class"]).T.round(3), use_container_width=True)
    st.write(f"PR-AUC with **Approved** as positive: {t['pr_auc_approved_positive']:.3f} "
             f"(baseline = approval rate {t['base_rate_approved']:.3f}); with **Rejected** as positive: "
             f"{t['pr_auc_rejected_positive']:.3f} (baseline {1 - t['base_rate_approved']:.3f}).")
    for img in ["test_confusion_matrix.png", "test_roc_pr_calibration.png", "global_importance.png"]:
        p = REP / "figures" / img
        if p.exists():
            st.image(str(p))
    for name, fn in [("Cross-validation comparison (2023 train split)", "cv_model_comparison.csv"),
                     ("Global importance (validation)", "global_importance_validation.csv"),
                     ("Real misclassified held-out examples", "error_examples_test.csv"),
                     ("Subgroup audit (evaluation only; protected attributes never used by the model)",
                      "subgroup_audit_test.csv")]:
        p = REP / fn
        if p.exists():
            st.subheader(name)
            st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)
    tc = jfile(ART / "threshold_curve.json")
    if tc:
        st.subheader("Threshold choice (validation)")
        st.plotly_chart(px.line(x=tc["thresholds"], y=tc["val_cost"], labels={"x": "threshold", "y": "expected disagreement cost"}),
                        use_container_width=True)
        st.caption(f"Costs: false-approve = {meta['threshold_policy']['cost_false_approve']}, false-reject = "
                   f"{meta['threshold_policy']['cost_false_reject']}. These are costs of disagreeing with history, not loan losses.")

# ===================================================== PROVENANCE & LIMITATIONS
else:
    st.title("Data Provenance & Limitations")
    prov = jfile(RAW / "provenance.json")
    if prov:
        st.json(prov)
    au = jfile(PROC / "data_audit.json")
    if au:
        st.subheader("Filtering log")
        st.dataframe(pd.DataFrame(au["filter_log"]), hide_index=True, use_container_width=True)
        st.subheader("Duplicates policy")
        st.json(au["duplicates"])
    st.markdown((ROOT / "docs" / "limitations.md").read_text() if (ROOT / "docs" / "limitations.md").exists() else "")
