"""Validate raw files, apply documented filters, derive the target, engineer
features, audit missingness/duplicates, and write the processed dataset plus a
split manifest. Deterministic: same raw files -> same outputs.

    python -m loanlens.build_dataset
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import features as F
from .common import CFG, path, raw_file, read_json, sha256, write_json

REQUIRED = [
    "activity_year", "state_code", "action_taken", "income", "loan_amount", "loan_type",
    "loan_purpose", "lien_status", "occupancy_type", "preapproval", "derived_dwelling_category",
    "open-end_line_of_credit", "derived_msa-md", "co-applicant_credit_score_type",
    "loan_to_value_ratio", "property_value", "loan_term", "debt_to_income_ratio",
    "business_or_commercial_purpose", "reverse_mortgage",
    "derived_race", "derived_ethnicity", "derived_sex",
]


class SchemaError(ValueError):
    pass


def validate_raw(df: pd.DataFrame, year: int) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"{year}: missing columns {missing}")
    years = set(df["activity_year"].astype(str).unique())
    if years != {str(year)}:
        raise SchemaError(f"{year}: activity_year values {years}")
    states = set(df["state_code"].astype(str).unique())
    if states != {CFG["source"]["state"]}:
        raise SchemaError(f"{year}: state_code values {states}")
    bad = set(df["action_taken"].astype(str)) - {"1", "2", "3"}
    if bad:
        raise SchemaError(f"{year}: unexpected action_taken codes {bad} (query should filter to 1,2,3)")


def load_year(year: int) -> tuple[pd.DataFrame, list[dict]]:
    fp = raw_file(year)
    if not fp.exists():
        raise SystemExit(f"Raw file missing: {fp}\nRun `python -m loanlens.download` or register a manual download.")
    prov = read_json(path("raw_dir") / "provenance.json")["files"][str(year)]
    if sha256(fp) != prov["sha256"]:
        raise SystemExit(f"Checksum mismatch for {fp}; raw files must not change.")
    df = pd.read_csv(fp, dtype=str, keep_default_na=False)
    validate_raw(df, year)
    df["row_id"] = [f"{year}-{i:06d}" for i in range(len(df))]  # line position in the raw file
    log = [{"year": year, "step": "raw rows (actions 1,2,3 from API query)", "rows": len(df)}]

    df = df[df["business_or_commercial_purpose"] != "1"]
    log.append({"year": year, "step": "exclude business/commercial-purpose applications", "rows": len(df)})
    df = df[df["reverse_mortgage"] != "1"]
    log.append({"year": year, "step": "exclude reverse mortgages", "rows": len(df)})
    amt = pd.to_numeric(df["loan_amount"], errors="coerce")
    df = df[amt.notna() & (amt > 0)]
    log.append({"year": year, "step": "require a positive loan amount", "rows": len(df)})
    return df.reset_index(drop=True), log


def missingness_audit(feat: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for c in F.CORE_NUMERIC + F.CANDIDATE_NUMERIC + F.CANDIDATE_CATEGORICAL + ["income_k"]:
        miss = feat[c].isna() if c not in F.CANDIDATE_CATEGORICAL else feat[c].isin(["NA", "Exempt"])
        ma, mr = miss[y == 1].mean(), miss[y == 0].mean()
        rows.append({"feature": c, "missing_overall": miss.mean(), "missing_approved": ma,
                     "missing_rejected": mr, "gap": abs(ma - mr)})
    out = pd.DataFrame(rows).drop_duplicates("feature")
    out["decision"] = ["core (missing values imputed + flagged by audit)" if f in F.CORE_NUMERIC
                       else ("include" if g <= F.MISSINGNESS_GAP_LIMIT else "EXCLUDE: missingness tracks the label")
                       for f, g in zip(out.feature, out.gap)]
    return out


def main() -> None:
    s, sp = CFG["source"], CFG["split"]
    frames, log = [], []
    for year in (s["train_year"], s["test_year"]):
        df, lg = load_year(year)
        frames.append(df)
        log += lg
    raw = pd.concat(frames, ignore_index=True)

    y = F.derive_target(raw["action_taken"])
    assert y.notna().all()
    feat = F.engineer(raw)
    data = pd.concat([raw[["row_id", "activity_year"]], feat], axis=1)
    data[F.TARGET] = y.astype(int)

    # ---- split: test = later year; validation carved from the earlier year only
    is_test = data["activity_year"] == str(s["test_year"])
    tr_idx, va_idx = train_test_split(data.index[~is_test], test_size=sp["validation_fraction"],
                                      stratify=data.loc[~is_test, F.TARGET], random_state=sp["random_state"])
    data["split"] = "test"
    data.loc[tr_idx, "split"] = "train"
    data.loc[va_idx, "split"] = "val"

    # ---- audits use TRAIN rows only
    tr = data["split"] == "train"
    audit = missingness_audit(feat[tr], data.loc[tr, F.TARGET])
    allowed = audit[audit.decision == "include"].feature.tolist()

    # duplicates: identical public rows are NOT assumed to be the same applicant
    feat_cols = F.CORE_NUMERIC + F.CORE_CATEGORICAL + F.CANDIDATE_NUMERIC + F.CANDIDATE_CATEGORICAL
    key = pd.util.hash_pandas_object(data[feat_cols].astype(str), index=False)
    full_dup = raw.drop(columns=["row_id"]).duplicated(keep=False)
    by_key = data.assign(k=key).groupby("k")["split"].nunique()
    dup_info = {
        "exact_full_row_duplicates": int(full_dup.sum()),
        "rows_sharing_identical_feature_vectors": int(key.duplicated(keep=False).sum()),
        "feature_vectors_appearing_in_more_than_one_split": int((by_key > 1).sum()),
        "policy": "Kept. HMDA privacy rounding/binning can make different applicants identical, and there is "
                  "no applicant ID to group on, so rows are not deduplicated. Counts are reported for transparency.",
    }

    out = path("processed_dir")
    model_cols = ["row_id", "activity_year", "split", F.TARGET] + feat_cols
    data[model_cols].to_parquet(out / "loanlens_dataset.parquet", index=False)
    prot = raw[["row_id"] + [c for c in F.PROTECTED_COLUMNS if c in raw.columns]]
    prot.to_parquet(out / "audit_only_protected.parquet", index=False)  # never joined into training
    data[["row_id", "split"]].to_csv(out / "split_manifest.csv", index=False)

    summary = {
        "filter_log": log,
        "class_counts": {sp_: data[data.split == sp_][F.TARGET].map(F.CLASS_NAMES).value_counts().to_dict()
                         for sp_ in ["train", "val", "test"]},
        "action_taken_counts": raw.groupby(["activity_year", "action_taken"]).size().rename("n")
                                  .reset_index().to_dict("records"),
        "missingness_audit": audit.round(4).to_dict("records"),
        "rich_features_allowed": allowed,
        "duplicates": dup_info,
        "income_units": "HMDA income is thousands of USD; income_k<=0 treated as missing; loan_to_income uses income_k*1000",
        "loan_amount_values_ending_5000_share": float((data.loan_amount_usd % 10000 == 5000).mean()),
    }
    write_json(summary, out / "data_audit.json")
    print(pd.DataFrame(log).to_string(index=False))
    print(summary["class_counts"])
    print(audit.round(3).to_string(index=False))
    print("rich features allowed:", allowed)
    print(dup_info)


if __name__ == "__main__":
    main()
