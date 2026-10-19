"""Feature and target definitions — the single source of truth for what the
model may see. Every column here was checked against the official HMDA LAR
field documentation (see config.toml -> source.field_docs).

Prediction moment: the application has been submitted and the listed
application facts are known, but no credit decision has been made.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- target
TARGET = "approved"            # 1 = Approved, 0 = Rejected
CLASS_NAMES = {1: "Approved", 0: "Rejected"}


def derive_target(action_taken: pd.Series) -> pd.Series:
    a = pd.to_numeric(action_taken, errors="coerce")
    out = pd.Series(np.nan, index=a.index)
    out[a.isin([1, 2])] = 1
    out[a.isin([3])] = 0
    return out


# ------------------------------------------------- columns NEVER used as inputs
LEAKAGE_COLUMNS = {
    # outcome itself and outcome explanations
    "action_taken": "the label",
    "denial_reason-1": "outcome explanation", "denial_reason-2": "outcome explanation",
    "denial_reason-3": "outcome explanation", "denial_reason-4": "outcome explanation",
    # post-decision pricing / closing / disposition (NA for denials -> perfect proxy)
    "interest_rate": "set at/after approval", "rate_spread": "post-decision pricing",
    "hoepa_status": "only determined for originated loans",
    "total_loan_costs": "closing cost", "total_points_and_fees": "closing cost",
    "origination_charges": "closing cost", "discount_points": "closing cost",
    "lender_credits": "closing cost", "purchaser_type": "post-origination sale",
    "prepayment_penalty_term": "pricing term mostly reported for originations",
    "intro_rate_period": "pricing term mostly reported for originations",
}
ID_GEO_COLUMNS = {
    "lei": "lender identifier", "census_tract": "granular geography",
    "county_code": "granular geography", "state_code": "constant (one state)",
    "activity_year": "defines the split, not a feature",
}
PROTECTED_COLUMNS = [  # kept ONLY in a separate audit table
    "derived_race", "derived_ethnicity", "derived_sex", "applicant_age",
    "applicant_age_above_62",
]
PROXY_COLUMNS = {
    "tract_minority_population_percent": "neighbourhood race proxy",
    "tract_to_msa_income_percentage": "neighbourhood income proxy (redlining risk)",
    "tract_population": "geography", "tract_owner_occupied_units": "geography",
    "tract_one_to_four_family_homes": "geography", "tract_median_age_of_housing_units": "geography",
    "ffiec_msa_md_median_family_income": "geography",
}

# ------------------------------------------------------ model input features
CORE_NUMERIC = ["income_k", "loan_amount_usd", "loan_to_income"]
CORE_CATEGORICAL = [
    "loan_type", "loan_purpose", "lien_status", "occupancy_type", "preapproval",
    "derived_dwelling_category", "open_end_line_of_credit", "metro_area", "has_coapplicant",
]
# Richer underwriting fields: used only if their missingness passes the audit
CANDIDATE_NUMERIC = ["loan_to_value_ratio", "property_value", "loan_term"]
CANDIDATE_CATEGORICAL = ["debt_to_income_ratio"]

MISSINGNESS_GAP_LIMIT = 0.10   # max |missing% Approved - missing% Rejected| on TRAIN

HUMAN_LABELS = {
    "income_k": "Gross annual income (thousand USD)",
    "loan_amount_usd": "Loan amount (USD)",
    "loan_to_income": "Loan amount ÷ annual income",
    "loan_type": "Loan type", "loan_purpose": "Loan purpose", "lien_status": "Lien status",
    "occupancy_type": "Occupancy", "preapproval": "Preapproval requested",
    "derived_dwelling_category": "Dwelling category",
    "open_end_line_of_credit": "Open-end line of credit",
    "metro_area": "Property in a metro area (MSA/MD)",
    "has_coapplicant": "Has co-applicant",
    "loan_to_value_ratio": "Loan-to-value (%)",
    "property_value": "Property value (USD, rounded by publisher)",
    "loan_term": "Loan term (months)",
    "debt_to_income_ratio": "Debt-to-income (published bins)",
}

CODE_LABELS = {
    "loan_type": {"1": "Conventional", "2": "FHA", "3": "VA", "4": "USDA RHS/FSA"},
    "loan_purpose": {"1": "Home purchase", "2": "Home improvement", "31": "Refinancing",
                     "32": "Cash-out refinancing", "4": "Other purpose", "5": "Not applicable"},
    "lien_status": {"1": "First lien", "2": "Subordinate lien"},
    "occupancy_type": {"1": "Principal residence", "2": "Second residence", "3": "Investment property"},
    "preapproval": {"1": "Requested", "2": "Not requested"},
    "open_end_line_of_credit": {"1": "Open-end", "2": "Not open-end", "1111": "Exempt"},
    "metro_area": {"Metro": "Metro (MSA/MD)", "Non-metro": "Not in an MSA/MD"},
    "has_coapplicant": {"Yes": "Yes", "No": "No", "Unknown": "Unknown (exempt reporter)"},
}

NON_METRO_CODES = {"99999", "NA", ""}


def _num(s: pd.Series) -> pd.Series:
    """Parse HMDA numeric fields: 'NA', 'Exempt' and blanks become NaN."""
    return pd.to_numeric(s.astype(str).str.strip().replace({"NA": None, "Exempt": None, "": None}),
                         errors="coerce")


def _cat(s: pd.Series) -> pd.Series:
    """Categorical codes kept as strings; NA/Exempt become explicit categories."""
    return s.astype(str).str.strip().replace({"nan": "NA", "": "NA"})


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw HMDA columns -> model-ready feature frame (no target, no protected attrs).

    Units: HMDA `income` is in THOUSANDS of USD; `loan_amount` is in USD.
    """
    out = pd.DataFrame(index=raw.index)
    income_k = _num(raw["income"])
    out["income_k"] = income_k.where(income_k > 0)            # <=0 is not a usable annual income
    out["loan_amount_usd"] = _num(raw["loan_amount"])
    income_usd = out["income_k"] * 1000.0                      # explicit unit conversion
    out["loan_to_income"] = (out["loan_amount_usd"] / income_usd).where(income_usd > 0)

    for c in ["loan_type", "loan_purpose", "lien_status", "occupancy_type", "preapproval",
              "derived_dwelling_category", "open-end_line_of_credit"]:
        out[c.replace("-", "_")] = _cat(raw[c])

    msa = raw["derived_msa-md"].astype(str).str.strip()
    out["metro_area"] = np.where(msa.isin(NON_METRO_CODES), "Non-metro", "Metro")
    co = raw["co-applicant_credit_score_type"].astype(str).str.strip()
    # 10 = "No co-applicant" (documented); 1111 = Exempt -> unknown, not "Yes"
    out["has_coapplicant"] = np.select([co == "10", co.isin(["1111", "NA", "", "nan"])],
                                       ["No", "Unknown"], default="Yes")

    for c in CANDIDATE_NUMERIC:
        out[c] = _num(raw[c])
    out["debt_to_income_ratio"] = _cat(raw["debt_to_income_ratio"])
    return out


def to_dollars(income_k: float) -> float:
    return float(income_k) * 1000.0
