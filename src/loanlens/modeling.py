"""Preprocessing + estimator definitions. Everything that learns from data
(imputation medians, one-hot vocabularies, scaling) lives inside one sklearn
Pipeline, so cross-validation refits it inside each training fold."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import features as F

LOG_COLS = {"income_k", "loan_amount_usd", "loan_to_income", "property_value"}


def _log1p_clip(x):
    return np.log1p(np.clip(x, 0, None))


def feature_lists(rich_allowed: list[str]) -> tuple[list[str], list[str]]:
    num = F.CORE_NUMERIC + [c for c in F.CANDIDATE_NUMERIC if c in rich_allowed]
    cat = F.CORE_CATEGORICAL + [c for c in F.CANDIDATE_CATEGORICAL if c in rich_allowed]
    return num, cat


def make_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    log_cols = [c for c in num if c in LOG_COLS]
    lin_cols = [c for c in num if c not in LOG_COLS]
    num_log = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("log1p", FunctionTransformer(_log1p_clip, feature_names_out="one-to-one")),
                        ("scale", StandardScaler())])
    num_lin = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cats = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="NA")),
                     ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                              sparse_output=False))])
    parts = [("num_log", num_log, log_cols), ("cat", cats, cat)]
    if lin_cols:
        parts.insert(1, ("num", num_lin, lin_cols))
    return ColumnTransformer(parts, verbose_feature_names_out=True)


def candidates(seed: int) -> dict:
    """name -> (estimator, small hyper-parameter grid). Grids are kept small on purpose."""
    return {
        "Dummy (majority prior)": (DummyClassifier(strategy="prior"), {}),
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                                {"clf__C": [0.1, 1.0, 10.0]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=seed),
                          {"clf__max_depth": [4, 6, 8], "clf__min_samples_leaf": [20, 100]}),
        "Random Forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=seed),
                          {"clf__max_depth": [8, 14], "clf__min_samples_leaf": [5, 20]}),
        "Gradient Boosting (HistGB)": (HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
                                       {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31]}),
    }


def make_pipeline(estimator, num, cat) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(num, cat)), ("clf", estimator)])
