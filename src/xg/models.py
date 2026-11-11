"""Models: a logistic-regression baseline and a gradient-boosting model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

GLM_FEATURES = ["log_distance", "angle"]

CATEGORICAL = ["body_part", "shot_type", "technique", "play_pattern"]
BOOLEAN = ["under_pressure", "first_time", "one_on_one", "follows_dribble", "aerial_won", "open_goal"]
NUMERIC = ["distance", "angle", "x", "y",
           "n_def_in_cone", "n_att_in_cone", "gk_in_cone", "gk_dist_to_goal",
           "gk_dist_to_line", "nearest_def_dist", "n_def_within_3"]
GBM_FEATURES = NUMERIC + BOOLEAN + CATEGORICAL


# ---------- GLM ----------------------------------------------------------

def fit_glm(df: pd.DataFrame) -> LogisticRegression:
    """Unpenalised logistic regression: logit P(goal) = b0 + b1 log(d) + b2 angle."""
    model = LogisticRegression(C=np.inf, max_iter=1000)   # C=inf: no regularisation
    model.fit(df[GLM_FEATURES].to_numpy(), df.is_goal.to_numpy())
    return model


def glm_coef_table(model: LogisticRegression, df: pd.DataFrame) -> pd.DataFrame:
    """Coefficients with standard errors from the inverse Fisher information.

    For logistic regression, Cov(beta) = (X^T W X)^-1 with W = diag(p(1-p)).
    """
    X = np.column_stack([np.ones(len(df)), df[GLM_FEATURES].to_numpy()])
    beta = np.concatenate([model.intercept_, model.coef_.ravel()])
    p = 1.0 / (1.0 + np.exp(-X @ beta))
    fisher = X.T @ (X * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(fisher)))
    return pd.DataFrame({"coef": beta, "std_err": se, "z": beta / se},
                        index=["intercept"] + GLM_FEATURES)


# ---------- Gradient boosting -------------------------------------------

class CategoryEncoder:
    """Map string categories to integer codes, using the training set's levels only.

    Levels unseen at fit time become NaN, which the GBM treats as 'missing'
    rather than silently matching them to an arbitrary training category.
    """

    def __init__(self, features: list[str]):
        self.features = features

    def fit(self, df: pd.DataFrame):
        self.cats = [c for c in self.features if c in CATEGORICAL]
        self.levels = {c: sorted(df[c].dropna().unique()) for c in self.cats}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for c in self.features:
            if c in BOOLEAN:
                X[c] = X[c].astype(float)
        for c in self.cats:
            X[c] = pd.Categorical(X[c], categories=self.levels[c]).codes.astype(float)
            X.loc[X[c] < 0, c] = np.nan
        return X.astype(float)


def make_gbm(features: list[str], **params) -> HistGradientBoostingClassifier:
    cat_mask = [c in CATEGORICAL for c in features]
    defaults = dict(loss="log_loss", max_iter=400, learning_rate=0.05,
                    max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0,
                    categorical_features=cat_mask, early_stopping=False, random_state=0)
    defaults.update(params)
    return HistGradientBoostingClassifier(**defaults)


class GBMModel:
    def __init__(self, features: list[str] = GBM_FEATURES, **params):
        self.features = list(features)
        self.params = params

    def fit(self, df: pd.DataFrame):
        self.enc = CategoryEncoder(self.features).fit(df)
        self.model = make_gbm(self.features, **self.params).fit(self.enc.transform(df), df.is_goal)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.enc.transform(df))[:, 1]
