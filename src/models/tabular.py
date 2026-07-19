"""Tabular arms: LightGBM (strong) and MLP (fair).

Both arms exist on purpose. LightGBM is the honest ceiling for CBC indices; the
MLP is the apples-to-apples deep baseline against the image encoder. Reporting
fusion-over-MLP while quietly omitting LightGBM is the standard way papers
manufacture a fusion win, and it is the thing this project is trying not to do.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

from src.config import SEED

# Heavily regularised on purpose. 800 training rows with 8 correlated features is
# tiny for a GBDT; library defaults (31 leaves, 100 unconstrained trees) memorise
# the fold. n_estimators is an upper bound -- early stopping picks the real count.
DEFAULT_LGB_PARAMS: dict = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_lambda": 1.0,
    "verbose": -1,
}


def inner_split(groups: np.ndarray, seed: int, val_frac: float = 0.2):
    """Carve a validation set out of the TRAINING rows, keeping groups intact.

    Early stopping needs a set to watch. Using the outer validation fold for that
    is the obvious move and it is wrong: the number of trees would then be chosen
    by looking at the very rows the arm is scored on, and every reported AUROC
    comes out optimistic. So we split the training rows again and early-stop on
    that, leaving the outer fold genuinely untouched until scoring.

    Grouped, for the same reason the outer split is: a patient must not appear on
    both sides.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))


def fit_lightgbm(
    X_train,
    y_train,
    X_val,
    groups_train,
    params: dict | None = None,
    seed: int = SEED,
):
    """Fit LightGBM, return (model, val_probabilities).

    Pass X with its NaNs intact -- do NOT impute first. LightGBM learns a default
    split direction for missing values, which beats a constant fill on this data:
    0.681 native vs 0.671 median-imputed, same folds and hyperparameters.

    `y_val` is deliberately not a parameter. There is no way for this function to
    peek at the outer fold's labels, which is the point.

    Expect ~0.68 AUROC. Below 0.64 something upstream is broken; above 0.80 you
    have leaked -- check HGB/HCT are excluded and no missingness indicator crept
    into X (see README.md, trap 3).
    """
    cfg = {**DEFAULT_LGB_PARAMS, **(params or {}), "random_state": seed}

    tr_idx, es_idx = inner_split(np.asarray(groups_train), seed)
    # Kept as DataFrames so the fitted model carries real column names: it
    # silences sklearn's feature-name warning and makes `feature_importances_`
    # readable when you plot it.
    y_train = np.asarray(y_train)

    model = lgb.LGBMClassifier(**cfg)
    model.fit(
        X_train.iloc[tr_idx],
        y_train[tr_idx],
        eval_set=[(X_train.iloc[es_idx], y_train[es_idx])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model, model.predict_proba(X_val)[:, 1]


class TabularMLP(nn.Module):
    """2-layer MLP on CBC indices. The 'fair' tabular arm.

    Also reused as the tabular branch of the fusion model -- `forward_features`
    returns the penultimate embedding for concatenation.
    """

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 1)
        self.out_dim = hidden

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)


def train_mlp(
    X_train,
    y_train,
    X_val,
    groups_train,
    seed: int = SEED,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    device: str | None = None,
):
    """Train `TabularMLP`, return (model, val_probabilities).

    X must already be imputed and standardised -- a network cannot ingest NaN,
    and un-scaled inputs (PLT ~300 next to MCHC ~32) make the first layer's job
    needlessly hard. Both must be fitted on training rows only; `train.py` does
    that with `CBCImputer` and train-fold mean/std.

    Early stopping watches an inner split carved from the training rows, exactly
    as in `fit_lightgbm`, so the outer fold stays untouched. Best-epoch weights
    are restored before predicting.
    """
    from sklearn.metrics import roc_auc_score

    from src.config import DEVICE

    device = device or DEVICE
    torch.manual_seed(seed)

    tr_idx, es_idx = inner_split(np.asarray(groups_train), seed)
    X_train, y_train = np.asarray(X_train, dtype=np.float32), np.asarray(y_train, dtype=np.float32)

    to = lambda a: torch.tensor(a, device=device)
    xt, yt = to(X_train[tr_idx]), to(y_train[tr_idx])
    xe, ye = to(X_train[es_idx]), y_train[es_idx]
    xv = to(np.asarray(X_val, dtype=np.float32))

    model = TabularMLP(X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, waited = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        # drop_last: BatchNorm needs >1 sample per batch, and a stray batch of 1
        # crashes rather than degrading gracefully.
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(ye, torch.sigmoid(model(xe)).cpu().numpy())

        if auc > best_auc:
            best_auc, waited = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        return model, torch.sigmoid(model(xv)).cpu().numpy()
