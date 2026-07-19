"""Grouped stratified cross-validation.

Every arm must use the SAME folds, or the fusion-minus-unimodal delta is
measuring fold noise instead of a modality effect. Build folds once, persist
them, and have every arm read the same file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import ARTIFACTS, N_FOLDS, SEED


def make_folds(df: pd.DataFrame, n_splits: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """Assign each row a fold index, grouped by `group` and stratified on `y`.

    `df` needs columns `y` and `group`. Returns a copy with an added `fold`
    column. Rows sharing a `group` always land in the same fold.
    """
    for col in ("y", "group"):
        if col not in df:
            raise KeyError(f"df must have a '{col}' column")

    out = df.copy()
    out["fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(splitter.split(out, out["y"], out["group"])):
        out.iloc[val_idx, out.columns.get_loc("fold")] = fold

    assert (out["fold"] >= 0).all(), "some rows were never assigned a fold"
    check_no_group_leak(out)
    return out


def check_no_group_leak(df: pd.DataFrame) -> None:
    """Fail loudly if any group spans more than one fold.

    This is the assertion that catches trap 2. Call it after every split, and
    especially after expanding parents to AneRBC-II crops -- the expansion is
    where a group silently gets torn apart.
    """
    spread = df.groupby("group")["fold"].nunique()
    bad = spread[spread > 1]
    if len(bad):
        raise AssertionError(
            f"{len(bad)} group(s) span multiple folds, e.g. {list(bad.index[:5])}. "
            "Folds are leaking; do not trust any metric computed on them."
        )


def expand_to_crops(parent_df: pd.DataFrame, crop_index: pd.DataFrame) -> pd.DataFrame:
    """Propagate parent-level fold assignments down to AneRBC-II crops.

    Assign folds at the PARENT level first, then expand. Never run the splitter
    on the 12k crop rows directly -- StratifiedGroupKFold would technically
    honour the groups, but any later reshuffle or merge can break the tie, and
    the failure is silent.

    `crop_index` needs `group` plus whatever crop-level columns you carry.
    """
    merged = crop_index.merge(
        parent_df[["group", "fold", "y"]], on="group", how="left", validate="many_to_one"
    )
    if merged["fold"].isna().any():
        missing = merged.loc[merged["fold"].isna(), "group"].unique()[:5]
        raise ValueError(f"crops with no parent fold, e.g. {list(missing)}")
    check_no_group_leak(merged)
    return merged


def save_folds(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or ARTIFACTS / "folds" / f"folds_seed{SEED}_k{N_FOLDS}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["sample_id", "group", "y", "fold"]].to_csv(path, index=False)
    return path


def load_folds(path: Path | None = None) -> pd.DataFrame:
    path = path or ARTIFACTS / "folds" / f"folds_seed{SEED}_k{N_FOLDS}.csv"
    return pd.read_csv(path)


def fold_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-fold size and class balance. Eyeball this before training anything."""
    return (
        df.groupby("fold")
        .agg(n=("y", "size"), n_groups=("group", "nunique"), pos_rate=("y", "mean"))
        .round(3)
    )


if __name__ == "__main__":
    from src.parse_cbc import load_cbc_table

    folds = make_folds(load_cbc_table())
    print(fold_summary(folds).to_string())
    print(f"\nsaved -> {save_folds(folds)}")
