"""Assemble the master dataframe: one row per sample, with image paths + CBC + fold.

Run this once; every arm reads its output. Writing it to disk (rather than
rebuilding per arm) is what guarantees all four arms see identical folds.

    python -m src.build_dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ANE_I, ANE_II, ARTIFACTS, CLASS_DIRS, group_id
from src.parse_cbc import audit_coverage, load_cbc_table
from src.splits import make_folds, save_folds


def index_parent_images() -> pd.DataFrame:
    """One row per AneRBC-I original image.

    Returns: sample_id, parent_id, cls, group, img_path
    """
    rows = []
    for cls, dirname in CLASS_DIRS.items():
        for p in sorted((ANE_I / dirname / "Original_images").glob("*.png")):
            pid = p.stem.split("_")[0]
            rows.append(
                {
                    "sample_id": p.stem,
                    "parent_id": pid,
                    "cls": cls,
                    "group": group_id(pid, cls),
                    "img_path": str(p),
                }
            )
    return pd.DataFrame(rows)


def index_crop_images() -> pd.DataFrame:
    """One row per AneRBC-II crop (12 per parent).

    Filenames are `{parent:04d}_{crop:02d}_{h|a}.png`. Join back to AneRBC-I via
    int(parent_id) -> f"{id:03d}". Verified 500/500 match, no orphans.

    Returns: crop_id, parent_id, crop_no, cls, group, img_path
    """
    rows = []
    for cls, dirname in CLASS_DIRS.items():
        for p in sorted((ANE_II / dirname / "Original_images").glob("*.png")):
            parent, crop_no, _ = p.stem.split("_")
            rows.append(
                {
                    "crop_id": p.stem,
                    "parent_id": f"{int(parent):03d}",
                    "crop_no": int(crop_no),
                    "cls": cls,
                    "group": group_id(parent, cls),
                    "img_path": str(p),
                }
            )
    return pd.DataFrame(rows)


def build_master() -> pd.DataFrame:
    """CBC + parent image paths + fold assignment, one row per sample."""
    cbc = load_cbc_table()
    imgs = index_parent_images()

    df = cbc.merge(
        imgs[["sample_id", "img_path"]], on="sample_id", how="inner", validate="one_to_one"
    )
    if len(df) != 1000:
        raise ValueError(f"expected 1000 joined samples, got {len(df)}")

    return make_folds(df)


class CBCImputer:
    """Fill missing CBC values. Fit on train folds, apply to val.

    The fit/transform split is the whole point: a median computed over all 1000
    samples leaks the validation distribution into training. Fit once per fold,
    on the training rows only, then transform both sides with those same stats.

    Usage inside a CV loop:

        imp = CBCImputer().fit(train_df, features)
        train_x = imp.transform(train_df)
        val_x = imp.transform(val_df)      # train medians, not val medians

    Do NOT use this for the LightGBM arm
    ------------------------------------
    LightGBM handles NaN natively -- it learns a default split direction for
    missing values, which strictly dominates a constant fill. Measured, 5-fold,
    same folds and hyperparameters:

        native NaN       0.6807
        median-imputed   0.6705

    Pass the raw columns with their NaNs to LightGBM. This class exists for the
    MLP, image and fusion arms, which cannot ingest NaN.

    On `add_indicators` -- read this before turning it on
    ----------------------------------------------------
    Adding RDW_missing / MPV_missing columns is a reasonable-sounding idea with a
    real clinical rationale: analysers do suppress indices when the underlying
    count is too low to be reliable. That rationale was tested against this
    dataset and does not hold here:

        corr(RDW_missing, RBC)     = +0.059   (mechanism predicts strongly negative)
        corr(RDW_missing, n_rows)  = -0.791   (report length explains it)
        corr(MPV_missing, PLT)     = -0.131
        corr(MPV_missing, n_rows)  = -0.442

    RDW is missing in 84% of short-format reports vs 7% of full-format ones, and
    RDW/MPV go missing together (phi = 0.389, p = 7e-35) despite being different
    analytes on different channels. The missingness tracks which report TEMPLATE
    the lab used, not patient physiology.

    That template happens to correlate with the label (+10.8% / +10.0% class gap),
    so indicators do measurably help -- by encoding a recording artifact that will
    not transfer to another hospital.

    Left switchable so you can measure it rather than take the above on trust. If
    you report a result with indicators on, you must also report it with them off
    and show the conclusion survives. See README.md, trap 3.
    """

    def __init__(self, strategy: str = "median", add_indicators: bool = False):
        if strategy not in ("median", "mean"):
            raise ValueError("strategy must be 'median' or 'mean'")
        self.strategy = strategy
        self.add_indicators = add_indicators
        self.features: list[str] = []
        self.fill_values: dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame, features: list[str]) -> "CBCImputer":
        """Learn fill values from TRAINING rows only."""
        missing = [f for f in features if f not in train_df]
        if missing:
            raise KeyError(f"features absent from dataframe: {missing}")

        self.features = list(features)
        agg = train_df[features].median() if self.strategy == "median" else train_df[features].mean()

        # A feature entirely absent from a training fold has no learnable fill.
        # Fall back to 0.0 post-standardisation rather than propagating NaN.
        self.fill_values = {f: (0.0 if pd.isna(v) else float(v)) for f, v in agg.items()}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned fill values. Returns a feature-matrix copy."""
        if not self.fill_values:
            raise RuntimeError("call fit() before transform()")

        out = df.copy()
        if self.add_indicators:
            for f in self.features:
                out[f"{f}_missing"] = out[f].isna().astype(np.int8)

        for f in self.features:
            out[f] = out[f].fillna(self.fill_values[f])
        return out

    def feature_names(self) -> list[str]:
        """Column order the model should consume, indicators included if on."""
        names = list(self.features)
        if self.add_indicators:
            names += [f"{f}_missing" for f in self.features]
        return names


if __name__ == "__main__":
    master = build_master()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print(audit_coverage(master).to_string(index=False), "\n")

    out = ARTIFACTS / "master.parquet"
    master.to_parquet(out, index=False)
    save_folds(master)
    print(f"{len(master)} samples -> {out}")
    print(master.groupby("fold").agg(n=("y", "size"), pos_rate=("y", "mean")).round(3))
