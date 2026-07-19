"""Arm runner: grouped CV loop, out-of-fold predictions, results table.

    python -m src.train --arm lgbm
    python -m src.train --arm mlp
    python -m src.train --arm image
    python -m src.train --arm concat
    python -m src.train --arm film
    python -m src.train --compare concat lgbm     # bootstrap CI on the delta

Every arm writes out-of-fold predictions to artifacts/preds/{arm}.csv, indexed by
sample_id. The comparison step reads those files rather than re-running anything,
which is what keeps the paired bootstrap genuinely paired.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.build_dataset import CBCImputer
from src.config import ARTIFACTS, FEATURES_DEFAULT, N_FOLDS, SEED
from src.data import build_image_cache, make_loader
from src.metrics import aggregate_folds, bootstrap_delta, summarize
from src.models.common import fit_deep
from src.models.fusion import ConcatFusion, FiLMFusion
from src.models.image import ImageEncoder
from src.models.tabular import TabularMLP, fit_lightgbm, inner_split, train_mlp

_IMAGES = None


def load_images():
    """Load the uint8 image cache once and reuse it across folds."""
    global _IMAGES
    if _IMAGES is None:
        _IMAGES = build_image_cache(load_master())
    return _IMAGES

ARMS = ("lgbm", "mlp", "image", "concat", "film")


def load_master() -> pd.DataFrame:
    path = ARTIFACTS / "master.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `python -m src.build_dataset` first")
    df = pd.read_parquet(path)
    # Row i of the dataframe is image i of the cache. Carrying the position
    # explicitly keeps that mapping intact through fold subsetting, where the
    # pandas index no longer matches the array offset.
    df["img_idx"] = np.arange(len(df))
    return df


def train_one_fold(arm: str, train_df, val_df, features: list[str], seed: int, stain_norm: bool = False):
    """Train `arm` on train_df, return validation probabilities aligned to val_df.

    Fold-local rules that decide whether the whole comparison is valid:
      - Impute and standardise using TRAIN-fold statistics only.
      - Give every deep arm the same epoch budget, optimiser and schedule.
      - Seed per fold (`seed + fold`) so arms are comparable but folds differ.
      - Early stopping watches an inner split of the training rows, never the
        outer fold. Both tabular arms already do this; keep it that way for the
        image and fusion arms or their scores are not comparable to these.
    """
    groups = train_df["group"].to_numpy()

    if arm == "lgbm":
        # Raw NaNs on purpose -- LightGBM handles them natively and scores better
        # than imputation (0.681 vs 0.671). See CBCImputer's docstring.
        return fit_lightgbm(
            train_df[features],
            train_df["y"].to_numpy(),
            val_df[features],
            groups,
            seed=seed,
        )[1]

    if arm == "mlp":
        X_train, X_val = prepare_tabular(train_df, val_df, features, standardize=True)
        return train_mlp(X_train, train_df["y"].to_numpy(), X_val, groups, seed=seed)[1]

    if arm in ("image", "concat", "film"):
        return train_deep_fold(arm, train_df, val_df, features, seed, stain_norm=stain_norm)

    raise NotImplementedError(f"arm '{arm}' not implemented yet")


def train_deep_fold(arm, train_df, val_df, features, seed, stain_norm: bool = False):
    """Image / concat / FiLM: split, build loaders, hand to the shared loop."""
    images = load_images()

    # Same inner-split discipline as the tabular arms: early stopping watches a
    # slice of the training rows, so the outer fold stays untouched until scoring.
    tr_idx, es_idx = inner_split(train_df["group"].to_numpy(), seed)
    inner_train, inner_es = train_df.iloc[tr_idx], train_df.iloc[es_idx]

    if arm == "image":
        feats, tab_train, tab_es, tab_val = [], None, None, None
        model = ImageEncoder()
    else:
        # Standardised on TRAIN-fold statistics, then sliced -- fitting the
        # scaler after the inner split would use different stats per arm.
        X_train, X_val = prepare_tabular(train_df, val_df, features, standardize=True)
        feats, tab_train, tab_es, tab_val = features, X_train[tr_idx], X_train[es_idx], X_val
        model = (
            ConcatFusion(ImageEncoder(), TabularMLP(len(features)))
            if arm == "concat"
            else FiLMFusion(len(features))
        )

    loader = lambda df, train, tab: make_loader(
        df, images, feats, train=train, tabular=tab, stain_norm=stain_norm
    )
    return fit_deep(
        model,
        loader(inner_train, True, tab_train),
        loader(inner_es, False, tab_es),
        loader(val_df, False, tab_val),
        seed=seed,
    )[1]


def prepare_tabular(train_df, val_df, features: list[str], standardize: bool):
    """Impute and optionally standardise, using TRAIN-fold statistics only.

    Both steps are fitted on the training rows and merely applied to val. A
    median or a mean computed over both sides leaks the validation distribution
    into training -- the single easiest way to inflate every number downstream.
    """
    imputer = CBCImputer().fit(train_df, features)
    cols = imputer.feature_names()
    X_train = imputer.transform(train_df)[cols].to_numpy(dtype=float)
    X_val = imputer.transform(val_df)[cols].to_numpy(dtype=float)

    if standardize:
        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        std = np.where(std == 0, 1.0, std)  # a constant column would divide by zero
        X_train, X_val = (X_train - mean) / std, (X_val - mean) / std

    return X_train, X_val


def run_arm(
    arm: str,
    features: list[str] = FEATURES_DEFAULT,
    seed: int = SEED,
    name: str | None = None,
    stain_norm: bool = False,
) -> pd.DataFrame:
    """Run one arm across all folds; persist out-of-fold predictions.

    `name` overrides the output filename. Pass it whenever the feature set is not
    the default, so an ablation cannot silently overwrite the headline run's
    predictions -- `compare()` reads those files and would otherwise be diffing
    two things that are not what their names claim.
    """
    name = name or arm
    df = load_master()
    oof = np.full(len(df), np.nan)
    per_fold = []

    for fold in range(N_FOLDS):
        is_val = df["fold"] == fold
        p = train_one_fold(arm, df[~is_val], df[is_val], features, seed + fold, stain_norm)
        oof[is_val.to_numpy()] = p
        per_fold.append(summarize(df.loc[is_val, "y"].to_numpy(), np.asarray(p)))
        print(f"  fold {fold}: " + "  ".join(f"{k}={v:.3f}" for k, v in per_fold[-1].items()))

    if np.isnan(oof).any():
        raise AssertionError("some samples never received an out-of-fold prediction")

    out = ARTIFACTS / "preds" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.assign(pred=oof)[["sample_id", "group", "y", "fold", "pred"]].to_csv(out, index=False)

    print(f"\n{name}: " + "  ".join(f"{k}={v}" for k, v in aggregate_folds(per_fold).items()))
    print(f"predictions -> {out}")
    return pd.read_csv(out)


def compare(arm_a: str, arm_b: str) -> dict:
    """Paired bootstrap CI on metric(arm_a) - metric(arm_b).

    `arm_a` is the challenger (fusion), `arm_b` the incumbent (best unimodal).
    The CI excluding zero is the result of the project.
    """
    a = pd.read_csv(ARTIFACTS / "preds" / f"{arm_a}.csv")
    b = pd.read_csv(ARTIFACTS / "preds" / f"{arm_b}.csv")
    merged = a.merge(b, on=["sample_id", "group", "y"], suffixes=("_a", "_b"))
    if len(merged) != len(a):
        raise ValueError("prediction files do not align -- were they built from the same folds?")

    res = bootstrap_delta(
        merged["y"].to_numpy(),
        merged["pred_a"].to_numpy(),
        merged["pred_b"].to_numpy(),
        merged["group"].to_numpy(),
    )
    verdict = "significant" if res["significant"] else "NOT significant (CI spans zero)"
    print(
        f"{arm_a} - {arm_b}: delta={res['delta']:+.4f} "
        f"95% CI [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}] -- {verdict}"
    )
    return res


def summary() -> pd.DataFrame:
    """Per-arm results table: mean +/- std across folds, for every saved run."""
    rows = []
    for path in sorted((ARTIFACTS / "preds").glob("*.csv")):
        d = pd.read_csv(path)
        per_fold = [
            summarize(g["y"].to_numpy(), g["pred"].to_numpy()) for _, g in d.groupby("fold")
        ]
        rows.append({"arm": path.stem, **aggregate_folds(per_fold)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--compare", nargs=2, metavar=("CHALLENGER", "INCUMBENT"))
    ap.add_argument("--summary", action="store_true", help="results table for all saved arms")
    ap.add_argument("--with-hb", action="store_true", help="include HGB/HCT (see README.md)")
    ap.add_argument(
        "--stain-norm",
        action="store_true",
        help="per-image colour standardisation; removes the 0.83-AUROC staining artifact",
    )
    args = ap.parse_args()

    if args.summary:
        summary()
        return

    features = list(FEATURES_DEFAULT)
    name = args.arm
    if args.with_hb:
        from src.config import FEATURES_HB

        features += FEATURES_HB
        name = f"{args.arm}_withhb"

    if args.compare:
        compare(*args.compare)
    elif args.arm:
        if args.stain_norm:
            name = f"{name}_stainnorm"
        run_arm(args.arm, features, name=name, stain_norm=args.stain_norm)
    else:
        ap.error("pass --arm or --compare")


if __name__ == "__main__":
    main()
