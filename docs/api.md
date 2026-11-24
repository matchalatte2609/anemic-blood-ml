# API reference

Function-level reference for the `src` package. For task-oriented instructions see the [user guide](user-guide.md).

**Always import `src` first.** The package `__init__` sets `OMP_NUM_THREADS=1` and imports LightGBM before PyTorch. Importing any submodule triggers it automatically, but in a notebook or script put `import src` at the top.

## Contents

- [`src.config`](#srcconfig): paths, constants, feature lists
- [`src.parse_cbc`](#srcparse_cbc): turn the OCR'd CBC reports into numbers
- [`src.build_dataset`](#srcbuild_dataset): master table and `CBCImputer`
- [`src.splits`](#srcsplits): grouped, stratified folds
- [`src.metrics`](#srcmetrics): AUROC, AUPRC, ECE, paired bootstrap
- [`src.data`](#srcdata): image cache, transforms, dataset, loaders
- [`src.models.tabular`](#srcmodelstabular): LightGBM and MLP arms
- [`src.models.image`](#srcmodelsimage): image encoder
- [`src.models.fusion`](#srcmodelsfusion): concat and FiLM fusion
- [`src.models.common`](#srcmodelscommon): the shared deep training loop
- [`src.train`](#srctrain): arm runner and command line
- [Artifact file formats](#artifact-file-formats)

---

## `src.config`

Paths, constants and feature lists. Import from here instead of hardcoding paths.

| Name | Value / meaning |
|---|---|
| `ROOT`, `DATA`, `ANE_I`, `ANE_II`, `ARTIFACTS` | Repository root, `data/AneRBC_dataset`, its two sub-datasets, and `artifacts/`. |
| `CLASS_DIRS` | `{"h": "Healthy_individuals", "a": "Anemic_individuals"}` |
| `LABEL` | `{"h": 0, "a": 1}` |
| `SEED`, `N_FOLDS`, `N_BOOTSTRAP` | `42`, `5`, `2000` |
| `DEVICE` | `"mps"`, `"cuda"` or `"cpu"`, chosen in that order of preference. |
| `FEATURES_CORE` | `["MCV", "MCH", "MCHC", "RDW", "RBC", "WBC", "PLT", "MPV"]` |
| `FEATURES_HB` | `["HGB", "HCT"]`, excluded by default. |
| `FEATURES_DEFAULT` | Same as `FEATURES_CORE`. |
| `EXPECTED_COVERAGE` | Share of the 1000 reports where each field should parse. Used to audit the parser. |
| `UNIVARIATE_AUROC` | Recorded single-feature AUROCs from the raw parse. |
| `IMG_SIZE`, `CROP_SIZE`, `CROPS_PER_PARENT` | `384`, `306`, `12` |
| `IMAGENET_MEAN`, `IMAGENET_STD` | Standard ImageNet normalization constants. |

### `group_id(parent_id: str, cls: str) -> str`

Returns the cross-validation group key, for example `group_id("1", "h")` gives `"001_h"`. The key must include the class because `001_h` and `001_a` are different patients.

---

## `src.parse_cbc`

Parses OCR'd CBC report text files into numeric rows. Value extraction is anchored on the **unit** (`g/dL`, `fL`, `pg`, `%`) rather than on column position, field names are normalized before alias lookup, and every value is range-filtered. Values outside the window become missing. No missing-value indicator columns are produced.

Module tables: `ALIASES` (spelling variants to canonical names), `UNITS` (the unit each field is reported in), `PLAUSIBLE` (the accepted range per field).

### `parse_cbc_file(path: Path) -> dict[str, float]`

Parses one report. Returns `{canonical_field: value}`. Fields that are absent or implausible are left out. May include `sex_inferred` (1.0 male range, 0.0 female range).

### `parse_reference_sex(text: str) -> str | None`

Infers `"M"` or `"F"` from the hemoglobin reference range printed in the report, or `None` if absent or ambiguous. The result is confounded with the class, so treat it as a covariate.

### `load_cbc_table() -> pd.DataFrame`

Parses all 1000 reports under `data/AneRBC_dataset`. Columns: `sample_id`, `parent_id`, `cls`, `y`, `group`, then one column per parsed field. Requires the dataset.

### `audit_coverage(df: pd.DataFrame) -> pd.DataFrame`

Compares parsed coverage with `EXPECTED_COVERAGE`. Columns: `field`, `coverage`, `expected`, `delta`, `cov_healthy`, `cov_anemic`, `class_gap`. Sorted by `class_gap`, because a gap between classes is a leakage channel. Run it after any parser change.

---

## `src.build_dataset`

### `index_parent_images() -> pd.DataFrame`

One row per AneRBC-I original image. Columns: `sample_id`, `parent_id`, `cls`, `group`, `img_path`.

### `index_crop_images() -> pd.DataFrame`

One row per AneRBC-II crop (12 per parent). Columns: `crop_id`, `parent_id`, `crop_no`, `cls`, `group`, `img_path`.

### `build_master() -> pd.DataFrame`

Joins the CBC table to the parent image paths and assigns folds with `make_folds`. Raises `ValueError` unless exactly 1000 samples join.

### `class CBCImputer(strategy="median", add_indicators=False)`

Fills missing CBC values. **Fit on training rows only**, then apply to both sides, so no validation statistics leak into training.

| Method | Description |
|---|---|
| `fit(train_df, features) -> CBCImputer` | Learns one fill value per feature from the training rows. A feature that is entirely missing in the training rows falls back to `0.0`. Raises `KeyError` if a feature is absent from the frame. |
| `transform(df) -> pd.DataFrame` | Returns a copy with the fill values applied. Raises `RuntimeError` if `fit` was not called. |
| `feature_names() -> list[str]` | Column order the model should consume, including indicator columns if enabled. |

`strategy` is `"median"` or `"mean"` (anything else raises `ValueError`).

**Do not use it for the LightGBM arm.** LightGBM handles NaN natively, and measured on this data native NaN scored 0.6807 against 0.6705 after median imputation. It exists for the MLP, image and fusion arms.

**Leave `add_indicators` off.** Missingness tracks the report template, which correlates with the label. The option exists so you can measure the effect. If you report a result with indicators on, also report it with them off.

Running the module (`python -m src.build_dataset`) builds the master table, prints the coverage audit and writes `artifacts/master.parquet` and the folds file.

---

## `src.splits`

Grouped, stratified cross-validation. Every arm must use the same folds.

### `make_folds(df, n_splits=5, seed=42) -> pd.DataFrame`

Assigns a `fold` column using `StratifiedGroupKFold`, grouped by `group` and stratified on `y`. `df` needs the columns `y` and `group` (`KeyError` otherwise). Calls `check_no_group_leak` before returning.

### `check_no_group_leak(df) -> None`

Raises `AssertionError` if any group spans more than one fold. Call it after every split and after expanding parents to crops.

### `expand_to_crops(parent_df, crop_index) -> pd.DataFrame`

Copies parent-level `fold` and `y` onto AneRBC-II crops through `group`. Assign folds at the parent level first, then expand. Raises `ValueError` if a crop has no parent fold.

### `save_folds(df, path=None) -> Path` and `load_folds(path=None) -> pd.DataFrame`

Write or read `sample_id, group, y, fold`. The default path is `artifacts/folds/folds_seed42_k5.csv`.

### `fold_summary(df) -> pd.DataFrame`

Per-fold size `n`, number of groups `n_groups` and positive rate `pos_rate`. Look at it before training anything.

---

## `src.metrics`

### `auroc(y, p)`, `auprc(y, p)`, `ece(y, p, n_bins=15)`

Return floats. `ece` is the expected calibration error with equal-width bins. Lower is better.

### `summarize(y, p) -> dict[str, float]`

Returns `{"auroc": ..., "auprc": ..., "ece": ...}`.

### `aggregate_folds(per_fold: list[dict]) -> dict[str, str]`

Formats a list of `summarize` results as `"mean +/- std"` strings (population standard deviation across folds).

### `bootstrap_delta(y, p_a, p_b, groups, metric=auroc, n_boot=2000, seed=42, alpha=0.05) -> dict`

Paired bootstrap interval for `metric(p_a) - metric(p_b)`.

- **Groups, not rows, are resampled** with replacement, because rows from one patient group are not independent.
- **Both arms are scored on the identical resample**, which makes it paired.
- `p_a` is the challenger and `p_b` the incumbent.

Returns:

| Key | Meaning |
|---|---|
| `delta` | Metric difference on the full data. |
| `ci_low`, `ci_high` | The `alpha/2` and `1 - alpha/2` quantiles of the bootstrap differences. |
| `p_delta_gt_0` | Share of resamples where the difference is above zero. |
| `significant` | `True` if the interval excludes zero. |
| `n_boot_valid` | Resamples kept. A resample containing only one class is skipped. |

---

## `src.data`

Image pipeline for the image and fusion arms. Requires PyTorch and torchvision.

### `cache_path(size=384) -> Path` and `build_image_cache(df, size=384, force=False) -> np.ndarray`

Decodes and resizes every image once into a `uint8` array of shape `(N, size, size, 3)`, saved as `artifacts/images_{size}.npy` (about 442 MB at 384). Row `i` of `df` is image `i`. The array is loaded fully into RAM and not memory-mapped, because memmapped random access crashed the process. Pass `force=True` if you change `size` or the row order of `df`, otherwise labels and images silently desynchronize.

### `class PerImageStandardize`

A transform that standardizes each image by its own per-channel mean and standard deviation. It removes the global color statistics that predict the label through the staining artifact. It replaces ImageNet normalization instead of following it.

### `build_transforms(train: bool, stain_norm: bool = False)`

Flips and mild color jitter for training (brightness, contrast and saturation 0.1, hue 0.02), then either `PerImageStandardize` or ImageNet normalization. Jitter is kept mild on purpose, because the label is literally "pale cells".

### `class SmearDataset(df, images, features, train, tabular=None, stain_norm=False)`

Returns `(image, tabular, y)` for each index. `df` must have `y` and `img_idx`. With `features=[]` and `tabular=None`, the tabular tensor is empty. During training it also applies a random multiple-of-90-degree rotation.

### `make_loader(df, images, features, train, tabular=None, stain_norm=False, batch_size=32, num_workers=0) -> DataLoader`

`train=True` turns on augmentation, shuffling and `drop_last`. Evaluation loaders keep the original row order, so predictions line up with the frame. `num_workers=0` is deliberate (see the [troubleshooting](user-guide.md#9-troubleshooting) section).

---

## `src.models.tabular`

### `DEFAULT_LGB_PARAMS`

A heavily regularized LightGBM configuration: 500 trees as an upper bound, learning rate 0.03, `num_leaves=7`, `min_child_samples=20`, feature and bagging fractions of 0.8, `reg_lambda=1.0`. Early stopping picks the real tree count.

### `inner_split(groups, seed, val_frac=0.2) -> (train_idx, es_idx)`

Splits the **training rows** again, grouped, so early stopping has a set to watch without touching the outer fold.

### `fit_lightgbm(X_train, y_train, X_val, groups_train, params=None, seed=42) -> (model, probabilities)`

Fits LightGBM with early stopping on an inner split. `X_train` must be a DataFrame and should keep its NaNs. There is no `y_val` parameter, so the function cannot see the outer fold's labels. Returns the fitted model and the validation probabilities.

### `class TabularMLP(n_features, hidden=64, dropout=0.3)`

A two-layer MLP with batch norm. `forward(x)` returns a logit. `forward_features(x)` returns the penultimate embedding (size `out_dim`), which fusion models reuse.

### `train_mlp(X_train, y_train, X_val, groups_train, seed=42, epochs=200, batch_size=64, lr=1e-3, weight_decay=1e-4, patience=30, device=None) -> (model, probabilities)`

Trains `TabularMLP`. `X` must already be imputed and standardized with training-fold statistics. Early stopping uses an inner split and the best-epoch weights are restored.

---

## `src.models.image`

### `class ImageEncoder(name="resnet18", pretrained=True, dropout=0.2)`

A `timm` backbone with the classifier removed, plus a linear head.

| Method | Output |
|---|---|
| `feature_map(x)` | Pre-pooling feature map `(B, C, H, W)`. FiLM conditions on it. |
| `forward_features(x)` | Pooled embedding `(B, out_dim)`. Concat fusion uses it. |
| `forward(img, tab=None)` | A logit. `tab` is accepted and ignored. |

### `pool_crop_embeddings(embeddings, parent_ids)`

Not implemented (raises `NotImplementedError`). It is a placeholder for mean-pooling AneRBC-II crop embeddings per parent. If you implement it, keep folds at the parent level.

---

## `src.models.fusion`

Every deep model takes `(img, tab)` and returns a logit.

### `class ConcatFusion(image_encoder, tabular_encoder, proj_dim=64, hidden=128, dropout=0.3)`

Projects both embeddings to `proj_dim`, concatenates them and applies an MLP head. The projection step stops the 512-dimensional image embedding from drowning out the 64-dimensional tabular one.

### `class FiLMLayer(n_channels, tab_dim, hidden=64)`

Computes `out = (1 + gamma(tab)) * feature_map + beta(tab)`, broadcast over height and width. The last layer is zero-initialized, so an untrained layer is an exact identity.

### `class FiLMFusion(tab_dim, backbone="resnet18", pretrained=True)`

An image backbone with FiLM applied to the final feature map, then mean pooling and a linear head.

---

## `src.models.common`

### `DEEP_DEFAULTS`

`{"epochs": 25, "lr": 3e-4, "weight_decay": 1e-4, "patience": 6}`. One set of defaults for every deep arm. If you change them, re-run all of them.

### `predict(model, loader, device) -> np.ndarray`

Sigmoid probabilities for every row of `loader`, in order.

### `fit_deep(model, train_loader, es_loader, predict_loader, seed=42, device=None, verbose=False, **overrides) -> (model, predictions)`

The single training loop shared by the image, concat and FiLM arms. It uses AdamW with `BCEWithLogitsLoss`, early-stops on `es_loader` (which must come from an inner split of the training rows), restores the best-epoch weights and then predicts on `predict_loader`. `overrides` replace entries of `DEEP_DEFAULTS`.

---

## `src.train`

The arm runner and the command-line entry point.

### Command line

```
python -m src.train --arm {lgbm,mlp,image,concat,film} [--stain-norm] [--with-hb]
python -m src.train --compare CHALLENGER INCUMBENT
python -m src.train --summary
```

| Flag | Effect |
|---|---|
| `--arm` | Train one arm across all folds and save its predictions. |
| `--stain-norm` | Per-image standardization. Appends `_stainnorm` to the output name. |
| `--with-hb` | Adds `HGB` and `HCT` to the features. Appends `_withhb` to the output name. |
| `--compare A B` | Paired bootstrap of `A - B` from two saved prediction files. |
| `--summary` | Results table for every file in `artifacts/preds/`. |

### Functions

| Function | Description |
|---|---|
| `load_master() -> pd.DataFrame` | Reads `artifacts/master.parquet` and adds `img_idx`. Raises `FileNotFoundError` if it is missing. |
| `load_images() -> np.ndarray` | Loads the image cache once and reuses it across folds. |
| `train_one_fold(arm, train_df, val_df, features, seed, stain_norm=False)` | Trains one arm on one fold and returns validation probabilities aligned to `val_df`. |
| `prepare_tabular(train_df, val_df, features, standardize) -> (X_train, X_val)` | Imputes, and optionally standardizes, using training-fold statistics only. |
| `run_arm(arm, features=FEATURES_DEFAULT, seed=42, name=None, stain_norm=False) -> pd.DataFrame` | Runs all folds, asserts every sample got a prediction and writes `artifacts/preds/{name}.csv`. Overwrites an existing file of that name. |
| `compare(arm_a, arm_b) -> dict` | Paired bootstrap on two saved prediction files. Raises `ValueError` if they do not align. |
| `summary() -> pd.DataFrame` | Per-arm mean and standard deviation over folds for every saved run. |

To retrain without overwriting saved predictions, call `train_one_fold` in your own loop, as `notebooks/01_cbc_exploration.ipynb` does.

---

## Artifact file formats

All under `artifacts/` and tracked in git, except the image cache.

**`master.parquet`** has one row per sample (1000 rows): `sample_id`, `parent_id`, `cls`, `y`, `group`, the parsed CBC columns (`WBC`, `RBC`, `HGB`, `HCT`, `MCV`, `MCH`, `MCHC`, `RDW`, `PLT`, `MPV`), `sex_inferred`, `img_path` and `fold`.

**`folds/folds_seed42_k5.csv`** has `sample_id, group, y, fold`.

**`preds/<name>.csv`** has `sample_id, group, y, fold, pred`, where `pred` is the out-of-fold probability of the anemic class. The comparison step merges two of these files on `sample_id, group, y`, which is what keeps the bootstrap paired.

**`images_384.npy`** is the decoded image cache (about 442 MB). It is git-ignored and regenerated on demand.
