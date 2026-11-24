# User guide

This guide covers three things: setting the project up locally, training a model, and evaluating its performance. For the function-level reference see the [API docs](api.md). For the findings behind the design see the [research notes](research-notes.md).

## Contents

1. [What you can do](#1-what-you-can-do)
2. [Set up locally](#2-set-up-locally)
3. [Get the dataset (image arms only)](#3-get-the-dataset-image-arms-only)
4. [Train a model](#4-train-a-model)
5. [Evaluate performance](#5-evaluate-performance)
6. [Use the code as a library](#6-use-the-code-as-a-library)
7. [Add your own model arm](#7-add-your-own-model-arm)
8. [Conventions](#8-conventions)
9. [Troubleshooting](#9-troubleshooting)

## 1. What you can do

| Goal | Needs the 7.5 GB dataset? | Time |
|---|---|---|
| Read and re-check the results (notebooks, `--summary`, `--compare`) | No | Seconds to a few minutes |
| Retrain the two tabular arms (LightGBM, MLP) | No | Seconds |
| Rebuild the master table and folds from raw reports | Yes | Not timed |
| Train the image and fusion arms | Yes | About 25 minutes per arm on Apple Silicon (the author's measurement) |

Everything in the first two rows works straight after cloning, because the parsed table, the folds and every arm's out-of-fold predictions are committed in `artifacts/`.

## 2. Set up locally

**Requirements**

- Python 3.11 or newer. The project was developed on 3.11, and installation and the test suite were verified on 3.12.
- About 1.5 GB of disk space for the Python packages (the virtual environment measured 1.1 GB on macOS, and PyTorch is the largest part).
- For the image and fusion arms: a GPU is strongly recommended. The code picks the device automatically in this order: Apple `mps`, then `cuda`, then `cpu` (`DEVICE` in `src/config.py`).

**Install**

```bash
git clone https://github.com/matchalatte2609/anemic-blood-ml.git
cd anemic-blood-ml

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt  # runtime packages plus Jupyter and pytest
```

The packages in `requirements.txt` are pinned to exact versions. Note that pandas is 3.x, which removed several older idioms.

**Verify the install**

```bash
pytest
```

You should see all tests pass in a few seconds. They need no dataset.

**Open the notebooks**

```bash
jupyter nbconvert --execute --to notebook --inplace notebooks/01_cbc_exploration.ipynb
```

or open them in any Jupyter front end. Run the cells top to bottom. They write figures to `docs/figures/`.

## 3. Get the dataset (image arms only)

The tabular arms and all notebooks work without this step.

1. Download the AneRBC dataset from https://data.mendeley.com/datasets/hms3sjzt7f/1 (CC BY 4.0, about 7.5 GB).
2. Unpack it so the folder layout is:

```
data/AneRBC_dataset/
  AneRBC-I/{Healthy,Anemic}_individuals/
    Original_images/     500 images per class, 1224x960 RGB, e.g. 001_h.png
    Binary_segmented/    ground-truth masks
    RGB_segmented/       masked RGB images
    CBC_reports/         500 text files per class, e.g. 001_h.txt
    Morphology_reports/  label in prose. Never use as a feature.
  AneRBC-II/{Healthy,Anemic}_individuals/
    Original_images/     6000 crops per class, 306x320, e.g. 0001_01_h.png
    ...
```

3. Rebuild the master table so the image paths point at your machine:

```bash
python -m src.build_dataset
```

This step matters. The committed `artifacts/master.parquet` stores `img_path` values from the author's computer, so the image arms will not find the files until you rebuild the table. The command parses all 1000 CBC reports, joins them to the image paths, assigns the 5 folds and overwrites `artifacts/master.parquet` and `artifacts/folds/folds_seed42_k5.csv`. The folds are seeded, so they should match the committed ones. You can confirm with `git diff artifacts/folds`.

`build_dataset` also prints a coverage audit: the share of reports where each field parsed, compared with the recorded targets. A large shortfall means the parser regressed.

## 4. Train a model

Each model is called an **arm**. Every arm runs 5-fold grouped cross-validation on the stored folds and writes its out-of-fold predictions to `artifacts/preds/<name>.csv`.

| Arm | What it is | Command |
|---|---|---|
| `lgbm` | LightGBM on CBC values (the strong tabular baseline) | `python -m src.train --arm lgbm` |
| `mlp` | Two-layer MLP on CBC values (the fair deep baseline) | `python -m src.train --arm mlp` |
| `image` | ImageNet-pretrained ResNet18 on the smear image | `python -m src.train --arm image --stain-norm` |
| `concat` | Image embedding and CBC embedding, concatenated | `python -m src.train --arm concat --stain-norm` |
| `film` | CBC values condition the image features (FiLM) | `python -m src.train --arm film --stain-norm` |

**Flags**

| Flag | Effect |
|---|---|
| `--stain-norm` | Standardizes each image by its own channel mean and standard deviation. Output file gets the suffix `_stainnorm`. |
| `--with-hb` | Adds hemoglobin and hematocrit to the features. Output file gets the suffix `_withhb`. |

**Important: training overwrites predictions.** The files in `artifacts/preds/` are tracked in git and back the reported results. Re-running an arm replaces its file. To undo, run `git checkout artifacts/preds/`. The file name is built from the arm and the flags, so an ablation never overwrites the headline run.

**What to expect**

| Step | Result on the author's machine |
|---|---|
| `--arm lgbm` | About 3 seconds. Reproduces the committed predictions exactly (checked on Apple Silicon). |
| `--arm mlp` | About 7 seconds on `mps`. Reproduces the committed predictions exactly (checked on Apple Silicon). |
| `--arm image --stain-norm` | About 25 minutes on `mps`. Not re-run when this guide was written, so the timing is the author's earlier measurement. |

On other hardware the deep arms can differ slightly from the committed numbers. The first run of an image arm downloads the ResNet18 weights through `timm`, so it needs an internet connection once.

The training code follows rules that make the comparison fair. They are enforced in code, and the [API docs](api.md) describe each one:

- Imputation and standardization are fitted on the training rows of each fold only.
- Early stopping watches an inner split of the training rows, never the outer fold.
- Every deep arm goes through one shared training loop with identical epochs, optimizer and patience.
- Each fold is seeded with `seed + fold`.

## 5. Evaluate performance

**Results table for every saved arm**

```bash
python -m src.train --summary
```

This prints AUROC, AUPRC and expected calibration error (ECE), each as mean and standard deviation over the 5 folds.

**Is one arm really better than another?**

```bash
python -m src.train --compare concat_stainnorm image_stainnorm
```

The first name is the challenger and the second is the incumbent. The command runs a paired bootstrap that resamples patient groups 2000 times, scores both arms on each resample, and prints the difference in AUROC with a 95% interval. **An interval that excludes zero is the result.** A bare AUROC table is not, because two arms scored on the same samples have correlated errors.

Example output:

```
concat_stainnorm - image_stainnorm: delta=+0.0159 95% CI [+0.0061, +0.0269] -- significant
```

**Reference numbers (committed predictions)**

| Arm | AUROC | Trustworthy as a morphology result? |
|---|---|---|
| Concat fusion (stain-norm) | 0.959 +/- 0.015 | No, confounded |
| FiLM fusion (stain-norm) | 0.954 +/- 0.016 | No, confounded |
| Image, raw | 0.948 +/- 0.010 | No, acquisition artifact |
| Image, stain-norm | 0.942 +/- 0.012 | No, acquisition artifact |
| LightGBM (CBC) | 0.690 +/- 0.031 | Yes |
| MLP (CBC) | 0.689 +/- 0.023 | Yes |
| LightGBM + HGB/HCT | 0.686 +/- 0.025 | Yes |

**Sanity checks for the tabular arms.** LightGBM should land near 0.68. Below about 0.64, something upstream is broken. Above about 0.80, you have probably leaked the label, for example by including a missing-value indicator column or morphology report text.

**Every image-involving number inherits an acquisition artifact.** Empty slide background alone predicts the diagnosis, so a large image score is not evidence about cell morphology. Read the [research notes](research-notes.md), trap 4, before quoting one.

**Notebooks.** [`02_results_and_comparisons.ipynb`](../notebooks/02_results_and_comparisons.ipynb) rebuilds the whole results table, ROC curves and every bootstrap comparison from the saved predictions, with explanations.

## 6. Use the code as a library

Import `src` before anything else. It sets `OMP_NUM_THREADS=1` and loads LightGBM before PyTorch, which prevents a crash on macOS (see [Troubleshooting](#9-troubleshooting)).

```python
import src                                   # keep first
import pandas as pd
from src.metrics import summarize, bootstrap_delta

a = pd.read_csv("artifacts/preds/concat_stainnorm.csv")
b = pd.read_csv("artifacts/preds/image_stainnorm.csv")

summarize(a["y"].to_numpy(), a["pred"].to_numpy())
# {'auroc': ..., 'auprc': ..., 'ece': ...}

m = a.merge(b, on=["sample_id", "group", "y"], suffixes=("_a", "_b"))
bootstrap_delta(m["y"], m["pred_a"], m["pred_b"], m["group"])
# {'delta': ..., 'ci_low': ..., 'ci_high': ..., 'significant': ...}
```

## 7. Add your own model arm

1. Write a model that follows the contract for its type. A tabular arm is a function that takes train and validation frames and returns validation probabilities (see `fit_lightgbm` and `train_mlp`). A deep arm is an `nn.Module` whose `forward(img, tab)` returns a logit, so it can use the shared loop `fit_deep`.
2. Use `inner_split` for early stopping and fit any imputer or scaler on the training rows only.
3. Add the arm name to `ARMS` in `src/train.py` and handle it in `train_one_fold`.
4. Run `python -m src.train --arm <name>`.
5. Compare it with an existing arm using `--compare`.

If you change anything in the shared training loop or the tabular hyperparameters, re-run every arm. Otherwise the comparison measures the schedule, not the model.

## 8. Conventions

- **Label:** `y = 1` for anemic, `y = 0` for healthy.
- **Group key:** `group = f"{parent_id}_{cls}"`, for example `001_h`. It must include the class, because `001_h` and `001_a` are different patients.
- **Folds:** assigned once at the patient level, stored on disk and read by every arm. Never re-split.
- **Features:** the default is 8 CBC columns (`FEATURES_DEFAULT`). Hemoglobin and hematocrit are optional, through `--with-hb`.
- **Never use** the text in `Morphology_reports/` as a feature. It states the class in words.
- **Report** mean and standard deviation over the 5 folds, plus the bootstrap interval on the difference between arms.

## 9. Troubleshooting

**The process dies with exit code 139 and no error message (macOS).** PyTorch and LightGBM each ship their own OpenMP runtime, and two copies in one process can crash it. Importing `src` first fixes this. Do not reorder `src/__init__.py`. Do not set `KMP_DUPLICATE_LIB_OK=TRUE`, which hides the conflict and is documented as able to produce silently wrong results.

**`FileNotFoundError: ... master.parquet missing`.** Run `python -m src.build_dataset`, or restore the file with `git checkout artifacts/master.parquet`.

**Image arms cannot find the image files.** The `img_path` column in the committed table points at the author's machine. Download the dataset and run `python -m src.build_dataset` (section 3).

**Image training hangs or crashes at data loading.** `make_loader` defaults to `num_workers=0` on purpose. Worker processes can hang with `mps` in some PyTorch versions. Also, do not memory-map the image cache (`np.load(..., mmap_mode="r")`): random access through the torchvision transforms crashed the process.

**Predictions changed after re-running an arm.** Restore the committed files with `git checkout artifacts/preds/`.

**A pandas error about `inplace`, `applymap` or downcasting.** The project pins pandas 3.x, which removed or changed these. Use the pinned versions in `requirements.txt`.
