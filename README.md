# anemic-blood

Does RBC morphology (image) add predictive signal over CBC indices (tabular) for anemia
classification — and vice versa? Four arms: tabular-strong, tabular-fair, image-only, fusion.

**Status:** complete. All four arms trained and compared; see Final results below.

> **Read the acquisition-artifact section before quoting any image-arm number.** Blank slide
> background predicts the diagnosis at 0.87 AUROC on this dataset.

---

## The finding that reframes the project

**The label is a morphology call, not a hemoglobin threshold.**

The original plan assumed "anemia *is* low hemoglobin, so including Hb gives ~1.0 AUROC and
the comparison is meaningless." **That is empirically false for this dataset.** Measured
univariate AUROCs over all 1000 samples:

| Feature | Coverage | AUROC |
|---|---|---|
| MCH | 92.9% | **0.697** |
| MCV | 98.1% | **0.690** |
| MCHC | 94.2% | 0.659 |
| RDW | 63.0% | 0.634 |
| **HGB** | 97.9% | **0.611** |
| HCT | 95.5% | 0.598 |
| PLT | 96.7% | 0.591 |
| MPV | 68.2% | 0.561 |
| RBC | 95.5% | 0.542 |
| WBC | 98.1% | 0.522 |

Hemoglobin is nearly the *weakest* useful feature. 52% of "healthy"-labelled samples have
Hb < 12 g/dL. The Hb distributions of the two classes overlap almost completely.

**Caveat on that table:** these are raw rank-AUROCs on *complete cases only* (each feature
scored over the samples where it parses), so they are not directly comparable to a
cross-validated model evaluated over all 1000. MCH scores 0.696 on its 929 complete cases but
0.677 under 5-fold CV over everything. Use the CV numbers below as the baseline.

**Measured 5-fold LightGBM baselines** (native NaN, 60 trees, lr 0.03, `num_leaves=7`):

| Feature set | AUROC |
|---|---|
| MCV + MCH + MCHC | **0.683** |
| all 8 (`FEATURES_DEFAULT`) | 0.682 |
| MCV + MCH | 0.682 |
| + RDW | 0.679 |
| MCH alone | 0.677 |

The extra features buy essentially nothing over the three red-cell indices. That is expected
rather than disappointing: MCH = HGB/RBC, MCV = HCT/RBC and MCHC = HGB/HCT are algebraically
interrelated, so the CBC panel carries far less independent information than its ten columns
suggest. **~0.68 is the honest tabular ceiling.**

Why: the class assignment comes from the pathologist's **smear morphology** reading, not from
a CBC threshold. Confirmed by keyword-counting all 1000 morphology reports:

| Descriptor | Healthy | Anemic |
|---|---|---|
| normocytic | 84.8% | 12.2% |
| normochromic | 82.8% | 8.8% |
| microcytic | 13.0% | 88.2% |
| hypochromic | 15.6% | 93.0% |
| elliptocytes | 9.4% | 77.0% |
| anisocytosis | 10.2% | 53.0% |

A patient can have low Hb with normocytic normochromic RBCs (acute blood loss, anaemia of
chronic disease) and still be labelled "healthy" here. The label answers *"does this smear
look microcytic/hypochromic?"*, not *"is this patient anaemic?"*.

### What follows from this

1. **Dropping Hb/HCT is optional, not mandatory — now settled empirically.** The ablation has
   been run (`--arm lgbm --with-hb`): adding HGB and HCT moves AUROC by **−0.0000, 95% CI
   [−0.021, +0.020]**. Not a small effect — *no* effect. The plan's predicted ~1.0 AUROC
   blow-up does not happen. Keep them excluded, but justify it as *"Hb is the clinical
   definition of anaemia and a deployed model would not need a model to read it"*, and cite
   the ablation rather than a leakage argument.
2. **MCV/MCH are the actual near-label features.** "Microcytic" *means* low MCV; "hypochromic"
   *means* low MCH. They are the instrument's measurement of the exact property the
   pathologist eyeballed. They are not an independent modality — they are the same
   measurement by another route. This is the honest framing of the tabular arm.
3. **The expected outcome flipped — then turned out to be unanswerable here.** The original
   plan predicted tabular wins. Since the label was generated *by looking at the image*, the
   image arm should have the direct path. It does score far higher (0.94 vs 0.69) — but that
   gap is substantially an acquisition artifact, not morphology (trap 4). The CBC's ~0.69 is
   real; the image arm's 0.94 is not interpretable as a morphology result on this dataset.

### Final results, all arms

| Arm | AUROC | AUPRC | ECE | Trustworthy? |
|---|---|---|---|---|
| Concat fusion (stain-norm) | **0.959 ± 0.015** | 0.962 ± 0.013 | 0.070 | confounded |
| FiLM fusion (stain-norm) | 0.954 ± 0.016 | 0.951 ± 0.022 | 0.077 | confounded |
| Image, raw | 0.948 ± 0.010 | 0.946 ± 0.015 | 0.095 | **no — artifact** |
| Image, stain-norm | 0.942 ± 0.012 | 0.946 ± 0.014 | 0.078 | **no — artifact** |
| LightGBM (CBC) | 0.690 ± 0.031 | 0.636 ± 0.036 | 0.097 | yes |
| MLP (CBC) | 0.689 ± 0.023 | 0.648 ± 0.030 | 0.097 | yes |
| LightGBM + HGB/HCT | 0.686 ± 0.025 | 0.644 ± 0.034 | 0.106 | yes |

Paired bootstrap deltas (2000 resamples, grouped by patient):

| Comparison | Delta | 95% CI | |
|---|---|---|---|
| concat − image | **+0.016** | [+0.006, +0.027] | significant |
| FiLM − image | **+0.011** | [+0.000, +0.023] | significant, marginally |
| FiLM − concat | −0.005 | [−0.012, +0.003] | no difference |
| concat − lgbm | +0.275 | [+0.240, +0.309] | significant **but confounded** |
| image − lgbm | +0.259 | [+0.221, +0.296] | significant **but confounded** |
| stain-norm − raw image | +0.003 | [−0.008, +0.015] | no effect |
| lgbm − mlp | −0.006 | [−0.026, +0.014] | no difference |
| lgbm+Hb − lgbm | −0.000 | [−0.021, +0.020] | no effect |

FiLM matches concat rather than beating it (CI spans zero), so the extra conditioning machinery
buys nothing here — report concat as the fusion arm and FiLM as the robustness check. Both
recover the same small positive CBC contribution over the image arm (+0.016 and +0.011), which
is the useful part: two architecturally different fusion schemes agree.

**How to read this table.** Any row involving an image arm inherits the acquisition confound
(trap 4) and cannot be quoted as a morphology-vs-CBC result. The `image − lgbm = +0.259` row is
the cautionary exhibit: a large, tightly-bounded, entirely publishable-looking effect that is
substantially measuring which imaging session a slide came from.

**The one defensible image-involving number is `concat − image = +0.016`.** Because the paired
bootstrap scores both arms on the same resamples and both arms carry the same artifact, the
difference largely cancels it and isolates what the CBC adds on top. Small, positive, CI
excludes zero: the blood indices contribute genuine physiological signal the image model does
not already have. That is the honest fusion finding.

### Stage 1 results (both tabular arms shipped)

Out-of-fold, 5 folds, inner-split early stopping so the outer fold is never used for fitting:

| Arm | AUROC | AUPRC | ECE |
|---|---|---|---|
| LightGBM (strong) | **0.690 ± 0.031** | 0.636 ± 0.036 | 0.097 ± 0.048 |
| MLP (fair) | **0.689 ± 0.023** | 0.648 ± 0.030 | 0.097 ± 0.014 |
| LightGBM + HGB/HCT | 0.686 ± 0.025 | 0.644 ± 0.034 | 0.106 ± 0.023 |

`lgbm − mlp = −0.006, 95% CI [−0.026, +0.014]` — indistinguishable.

**That equivalence is load-bearing for the whole project.** The standard way a fusion paper
manufactures a win is to pair a strong image encoder against a weak tabular branch. Here the
"fair" MLP matches the GBDT ceiling, so a later fusion gain cannot be dismissed as the tabular
arm having been crippled. Keep it that way: if you retune LightGBM, re-run the MLP too.

Predictions live in `artifacts/preds/{lgbm,mlp,lgbm_withhb}.csv` and are what `compare()` reads.

---

## Leakage traps

### 1. Morphology reports are the label in prose — never use as features
`Morphology_reports/*.txt` state the class in words ("microcytic hypochromic picture of
RBCs"). Using their text, or any feature derived from it, is a ~90% direct label leak.

**Legitimate uses:** stratification, error analysis (the ~13% of samples whose morphology text
disagrees with the folder label are the interesting hard cases), and building an auxiliary
multi-label target if you want a stretch goal.

### 2. AneRBC-II subimages — split by parent
12 crops per parent, tiling the 1224×960 parent as a clean 4×3 grid of 306×320, no overlap.
Verified: 500 parents × 12 = 6000 per class, both classes.

**The group key is `{parent_id}_{class}`, not `{parent_id}`.** `0001_h` and `0001_a` are
different patients that share a numeric ID. Grouping on the number alone silently merges two
patients into one group and corrupts the folds.

### 3. Differential missingness — the trap not in the original plan
The CBC reports are inconsistent in *which* fields they contain, and the inconsistency
correlates with the label:

| Artifact | AUROC as a standalone predictor |
|---|---|
| "RDW is missing" flag | 0.561 |
| CBC report row count | 0.564 |
| "has WBC differential" flag | 0.543 |

RDW is present in 71.6% of healthy vs 59.8% of anemic reports — a 12-point gap. That artifact
alone is as predictive as MPV's real signal (0.561). When the whole result hinges on a
fusion-minus-unimodal delta of a few points, a 0.56 channel of pure recording artifact is not
negligible.

**Mitigation:** impute missing values; do **not** add missingness-indicator columns; do not
feed row counts or any report-shape feature. If you want indicators, you must show the result
survives without them.

#### The clinical-suppression hypothesis, tested and rejected

There is a real lab practice where analysers suppress RDW/MPV when the underlying RBC/PLT
count is too low to be reliable. If that were generating the missingness here, it would be
legitimate clinical signal and indicators would be defensible. It is not what is happening:

| | correlation |
|---|---|
| RDW missing vs RBC | **+0.059** (mechanism predicts strongly negative) |
| RDW missing vs report row count | **−0.791** |
| MPV missing vs PLT | −0.131 |
| MPV missing vs report row count | −0.442 |

RDW is missing in 84% of short-format reports vs 7% of full-format ones. RDW and MPV go
missing *together* (φ = 0.389, p = 7e-35) despite being different analytes on different
channels — clinical suppression of one gives no reason for the other to follow. The lowest-RBC
sample in the dataset (2.06) has RDW reported.

The missingness tracks **which report template the lab used**, which correlates with class
(+10.8% / +10.0%) through cohort composition rather than physiology.

Measured cost/benefit, 5-fold, identical folds: indicators ON 0.6388, OFF 0.6346 — a **+0.004**
gain, while the two flags alone score **0.553**. Nearly all of the (negligible) gain is
artifact. `CBCImputer(add_indicators=True)` exists so this stays testable, not because it should
be on.

---

### 4. Staining / acquisition artifact in the images — the big one
Discovered when the image arm returned 0.946 against a 0.690 tabular ceiling. That gap was
mostly not morphology.

| Feature set | AUROC |
|---|---|
| Full ResNet18 on raw images | 0.946 |
| **6 numbers: per-channel mean + std, no spatial info, no cells** | **0.905** |
| Same 6 numbers over **background pixels only** (no cells at all) | **0.833** |
| Same 6 numbers over **cell pixels only** | 0.806 |
| Cell-minus-background contrast (stain-invariant) | **0.734** |
| Cell/background ratio (stain-invariant) | 0.765 |

Background pixels carry no information about a patient's red cells, yet they out-predict the
cell regions. That is staining batch, illumination and exposure — an acquisition artifact
correlated with class, almost certainly because the two classes were collected or stained in
different sessions. Any model free to use raw colour will take it.

Note the mean shifts are tiny (red differs by 2.9/255) — this is not visible by eye and will
not show up in a spot check of a few images. It only appears when you measure it.

**What is real:** the stain-invariant contrast still predicts at 0.734, in the correct
direction — anemic cells sit closer to background brightness, i.e. paler. That is genuine
hypochromia, and it is roughly on par with the CBC's 0.690.

#### Attempted mitigation — and why it is not enough

`--stain-norm` (`data.PerImageStandardize`) forces every image to per-channel mean 0 / std 1.
It provably destroys the six features above, since they become constants by construction. It is
**not sufficient**. Measured after standardisation:

| Feature set, post-standardisation | AUROC |
|---|---|
| **Background-only pixels (mean/std/skew/kurtosis)** | **0.872** |
| Colour-distribution shape (skew + kurtosis, whole image) | 0.876 |
| Sharpness / focus, 2 numbers | 0.724 |
| Colour percentiles | 0.686 |
| ResNet18 with `--stain-norm` | ~0.94 (unchanged from raw) |

**The background-only number is the one that settles it.** Those pixels contain no cells. No
biological pathway exists from empty slide to a patient's red-cell morphology. 0.872 there is
pure acquisition signature, and it survives standardisation — the two classes differ in
higher-order image statistics (distribution shape, focus), not merely in average colour.

The stain-normalised CNN scoring the same as the raw CNN is consistent with this: it never
needed the mean colour, because plenty of other artifact channels remain.

#### What this means for the project

**The image arm on this dataset cannot support a claim about morphology.** Healthy and anemic
slides were evidently captured in systematically different sessions, and that difference is
recoverable from regions containing no cells at all. Any image result here — 0.948 raw, ~0.94
stain-normalised — is measuring the acquisition protocol to an unknown but large degree.

Genuine biology is present too: the stain-invariant cell-minus-background contrast predicts at
0.734 in the correct direction (anemic cells sit closer to background brightness = paler =
hypochromic). But it cannot be separated from the confound by any post-hoc normalisation tried
here, and 0.734 is not distinguishable from the CBC's 0.690.

**Report this as the finding, not as a failure.** "A widely-used public anemia dataset carries a
class-correlated acquisition artifact strong enough that empty slide background predicts the
diagnosis at 0.87" is a more useful contribution than another fusion delta, and it invalidates
naive benchmark numbers on AneRBC.

If you want a defensible image arm, the next things to try, in order: train on `RGB_segmented`
(background masked out entirely), then per-cell colour normalisation, and validate by checking
the background/artifact baselines drop to chance **before** reporting any headline number.

This generalises: whenever an image arm dramatically beats a tabular arm on medical data,
measure the colour-only, background-only and focus-only baselines before believing it.

---

## Data

**Source:** AneRBC dataset, Shahzad et al. (2024), *Database*, Volume 2024, article baae120,
[doi.org/10.1093/database/baae120](https://doi.org/10.1093/database/baae120). Data hosted on
Mendeley Data: [doi.org/10.17632/hms3sjzt7f.1](https://doi.org/10.17632/hms3sjzt7f.1), licensed
CC BY 4.0.

```bibtex
@article{shahzad2024anerbc,
  title   = {AneRBC dataset: a benchmark dataset for computer-aided anemia diagnosis using RBC images},
  author  = {Shahzad, Muhammad and Shirazi, Syed Hamad and Yaqoob, Muhammad and Khan, Zakir
             and Rasheed, Assad and Sheikh, Israr Ahmed and Hayat, Asad and Zhou, Huiyu},
  journal = {Database},
  volume  = {2024},
  pages   = {baae120},
  year    = {2024},
  doi     = {10.1093/database/baae120}
}
```

Not redistributed here (7.5 GB, CC BY 4.0 requires attribution rather than permitting
rehosting as part of this repo) — download from the Mendeley link above and unpack to
`data/AneRBC_dataset/`. See **Reproducing** below.

`data/AneRBC_dataset/` — 7.5 GB total (AneRBC-I 3.7 GB, AneRBC-II 3.8 GB). Note: larger than
the 5 GB the plan assumed; matters for Colab/Drive.

```
AneRBC-I/{Healthy,Anemic}_individuals/
  Original_images/     500 × 1224×960 RGB PNG   001_h.png … 500_h.png
  Binary_segmented/    500  (ground-truth masks, 1224×960)
  RGB_segmented/       500  (masked RGB, 1224×960)
  CBC_reports/         500 × .txt                001_h.txt …
  Morphology_reports/  500 × .txt   <-- LABEL IN PROSE, NOT A FEATURE
AneRBC-II/{Healthy,Anemic}_individuals/
  Original_images/     6000 × 306×320  0001_01_h.png … 0500_12_h.png
  Binary_segmented/, RGB_segmented/, CBC_reports/ (same 500 .txt), Morphology_reports/
```

Naming: AneRBC-I `{id:03d}_{h|a}.png`; AneRBC-II `{parent:04d}_{crop:02d}_{h|a}.png`.
Join II→I by `int(parent_id)` → `f"{id:03d}"`. Verified 500/500 match, no orphans either way.

`upload_data_for_training.ipynb` ships with the dataset — the authors' loader example. Not
used by this project's pipeline.

### The CBC files are machine-readable — no transcription day needed
They are comma-separated text, **not** scanned images. The plan's "budget one day to
transcribe 1000 rows" contingency does not apply. But they are OCR output and dirty:

```
Test, Results/Units, Normal Range        <- header wording varies between files
WBC,7.39 x10.e 3/μl,4 --- 10
HGB,* 10.1 g/dL,12 --- 15                <- '*' = abnormal flag, inside the value field
RDW---CV,12.8 %,11.5 --- 14.5
```

Corruption modes found, in rough order of nastiness:

1. **Column-shift / fragment injection.** `056_h.txt`: `HGB,6/μl,13.5 g/dL,11 --- 14` — a
   fragment of the previous row leaked into the value column. Naive "first number in field 2"
   parsing returns `6` instead of `13.5`. **Anchor on the unit** (`g/dL`, `fL`, `pg`, `%`)
   rather than on column position.
2. **Field-name corruption.** The `#` absolute-count prefix is OCR'd as `$`, `%`, `&`. RDW
   appears as `RDW---CV`, `%RDW---CV`, `$RDW---CV`, `&RDW---CV`, `%RDW-CV`, `RDW-CV`.
   Strip all leading non-alpha, uppercase, drop `-`/`.`/space, then alias-map.
3. **Name variants.** HGB is `HGB` (83%) *or* `Hemoglobin` (17%); both needed for full
   coverage. Same for `Neut`/`NEUT`/`Neutrophil`/`Neutrophils`. One file has `MCHO` for MCHC.
4. **Missing commas** merging name and value: `#EOS0.13 x10.e 3/μl`.
5. **Physiologically impossible values** that survive parsing: RDW ranging 0.6 – 396.0
   (real RDW-CV is ~11–25). Decimal misplacement, e.g. `HGB,1.48 g/dL` in `066_h.txt` /
   `067_h.txt` is almost certainly 14.8. Out-of-range rates after plausibility filtering are
   ~2–7% per field. **Range-filter to NaN, then impute. Do not trust raw parsed values.**
6. **Unicode.** `μ` U+03BC (Greek mu) and `µ` U+00B5 (micro sign) both appear. Normalise.
7. Row count varies 7–21 per file, bimodally (short reports lack the WBC differential).

### Free feature hiding in the reference-range column
The per-file "normal range" for HGB encodes sex: `13 --- 17` is the adult male reference
range, `12 --- 15` female. Recoverable for most files — the plan's "age/sex if present" is
partly satisfiable this way.

**But it is confounded:** healthy splits ~167 female-range / 165 male-range, anemic ~117 / 56.
Inferred sex is predictive partly through dataset composition, not physiology. Treat as a
covariate to control for and report with/without, not as a free win. Age is not recoverable.

---

## Environment

conda env **`tomcat`** (`/opt/homebrew/Caskroom/miniconda/base/envs/tomcat`), Python 3.11.15.

Installed and verified: torch 2.13.0, torchvision 0.28.0, timm 1.0.28, tqdm 4.69.0,
lightgbm 4.6.0, scikit-learn 1.9.0, pandas 3.0.3, numpy 1.26.4, scipy 1.17.1, pillow 12.3.0,
matplotlib 3.11.0.

```bash
conda activate tomcat
```

Apple Silicon: **MPS is available and verified** (ResNet18 forward pass on `mps` passes).
Use `mps` locally; the plan's Colab T4 is optional. Set device via `src/config.py:DEVICE`.

### The OpenMP crash — `src/__init__.py` handles it, don't reorder that file
torch and LightGBM each ship their own `libomp.dylib`. The two copies fight and the process
dies with **SIGSEGV (exit 139) and no traceback**. It is nondeterministic and surfaces wherever
a thread pool spawns first, so the *same* bug appears as a crash in `LGBMClassifier.fit`, or
inside a torchvision v2 transform, or during imports. It reads as a data bug. It is not.

Two things are required, both before torch or lightgbm load:

1. **`OMP_NUM_THREADS=1`** — stops the duplicate runtimes contending. Measured cost here:
   none. 50 img/s single-threaded vs 47 img/s default, because the work sits on MPS and the
   threads only added contention.
2. **lightgbm imported ahead of torch** — necessary but **not sufficient alone**. It protects
   `LGBMClassifier.fit` while leaving torchvision transforms crashing, which is exactly how
   this was first misdiagnosed as fixed after the tabular arms started passing.

Avoid `KMP_DUPLICATE_LIB_OK=TRUE`: it lets two OpenMP runtimes coexist and is documented as
capable of producing silently wrong results.

### Do not memmap the image cache
`np.load(..., mmap_mode="r")` on `images_384.npy` is tempting at 442 MB but the array fits in
RAM many times over, so paging buys nothing and adds a failure mode. `build_image_cache` loads
it fully.

Two version notes: **pandas is 3.0.x**, a major release with breaking changes — some
StackOverflow-era idioms (`inplace=`, silent downcasting, `applymap`) are removed or changed.
numpy is pinned <2 by the existing env.

---

## Layout

```
src/config.py          paths, feature lists, CV/seed constants
src/parse_cbc.py       CBC .txt -> tidy rows; unit-anchored, range-filtered
src/build_dataset.py   master parquet + CBCImputer (train-fold-only fit)
src/splits.py          grouped stratified CV + group-leak assertion
src/metrics.py         AUROC/AUPRC/ECE + paired group bootstrap on deltas
src/data.py            image cache, transforms, PerImageStandardize, loaders
src/models/tabular.py  LightGBM + MLP arms, inner-split early stopping
src/models/image.py    ResNet18 encoder (timm)
src/models/fusion.py   concat fusion + FiLM (identity-initialised)
src/models/common.py   ONE training loop shared by all deep arms
src/train.py           arm runner, CV loop, comparisons, summary table
```

## Reproducing

```bash
conda activate tomcat
python -m src.build_dataset            # parse CBC, join images, build folds
python -m src.train --arm lgbm         # ~1 min
python -m src.train --arm mlp
python -m src.train --arm image --stain-norm     # ~25 min on MPS
python -m src.train --arm concat --stain-norm
python -m src.train --arm film --stain-norm
python -m src.train --compare concat_stainnorm image_stainnorm
python -m src.train --summary
```

The dataset is not in this repo (7.5 GB). Download from
https://data.mendeley.com/datasets/hms3sjzt7f/1 and unpack to `data/AneRBC_dataset/`.

Out-of-fold predictions for every arm are tracked in `artifacts/preds/`, so the reported deltas
can be re-checked with `--compare` without re-training anything.

## Conventions

- Group key for every split: `group_id = f"{parent_id}_{cls}"`.
- Label: `y = 1` for anemic, `0` for healthy.
- Never let `Morphology_reports` text reach a feature matrix.
- Report mean ± std across 5 folds, plus bootstrap 95% CI on the fusion-minus-best-unimodal
  delta. The CI excluding zero is the result; a bare AUROC table is not.
