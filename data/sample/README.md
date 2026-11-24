# Sample data

`cbc_sample.csv` is a 100-row sample (50 healthy, 50 anemic) of the complete blood count (CBC) values parsed from the public AneRBC dataset. It lets you inspect the data, run the notebooks' first section and try the library without downloading 7.5 GB of images.

```python
import pandas as pd

sample = pd.read_csv("data/sample/cbc_sample.csv")
sample.groupby("y").size()   # 0 = healthy, 1 = anemic, 50 each
```

The full parsed table with all 1000 samples is `artifacts/master.parquet`. It has the same columns plus `parent_id` and `img_path`, and notebook 01 loads it.

## Columns

| Column | Meaning |
|---|---|
| `sample_id` | Image and report identifier, for example `001_h`. The suffix is `h` for healthy and `a` for anemic. |
| `cls` | Class letter, `h` or `a`. |
| `y` | Label: 1 for anemic, 0 for healthy. |
| `group` | Patient group key `{id}_{cls}`. Always split on this, never on the number alone, because `001_h` and `001_a` are different patients. |
| `fold` | Cross-validation fold, 0 to 4, from `artifacts/folds/folds_seed42_k5.csv`. |
| `MCV` | Mean corpuscular volume, fL. |
| `MCH` | Mean corpuscular hemoglobin, pg. |
| `MCHC` | Mean corpuscular hemoglobin concentration, g/dL. |
| `RDW` | Red cell distribution width (CV), %. |
| `RBC` | Red blood cell count, 10^6 per uL. |
| `WBC` | White blood cell count, 10^3 per uL. |
| `PLT` | Platelet count, 10^3 per uL. |
| `MPV` | Mean platelet volume, fL. |
| `HGB` | Hemoglobin, g/dL. Excluded from the default model features. |
| `HCT` | Hematocrit, %. Excluded from the default model features. |
| `sex_inferred` | 1 if the report's hemoglobin reference range is the adult male range, 0 if it is the female range. Blank when the range is absent or ambiguous. Confounded with the class (see below). Not a model feature by default. |

A blank cell means the field is not in the report, or the value failed a plausibility check and was discarded (the source reports are OCR text and contain impossible values such as an RDW of 396). In this sample, RDW is blank in 31 rows, MPV in 27 and `sex_inferred` in 39. The other measurements are blank in 6 rows or fewer.

## Things to know before modeling

- **The label is a morphology call, not a hemoglobin threshold.** It comes from a pathologist's reading of the blood smear, so hemoglobin alone is a weak predictor here.
- **Missingness is informative.** Which fields a report contains differs by class, and that pattern predicts the label better than chance. Impute without adding missing-value indicator columns. See `notebooks/01_cbc_exploration.ipynb`, section 5.
- **`sex_inferred` is confounded.** It reflects the make-up of the two cohorts as well as physiology.
- The morphology report text states the class in words. It is deliberately not included, and it must never be used as a feature.

## How it was made

`python scripts/make_sample.py` draws 50 rows per class from `artifacts/master.parquet` with random seed 42 and drops the local image paths.

## Source and license

The values come from the AneRBC dataset:

> Shahzad, M., Shirazi, S. H., Yaqoob, M., Khan, Z., Rasheed, A., Sheikh, I. A., Hayat, A., and Zhou, H. (2024). AneRBC dataset: a benchmark dataset for computer-aided anemia diagnosis using RBC images. *Database*, 2024, baae120. https://doi.org/10.1093/database/baae120

The dataset is hosted on Mendeley Data (https://doi.org/10.17632/hms3sjzt7f.1) under the Creative Commons Attribution 4.0 license (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.

**Changes made:** the numbers in this folder were parsed from the dataset's CBC report text files by `src/parse_cbc.py`. Field names were normalized, values were anchored on their units, and implausible values were removed. Images and report text are not redistributed here.

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

This data is for research and education only. It is not for clinical use.
