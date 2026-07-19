"""Paths, feature lists and constants. Import from here; don't hardcode paths elsewhere."""

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "AneRBC_dataset"
ANE_I = DATA / "AneRBC-I"
ANE_II = DATA / "AneRBC-II"
ARTIFACTS = ROOT / "artifacts"

CLASS_DIRS = {"h": "Healthy_individuals", "a": "Anemic_individuals"}
LABEL = {"h": 0, "a": 1}

SEED = 42
N_FOLDS = 5
N_BOOTSTRAP = 2000

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# --- CBC features ---------------------------------------------------------
# See README.md. HGB/HCT are excluded by default: Hb is the clinical *definition*
# of anaemia, so a deployed model would not need to predict it. Note this is a
# framing choice, not leakage control -- measured HGB AUROC here is only 0.611.
# Run both ways; the delta is a reportable result.
FEATURES_CORE = ["MCV", "MCH", "MCHC", "RDW", "RBC", "WBC", "PLT", "MPV"]
FEATURES_HB = ["HGB", "HCT"]
FEATURES_DEFAULT = FEATURES_CORE

# Coverage measured over all 1000 reports (fraction of files where the field
# parses to a plausible value). Sanity-check your parser against these -- a big
# shortfall means an alias or a unit anchor is missing.
EXPECTED_COVERAGE = {
    "MCV": 0.981, "MCH": 0.929, "MCHC": 0.942, "RDW": 0.630, "RBC": 0.955,
    "WBC": 0.981, "PLT": 0.967, "MPV": 0.682, "HGB": 0.979, "HCT": 0.955,
}

# Univariate AUROC measured on the raw parse. Your LightGBM arm should beat the
# best of these (0.697); if it doesn't, something is wrong upstream.
UNIVARIATE_AUROC = {
    "MCH": 0.697, "MCV": 0.690, "MCHC": 0.659, "RDW": 0.634, "HGB": 0.611,
    "HCT": 0.598, "PLT": 0.591, "MPV": 0.561, "RBC": 0.542, "WBC": 0.522,
}

# --- image ---------------------------------------------------------------
IMG_SIZE = 384          # for AneRBC-I originals (1224x960)
CROP_SIZE = 306         # AneRBC-II crops are 306x320
CROPS_PER_PARENT = 12   # clean 4x3 tiling of the parent, no overlap
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def group_id(parent_id: str, cls: str) -> str:
    """Group key for CV. MUST include class.

    '0001_h' and '0001_a' are different patients that share a numeric ID.
    Grouping on the number alone merges two patients into one group.
    """
    return f"{int(parent_id):03d}_{cls}"
