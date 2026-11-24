"""Build the small public sample in data/sample/ from artifacts/master.parquet.

    python scripts/make_sample.py

The sample is 100 rows, 50 healthy and 50 anemic, drawn with a fixed seed. It
keeps only the parsed CBC values and the identifiers needed to join back to the
full table. Local image paths and report text are dropped on purpose: the
morphology reports state the class in words and must never reach a feature
matrix (README, leakage trap 1).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
PER_CLASS = 50

KEEP = [
    "sample_id", "cls", "y", "group", "fold",
    "MCV", "MCH", "MCHC", "RDW", "RBC", "WBC", "PLT", "MPV", "HGB", "HCT",
    "sex_inferred",
]


def main() -> None:
    master = pd.read_parquet(ROOT / "artifacts" / "master.parquet")
    sample = (
        master.groupby("y", group_keys=False)
        .sample(n=PER_CLASS, random_state=SEED)
        .sort_values("sample_id")[KEEP]
        .reset_index(drop=True)
    )

    out = ROOT / "data" / "sample" / "cbc_sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(f"{len(sample)} rows -> {out.relative_to(ROOT)}")
    print(sample["y"].value_counts().rename({0: "healthy", 1: "anemic"}).to_string())


if __name__ == "__main__":
    main()
