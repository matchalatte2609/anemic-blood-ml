"""Parse the OCR'd CBC .txt reports into tidy numeric rows.

This is the critical path of the project. The reports look like CSV but are OCR
output with at least seven distinct corruption modes -- see README.md for the
full catalogue and worked examples.

Design rules, learned the hard way from the audit:

1. Anchor value extraction on the UNIT, never on column position. File 056_h.txt
   reads `HGB,6/ul,13.5 g/dL,11 --- 14` -- a fragment of the previous row leaked
   into the value column, so "first number after the first comma" returns 6
   instead of 13.5.
2. Normalise field names hard (strip leading non-alpha, uppercase, drop
   separators) before alias lookup. The '#' absolute-count prefix is variously
   OCR'd as '$', '%', '&'.
3. Range-filter every value to a physiological window and set failures to NaN.
   Raw parses include RDW = 396.0 and HGB = 1.48.
4. Do NOT emit missingness indicators. Which fields a report contains correlates
   with the label at ~0.56 AUROC (see README.md, trap 3).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ANE_I, CLASS_DIRS, EXPECTED_COVERAGE, LABEL, group_id

# --- field name aliases ---------------------------------------------------
# Key = normalised name (see _norm_key). Value = canonical feature name.
# RDW is handled separately by prefix match: it has six observed spellings.
ALIASES: dict[str, str] = {
    "HGB": "HGB", "HB": "HGB", "HEMOGLOBIN": "HGB", "HAEMOGLOBIN": "HGB",
    "HCT": "HCT", "HAEMATOCRIT": "HCT", "HEMATOCRIT": "HCT",
    "MCV": "MCV",
    "MCH": "MCH",
    "MCHC": "MCHC", "MCHO": "MCHC",  # single-file OCR error
    "RBC": "RBC",
    "WBC": "WBC",
    "PLT": "PLT", "PLATELET": "PLT", "PLATELETS": "PLT",
    "MPV": "MPV",
    # TODO: the WBC differential (NEUT/LYMP/MONO/EOS/BASO, each with % and
    # absolute forms) is present in only ~50% of files and its presence is
    # label-correlated. Decide whether to parse it at all. Recommendation: skip
    # for v1 -- the coverage is too low and too confounded to be worth it.
}

# The unit each field is reported in. Used to anchor the number.
UNITS: dict[str, tuple[str, ...]] = {
    "HGB": ("g/dL",),
    "HCT": ("%",),
    "MCV": ("fL",),
    "MCH": ("pg",),
    "MCHC": ("g/dL",),
    "RBC": ("x10.e 6", "x10.e6"),
    "WBC": ("x10.e 3", "x10.e3"),
    "PLT": ("x10.e 3", "x10.e3"),
    "MPV": ("fL",),
    "RDW": ("%",),
}

# Physiological plausibility windows. Values outside -> NaN.
# Deliberately generous: these reject OCR garbage, not unusual patients.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "HGB": (4.0, 20.0),
    "HCT": (15.0, 60.0),
    "MCV": (50.0, 120.0),
    "MCH": (15.0, 40.0),
    "MCHC": (25.0, 40.0),
    "RBC": (2.0, 7.0),
    "WBC": (1.0, 30.0),
    "PLT": (20.0, 900.0),
    "MPV": (5.0, 15.0),
    "RDW": (10.0, 30.0),
}

_NUM = r"(\d+(?:\.\d+)?)"


def _clean_text(s: str) -> str:
    """Normalise unicode. Both U+03BC (Greek mu) and U+00B5 (micro) appear."""
    s = unicodedata.normalize("NFKC", s)
    return s.replace("μ", "u").replace("µ", "u")


def _norm_key(raw: str) -> str:
    """Strip OCR prefix junk and separators so aliases can match.

    '#NEUT' / '$NEUT' / '%RDW---CV' all lose their leading marker; '-', '.' and
    spaces are dropped so 'RDW---CV' and 'RDW-CV' collapse together.
    """
    k = re.sub(r"^[^A-Za-z]+", "", raw.strip())
    return k.upper().replace("-", "").replace(".", "").replace(" ", "")


def _canonical(raw_key: str) -> str | None:
    k = _norm_key(raw_key)
    if k.startswith("RDW"):
        return "RDW"
    return ALIASES.get(k)


def _extract_value(line: str, field: str) -> float | None:
    """Pull the numeric value for `field` out of a whole raw line.

    Tries unit-anchored match first (robust to column shift), then falls back to
    the first number after the first comma.
    """
    for unit in UNITS.get(field, ()):
        m = re.search(_NUM + r"\s*" + re.escape(unit), line, re.I)
        if m:
            return float(m.group(1))

    # Fallback: first number in the value column. Strips the '*' abnormal flag.
    parts = line.split(",")
    if len(parts) >= 2:
        m = re.search(_NUM, parts[1])
        if m:
            return float(m.group(1))
    return None


def _in_range(field: str, value: float | None) -> float:
    if value is None:
        return np.nan
    lo, hi = PLAUSIBLE[field]
    return float(value) if lo <= value <= hi else np.nan


def parse_reference_sex(text: str) -> str | None:
    """Infer sex from the HGB reference range printed in the report.

    '13 --- 17' is the adult male range, '12 --- 15' female. Recoverable for most
    files -- partially satisfies the plan's "age/sex if present".

    CAUTION: confounded. Healthy is ~167 female / 165 male, anemic ~117 / 56, so
    inferred sex is predictive partly through cohort composition rather than
    physiology. Report models with and without it. See README.md.
    """
    for line in text.splitlines():
        if _canonical(line.split(",")[0]) != "HGB":
            continue
        m = re.search(_NUM + r"\s*-+\s*" + _NUM + r"\s*$", line.strip())
        if not m:
            return None
        lo = float(m.group(1))
        # TODO: verify this cut against the full observed set of ranges.
        # Observed: 12--15, 13--17, 11--14, 11.1--14.1, 11.5--15.5. The 11.x
        # ranges are ambiguous (paediatric? another lab's female range?) and are
        # currently bucketed as unknown rather than guessed.
        if lo >= 12.5:
            return "M"
        if 11.5 <= lo < 12.5:
            return "F"
        return None
    return None


def parse_cbc_file(path: Path) -> dict[str, float]:
    """Parse one CBC report into {canonical_field: value}. Missing -> absent."""
    text = _clean_text(path.read_text(encoding="utf-8", errors="replace"))
    out: dict[str, float] = {}

    for line in text.splitlines():
        if not line.strip() or "," not in line:
            continue
        field = _canonical(line.split(",")[0])
        if field is None or field in out:
            continue  # first occurrence wins; duplicates are OCR artifacts
        value = _in_range(field, _extract_value(line, field))
        if not np.isnan(value):
            out[field] = value

    sex = parse_reference_sex(text)
    if sex is not None:
        out["sex_inferred"] = 1.0 if sex == "M" else 0.0
    return out


def load_cbc_table() -> pd.DataFrame:
    """Parse all 1000 CBC reports into one DataFrame.

    Returns columns: sample_id, cls, y, group, <feature columns...>
    """
    rows = []
    for cls, dirname in CLASS_DIRS.items():
        for path in sorted((ANE_I / dirname / "CBC_reports").glob("*.txt")):
            pid = path.stem.split("_")[0]
            rows.append(
                {
                    "sample_id": path.stem,
                    "parent_id": pid,
                    "cls": cls,
                    "y": LABEL[cls],
                    "group": group_id(pid, cls),
                    **parse_cbc_file(path),
                }
            )
    return pd.DataFrame(rows)


def audit_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Compare parsed coverage against the audited baseline.

    Run this after ANY parser change. A field coming in materially below
    `EXPECTED_COVERAGE` means an alias or unit anchor regressed.
    """
    recs = []
    for field, expected in EXPECTED_COVERAGE.items():
        got = df[field].notna().mean() if field in df else 0.0
        by_class = (
            df.groupby("cls")[field].apply(lambda s: s.notna().mean())
            if field in df else pd.Series(dtype=float)
        )
        recs.append(
            {
                "field": field,
                "coverage": round(float(got), 3),
                "expected": expected,
                "delta": round(float(got) - expected, 3),
                "cov_healthy": round(float(by_class.get("h", np.nan)), 3),
                "cov_anemic": round(float(by_class.get("a", np.nan)), 3),
                # Differential missingness is a leak channel, not a curiosity.
                "class_gap": round(
                    float(abs(by_class.get("h", np.nan) - by_class.get("a", np.nan))), 3
                ),
            }
        )
    return pd.DataFrame(recs).sort_values("class_gap", ascending=False)


if __name__ == "__main__":
    table = load_cbc_table()
    print(f"parsed {len(table)} reports\n")
    print(audit_coverage(table).to_string(index=False))
