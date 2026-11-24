"""Fast checks for the core library. No dataset, GPU or network needed.

    pytest
"""

import numpy as np
import pandas as pd
import pytest

from src.build_dataset import CBCImputer
from src.config import group_id
from src.metrics import aggregate_folds, auroc, bootstrap_delta, ece, summarize
from src.parse_cbc import _canonical, _clean_text, _extract_value, parse_cbc_file, parse_reference_sex
from src.splits import check_no_group_leak, expand_to_crops, make_folds


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_group_id_includes_class():
    """0001_h and 0001_a are different patients and must not share a group."""
    assert group_id("1", "h") == "001_h"
    assert group_id("0001", "a") == "001_a"
    assert group_id("1", "h") != group_id("1", "a")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_auroc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_ece_zero_when_calibrated_and_large_when_confidently_wrong():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert ece(y, y.astype(float)) == 0.0
    assert ece(y, 1.0 - y) == pytest.approx(1.0)


def test_summarize_and_aggregate_format():
    y = np.array([0, 1, 0, 1])
    s = summarize(y, np.array([0.2, 0.8, 0.3, 0.7]))
    assert set(s) == {"auroc", "auprc", "ece"}
    out = aggregate_folds([s, s])
    assert out["auroc"] == "1.000 +/- 0.000"


def _paired_data(n_groups=120, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n_groups // 2)
    groups = np.array([f"g{i}" for i in range(n_groups)])
    return y, groups, rng


def test_bootstrap_delta_identical_arms_is_not_significant():
    y, groups, rng = _paired_data()
    p = rng.random(len(y)) * 0.2 + 0.4 * y
    res = bootstrap_delta(y, p, p, groups, n_boot=200)
    assert res["delta"] == 0.0
    assert res["ci_low"] <= 0.0 <= res["ci_high"]
    assert not res["significant"]


def test_bootstrap_delta_detects_a_clearly_better_arm():
    y, groups, rng = _paired_data()
    strong = y + rng.normal(0, 0.3, len(y))
    weak = rng.random(len(y))
    res = bootstrap_delta(y, strong, weak, groups, n_boot=200)
    assert res["delta"] > 0.2
    assert res["significant"] and res["ci_low"] > 0


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def _toy_frame(n=100):
    return pd.DataFrame(
        {
            "sample_id": [f"{i:03d}" for i in range(n)],
            "group": [f"{i:03d}_x" for i in range(n)],
            "y": [i % 2 for i in range(n)],
        }
    )


def test_make_folds_assigns_every_row_and_keeps_groups_whole():
    out = make_folds(_toy_frame(), n_splits=5, seed=42)
    assert set(out["fold"]) == {0, 1, 2, 3, 4}
    check_no_group_leak(out)


def test_check_no_group_leak_raises_when_a_group_spans_two_folds():
    df = _toy_frame(10)
    df["group"] = "same"
    df["fold"] = [0, 1] * 5
    with pytest.raises(AssertionError, match="span multiple folds"):
        check_no_group_leak(df)


def test_expand_to_crops_propagates_parent_folds():
    parents = make_folds(_toy_frame(20), n_splits=5, seed=1)
    crops = pd.DataFrame({"crop_id": [f"{g}_{k}" for g in parents["group"] for k in range(3)]})
    crops["group"] = crops["crop_id"].str.rsplit("_", n=1).str[0]
    merged = expand_to_crops(parents, crops)
    assert len(merged) == 60
    check_no_group_leak(merged)


# --------------------------------------------------------------------------- #
# imputer
# --------------------------------------------------------------------------- #
def test_imputer_uses_train_statistics_only():
    train = pd.DataFrame({"MCV": [80.0, 90.0, np.nan], "MCH": [27.0, 29.0, 31.0]})
    val = pd.DataFrame({"MCV": [np.nan, 200.0], "MCH": [np.nan, 100.0]})
    imp = CBCImputer().fit(train, ["MCV", "MCH"])
    out = imp.transform(val)
    assert out["MCV"].iloc[0] == 85.0  # train median, not influenced by the 200.0 in val
    assert out["MCH"].iloc[0] == 29.0
    assert out[["MCV", "MCH"]].notna().all().all()


def test_imputer_requires_fit_and_can_add_indicators():
    df = pd.DataFrame({"MCV": [80.0, np.nan]})
    with pytest.raises(RuntimeError):
        CBCImputer().transform(df)
    imp = CBCImputer(add_indicators=True).fit(df, ["MCV"])
    assert imp.feature_names() == ["MCV", "MCV_missing"]
    assert imp.transform(df)["MCV_missing"].tolist() == [0, 1]


# --------------------------------------------------------------------------- #
# CBC parser
# --------------------------------------------------------------------------- #
def test_unit_anchor_survives_a_column_shift():
    """A fragment of the previous row leaked into the value column (file 056_h)."""
    line = _clean_text("HGB,6/μl,13.5 g/dL,11 --- 14")
    assert _extract_value(line, "HGB") == 13.5


@pytest.mark.parametrize(
    "raw", ["RDW---CV", "%RDW---CV", "$RDW---CV", "&RDW---CV", "%RDW-CV", "RDW-CV"]
)
def test_rdw_spelling_variants_map_to_one_field(raw):
    assert _canonical(raw) == "RDW"


def test_hemoglobin_aliases():
    assert _canonical("HGB") == "HGB"
    assert _canonical("Hemoglobin") == "HGB"
    assert _canonical("MCHO") == "MCHC"  # single-file OCR error


def test_parse_cbc_file_filters_implausible_values(tmp_path):
    report = tmp_path / "001_h.txt"
    report.write_text(
        "Test, Results/Units, Normal Range\n"
        "WBC,7.39 x10.e 3/μl,4 --- 10\n"
        "HGB,* 10.1 g/dL,12 --- 15\n"
        "MCV,88.2 fL,80 --- 100\n"
        "RDW---CV,396.0 %,11.5 --- 14.5\n",  # decimal misplacement, must become missing
        encoding="utf-8",
    )
    parsed = parse_cbc_file(report)
    assert parsed["WBC"] == 7.39
    assert parsed["HGB"] == 10.1
    assert parsed["MCV"] == 88.2
    assert "RDW" not in parsed


def test_reference_range_encodes_sex():
    assert parse_reference_sex("HGB,13.5 g/dL,13 --- 17") == "M"
    assert parse_reference_sex("HGB,11.0 g/dL,12 --- 15") == "F"
