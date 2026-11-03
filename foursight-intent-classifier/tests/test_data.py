import numpy as np

from foursight_intent.data import class_weights, cv_folds, leakage_report, load_test, load_train
from foursight_intent.labels import LABELS
from foursight_intent.text import clean_text, dedup_key, extract_reference_ids, head_tail_truncate


def test_load_train_shapes(root):
    df = load_train(root / "data" / "train.csv")
    assert len(df) == 1904
    assert set(df["intent"]) <= set(LABELS)
    assert df["id"].is_unique


def test_load_test_shapes(root):
    df = load_test(root / "data" / "test.csv")
    assert len(df) == 566
    assert df["id"].is_unique


def test_grouped_folds_have_no_text_leakage(root):
    df = load_train(root / "data" / "train.csv")
    folds = cv_folds(df, 5, 42, "grouped")
    rep = leakage_report(df, folds)
    assert rep["total_leaked_rows"] == 0
    assert sorted(np.concatenate([f.val_idx for f in folds])) == list(range(len(df)))


def test_folds_are_seeded(root):
    df = load_train(root / "data" / "train.csv")
    a = cv_folds(df, 5, 42, "grouped")
    b = cv_folds(df, 5, 42, "grouped")
    for fa, fb in zip(a, b):
        assert np.array_equal(fa.val_idx, fb.val_idx)


def test_clean_text_and_keys():
    assert clean_text("  Where\n is\tmy   load? ") == "Where is my load?"
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("　ＥＴＡ ") == "ETA"
    assert dedup_key("Where IS my load?") == dedup_key("where is my load")


def test_reference_ids():
    ids = extract_reference_ids("status of 00332009115438339 and SOU1001357772 please, PO 823180")
    assert "00332009115438339" in ids and "SOU1001357772" in ids and "823180" in ids
    assert extract_reference_ids("how do I add a user") == []


def test_head_tail_truncate():
    ids = list(range(100))
    assert head_tail_truncate(ids, 10, tail=0) == list(range(10))
    assert head_tail_truncate(ids, 10, tail=3) == [0, 1, 2, 3, 4, 5, 6, 97, 98, 99]
    assert head_tail_truncate(ids[:5], 10, tail=3) == ids[:5]


def test_class_weights_monotone():
    labels = np.array([0] * 100 + [1] * 10 + [2] * 1 + [3, 4, 5, 6])
    w = class_weights(labels, "sqrt_inv")
    assert w[0] < w[1] < w[2]
    assert np.allclose(class_weights(labels, "none"), 1.0)
