import json

from foursight_intent.labels import ID2LABEL, LABEL2ID, LABELS, validate_label
from foursight_intent.metrics import official_scores


def test_seven_labels_in_official_order(root):
    official = json.load(open(root / "data" / "labels.json"))["labels"]
    assert LABELS == official
    assert LABELS == [
        "shipment_information", "realtime_query", "analytics", "disruptions", "chitchat", "knowledge_base", "other",
    ]


def test_label_mappings_round_trip():
    for i, label in enumerate(LABELS):
        assert LABEL2ID[label] == i
        assert ID2LABEL[i] == label
    assert len(LABEL2ID) == len(ID2LABEL) == 7


def test_validate_label_normalises_and_rejects():
    assert validate_label(" Analytics ") == "analytics"
    import pytest

    with pytest.raises(ValueError):
        validate_label("new_intent")


def test_official_metric_formula():
    gold = ["analytics"] * 4 + ["chitchat"] * 2 + ["other"] * 2
    pred = ["analytics"] * 3 + ["chitchat"] + ["chitchat"] * 2 + ["other"] + ["analytics"]
    s = official_scores(gold, pred)
    assert abs(s["accuracy"] - 6 / 8) < 1e-12
    # classes with no support count as F1 = 0 in the macro average (7 classes)
    f1s = [s["per_class"][c]["f1"] for c in LABELS]
    assert abs(s["macro_f1"] - sum(f1s) / 7) < 1e-12
    assert abs(s["OVERALL"] - (0.6 * s["macro_f1"] + 0.4 * s["accuracy"])) < 1e-12


def test_invalid_prediction_counts_as_wrong():
    s = official_scores(["other", "other"], ["other", "not_a_label"])
    assert s["accuracy"] == 0.5
    assert s["invalid_predictions"] == 1
    assert s["per_class"]["other"]["recall"] == 0.5


def test_metric_matches_score_py(root, tmp_path):
    """Cross-check our implementation against the official scorer on a random labelling."""
    import csv
    import random
    import subprocess
    import sys

    rng = random.Random(0)
    gold = [rng.choice(LABELS) for _ in range(300)]
    pred = [g if rng.random() < 0.7 else rng.choice(LABELS + ["bogus"]) for g in gold]
    g, p, o = tmp_path / "g.csv", tmp_path / "p.csv", tmp_path / "o.json"
    with open(g, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "intent"])
        w.writerows([[i, "q", x] for i, x in enumerate(gold)])
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "intent"])
        w.writerows([[i, x] for i, x in enumerate(pred)])
    subprocess.run([sys.executable, str(root / "scripts" / "score.py"), "--gold", g, "--pred", p, "--out", o], check=True, capture_output=True)
    ref = json.load(open(o))
    mine = official_scores(gold, pred)
    for k in ("accuracy", "macro_f1", "weighted_f1", "OVERALL"):
        assert abs(ref[k] - mine[k]) < 1e-12, k
    for c in LABELS:
        for m in ("precision", "recall", "f1", "support"):
            assert abs(ref["per_class"][c][m] - mine["per_class"][c][m]) < 1e-12, (c, m)
