import csv

import pytest

from foursight_intent.labels import LABELS
from foursight_intent.submission import SubmissionError, validate_submission, write_submission


@pytest.fixture
def test_csv(tmp_path):
    p = tmp_path / "test.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question"])
        for i in range(1, 11):
            w.writerow([i, f"question {i}"])
    return p


def test_valid_submission_passes(tmp_path, test_csv):
    out = tmp_path / "pred.csv"
    write_submission([str(i) for i in range(1, 11)], [LABELS[i % 7] for i in range(10)], out)
    summary = validate_submission(out, test_csv, expected_rows=10)
    assert summary["rows"] == 10
    with open(out, newline="") as f:
        assert next(csv.reader(f)) == ["id", "intent"]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rows: rows[:-1], "rows"),
        (lambda rows: rows + [["11", "other"]], "rows"),
        (lambda rows: [[r[0], "unknown_intent"] for r in rows], "taxonomy"),
        (lambda rows: [[r[0], ""] for r in rows], "empty"),
        (lambda rows: list(reversed(rows)), "order"),
        (lambda rows: [["1", "other"]] + rows[1:-1] + [["1", "other"]], "duplicate"),
        (lambda rows: [[r[0], "Analytics"] for r in rows], "canonical"),
    ],
)
def test_invalid_submissions_rejected(tmp_path, test_csv, mutate, message):
    rows = [[str(i), "other"] for i in range(1, 11)]
    rows = mutate(rows)
    out = tmp_path / "pred.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "intent"])
        w.writerows(rows)
    with pytest.raises(SubmissionError) as exc:
        validate_submission(out, test_csv, expected_rows=10)
    assert message in str(exc.value).lower()


def test_wrong_header_rejected(tmp_path, test_csv):
    out = tmp_path / "pred.csv"
    out.write_text("id,label\n1,other\n")
    with pytest.raises(SubmissionError):
        validate_submission(out, test_csv)


def test_real_submission_if_present(root):
    sub = root / "test_predictions.csv"
    if not sub.exists():
        pytest.skip("test_predictions.csv not generated yet")
    summary = validate_submission(sub, root / "data" / "test.csv", expected_rows=566)
    assert summary["rows"] == 566
    assert set(summary["label_counts"]) <= set(LABELS)
