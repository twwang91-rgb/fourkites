"""Build and validate the official submission file (id,intent)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import pandas as pd

from foursight_intent.labels import LABEL_SET


class SubmissionError(ValueError):
    pass


def write_submission(ids: Sequence[str], intents: Sequence[str], path: str | Path) -> Path:
    if len(ids) != len(intents):
        raise SubmissionError("ids and intents must have equal length")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "intent"])
        for i, lab in zip(ids, intents):
            w.writerow([str(i), lab])
    return path


def validate_submission(pred_path: str | Path, test_path: str | Path, expected_rows: int | None = None) -> dict:
    """Raise SubmissionError on any schema violation; return a summary dict otherwise.

    Checks: header exactly ["id", "intent"], same row count as test.csv, ids identical and
    in the same order, no duplicates, no empty labels, every label inside the closed set.
    """
    pred_path, test_path = Path(pred_path), Path(test_path)
    if not pred_path.exists():
        raise SubmissionError(f"{pred_path} does not exist")
    with open(pred_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    if header != ["id", "intent"]:
        raise SubmissionError(f"header must be exactly ['id','intent'], got {header}")
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False)
    test_ids = [str(x).strip() for x in test["id"].tolist()]
    if expected_rows is not None and len(test_ids) != expected_rows:
        raise SubmissionError(f"test file has {len(test_ids)} rows, expected {expected_rows}")
    if len(rows) != len(test_ids):
        raise SubmissionError(f"submission has {len(rows)} rows, test has {len(test_ids)}")
    pred_ids = [r[0].strip() if r else "" for r in rows]
    if len(set(pred_ids)) != len(pred_ids):
        raise SubmissionError("duplicate ids in submission")
    if pred_ids != test_ids:
        missing = sorted(set(test_ids) - set(pred_ids))
        extra = sorted(set(pred_ids) - set(test_ids))
        if missing or extra:
            raise SubmissionError(f"id mismatch; missing={missing[:5]} extra={extra[:5]}")
        raise SubmissionError("ids present but not in test.csv row order")
    bad_rows = [r for r in rows if len(r) != 2]
    if bad_rows:
        raise SubmissionError(f"rows with wrong column count: {bad_rows[:3]}")
    labels = [r[1] for r in rows]
    empty = [i for i, l in zip(pred_ids, labels) if not l.strip()]
    if empty:
        raise SubmissionError(f"empty labels for ids {empty[:5]}")
    invalid = sorted({l for l in labels if l.strip().lower() not in LABEL_SET})
    if invalid:
        raise SubmissionError(f"labels outside taxonomy: {invalid}")
    not_canonical = sorted({l for l in labels if l != l.strip().lower()})
    if not_canonical:
        raise SubmissionError(f"labels must be canonical lower-case: {not_canonical}")
    counts = pd.Series(labels).value_counts().to_dict()
    return {"rows": len(rows), "label_counts": counts, "path": str(pred_path)}
