#!/usr/bin/env python
"""Predict test.csv with the final artefact and write the official submission.

Outputs
  test_predictions.csv                      id,intent  (official format)
  reports/test_predictions_with_confidence.csv  id,intent,confidence,margin,needs_review,second_intent  (metadata only)
The submission is validated against data/test.csv before the script exits 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from foursight_intent.data import load_test
from foursight_intent.inference import load_classifier
from foursight_intent.labels import ID2LABEL, LABELS
from foursight_intent.submission import validate_submission, write_submission
from foursight_intent.utils import get_logger, save_json, seed_everything

log = get_logger("submission")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models/final"))
    ap.add_argument("--test", default=str(ROOT / "data/test.csv"))
    ap.add_argument("--out", default=str(ROOT / "test_predictions.csv"))
    ap.add_argument("--meta-out", default=str(ROOT / "reports/test_predictions_with_confidence.csv"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--expected-rows", type=int, default=566)
    ap.add_argument("--summary-out", default=str(ROOT / "reports/results/submission_summary.json"))
    a = ap.parse_args()
    seed_everything(42)

    test = load_test(a.test)
    clf = load_classifier(a.model_dir, device="cpu", num_threads=a.threads)
    import time

    t0 = time.perf_counter()
    probs = clf.probabilities(test["text"].tolist())
    ms_per_row = 1000 * (time.perf_counter() - t0) / len(test)
    fired = clf.last_rules_fired
    preds = [clf._to_prediction(p) for p in probs]
    intents = [p.intent for p in preds]

    write_submission(test["id"].tolist(), intents, a.out)
    summary = validate_submission(a.out, a.test, expected_rows=a.expected_rows)
    log.info("submission valid: %s", summary)

    meta = pd.DataFrame({
        "id": test["id"], "intent": intents, "confidence": [round(p.confidence, 4) for p in preds], "margin": [round(p.margin, 4) for p in preds],
        "needs_review": [p.needs_review for p in preds], "second_intent": [p.second_intent for p in preds], "rules_fired": ["|".join(f) for f in fired],
    })
    Path(a.meta_out).parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(a.meta_out, index=False)
    log.info("label distribution: %s", meta["intent"].value_counts().to_dict())
    log.info("needs_review flagged: %d / %d (%.1f%%)  mean confidence %.3f  (%.1f ms/row on CPU, %d threads)", int(meta["needs_review"].sum()), len(meta), 100 * meta["needs_review"].mean(), meta["confidence"].mean(), ms_per_row, a.threads)
    save_json({**summary, "needs_review_flagged": int(meta["needs_review"].sum()), "flag_rate": float(meta["needs_review"].mean()), "mean_confidence": float(meta["confidence"].mean()),
               "ms_per_row": ms_per_row, "threads": a.threads, "rules_fired": int(sum(bool(f) for f in fired)), "model_dir": str(a.model_dir)}, a.summary_out)


if __name__ == "__main__":
    main()
