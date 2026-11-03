#!/usr/bin/env python
"""Write out-of-fold validation files in the official format and run the official scorer.

This is the supported self-check workflow from the challenge README:

    python3 score.py --gold my_val.csv --pred my_val_predictions.csv

Here ``my_val.csv`` is every training row (each predicted by the fold model that never
saw it) and the predictions come from the selected candidate's OOF probabilities with the
same calibration / rule settings as the shipped artefact. The scorer's full text output is
saved to reports/results/score_py_oof_<name>.txt and its JSON to score_py_oof_<name>.json.
Per-fold scorer runs are also written (reports/validation/fold<k>_*.csv) so a single
hold-out fold can be re-scored by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from foursight_intent.calibration import fit_temperature, softmax
from foursight_intent.data import cv_folds, load_train
from foursight_intent.labels import ID2LABEL
from foursight_intent.rules import apply_rules
from foursight_intent.utils import get_logger

log = get_logger("score_oof")


def write_gold(df, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "intent"])
        for i, q, l in zip(df["id"], df["question"], df["intent"]):
            w.writerow([i, q, l])


def write_pred(ids, labels, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "intent"])
        for i, l in zip(ids, labels):
            w.writerow([i, l])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="experiment name (reports/results/oof_logits_<name>.npy)")
    ap.add_argument("--use-rules", action="store_true")
    ap.add_argument("--rules-max-conf", type=float, default=0.6)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    df = load_train(ROOT / cfg["data"]["train_csv"])
    folds = cv_folds(df, int(cfg["validation"]["n_splits"]), int(cfg["seed"]), cfg["validation"]["protocol"])
    y = df["label"].to_numpy()
    logits = np.load(ROOT / "reports/results" / f"oof_logits_{a.name}.npy")
    probs = np.zeros_like(logits)
    for f in folds:  # cross-fitted temperature, as in analysis / selection
        T = fit_temperature(logits[f.train_idx], y[f.train_idx])
        probs[f.val_idx] = softmax(logits[f.val_idx] / T)
    if a.use_rules:
        probs, _ = apply_rules(df["text"].tolist(), probs, max_conf=a.rules_max_conf)
    labels = [ID2LABEL[int(i)] for i in probs.argmax(1)]

    val_dir = ROOT / "reports" / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    gold_p, pred_p = val_dir / "oof_gold.csv", val_dir / f"oof_pred_{a.name}.csv"
    write_gold(df, gold_p)
    write_pred(df["id"], labels, pred_p)
    out_json = ROOT / "reports/results" / f"score_py_oof_{a.name}.json"
    res = subprocess.run([sys.executable, str(ROOT / "scripts/score.py"), "--gold", str(gold_p), "--pred", str(pred_p), "--out", str(out_json)], capture_output=True, text=True, check=True)
    (ROOT / "reports/results" / f"score_py_oof_{a.name}.txt").write_text(res.stdout)
    log.info("official scorer on pooled OOF (%s%s):\n%s", a.name, " + rules" if a.use_rules else "", res.stdout)

    per_fold = []
    for f in folds:
        sub = df.iloc[f.val_idx]
        g, p = val_dir / f"fold{f.index}_gold.csv", val_dir / f"fold{f.index}_pred_{a.name}.csv"
        write_gold(sub, g)
        write_pred(sub["id"], [labels[i] for i in f.val_idx], p)
        o = val_dir / f"fold{f.index}_score_{a.name}.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/score.py"), "--gold", str(g), "--pred", str(p), "--out", str(o)], capture_output=True, text=True, check=True)
        s = json.load(open(o))
        per_fold.append({"fold": f.index, "n": s["n"], "accuracy": s["accuracy"], "macro_f1": s["macro_f1"], "OVERALL": s["OVERALL"]})
        log.info("fold %d (n=%d): acc=%.4f macroF1=%.4f OVERALL=%.4f", f.index, s["n"], s["accuracy"], s["macro_f1"], s["OVERALL"])
    json.dump(per_fold, open(ROOT / "reports/results" / f"score_py_folds_{a.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
