#!/usr/bin/env python
"""Negative-result study: per-class logit bias (decision-threshold tuning for macro-F1).

For every finished encoder experiment, fit a per-class additive logit bias that maximises
OVERALL on four folds' out-of-fold logits and apply it to the fifth (cross-fitted), with
and without the rule layer. Writes reports/results/logit_bias_study.json. The shipped
artefact only carries a non-zero bias if this study shows a cross-fitted gain for the
selected model — otherwise `logit_bias` stays all zeros.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from foursight_intent.calibration import cross_fitted_bias, fit_temperature, softmax
from foursight_intent.data import cv_folds, load_train
from foursight_intent.labels import ID2LABEL, LABELS
from foursight_intent.metrics import official_scores
from foursight_intent.rules import apply_rules
from foursight_intent.utils import get_logger, save_json

log = get_logger("bias_study")


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    df = load_train(ROOT / cfg["data"]["train_csv"])
    protocol = cfg["validation"]["protocol"]
    folds = cv_folds(df, int(cfg["validation"]["n_splits"]), int(cfg["seed"]), protocol)
    y = df["label"].to_numpy(); gold = df["intent"].tolist(); texts = df["text"].tolist()
    out = {}
    for path in sorted(glob.glob(str(ROOT / "reports/results/transformer_*.json"))):
        r = json.load(open(path))
        if r.get("kind") == "ensemble":
            continue
        if r.get("protocol", protocol) != protocol:
            log.info("skipping %s: protocol %s != %s", path, r.get("protocol"), protocol)
            continue
        if r["rows_evaluated"] != len(df):
            continue
        name = r["name"]
        logits = np.load(ROOT / "reports/results" / f"oof_logits_{name}.npy")
        cal = np.zeros_like(logits)
        for f in folds:
            T = fit_temperature(logits[f.train_idx], y[f.train_idx]); cal[f.val_idx] = logits[f.val_idx] / T
        base = official_scores(gold, [ID2LABEL[i] for i in cal.argmax(1)])
        biased, biases = cross_fitted_bias(cal, y, folds)
        sc = official_scores(gold, [ID2LABEL[i] for i in biased.argmax(1)])
        adj, _ = apply_rules(texts, softmax(biased), max_conf=0.6)
        sc_r = official_scores(gold, [ID2LABEL[i] for i in adj.argmax(1)])
        adj0, _ = apply_rules(texts, softmax(cal), max_conf=0.6)
        sc_r0 = official_scores(gold, [ID2LABEL[i] for i in adj0.argmax(1)])
        out[name] = {
            "OVERALL_no_bias": base["OVERALL"], "OVERALL_bias_cross_fitted": sc["OVERALL"], "delta": sc["OVERALL"] - base["OVERALL"],
            "OVERALL_rules_only": sc_r0["OVERALL"], "OVERALL_bias_plus_rules": sc_r["OVERALL"],
            "macro_f1_no_bias": base["macro_f1"], "macro_f1_bias": sc["macro_f1"], "accuracy_no_bias": base["accuracy"], "accuracy_bias": sc["accuracy"],
            "per_class_f1_no_bias": {c: base["per_class"][c]["f1"] for c in LABELS}, "per_class_f1_bias": {c: sc["per_class"][c]["f1"] for c in LABELS},
            "fold_bias_mean": dict(zip(LABELS, np.mean(biases, 0).round(3).tolist())), "fold_bias_std": dict(zip(LABELS, np.std(biases, 0).round(3).tolist())),
        }
        log.info("%-20s no-bias %.4f  cross-fitted bias %.4f (%+.4f)  rules-only %.4f  bias+rules %.4f", name, base["OVERALL"], sc["OVERALL"], sc["OVERALL"] - base["OVERALL"], sc_r0["OVERALL"], sc_r["OVERALL"])
    helped = [n for n, v in out.items() if v["delta"] > 0.005]
    verdict = ("Per-class logit bias improves cross-fitted OVERALL by > 0.005 for: " + ", ".join(helped)) if helped else "Per-class logit bias does not improve cross-fitted OVERALL for any candidate (gains seen in-sample do not transfer across folds: the fitted biases vary a lot between folds, see fold_bias_std). Not shipped."
    save_json({"candidates": out, "verdict": verdict}, ROOT / "reports/results/logit_bias_study.json")
    log.info(verdict)


if __name__ == "__main__":
    main()
