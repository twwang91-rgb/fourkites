#!/usr/bin/env python
"""Run the classical baselines under grouped 5-fold CV and record results.

Baseline A: TF-IDF (word / char / word+char) x {logistic regression, linear SVM} x C grid.
Baseline B: frozen sentence embeddings + logistic regression (several small encoders).

Outputs
  reports/results/baselines.json      every configuration, per-fold + pooled OOF scores
  reports/results/oof_<name>.npy      out-of-fold probabilities of the best config per family
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from foursight_intent.baselines import EmbeddingCache, EmbeddingLinear, TfidfLinear
from foursight_intent.data import cv_folds, load_train
from foursight_intent.evaluate import run_cv
from foursight_intent.labels import LABELS
from foursight_intent.metrics import triad_confusion
from foursight_intent.utils import get_logger, save_json, seed_everything

log = get_logger("baselines")


def record(name: str, fs, timing, extra=None) -> dict:
    s = fs.summary()
    pooled = fs.pooled
    rec = {
        "model": name,
        "accuracy_oof": pooled["accuracy"],
        "macro_f1_oof": pooled["macro_f1"],
        "OVERALL_oof": pooled["OVERALL"],
        "OVERALL_fold_mean": s["OVERALL_mean"],
        "OVERALL_fold_std": s["OVERALL_std"],
        "per_class_f1": {c: pooled["per_class"][c]["f1"] for c in LABELS},
        "triad_errors": triad_confusion(pooled),
        "predict_ms_per_row": timing["predict_ms_per_row"],
        "fit_seconds_total": timing["fit_seconds_total"],
    }
    if extra:
        rec.update(extra)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--protocol", default=None, help="override validation protocol (grouped|stratified)")
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smaller grids for smoke testing")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    seed = int(cfg["seed"])
    seed_everything(seed)
    protocol = a.protocol or cfg["validation"]["protocol"]
    n_splits = int(cfg["validation"]["n_splits"])

    df = load_train(ROOT / cfg["data"]["train_csv"])
    folds = cv_folds(df, n_splits=n_splits, seed=seed, protocol=protocol)
    out_dir = ROOT / "reports" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    best_oof: dict[str, tuple[float, np.ndarray, str]] = {}

    # ---------------- Baseline A: TF-IDF --------------------------------------------
    bcfg = cfg["baselines"]["tfidf"]
    C_grid = [1.0, 4.0] if a.quick else bcfg["C_grid"]
    feats_grid = ["word+char"] if a.quick else ["word", "char", "word+char"]
    for features in feats_grid:
        for clf in ["logreg", "svm"]:
            for cw in ["balanced", None]:
                if a.quick and cw is None:
                    continue
                for C in C_grid:
                    name = TfidfLinear(features=features, clf=clf, C=C, class_weight=cw).name
                    t0 = time.perf_counter()
                    oof, fs, timing = run_cv(
                        df, folds,
                        lambda: TfidfLinear(
                            features=features, clf=clf, C=C, class_weight=cw, seed=seed,
                            word_ngrams=tuple(bcfg["word_ngrams"]), char_ngrams=tuple(bcfg["char_ngrams"]),
                        ),
                        verbose=False,
                    )
                    rec = record(name, fs, timing, {"family": "tfidf", "features": features, "clf": clf, "C": C, "class_weight": cw})
                    results.append(rec)
                    log.info("%-55s OOF acc=%.4f macroF1=%.4f OVERALL=%.4f (%.1fs)", name, rec["accuracy_oof"], rec["macro_f1_oof"], rec["OVERALL_oof"], time.perf_counter() - t0)
                    key = f"tfidf_{clf}"
                    if key not in best_oof or rec["OVERALL_oof"] > best_oof[key][0]:
                        best_oof[key] = (rec["OVERALL_oof"], oof, name)

    # ---------------- Baseline B: sentence embeddings --------------------------------
    if not a.skip_embeddings:
        ecfg = cfg["baselines"]["embeddings"]
        models = ecfg["models"][:1] if a.quick else ecfg["models"]
        for model_name in models:
            try:
                t0 = time.perf_counter()
                cache = EmbeddingCache(model_name, device="cpu")
                cache.encode(df["text"].tolist())  # encode once, reuse across folds
                enc_s = time.perf_counter() - t0
                log.info("encoded %d texts with %s in %.1fs", len(df), model_name, enc_s)
            except Exception as exc:  # model download failure etc.
                log.error("skipping %s: %s", model_name, exc)
                results.append({"model": f"emb[{model_name}]", "family": "embedding", "error": str(exc)})
                continue
            for cw in ["balanced", None]:
                for C in ([1.0, 4.0] if a.quick else ecfg["C_grid"]):
                    name = EmbeddingLinear(cache, C=C, class_weight=cw).name
                    oof, fs, timing = run_cv(df, folds, lambda: EmbeddingLinear(cache, C=C, class_weight=cw, seed=seed), verbose=False)
                    timing["predict_ms_per_row"] = 1000 * enc_s / len(df) + timing["predict_ms_per_row"]
                    rec = record(name, fs, timing, {"family": "embedding", "encoder": model_name, "C": C, "class_weight": cw, "encode_seconds": enc_s})
                    results.append(rec)
                    log.info("%-55s OOF acc=%.4f macroF1=%.4f OVERALL=%.4f", name, rec["accuracy_oof"], rec["macro_f1_oof"], rec["OVERALL_oof"])
                    key = "embedding"
                    if key not in best_oof or rec["OVERALL_oof"] > best_oof[key][0]:
                        best_oof[key] = (rec["OVERALL_oof"], oof, name)

    results.sort(key=lambda r: r.get("OVERALL_oof", -1), reverse=True)
    save_json({"protocol": protocol, "n_splits": n_splits, "seed": seed, "results": results}, out_dir / f"baselines_{protocol}.json")
    for key, (score, oof, name) in best_oof.items():
        np.save(out_dir / f"oof_{key}_{protocol}.npy", oof)
        (out_dir / f"oof_{key}_{protocol}.name").write_text(name)
    log.info("top 10 configurations (%s CV):", protocol)
    for r in results[:10]:
        log.info("  %-55s OVERALL=%.4f  macroF1=%.4f  acc=%.4f", r["model"], r["OVERALL_oof"], r["macro_f1_oof"], r["accuracy_oof"])


if __name__ == "__main__":
    main()
