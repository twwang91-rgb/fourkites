#!/usr/bin/env python
"""Cross-validated fine-tuning experiments for encoder candidates.

Example
  python scripts/run_experiments.py --name distilbert --model distilbert-base-uncased
  python scripts/run_experiments.py --name deberta_small --model microsoft/deberta-v3-small --lr 3e-5

For every fold the best-epoch validation logits are kept, so the script writes pooled
out-of-fold logits (reports/results/oof_logits_<name>.npy) plus per-fold metrics and
training curves (reports/results/transformer_<name>.json). One W&B run per fold
(offline unless WANDB_MODE=online and credentials exist).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from foursight_intent.data import cv_folds, load_train
from foursight_intent.labels import ID2LABEL, LABELS
from foursight_intent.metrics import official_scores, triad_confusion
from foursight_intent.train import TrainConfig, train_one
from foursight_intent.utils import get_logger, save_json

log = get_logger("experiments")


def init_wandb(cfg: dict, name: str, fold: int, tcfg: TrainConfig, protocol: str):
    mode = os.environ.get("WANDB_MODE", cfg.get("wandb", {}).get("mode", "offline"))
    if mode == "disabled":
        return None
    try:
        import wandb

        run = wandb.init(
            project=cfg.get("wandb", {}).get("project", "foursight-intent"), name=f"{name}-fold{fold}", group=name,
            job_type="cv", mode=mode, dir=str(ROOT), reinit="finish_previous",
            config={**tcfg.__dict__, "fold": fold, "protocol": protocol, "n_splits": cfg["validation"]["n_splits"], "labels": LABELS},
        )
        return run
    except Exception as exc:  # never let tracking break training
        log.warning("wandb disabled: %s", exc)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--class-weight", default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--label-smoothing", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--folds", default=None, help="comma separated subset of folds, e.g. 0 or 0,1")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    tcfg = TrainConfig.from_yaml(
        cfg, model_name=a.model, learning_rate=a.lr, max_epochs=a.epochs, batch_size=a.batch_size, class_weight=a.class_weight,
        max_length=a.max_length, label_smoothing=a.label_smoothing, patience=a.patience, device=a.device, seed=a.seed,
    )
    protocol = a.protocol or cfg["validation"]["protocol"]
    df = load_train(ROOT / cfg["data"]["train_csv"])
    folds = cv_folds(df, n_splits=int(cfg["validation"]["n_splits"]), seed=tcfg.seed, protocol=protocol)
    if a.folds:
        keep = {int(x) for x in a.folds.split(",")}
        folds = [f for f in folds if f.index in keep]
    out_dir = ROOT / "reports" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    oof = np.full((len(df), len(LABELS)), np.nan)
    fold_records = []
    t_start = time.perf_counter()
    for f in folds:
        run = init_wandb(cfg, a.name, f.index, tcfg, protocol)
        res = train_one(df.iloc[f.train_idx], df.iloc[f.val_idx], tcfg, wandb_run=run, log_prefix=f"[{a.name} fold {f.index}] ")
        oof[f.val_idx] = res.val_logits
        fold_records.append({
            "fold": f.index, "best_epoch": res.best_epoch, "accuracy": res.best_scores["accuracy"], "macro_f1": res.best_scores["macro_f1"],
            "OVERALL": res.best_scores["OVERALL"], "per_class_f1": {c: res.best_scores["per_class"][c]["f1"] for c in LABELS},
            "history": res.history, "train_seconds": res.train_seconds,
        })
        log.info("[%s] fold %d done: best epoch %d OVERALL=%.4f (%.0fs)", a.name, f.index, res.best_epoch, res.best_scores["OVERALL"], res.train_seconds)
        if run is not None:
            run.finish()

    done = ~np.isnan(oof).any(axis=1)
    pooled = official_scores(df.loc[done, "intent"].tolist(), [ID2LABEL[int(i)] for i in oof[done].argmax(1)])
    summary = {
        "name": a.name, "config": tcfg.__dict__, "protocol": protocol, "folds_run": [f.index for f in folds],
        "rows_evaluated": int(done.sum()),
        "accuracy_oof": pooled["accuracy"], "macro_f1_oof": pooled["macro_f1"], "OVERALL_oof": pooled["OVERALL"],
        "OVERALL_fold_mean": float(np.mean([r["OVERALL"] for r in fold_records])), "OVERALL_fold_std": float(np.std([r["OVERALL"] for r in fold_records])),
        "per_class": pooled["per_class"], "confusion": pooled["confusion"], "confused_pairs": pooled["confused_pairs"][:12],
        "triad_errors": triad_confusion(pooled), "best_epochs": [r["best_epoch"] for r in fold_records],
        "train_seconds_total": float(sum(r["train_seconds"] for r in fold_records)), "wall_seconds": time.perf_counter() - t_start,
        "folds": fold_records,
    }
    save_json(summary, out_dir / f"transformer_{a.name}.json")
    np.save(out_dir / f"oof_logits_{a.name}.npy", oof)
    log.info("[%s] pooled OOF  acc=%.4f  macroF1=%.4f  OVERALL=%.4f  (fold mean %.4f ± %.4f)", a.name, pooled["accuracy"], pooled["macro_f1"], pooled["OVERALL"], summary["OVERALL_fold_mean"], summary["OVERALL_fold_std"])
    for c in LABELS:
        log.info("    %-22s f1=%.3f  (support %d)", c, pooled["per_class"][c]["f1"], pooled["per_class"][c]["support"])


if __name__ == "__main__":
    main()
