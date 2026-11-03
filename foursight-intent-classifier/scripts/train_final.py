#!/usr/bin/env python
"""Retrain the selected configuration on ALL labelled rows and save the deployable artefact.

Run only after model selection is frozen (reports/results/final_selection.json). Works for
a single model or an ensemble:

  single   : models/final/ is one transformers model dir (+ label_map, inference_config,
             training_summary, README)
  ensemble : models/final/<member>/ per member, plus models/final/ensemble.json that lists
             members, per-member temperatures and the ensemble-level review policy

Epoch counts come from each member's CV best epochs (median). Temperatures and the review
threshold come from the out-of-fold analyses (analysis_<name>.json), because no held-out
data is left once we train on everything.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml

from foursight_intent.data import load_train
from foursight_intent.labels import ID2LABEL, LABEL2ID, LABELS
from foursight_intent.train import TrainConfig, fit_full, save_model
from foursight_intent.utils import get_logger, load_json, save_json

log = get_logger("train_final")


def wandb_run(cfg: dict, name: str, group: str, config: dict):
    mode = os.environ.get("WANDB_MODE", cfg.get("wandb", {}).get("mode", "offline"))
    if mode == "disabled":
        return None
    try:
        import wandb

        return wandb.init(project=cfg["wandb"]["project"], name=name, group=group, job_type="final", mode=mode, dir=str(ROOT), reinit="finish_previous", config=config)
    except Exception as exc:
        log.warning("wandb disabled: %s", exc)
        return None


def environment() -> dict:
    env = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()}
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except Exception:
        pass
    return env


def train_member(name: str, cfg: dict, df, out_dir: Path, device: str | None, epochs_override: int | None, use_rules: bool, rules_max_conf: float, standalone: bool, threads: int | None = None) -> dict:
    exp = load_json(ROOT / "reports/results" / f"transformer_{name}.json")
    ana_path = ROOT / "reports/results" / f"analysis_{name}.json"
    ana = load_json(ana_path) if ana_path.exists() else None
    tcfg = TrainConfig(**{k: v for k, v in exp["config"].items() if k in TrainConfig.__dataclass_fields__})
    if device:
        tcfg.device = device
    if threads:
        tcfg.num_threads = threads
    epochs = epochs_override or int(round(statistics.median(exp["best_epochs"])))
    log.info("[%s] retraining %s on all %d rows for %d epochs (CV best epochs %s)", name, tcfg.model_name, len(df), epochs, exp["best_epochs"])
    run = wandb_run(cfg, f"{name}-final-full-train", name, {**tcfg.__dict__, "epochs": epochs, "rows": "all", "selected_from": name})
    t0 = time.perf_counter()
    result, model, tok = fit_full(df, tcfg, epochs, wandb_run=run)
    train_s = time.perf_counter() - t0
    temperature = ana["calibration"]["fitted_on_all_oof"]["temperature"] if ana else 1.0
    review = {"threshold": 0.0, "metric": "confidence"}
    if ana:
        review = ana["review"]["deployment_policy_with_rules"] if (use_rules and standalone and "deployment_policy_with_rules" in ana["review"]) else ana["review"]["deployment_policy"]
    inference_config = {
        "kind": "single", "max_length": tcfg.max_length, "tail_tokens": tcfg.tail_tokens, "temperature": temperature,
        "review_threshold": review["threshold"], "review_metric": review["metric"],
        "use_rules": bool(use_rules and standalone), "rules_max_conf": rules_max_conf,
        "logit_bias": [0.0] * len(LABELS), "logit_bias_note": "all zeros: the cross-fitted per-class bias study (reports/results/logit_bias_study.json) showed no out-of-fold gain",
        "labels": LABELS, "taxonomy_version": "v1", "base_model": tcfg.model_name, "seed": tcfg.seed, "epochs": epochs,
        "trained_on": f"all {len(df):,} labelled rows of train.csv", "selected_experiment": name,
    }
    summary = {
        "experiment": name, "kind": "single", "epochs": epochs, "train_seconds": train_s, "history": result.history, "config": tcfg.__dict__,
        "cv_reference": {"OVERALL_oof": exp["OVERALL_oof"], "macro_f1_oof": exp["macro_f1_oof"], "accuracy_oof": exp["accuracy_oof"], "best_epochs": exp["best_epochs"]},
        "environment": {**environment(), "device_used": tcfg.device, "num_threads": tcfg.num_threads},
    }
    save_model(model, tok, out_dir, inference_config, {"training_summary.json": summary})
    if run is not None:
        run.summary.update({"final_train_loss": result.history[-1]["train_loss"], "epochs": epochs})
        run.finish()
    log.info("[%s] saved to %s (%.0fs)", name, out_dir, train_s)
    del model
    return {"name": name, "dir": out_dir.name, "base_model": tcfg.model_name, "epochs": epochs, "temperature": temperature, "train_seconds": train_s, "history": result.history, "config": tcfg.__dict__, "cv_reference": summary["cv_reference"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--experiment", default=None, help="CV experiment name; default: the selection in reports/results/final_selection.json")
    ap.add_argument("--out", default=str(ROOT / "models/final"))
    ap.add_argument("--epochs", type=int, default=None, help="override the epoch count (applies to every member)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--use-rules", action="store_true", help="enable the rule post-processing layer (default: follow final_selection.json)")
    ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--rules-max-conf", type=float, default=0.6)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads for training (fix it for bit-reproducible CPU training)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    sel_path = ROOT / "reports/results/final_selection.json"
    selected = load_json(sel_path)["selected"] if sel_path.exists() else None
    name = a.experiment or (selected["name"] if selected else None)
    if name is None:
        sys.exit("give --experiment or run scripts/select_final.py first")
    exp = load_json(ROOT / "reports/results" / f"transformer_{name}.json")
    kind = exp.get("kind", "single")
    use_rules = (selected["use_rules"] if (selected and selected["name"] == name) else False)
    if a.use_rules:
        use_rules = True
    if a.no_rules:
        use_rules = False
    df = load_train(ROOT / cfg["data"]["train_csv"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if kind == "single":
        train_member(name, cfg, df, out, a.device, a.epochs, use_rules, a.rules_max_conf, standalone=True, threads=a.threads)
        return

    # ---------------------------------------------------------------- ensemble
    members = exp["members"]
    ana_path = ROOT / "reports/results" / f"analysis_{name}.json"
    if not ana_path.exists():
        sys.exit(f"run scripts/analyze_results.py --name {name} first (needed for the ensemble review threshold)")
    ana = load_json(ana_path)
    for m in members:
        if not (ROOT / "reports/results" / f"analysis_{m}.json").exists():
            sys.exit(f"run scripts/analyze_results.py --name {m} first (needed for member temperature)")
    recs = [train_member(m, cfg, df, out / m, a.device, a.epochs, False, a.rules_max_conf, standalone=False, threads=a.threads) for m in members]
    review = ana["review"]["deployment_policy_with_rules"] if (use_rules and "deployment_policy_with_rules" in ana["review"]) else ana["review"]["deployment_policy"]
    ens_T = float(ana["calibration"]["fitted_on_all_oof"]["temperature"])
    spec = {
        "kind": "ensemble", "members": [{"name": r["name"], "dir": r["dir"], "weight": 1.0, "temperature": r["temperature"], "base_model": r["base_model"], "epochs": r["epochs"]} for r in recs],
        "ensemble_temperature": ens_T, "temperature": ens_T, "review_threshold": review["threshold"], "review_metric": review["metric"],
        "use_rules": bool(use_rules), "rules_max_conf": a.rules_max_conf, "labels": LABELS, "taxonomy_version": "v1",
        "base_model": "ensemble (probability average) of " + " + ".join(r["base_model"] for r in recs),
        "max_length": recs[0]["config"]["max_length"], "tail_tokens": recs[0]["config"]["tail_tokens"], "seed": recs[0]["config"]["seed"],
        "epochs": {r["name"]: r["epochs"] for r in recs}, "trained_on": f"all {len(df):,} labelled rows of train.csv", "selected_experiment": name,
    }
    save_json(spec, out / "ensemble.json")
    save_json(spec, out / "inference_config.json")
    save_json({"labels": LABELS, "label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}, out / "label_map.json")
    summary = {
        "experiment": name, "kind": "ensemble", "members": {r["name"]: {k: r[k] for k in ("epochs", "train_seconds", "history", "config", "cv_reference", "temperature")} for r in recs},
        "epochs": {r["name"]: r["epochs"] for r in recs}, "train_seconds": float(sum(r["train_seconds"] for r in recs)), "config": recs[0]["config"],
        "cv_reference": {"OVERALL_oof": exp["OVERALL_oof"], "macro_f1_oof": exp["macro_f1_oof"], "accuracy_oof": exp["accuracy_oof"], "best_epochs": exp["best_epochs"]},
        "environment": {**environment(), "device_used": a.device or recs[0]["config"]["device"]},
    }
    save_json(summary, out / "training_summary.json")
    log.info("saved ensemble artefact to %s (%d members, %.0fs total)", out, len(recs), summary["train_seconds"])


if __name__ == "__main__":
    main()
