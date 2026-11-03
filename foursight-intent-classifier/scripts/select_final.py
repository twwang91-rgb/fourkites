#!/usr/bin/env python
"""Freeze the final model choice from the recorded comparison and write
reports/results/final_selection.json (consumed by train_final.py and build_report.py).

Selection policy (stated up front, applied mechanically by this script):
  1. candidates = primary fine-tuned encoders with full OOF coverage, each with and without
     the rule layer (rules applied only when calibrated confidence < 0.6), plus the
     greedy top-k probability-averaging ensembles from compare_candidates.py;
  2. rank by pooled OOF OVERALL;
  3. the rule layer is kept only if it improves OVERALL AND lowers no class F1 by more
     than 0.03 (a rule that trades one rare class for another is rejected);
  4. an ensemble is eligible only if, against the best single-model option, it
       (a) gains more than 0.02 pooled OOF OVERALL,
       (b) wins on at least 4 of the 5 folds, and
       (c) lowers no class F1 by more than 0.03;
     otherwise the extra inference cost and artefact complexity are not justified and the
     best single model ships (the ensemble is still reported);
  5. ties within 0.005 OVERALL are broken, in order, by (a) the higher OVERALL of the *same
     option without the rule layer* (a regex layer is the most brittle component under
     distribution shift, so the stronger learned model wins), (b) lower measured CPU
     latency, (c) fewer parameters / members.
     History: the first draft of this rule said only "smaller model"; parameter count made
     DeBERTa-v3-small (142M, of which 98M is its 128k-token embedding table) look larger
     than RoBERTa-base (125M) although it is faster on CPU. The tie-break was made explicit
     when that ambiguity surfaced, and the outcome under the original wording is recorded in
     final_selection.json ("tie_break_audit") so the change is auditable.
Robustness and ablation runs (other seeds, other class weights) are never selected; they
exist to check that the ranking is stable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from foursight_intent.calibration import fit_temperature, softmax
from foursight_intent.data import cv_folds, load_train
from foursight_intent.labels import ID2LABEL, LABELS
from foursight_intent.metrics import official_scores, triad_confusion
from foursight_intent.rules import apply_rules
from foursight_intent.utils import get_logger, save_json

log = get_logger("select")

RULE_MAX_CLASS_DROP = 0.03
ENSEMBLE_MIN_GAIN = 0.02
ENSEMBLE_MIN_FOLD_WINS = 4
TIE_BAND = 0.005


def per_fold(pred: np.ndarray, gold: list[str], folds) -> list[float]:
    return [official_scores([gold[i] for i in f.val_idx], [ID2LABEL[int(pred[i])] for i in f.val_idx])["OVERALL"] for f in folds]


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    df = load_train(ROOT / cfg["data"]["train_csv"])
    folds = cv_folds(df, int(cfg["validation"]["n_splits"]), int(cfg["seed"]), cfg["validation"]["protocol"])
    y = df["label"].to_numpy(); gold = df["intent"].tolist(); texts = df["text"].tolist()
    res = ROOT / "reports/results"
    cmp_ = json.load(open(res / "comparison.json"))
    boot = cmp_["bootstrap"]
    primary = set(cmp_.get("primary_candidates", []))
    singles = [c for c in cmp_["candidates"] if c["kind"] == "single" and c["model"] in primary]
    ensembles = cmp_.get("ensembles", [])

    def evaluate(name: str, logits_or_logp: np.ndarray, params_M, kind: str, members=None) -> list[dict]:
        cal = np.zeros_like(logits_or_logp)
        for f in folds:  # cross-fitted temperature so the rule threshold sees calibrated confidences
            T = fit_temperature(logits_or_logp[f.train_idx], y[f.train_idx]); cal[f.val_idx] = softmax(logits_or_logp[f.val_idx] / T)
        base_pred = cal.argmax(1)
        base_sc = official_scores(gold, [ID2LABEL[int(i)] for i in base_pred])
        adj, _ = apply_rules(texts, cal, max_conf=0.6)
        rule_pred = adj.argmax(1)
        rule_sc = official_scores(gold, [ID2LABEL[int(i)] for i in rule_pred])
        drops = {k: rule_sc["per_class"][k]["f1"] - base_sc["per_class"][k]["f1"] for k in LABELS}
        rules_ok = rule_sc["OVERALL"] > base_sc["OVERALL"] and min(drops.values()) >= -RULE_MAX_CLASS_DROP
        common = {"name": name, "kind": kind, "members": members or [name], "params_M": params_M}
        return [
            {**common, "use_rules": False, "scores": base_sc, "per_fold": per_fold(base_pred, gold, folds), "bootstrap_ci": boot.get(name, {}).get("ci95"), "triad_errors": triad_confusion(base_sc)["total_triad_errors"], "eligible": True},
            {**common, "use_rules": True, "scores": rule_sc, "per_fold": per_fold(rule_pred, gold, folds), "bootstrap_ci": boot.get(name + "+rules", {}).get("ci95"), "triad_errors": triad_confusion(rule_sc)["total_triad_errors"],
             "rules_ok": rules_ok, "rule_class_f1_deltas": drops, "eligible": rules_ok},
        ]

    options: list[dict] = []
    for c in singles:
        options += evaluate(c["model"], np.load(res / f"oof_logits_{c['model']}.npy"), c.get("params_M"), "single")
    for e in ensembles:
        options += evaluate(e["model"], np.load(res / f"oof_logits_{e['model']}.npy"), e.get("params_M"), "ensemble", e["members"])

    # ---- best single (rule 3 + rule 5)
    latency = {c["model"]: c.get("latency", {}).get("cpu_ms_per_row_batched", float("inf")) for c in cmp_["candidates"]}
    no_rules_overall = {o["name"]: o["scores"]["OVERALL"] for o in options if not o["use_rules"]}

    def tie_key(o):
        return (-no_rules_overall[o["name"]], latency.get(o["name"], float("inf")), o["params_M"] or 0, len(o["members"]))

    single_opts = [o for o in options if o["kind"] == "single" and o["eligible"]]
    single_opts.sort(key=lambda o: (-round(o["scores"]["OVERALL"], 3),) + tie_key(o))
    top = single_opts[0]["scores"]["OVERALL"]
    tied = [o for o in single_opts if top - o["scores"]["OVERALL"] <= TIE_BAND]
    best_single = min(tied, key=tie_key)
    tie_break_audit = {
        "tied_options": [{"name": o["name"], "use_rules": o["use_rules"], "OVERALL": o["scores"]["OVERALL"], "OVERALL_without_rules": no_rules_overall[o["name"]],
                          "cpu_ms_per_row_batched": latency.get(o["name"]), "params_M": o["params_M"]} for o in tied],
        "chosen_by_current_rule": best_single["name"],
        "chosen_by_original_rule_smaller_param_count": min(tied, key=lambda o: (o["params_M"] or 0, -o["scores"]["OVERALL"]))["name"],
    }

    # ---- ensemble gate (rule 4)
    gated = []
    for o in options:
        if o["kind"] != "ensemble" or not o["eligible"]:
            continue
        gain = o["scores"]["OVERALL"] - best_single["scores"]["OVERALL"]
        wins = int(sum(a > b for a, b in zip(o["per_fold"], best_single["per_fold"])))
        drops = {k: o["scores"]["per_class"][k]["f1"] - best_single["scores"]["per_class"][k]["f1"] for k in LABELS}
        o["gate"] = {"gain": gain, "fold_wins": wins, "min_class_delta": min(drops.values()), "class_deltas": drops,
                     "passes": gain > ENSEMBLE_MIN_GAIN and wins >= ENSEMBLE_MIN_FOLD_WINS and min(drops.values()) >= -RULE_MAX_CLASS_DROP}
        if o["gate"]["passes"]:
            gated.append(o)
    if gated:
        gated.sort(key=lambda o: (-round(o["scores"]["OVERALL"], 3), len(o["members"])))
        top_e = gated[0]["scores"]["OVERALL"]
        best = min([o for o in gated if top_e - o["scores"]["OVERALL"] <= TIE_BAND], key=tie_key)
        g = best["gate"]
        why = (f"Probability-averaging ensemble of {len(best['members'])} fine-tuned encoders ({' + '.join(best['members'])}). It passes the pre-stated ensemble gate against the best single model "
               f"({best_single['name']}{' + rules' if best_single['use_rules'] else ''}, OVERALL {best_single['scores']['OVERALL']:.3f}): gain {g['gain']:+.3f} (> {ENSEMBLE_MIN_GAIN}), "
               f"wins {g['fold_wins']}/5 folds (≥ {ENSEMBLE_MIN_FOLD_WINS}), smallest per-class F1 change {g['min_class_delta']:+.3f} (≥ -{RULE_MAX_CLASS_DROP}). "
               + (f"The rule layer is kept ({best['scores']['OVERALL'] - [o for o in options if o['name'] == best['name'] and not o['use_rules']][0]['scores']['OVERALL']:+.3f} OVERALL, no class loses more than {RULE_MAX_CLASS_DROP} F1)." if best["use_rules"] else "The rule layer is not kept for the ensemble (no OVERALL gain or a class regression)."))
    else:
        best = best_single
        base_for_best = [o for o in options if o["name"] == best["name"] and not o["use_rules"]][0]
        ens_note = ""
        ens_opts = [o for o in options if o["kind"] == "ensemble"]
        if ens_opts:
            be = max(ens_opts, key=lambda o: o["scores"]["OVERALL"])
            g = be.get("gate", {})
            ens_note = (f" The best ensemble ({be['name']}{' + rules' if be['use_rules'] else ''}) scores {be['scores']['OVERALL']:.3f} ({g.get('gain', float('nan')):+.3f}, {g.get('fold_wins', '?')}/5 fold wins, "
                        f"min class ΔF1 {g.get('min_class_delta', float('nan')):+.3f}) and fails the ensemble gate, so the single model ships.")
        tie_note = ""
        if len(tied) > 1:
            others = [o for o in tied if o is not best]
            tie_note = (" It is within the 0.005 tie band of " + ", ".join(f"{o['name']}{' + rules' if o['use_rules'] else ''} ({o['scores']['OVERALL']:.4f})" for o in others)
                        + f"; the tie-break prefers the stronger learned model without rules ({best['name']} {no_rules_overall[best['name']]:.3f} vs " + ", ".join(f"{o['name']} {no_rules_overall[o['name']]:.3f}" for o in others) + ")"
                        + (f" and lower CPU latency ({latency[best['name']]:.1f} ms/row vs " + ", ".join(f"{latency[o['name']]:.1f}" for o in others) + ")." if latency.get(best['name'], float('inf')) < float('inf') else "."))
        why = ("Highest pooled OOF OVERALL among deployable single-model options under the pre-stated policy." + tie_note + " "
               + (f"The rule layer adds {best['scores']['OVERALL'] - base_for_best['scores']['OVERALL']:+.3f} OVERALL (largest class deltas: " + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(best.get('rule_class_f1_deltas', {}).items(), key=lambda kv: -abs(kv[1]))[:3]) + f") and no class loses more than {RULE_MAX_CLASS_DROP} F1." if best["use_rules"] else "The rule layer was not kept because it did not improve OVERALL without a class regression.")
               + ens_note)
    desc = ("fine-tuned " + best["name"] if best["kind"] == "single" else f"{best['name']} ({' + '.join(best['members'])})") + (" + rule layer" if best["use_rules"] else "")
    out = {"policy": __doc__, "thresholds": {"rule_max_class_drop": RULE_MAX_CLASS_DROP, "ensemble_min_gain": ENSEMBLE_MIN_GAIN, "ensemble_min_fold_wins": ENSEMBLE_MIN_FOLD_WINS, "tie_band": TIE_BAND},
           "options": options, "tie_break_audit": tie_break_audit, "best_single": {k: best_single[k] for k in ("name", "use_rules", "scores", "per_fold", "params_M")}, "selected": {**best, "description": desc, "why": why}}
    save_json(out, res / "final_selection.json")
    log.info("SELECTED: %s  OVERALL=%.4f  macroF1=%.4f  acc=%.4f", desc, best["scores"]["OVERALL"], best["scores"]["macro_f1"], best["scores"]["accuracy"])
    log.info("why: %s", why)
    for o in sorted(options, key=lambda o: -o["scores"]["OVERALL"]):
        flag = "" if o["eligible"] else "(rules rejected)"
        if o["kind"] == "ensemble" and "gate" in o:
            flag += " gate=" + ("PASS" if o["gate"]["passes"] else f"fail(gain {o['gate']['gain']:+.3f}, wins {o['gate']['fold_wins']}/5, minΔ {o['gate']['min_class_delta']:+.3f})")
        log.info("   %-20s %-8s rules=%-5s OVERALL=%.4f macroF1=%.4f acc=%.4f folds=%s %s", o["name"], o["kind"], o["use_rules"], o["scores"]["OVERALL"], o["scores"]["macro_f1"], o["scores"]["accuracy"], [round(x, 3) for x in o["per_fold"]], flag)


if __name__ == "__main__":
    main()
