#!/usr/bin/env python
"""Figures derived from recorded results: training curves of the selected model's CV folds,
and the candidate comparison (OVERALL with bootstrap CI). Written to reports/figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#0b0b0b"
MUTED = "#52514e"      # text-secondary: neutral for non-series elements
GRID = "#e4e3df"
BLUE = "#2a78d6"       # categorical slot 1
ORANGE = "#eb6834"     # categorical slot 2
GREEN = "#1baf7a"      # categorical slot 3 (aqua)
NEUTRAL_BAR = "#b9b7b0"  # baselines: de-emphasised, identity carried by the axis label


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=7)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def training_curves(name: str, out: Path) -> None:
    r = json.load(open(ROOT / "reports/results" / f"transformer_{name}.json"))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for f in r["folds"]:
        h = f["history"]
        ep = [x["epoch"] for x in h]
        axes[0].plot(ep, [x["train_loss"] for x in h], color=BLUE, lw=1.4, alpha=0.75)
        axes[0].plot(ep, [x["val_loss"] for x in h], color=ORANGE, lw=1.4, alpha=0.75)
        axes[1].plot(ep, [x["val_OVERALL"] for x in h], color=GREEN, lw=1.4, alpha=0.75)
        axes[1].scatter([f["best_epoch"]], [f["OVERALL"]], s=18, color=GREEN, zorder=3, edgecolor="white", linewidth=0.8)
    axes[0].plot([], [], color=BLUE, label="train loss (per fold)")
    axes[0].plot([], [], color=ORANGE, label="validation loss (per fold)")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_xlabel("epoch", fontsize=8); axes[0].set_ylabel("weighted cross-entropy", fontsize=8)
    axes[0].set_title(f"{name}: loss curves, 5 grouped folds", fontsize=9, color=INK)
    axes[1].set_xlabel("epoch", fontsize=8); axes[1].set_ylabel("validation OVERALL", fontsize=8)
    axes[1].set_title("validation OVERALL per epoch (dot = selected checkpoint)", fontsize=9, color=INK)
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(out / f"training_curves_{name}.png", dpi=170)
    plt.close(fig)


def comparison(out: Path) -> None:
    c = json.load(open(ROOT / "reports/results/comparison.json"))
    rows = c["candidates"]
    boot = c["bootstrap"]
    sel = json.load(open(ROOT / "reports/results/final_selection.json"))["selected"] if (ROOT / "reports/results/final_selection.json").exists() else None
    labels, vals, lo, hi, colors = [], [], [], [], []
    for r in rows:
        labels.append(r["model"].replace("+logreg", "").replace("(C=", " (C=")[:40])
        vals.append(r["OVERALL"])
        ci = boot.get(r["model"], {}).get("ci95")
        lo.append(r["OVERALL"] - ci[0] if ci else 0); hi.append(ci[1] - r["OVERALL"] if ci else 0)
        colors.append(BLUE if r["family"] == "fine-tuned encoder" else NEUTRAL_BAR)
    for r in c.get("ensembles", []):
        if sel and r["model"] == sel["name"]:
            continue
        labels.append(r["model"] + " (" + "+".join(m.split("_")[0] for m in r["members"]) + ")"); vals.append(r["OVERALL"])
        ci = boot.get(r["model"], {}).get("ci95"); lo.append(r["OVERALL"] - ci[0] if ci else 0); hi.append(ci[1] - r["OVERALL"] if ci else 0); colors.append(ORANGE)
    if sel:
        labels.append(f"{sel['name']}{' + rules' if sel['use_rules'] else ''} (selected)"); vals.append(sel["scores"]["OVERALL"])
        ci = sel.get("bootstrap_ci"); lo.append(vals[-1] - ci[0] if ci else 0); hi.append(ci[1] - vals[-1] if ci else 0); colors.append(GREEN)
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(vals) + 1.2))
    y = np.arange(len(vals))
    ax.barh(y, [vals[i] for i in order], color=[colors[i] for i in order], height=0.62)
    ax.errorbar([vals[i] for i in order], y, xerr=[[lo[i] for i in order], [hi[i] for i in order]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    for yi, i in zip(y, order):
        ax.text(vals[i] + 0.012 + hi[i], yi, f"{vals[i]:.3f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([labels[i] for i in order], fontsize=7)
    ax.set_xlim(0.45, max(vals) + 0.12)
    ax.set_xlabel("pooled out-of-fold OVERALL (grouped 5-fold CV, 95% paired-bootstrap CI)", fontsize=8)
    ax.set_title("Candidate comparison (blue: fine-tuned, orange: ensemble, grey: baseline, aqua: shipped)", fontsize=9, color=INK, loc="left")
    style(ax); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(out / "candidate_comparison.png", dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="experiment whose training curves to plot (default: selected)")
    a = ap.parse_args()
    out = ROOT / "reports/figures"; out.mkdir(parents=True, exist_ok=True)
    name = a.name
    if name is None:
        name = json.load(open(ROOT / "reports/results/final_selection.json"))["selected"]["name"]
    exp = json.load(open(ROOT / "reports/results" / f"transformer_{name}.json"))
    for member in (exp["members"] if exp.get("kind") == "ensemble" else [name]):
        training_curves(member, out)
    if (ROOT / "reports/results/comparison.json").exists():
        comparison(out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
