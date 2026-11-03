#!/usr/bin/env python
"""Prepare (and, only when explicitly authorised, upload) the Hugging Face model repo and
sync offline W&B runs.

By default this script is DRY-RUN: it validates the artefact directory, prints the exact
commands that would publish it, and exits. Nothing leaves the machine unless you pass
--push together with the required credentials (HF_TOKEN / WANDB_API_KEY).

  python scripts/publish_artifacts.py                    # dry run, prints commands
  python scripts/publish_artifacts.py --push --repo-id <user>/foursight-intent-distilbert

The challenge data is confidential: the model card must not embed training rows. The
model weights themselves are derived artefacts; publishing them is the candidate's call
once FourKites confirms it is permitted.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["config.json", "model.safetensors", "tokenizer_config.json", "label_map.json", "inference_config.json", "README.md"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models/final"))
    ap.add_argument("--repo-id", default="<hf-username>/foursight-intent-classifier")
    ap.add_argument("--push", action="store_true", help="actually upload (requires HF_TOKEN) and sync W&B (requires WANDB_API_KEY)")
    a = ap.parse_args()
    md = Path(a.model_dir)
    if (md / "ensemble.json").exists():
        import json

        spec = json.load(open(md / "ensemble.json"))
        missing = [f for f in ("label_map.json", "inference_config.json", "README.md") if not (md / f).exists()]
        for m in spec["members"]:
            missing += [f"{m['dir']}/{f}" for f in ("config.json", "model.safetensors", "tokenizer_config.json") if not (md / m["dir"] / f).exists()]
        print(f"ensemble artefact with {len(spec['members'])} members")
    else:
        missing = [f for f in REQUIRED if not (md / f).exists()]
    if missing:
        sys.exit(f"artefact incomplete, missing: {missing}")
    size_mb = sum(p.stat().st_size for p in md.rglob("*") if p.is_file()) / 1e6
    print(f"artefact OK: {md}  ({size_mb:.0f} MB)")
    quoted = [str(q.relative_to(ROOT)) for pat in ("reports/eda.json", "reports/eda.md", "reports/error_analysis_*.md", "reports/results/analysis_*.json", "reports/results/oof_predictions_*.csv", "reports/validation/*.csv") for q in ROOT.glob(pat)]
    if quoted:
        print("NOTE: these report files quote anonymised training questions verbatim (confidential dataset derivatives);")
        print("      keep them out of any *public* repository unless FourKites permits:")
        for q in quoted[:12]:
            print("        ", q)
        if len(quoted) > 12:
            print(f"         ... and {len(quoted) - 12} more")
    wandb_runs = sorted((ROOT / "wandb").glob("offline-run-*")) if (ROOT / "wandb").exists() else []
    print(f"offline W&B runs: {len(wandb_runs)}")

    print("\n# Hugging Face (public model repo, no data):")
    print(f"huggingface-cli login            # or export HF_TOKEN=...")
    print(f"huggingface-cli upload {a.repo_id} {md} . --repo-type model --commit-message 'FourSight intent classifier'")
    print("\n# Weights & Biases (sync offline runs):")
    print("wandb login                      # or export WANDB_API_KEY=...")
    print("wandb sync wandb/offline-run-*   # prints the run URLs")

    if not a.push:
        print("\nDRY RUN - nothing uploaded. Re-run with --push once publication is authorised.")
        return
    if not os.environ.get("HF_TOKEN"):
        sys.exit("--push requested but HF_TOKEN is not set")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.repo_id, repo_type="model", exist_ok=True)
    url = api.upload_folder(repo_id=a.repo_id, folder_path=str(md), repo_type="model", commit_message="FourSight intent classifier")
    print("uploaded:", url)
    if os.environ.get("WANDB_API_KEY") and wandb_runs:
        subprocess.run(["wandb", "sync", *map(str, wandb_runs)], check=False)


if __name__ == "__main__":
    main()
