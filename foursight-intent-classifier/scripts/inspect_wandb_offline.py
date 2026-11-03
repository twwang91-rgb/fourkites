#!/usr/bin/env python
"""Read offline W&B run files (*.wandb, LevelDB-log framed protobuf records) and print what
was logged: run name, config keys, number of history rows, last history row and summary.
Lets a reviewer verify the offline runs without syncing them to wandb.ai.

    python scripts/inspect_wandb_offline.py            # all runs under wandb/
    python scripts/inspect_wandb_offline.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import glob
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 32768
HEADER = 7


def iter_records(path: str):
    data = Path(path).read_bytes()
    pos = HEADER  # skip the ':W&B' + magic + version header (exactly one header-length)
    buf = b""
    while pos + HEADER <= len(data):
        block_rem = BLOCK - (pos % BLOCK)
        if block_rem < HEADER:  # zero padding at block end
            pos += block_rem
            continue
        _crc, length, rtype = struct.unpack("<IHB", data[pos : pos + HEADER])
        pos += HEADER
        chunk = data[pos : pos + length]
        pos += length
        if rtype == 0:  # padding
            continue
        if rtype == 1:  # FULL
            yield chunk
        elif rtype == 2:  # FIRST
            buf = chunk
        elif rtype == 3:  # MIDDLE
            buf += chunk
        elif rtype == 4:  # LAST
            buf += chunk
            yield buf
            buf = b""


def summarise(path: str) -> dict:
    from wandb.proto import wandb_internal_pb2 as pb

    out = {"file": path, "history_rows": 0, "config_keys": [], "summary": {}, "run_name": None, "project": None}
    last_hist = {}
    for raw in iter_records(path):
        rec = pb.Record()
        try:
            rec.ParseFromString(raw)
        except Exception:
            continue
        kind = rec.WhichOneof("record_type")
        if kind == "run":
            out["run_name"] = rec.run.display_name
            out["project"] = rec.run.project
            out["config_keys"] = sorted({it.key for it in rec.run.config.update})
        elif kind == "config":
            out["config_keys"] = sorted(set(out["config_keys"]) | {it.key for it in rec.config.update})
        elif kind == "history":
            out["history_rows"] += 1
            last_hist = {}
            for it in rec.history.item:
                key = it.key or ".".join(it.nested_key)
                if key and not key.startswith("_"):
                    try:
                        last_hist[key] = json.loads(it.value_json)
                    except Exception:
                        last_hist[key] = it.value_json
        elif kind == "summary":
            for it in rec.summary.update:
                if not it.key.startswith("_"):
                    try:
                        out["summary"][it.key] = json.loads(it.value_json)
                    except Exception:
                        out["summary"][it.key] = it.value_json
    out["last_history"] = last_hist
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args()
    files = a.paths or sorted(glob.glob(str(ROOT / "wandb" / "offline-run-*" / "run-*.wandb")))
    results = [summarise(f) for f in files]
    if a.json:
        print(json.dumps(results, indent=1, default=str))
        return
    for r in results:
        hist_keys = sorted(r["last_history"])
        print(f"{Path(r['file']).parent.name}  run={r['run_name']}  project={r['project']}  history_rows={r['history_rows']}")
        print(f"   config keys ({len(r['config_keys'])}): {', '.join(r['config_keys'][:12])}{' ...' if len(r['config_keys']) > 12 else ''}")
        print(f"   history keys: {', '.join(hist_keys[:10])}{' ...' if len(hist_keys) > 10 else ''}")
        if r["summary"]:
            keep = {k: v for k, v in r["summary"].items() if k.startswith("best_") or k in ("final_train_loss", "epochs")}
            print(f"   summary: {json.dumps(keep)[:300]}")


if __name__ == "__main__":
    main()
