#!/usr/bin/env python
"""Ad-hoc prediction CLI.

  python scripts/predict.py "where is my load 8006907462"
  python scripts/predict.py --file questions.txt
  python scripts/predict.py --conversation        # interactive multi-turn demo with the contextual resolver
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from foursight_intent.conversation import ContextualIntentResolver, ConversationState
from foursight_intent.inference import load_classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--model-dir", default=str(ROOT / "models/final"))
    ap.add_argument("--conversation", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    clf = load_classifier(a.model_dir, device="cpu")

    def single_turn(text: str) -> np.ndarray:
        return clf.probabilities([text])[0]

    if a.conversation:
        resolver = ContextualIntentResolver(single_turn, review_threshold=clf.review_threshold)
        state = ConversationState(session_id="cli")
        print("multi-turn demo (empty line to quit)")
        while True:
            try:
                text = input("you> ").strip()
            except EOFError:
                break
            if not text:
                break
            pred = resolver.resolve(text, state)
            print(f"  -> {pred.intent:22s} conf={pred.confidence:.2f}  single_turn={pred.single_turn_intent}({pred.single_turn_confidence:.2f})  "
                  f"context={'yes' if pred.used_context else 'no'}  review={'YES' if pred.needs_review else 'no'}  ids={pred.resolved_reference_ids}  time={pred.resolved_time_expression}")
            print(f"     reason: {pred.reason}")
        return

    texts = list(a.texts)
    if a.file:
        texts += [l.rstrip("\n") for l in open(a.file, encoding="utf-8") if l.strip()]
    if not texts:
        ap.error("give one or more texts, --file or --conversation")
    probs = clf.probabilities(texts)
    for text, p in zip(texts, probs):
        pred = clf._to_prediction(p)
        if a.json:
            print(json.dumps({"text": text, **pred.to_dict()}, ensure_ascii=False))
        else:
            print(f"{pred.intent:22s} conf={pred.confidence:.3f} margin={pred.margin:.3f} review={'YES' if pred.needs_review else 'no '}  | {text[:90]}")


if __name__ == "__main__":
    main()
