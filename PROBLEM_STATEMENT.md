# FourSight Intent Classification Challenge

## Background

FourKites' assistant, **FourSight**, is a chat interface that sits in front of several
different products. When a user types a question, the very first decision the system has to
make is *where does this question belong* — because the answer lives in a different place
depending on what is being asked:

1. **shipment_information** — Shipment Visibility. Answers *"where is my shipment?"* — the
   current location and visibility state of a load.
2. **realtime_query** — shipment questions about **ETAs, statuses, and reference numbers** for
   specific loads (live lookups against the operational store).
3. **analytics** — **aggregated** answers produced by churning and reasoning over historic
   data ("how many late deliveries last quarter", counts, trends, breakdowns).
4. **disruptions** — questions specifically about **disruptions** affecting shipments or the
   supply chain in general (weather, ports, borders, capacity).
5. **chitchat** — generic conversation with the bot ("hi", "how are you", "what can you do").
6. **knowledge_base** — questions to **understand a product or a feature** ("how do I add a
   user", "how often does FourKites update").
7. **other** — anything out of FourKites' scope, or that fits none of the classes above.

**Why this separation matters.** The data behind these answers is spread across multiple
databases and products. There is no single backend that can serve all of them, so there is no
unified solution — the assistant *must* route each question to the right product first. A wrong
route means the right system is never even queried. That makes this first-hop intent
classification the foundation the whole assistant stands on.

Your task is to build a system that reads a user's question and returns its intent — one of the
seven classes above — **deterministically** (the same input always produces the same label).

## The task

Build a classifier that maps a **user question** to exactly **one** of the seven intents.

- This is **not** a "call an LLM and hope" exercise. A fine-tuned encoder (BERT / DistilBERT /
  RoBERTa / DeBERTa or similar), a smaller sequence classifier, or any approach that *fits the
  use case* is encouraged. If you do use a generative model, it must produce a deterministic,
  constrained label — not free text.
- Inference must be **deterministic**: same question in, same label out, every time.
- The seven labels are a **closed set**. Every prediction must be one of them.

### This is a chat system — intent depends on conversational history

In production FourSight is a multi-turn chat, and the intent of the *current* question often
depends on **what was said earlier in the conversation** (e.g. "what about last month?" is
`analytics` only because of the turn before it). **The dataset shared with you is single-turn**
— each row is one question with its label, with no conversation context attached.

Your report **must** address how your approach would incorporate conversational history, and
**you will be evaluated on this**: the design of your solution for the multi-turn setting, not
only its accuracy on isolated questions. A model that is accurate on single turns *and* has a
credible, concrete story for using history is worth far more to us than one that ignores it.

## The data

```
data/
  train.csv    1,904 labelled questions   (id, question, intent)
  test.csv       566 questions            (id, question)  — no labels; we score these
```

- Each row is one real user question (anonymised — all personal and company identifiers have
  been removed).
- `train.csv` has the gold `intent`. `test.csv` does **not** — you predict it and submit.
- There is **no separate dev split**. Hold out your own validation set from `train.csv` (a
  stratified slice, since the classes are imbalanced) and use `score.py` on it to self-check.
- The classes are **imbalanced** (see `labels.json` for the training distribution). Rare classes
  — `chitchat`, `disruptions`, `other` — matter as much as the common ones in scoring.
- Some questions are genuinely ambiguous between neighbouring classes (notably
  `shipment_information` vs `realtime_query` vs `analytics`). That boundary is deliberately hard
  and is where good solutions separate from memorised ones.

See `labels.json` for the class definitions and distribution, and `schema.json` for the exact
submission format.

## Rules

- Predict exactly **one** of the seven labels for every test row. Every row counts.
- A prediction outside the seven-label set, or a missing row, scores as **wrong**.
- Don't label the test rows by hand.
- Don't try to de-anonymise the data or reconstruct the original users.

## Scoring

```
python3 score.py --gold data/test.csv --pred your_predictions.csv
```

`score.py` is the exact scorer we run on the hidden test labels. Because the classes are
imbalanced, the headline number weights the macro average heavily so rare classes count:

```
OVERALL = 0.6 * macro-F1   (every class weighted equally; rewards rare-class handling)
        + 0.4 * accuracy    (overall correctness)
```

We also report **per-class precision / recall / F1**, the **confusion matrix**, and the most
confused class pairs — we look closely at the `shipment_information` / `realtime_query` /
`analytics` boundary and at the rare classes. A classifier that is 90% right and *knows which
questions it is unsure about* (a calibrated "needs review" / abstention signal) is worth more to
us than one that is 93% right and can't tell.

## What to submit

1. **`test_predictions.csv`** — your predictions for all 566 test rows, in the format of
   `sample_submission.csv` (`id,intent`).

2. **`report.pdf`** (about 2 pages):
   - your approach and why you chose it (and why it's deterministic);
   - your validation results: overall, **per-class**, and the confusion matrix, and where it
     fails;
   - **how you would incorporate conversational history** in the multi-turn chat setting
     (required — see above);
   - how you would handle a brand-new intent appearing in production;
   - *(optional)* a per-prediction confidence or "needs human review" flag, with its precision
     and recall on your validation set;
   - the **GitHub, Hugging Face and Weights & Biases** links (below) so we can reproduce it.

3. **A GitHub repository link** with all your code and instructions to reproduce the predictions
   from `test.csv`: seeded and documented, so we can re-run it end to end.

4. **A Hugging Face Hub link** to any model you train or fine-tune, with a short model card
   (training data, intended use, metrics, limitations) so we can load and run it. If your
   approach trains nothing, state that; this item is then not required.

5. **A Weights & Biases run link** if you train or fine-tune anything, logging the relevant
   metrics: training and validation loss curves, dev `OVERALL` and its per-class breakdown, and
   the run config / hyper-parameters. If nothing is trained, this is not required.

Make the GitHub repo, Hugging Face model and W&B project **public**, or share access so we can
open them, and include all links in `report.pdf`.

## Constraints

- Any open-weight model, library or technique is allowed — fine-tuning, classical ML on
  embeddings, encoder models, or distilled small LMs.
- Inference must run on a **single GPU with 24 GB of memory or less, or on CPU**. **No paid
  hosted APIs at inference time.**
- You may use free GPUs (Google Colab / Kaggle) for training and evaluation.
- Suggested time: about **6–8 hours**. We value a clear, honest analysis over a leaderboard
  number.

## What we look for

- correctness on the hard boundaries, not just the easy, memorisable questions;
- honest handling of uncertainty (calibration, abstention, "needs review");
- a concrete, credible design for the multi-turn / conversational-history setting;
- clean, reproducible, deterministic code;
- clear reasoning about failure modes and trade-offs.
