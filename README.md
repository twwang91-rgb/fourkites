# FourSight Intent Classification Challenge — candidate package

| File | What it is |
|---|---|
| `intent_classification_challenge.pdf` | the task, rules, scoring and deliverables. **Start here.** |
| `PROBLEM_STATEMENT.md` | the same problem statement in Markdown |
| `data/train.csv` | 1,904 labelled questions — columns `id, question, intent` |
| `data/test.csv` | 566 questions — columns `id, question` (no labels; you predict these) |
| `labels.json` | the seven class definitions and the training distribution |
| `schema.json` | the submission format |
| `sample_submission.csv` | a valid submission covering all 566 test ids (default label `other`) |
| `score.py` | the official scorer (Python 3, standard library only) |

## The classes (closed set of 7)

`shipment_information` · `realtime_query` · `analytics` · `disruptions` · `chitchat` ·
`knowledge_base` · `other` — see `labels.json` for definitions.

## Scoring yourself

There is no separate dev split. Hold out a stratified slice of `data/train.csv` as your own
validation set, predict it, and run:

```
python3 score.py --gold my_val.csv --pred my_val_predictions.csv
```

`my_val.csv` needs columns `id,question,intent`; your prediction file needs `id,intent` (see
`sample_submission.csv`).

```
OVERALL = 0.6 * macro-F1 + 0.4 * accuracy
```

## Note: this is a chat system

The data here is **single-turn**, but in production the intent of a question depends on the
**conversational history** before it. Your report must explain how your approach would use that
history — you are evaluated on this. See the problem statement.

## Submit

`test_predictions.csv` (all 566 test rows), `report.pdf`, and public **GitHub**, **Hugging Face**
and **Weights & Biases** links. Details in the problem statement.

All questions are anonymised. Do not attempt to de-anonymise the data or label test rows by hand.
