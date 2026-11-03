"""Supervised fine-tuning of a compact encoder for 7-way intent classification.

A small explicit training loop is used instead of ``transformers.Trainer`` so that every
source of randomness is visible: seeded shuffling via a torch Generator, no workers,
fp32 everywhere, deterministic algorithms where the backend supports them, and the
best checkpoint selected on the *official* OVERALL metric of the validation fold.

Public API
    TrainConfig            all hyper-parameters (mirrors configs/default.yaml)
    IntentDataset/collate  head+tail truncation + dynamic padding
    train_one(...)         fit on a train split, early-stop on a validation split
    fit_full(...)          fit on all labelled data for a fixed number of epochs
    save_model(...)        write weights + tokenizer + label map + inference config
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from foursight_intent.data import class_weights
from foursight_intent.labels import ID2LABEL, LABEL2ID, LABELS
from foursight_intent.metrics import official_scores
from foursight_intent.tokenization import encode_texts, pad_batch, special_token_template
from foursight_intent.utils import get_logger, resolve_device, save_json, seed_everything

log = get_logger("train")


@dataclass
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    tail_tokens: int = 16
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_epochs: int = 8
    patience: int = 2
    grad_clip: float = 1.0
    class_weight: str = "sqrt_inv"  # none | sqrt_inv | inv
    label_smoothing: float = 0.0
    seed: int = 42
    device: str = "auto"
    num_threads: int | None = None
    # fixed-epoch training for the final artefact (set from CV best epochs)
    fixed_epochs: int | None = None

    @classmethod
    def from_yaml(cls, cfg: dict, **overrides: Any) -> "TrainConfig":
        t = cfg.get("transformer", {})
        kw = dict(
            model_name=t.get("model_name", cls.model_name),
            max_length=cfg.get("text", {}).get("max_length", cls.max_length),
            tail_tokens=cfg.get("text", {}).get("tail_tokens", cls.tail_tokens),
            batch_size=t.get("batch_size", cls.batch_size),
            eval_batch_size=t.get("eval_batch_size", cls.eval_batch_size),
            learning_rate=float(t.get("learning_rate", cls.learning_rate)),
            weight_decay=float(t.get("weight_decay", cls.weight_decay)),
            warmup_ratio=float(t.get("warmup_ratio", cls.warmup_ratio)),
            max_epochs=int(t.get("max_epochs", cls.max_epochs)),
            patience=int(t.get("patience", cls.patience)),
            grad_clip=float(t.get("grad_clip", cls.grad_clip)),
            class_weight=t.get("class_weight", cls.class_weight) or "none",
            label_smoothing=float(t.get("label_smoothing", cls.label_smoothing)),
            seed=int(cfg.get("seed", cls.seed)),
            device=t.get("device", cls.device),
        )
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)


class IntentDataset(Dataset):
    """Pre-tokenised dataset. Tokenisation happens once, deterministically, up front."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int] | None, tokenizer, max_length: int, tail_tokens: int):
        self.labels = None if labels is None else [int(x) for x in labels]
        self.input_ids: list[list[int]] = encode_texts(tokenizer, texts, max_length, tail_tokens, special_token_template(tokenizer))
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int):
        item = {"input_ids": self.input_ids[i]}
        if self.labels is not None:
            item["label"] = self.labels[i]
        return item

    def collate(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        ids, mask = pad_batch([b["input_ids"] for b in batch], self.pad_id)
        out = {"input_ids": ids, "attention_mask": mask}
        if "label" in batch[0]:
            out["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return out


def load_tokenizer_and_model(model_name: str, device: str):
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    transformers.logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label={i: l for i, l in ID2LABEL.items()},
        label2id=dict(LABEL2ID),
        dtype=torch.float32,  # some checkpoints (DeBERTa-v3) default to fp16; we train and serve in fp32
    )
    model = model.float().to(device)
    return tok, model


def _param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (n.endswith("bias") or "LayerNorm" in n or "layer_norm" in n) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def predict_logits(model: nn.Module, ds: IntentDataset, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=ds.collate, num_workers=0)
    outs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outs.append(model(**batch).logits.float().cpu())
    return torch.cat(outs).numpy() if outs else np.zeros((0, len(LABELS)))


def _eval_split(model, ds: IntentDataset, gold: Sequence[str], cfg: TrainConfig, device: str, loss_fn) -> tuple[dict, np.ndarray, float]:
    logits = predict_logits(model, ds, cfg.eval_batch_size, device)
    pred = [ID2LABEL[int(i)] for i in logits.argmax(1)]
    scores = official_scores(list(gold), pred)
    with torch.no_grad():
        val_loss = float(loss_fn(torch.tensor(logits), torch.tensor(ds.labels)).item()) if ds.labels is not None else math.nan
    return scores, logits, val_loss


@dataclass
class TrainResult:
    best_epoch: int
    best_scores: dict
    val_logits: np.ndarray
    history: list[dict] = field(default_factory=list)
    train_seconds: float = 0.0
    config: dict = field(default_factory=dict)


def train_one(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    cfg: TrainConfig,
    wandb_run=None,
    log_prefix: str = "",
    return_model: bool = False,
):
    """Fine-tune on ``train_df``; if ``val_df`` is given, early-stop on its OVERALL score.

    Returns ``TrainResult`` (and the model + tokenizer when ``return_model``)."""
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    tok, model = load_tokenizer_and_model(cfg.model_name, device)

    train_ds = IntentDataset(train_df["text"].tolist(), train_df["label"].tolist(), tok, cfg.max_length, cfg.tail_tokens)
    val_ds = None
    if val_df is not None:
        val_ds = IntentDataset(val_df["text"].tolist(), val_df["label"].tolist(), tok, cfg.max_length, cfg.tail_tokens)

    w = torch.tensor(class_weights(train_df["label"].to_numpy(), cfg.class_weight), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=cfg.label_smoothing)
    cpu_loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)

    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=train_ds.collate, generator=gen, num_workers=0)

    n_epochs = cfg.fixed_epochs or cfg.max_epochs
    total_steps = n_epochs * len(loader)
    optim = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    from transformers import get_linear_schedule_with_warmup

    sched = get_linear_schedule_with_warmup(optim, int(cfg.warmup_ratio * total_steps), total_steps)

    best = {"score": -1.0, "loss": math.inf, "epoch": -1, "state": None, "scores": None, "logits": None}
    history: list[dict] = []
    bad_epochs = 0
    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()
        running, steps = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()
            running += loss.item()
            steps += 1
        rec = {"epoch": epoch, "train_loss": running / max(steps, 1), "lr": sched.get_last_lr()[0], "elapsed_s": time.perf_counter() - t0}
        if val_ds is not None:
            scores, logits, val_loss = _eval_split(model, val_ds, val_df["intent"].tolist(), cfg, device, cpu_loss_fn)
            rec.update(
                val_loss=val_loss, val_accuracy=scores["accuracy"], val_macro_f1=scores["macro_f1"], val_OVERALL=scores["OVERALL"],
                **{f"val_f1/{c}": scores["per_class"][c]["f1"] for c in LABELS},
            )
            improved = scores["OVERALL"] > best["score"] + 1e-9 or (abs(scores["OVERALL"] - best["score"]) <= 1e-9 and val_loss < best["loss"])
            if improved:
                best.update(score=scores["OVERALL"], loss=val_loss, epoch=epoch, scores=scores, logits=logits,
                            state={k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
                bad_epochs = 0
            else:
                bad_epochs += 1
            log.info("%sepoch %d  train_loss=%.4f  val_loss=%.4f  acc=%.4f  macroF1=%.4f  OVERALL=%.4f%s",
                     log_prefix, epoch, rec["train_loss"], val_loss, scores["accuracy"], scores["macro_f1"], scores["OVERALL"], "  *" if improved else "")
        else:
            log.info("%sepoch %d  train_loss=%.4f", log_prefix, epoch, rec["train_loss"])
        history.append(rec)
        if wandb_run is not None:
            wandb_run.log(rec, step=epoch)
        if val_ds is not None and cfg.fixed_epochs is None and bad_epochs >= cfg.patience:
            log.info("%searly stopping at epoch %d (best epoch %d)", log_prefix, epoch, best["epoch"])
            break
    train_seconds = time.perf_counter() - t0

    if val_ds is not None and best["state"] is not None:
        model.load_state_dict(best["state"])
        result = TrainResult(best["epoch"], best["scores"], best["logits"], history, train_seconds, asdict(cfg))
    else:
        result = TrainResult(n_epochs, {}, np.zeros((0, len(LABELS))), history, train_seconds, asdict(cfg))
    if wandb_run is not None and result.best_scores:
        wandb_run.summary.update({"best_epoch": result.best_epoch, "best_val_OVERALL": result.best_scores["OVERALL"],
                                  "best_val_macro_f1": result.best_scores["macro_f1"], "best_val_accuracy": result.best_scores["accuracy"],
                                  **{f"best_val_f1/{c}": result.best_scores["per_class"][c]["f1"] for c in LABELS}})
    if return_model:
        return result, model, tok
    del model
    return result


def fit_full(df: pd.DataFrame, cfg: TrainConfig, epochs: int, wandb_run=None):
    """Train on every labelled row for a fixed number of epochs (chosen from CV)."""
    cfg = copy.replace(cfg, fixed_epochs=epochs) if hasattr(copy, "replace") else TrainConfig(**{**asdict(cfg), "fixed_epochs": epochs})
    result, model, tok = train_one(df, None, cfg, wandb_run=wandb_run, log_prefix="[full] ", return_model=True)
    return result, model, tok


def save_model(model, tok, out_dir: str | Path, inference_config: dict, extra_files: dict[str, Any] | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu")
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    save_json({"labels": LABELS, "label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}, out_dir / "label_map.json")
    save_json(inference_config, out_dir / "inference_config.json")
    for name, obj in (extra_files or {}).items():
        save_json(obj, out_dir / name)
    return out_dir
