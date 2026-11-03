"""Tokenizer-agnostic encoding with head+tail truncation.

transformers v5 fast tokenizers no longer expose ``build_inputs_with_special_tokens``,
so we derive the special-token template (prefix / suffix ids) empirically once per
tokenizer and apply it ourselves. Works for BERT-style ([CLS] x [SEP]), RoBERTa-style
(<s> x </s>) and DeBERTa-v3 tokenizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from foursight_intent.text import clean_text, head_tail_truncate


@dataclass(frozen=True)
class SpecialTokenTemplate:
    prefix: tuple[int, ...]
    suffix: tuple[int, ...]

    @property
    def n_special(self) -> int:
        return len(self.prefix) + len(self.suffix)

    def wrap(self, ids: Sequence[int]) -> list[int]:
        return [*self.prefix, *ids, *self.suffix]


def special_token_template(tokenizer) -> SpecialTokenTemplate:
    probe = "shipment"
    with_sp = list(tokenizer(probe, add_special_tokens=True)["input_ids"])
    without = list(tokenizer(probe, add_special_tokens=False)["input_ids"])
    n = len(without)
    for start in range(0, len(with_sp) - n + 1):
        if with_sp[start : start + n] == without:
            return SpecialTokenTemplate(tuple(with_sp[:start]), tuple(with_sp[start + n :]))
    # Fallback: assume single CLS / SEP style tokens if available.
    prefix = (tokenizer.cls_token_id,) if getattr(tokenizer, "cls_token_id", None) is not None else ()
    suffix = (tokenizer.sep_token_id,) if getattr(tokenizer, "sep_token_id", None) is not None else ()
    return SpecialTokenTemplate(prefix, suffix)


def encode_texts(tokenizer, texts: Sequence[str], max_length: int, tail_tokens: int, template: SpecialTokenTemplate | None = None) -> list[list[int]]:
    """Clean, tokenise and truncate (head + tail) each text, then add special tokens.

    Empty strings are mapped to the tokenizer's UNK token so every row produces a
    non-empty sequence (and therefore a prediction)."""
    template = template or special_token_template(tokenizer)
    budget = max(1, max_length - template.n_special)
    tail = min(tail_tokens, budget // 2)
    clean = [clean_text(t) for t in texts]
    unk = tokenizer.unk_token or "?"
    clean = [t if t else unk for t in clean]
    enc = tokenizer(clean, add_special_tokens=False, truncation=False, verbose=False)
    out = []
    for ids in enc["input_ids"]:
        ids = list(ids) or list(tokenizer(unk, add_special_tokens=False)["input_ids"])
        out.append(template.wrap(head_tail_truncate(ids, budget, tail=tail)))
    return out


def pad_batch(seqs: Sequence[Sequence[int]], pad_id: int):
    """Right-pad to the longest sequence; returns (input_ids, attention_mask) tensors."""
    import torch

    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(list(s), dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask
