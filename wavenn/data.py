"""Data pipeline for TinyStories (docs/02 §8, docs/05 §2).

Trains (or loads) a byte-level BPE tokenizer on the corpus, tokenizes it into a flat
token array, and serves random contiguous (B, T) batches. The RIPPLE trainer and its
BPTT twin draw from the *same* loader with the same seed so the comparison is a pure
learning-rule comparison (docs/05 §3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _corpus_path() -> str:
    p = os.path.join(_DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. Download it from HuggingFace roneneldan/TinyStories."
        )
    return p


def train_or_load_tokenizer(vocab_size: int = 8192):
    """Byte-level BPE tokenizer trained on the corpus; cached to data/ (docs/02 §8)."""
    from tokenizers import ByteLevelBPETokenizer

    tok_path = os.path.join(_DATA_DIR, f"tinystories_bpe_{vocab_size}.json")
    if os.path.exists(tok_path):
        from tokenizers import Tokenizer

        return Tokenizer.from_file(tok_path)
    tok = ByteLevelBPETokenizer()
    tok.train(
        [_corpus_path()],
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        min_frequency=2,
    )
    tok._tokenizer.save(tok_path)
    from tokenizers import Tokenizer

    return Tokenizer.from_file(tok_path)


def build_token_array(vocab_size: int = 8192, cache: bool = True) -> tuple[np.ndarray, object]:
    """Tokenize the whole corpus into a uint16 array (docs/05 §2). Returns (ids, tokenizer)."""
    cache_path = os.path.join(_DATA_DIR, f"tinystories_ids_{vocab_size}.npy")
    tok = train_or_load_tokenizer(vocab_size)
    if cache and os.path.exists(cache_path):
        return np.load(cache_path), tok
    with open(_corpus_path(), encoding="utf-8") as f:
        text = f.read()
    eot = tok.token_to_id("<|endoftext|>")
    # split on the story separator to insert <|endoftext|> between stories
    ids: list[int] = []
    for story in text.split("<|endoftext|>"):
        story = story.strip()
        if not story:
            continue
        ids.extend(tok.encode(story).ids)
        ids.append(eot)
    arr = np.asarray(ids, dtype=np.uint16)
    if cache:
        np.save(cache_path, arr)
    return arr, tok


@dataclass
class Batcher:
    """Serves random contiguous (B, T+1) windows; last column is the shifted target."""

    ids: np.ndarray
    B: int
    T: int
    device: str = "cuda"
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.data = torch.from_numpy(self.ids.astype(np.int64))

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self.data) - self.T - 1
        idx = self.rng.integers(0, n, size=self.B)
        x = torch.stack([self.data[i : i + self.T] for i in idx])
        y = torch.stack([self.data[i + 1 : i + 1 + self.T] for i in idx])
        return x.to(self.device), y.to(self.device)


def split_batchers(
    vocab_size: int, B: int, T: int, device: str, val_frac: float = 0.05, seed: int = 0
):
    """Train/val Batchers over disjoint spans of the corpus (docs/05 §7 reproducibility)."""
    ids, tok = build_token_array(vocab_size)
    n_val = int(len(ids) * val_frac)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    train = Batcher(train_ids, B, T, device, seed=seed)
    val = Batcher(val_ids, B, T, device, seed=seed + 1)
    return train, val, tok
