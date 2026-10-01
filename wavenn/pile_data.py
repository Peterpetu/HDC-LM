"""S6 Pile data pipeline — reference-matched loader over the NCALM token cache.

Implements docs/10-s6-protocol.md (plan §M.5, §1): the S6 comparison consumes the
IDENTICAL tokenized data as the reference GPT-2/NeoX-162M run — the int32 memmap
``pile_neox50k_tokens_250m.bin`` (250,000,000 NeoX-BPE tokens of
monology/pile-uncopyrighted, docs joined by EOS id 0) — with the reference's exact
positional 99.5/0.5 train/val split and window semantics, plus the S6 FROZEN-EVAL
deterministic evaluation set. The reference cache is opened READ-ONLY and never
modified (plan §1).

Reference semantics ported from ``C:/koodi/NCALM/ncalm/data/dataset.py``:
  * ``MemmapTokenStore``: np.memmap(dtype=int32, mode="r"); length = min(meta, max).
  * Split (dataset.py:444-447): train = [0, int(0.995*total)), val = [int(0.995*total), total).
  * Train sampling (dataset.py:458-469): uniform window starts
    ``torch.randint(0, n - seq_len - 1, (B,))`` over the split, windows of ``seq_len``
    tokens. DOCUMENTED DEVIATION: the reference drew from the *global* torch RNG; we
    draw from a dedicated seeded ``torch.Generator`` so runs are reproducible. The
    sampling distribution is identical (same uniform range).
  * Supervision (gpt2/model.py:297-301): a (B, T) window is trained with
    CE(logits[:, :-1], window[:, 1:]) — T-1 supervised positions, no cross-window
    target. We express the same set as (x = window, y = shift-left with
    ignore_index=-100 at the last position), matching the WNN trainer interface.

FROZEN-EVAL (plan §M.5): the entire val region as deterministic, non-overlapping
seq_len-token windows in fixed order — zero sampling noise (the reference's own eval
drew random val batches from the unseeded global RNG; see docs/10 §anchor-caveat).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

# Reference data location (plan §1; read-only). Overridable for tests/CL-domain caches.
NCALM_ROOT = os.environ.get("NCALM_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "NCALM"))  # a checkout of the NCALM repository (its data cache supplies the Pile tokens)
PILE_250M_BIN = os.path.join(NCALM_ROOT, "data_cache", "pile_neox50k_tokens_250m.bin")

IGNORE_INDEX = -100
TRAIN_FRACTION = 0.995  # reference dataset.py:445 — positional split


def _meta_path(bin_path: str) -> str:
    return bin_path + ".meta.json"


class PileMemmap:
    """Read-only int32 memmap over a reference-format token cache.

    Mirrors ``MemmapTokenStore`` (dataset.py:91-109): dtype int32, mode="r",
    length = min(meta["length"], max_tokens). ``mode="r"`` guarantees the reference
    file cannot be modified through this object (plan §1 read-only requirement).
    """

    def __init__(self, bin_path: str = PILE_250M_BIN, max_tokens: int | None = None):
        with open(_meta_path(bin_path), "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta.get("dtype", "int32") == "int32", f"unexpected cache dtype: {meta}"
        length = int(meta["length"])
        if max_tokens is not None:
            length = min(length, int(max_tokens))
        self.length = length
        self.arr = np.memmap(bin_path, dtype=np.int32, mode="r", shape=(self.length,))
        self.bin_path = bin_path

    def __len__(self) -> int:
        return self.length

    def windows(self, starts: np.ndarray, seq_len: int) -> np.ndarray:
        """Gather ``len(starts)`` windows of ``seq_len`` tokens as int64 (reference
        batch_windows, dataset.py:105-109)."""
        out = np.empty((len(starts), seq_len), dtype=np.int64)
        for i, s in enumerate(starts.tolist()):
            out[i] = self.arr[s : s + seq_len]
        return out


def split_bounds(total: int) -> tuple[int, int, int]:
    """(train_end, val_start, val_end) under the reference positional split.

    Reference (dataset.py:444-447): train = [0, int(total*0.995)), val = rest.
    For total = 250,000,000 this is exactly 248,750,000 (0.995*250e6 is integral).
    """
    train_end = int(total * TRAIN_FRACTION)
    return train_end, train_end, total


def window_to_xy(windows: Tensor) -> tuple[Tensor, Tensor]:
    """(B, T) token windows -> (x, y) with the reference supervision set.

    x = the full window (the model consumes all T positions, as the reference forward
    does); y[t] = window[t+1] for t < T-1 and IGNORE_INDEX at t = T-1 — exactly the
    reference's CE(logits[:, :-1], window[:, 1:]) supervision (T-1 targets/window,
    no cross-window target). Shapes: both (B, T), dtype long.
    """
    assert windows.ndim == 2 and windows.dtype == torch.long
    y = torch.full_like(windows, IGNORE_INDEX)
    y[:, :-1] = windows[:, 1:]
    return windows, y


@dataclass
class PileTrainBatcher:
    """Reference-matched infinite train batcher over the memmap's train split.

    Sampling matches dataset.py:458-469: uniform starts in [0, n - seq_len - 1)
    over the train region (n = train-region length), windows of ``seq_len`` tokens —
    drawn from a dedicated ``torch.Generator(seed)`` (documented deviation: the
    reference used the unseeded global RNG; the distribution is identical).
    """

    store: PileMemmap
    B: int
    seq_len: int
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        train_end, _, _ = split_bounds(len(self.store))
        self.n = train_end  # train-region length; region starts at global offset 0
        if self.n <= self.seq_len + 1:
            raise ValueError(f"train split too small: {self.n} tokens for seq_len={self.seq_len}")
        self.gen = torch.Generator().manual_seed(self.seed)
        self.batches_served = 0  # data cursor for checkpoint/resume (plan A.4)

    def batch(self) -> tuple[Tensor, Tensor]:
        # Reference bound: randint(0, n - seq_len - 1) — high bound exclusive.
        starts = torch.randint(0, self.n - self.seq_len - 1, (self.B,), generator=self.gen)
        w = self.store.windows(starts.numpy(), self.seq_len)
        windows = torch.from_numpy(w).to(dtype=torch.long, device=self.device)
        self.batches_served += 1
        return window_to_xy(windows)

    # -- checkpoint/resume support (plan A.4): the generator state IS the cursor --
    def state_dict(self) -> dict:
        return {"gen_state": self.gen.get_state(), "batches_served": self.batches_served}

    def load_state_dict(self, state: dict) -> None:
        self.gen.set_state(state["gen_state"])
        self.batches_served = int(state["batches_served"])


class FrozenEval:
    """FROZEN-EVAL (plan §M.5): the entire val region as deterministic windows.

    val region V = ids[N0 : N), N0 = int(0.995*N). Windows k = 0..K-1 with
    K = (|V| - 1) // seq_len; x_k = V[seq_len*k : seq_len*(k+1)) — non-overlapping,
    fixed order, fully inside V; supervision = the within-window shift (T-1 targets).
    For the 250M cache: |V| = 1,250,000, K = 1,220 windows = 153 batches of 8 (last
    partial). Deterministic by construction: no RNG anywhere.
    """

    def __init__(self, store: PileMemmap, seq_len: int, batch_size: int = 8):
        _, val_start, val_end = split_bounds(len(store))
        self.store = store
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.val_start = val_start
        self.n_val = val_end - val_start
        self.K = (self.n_val - 1) // seq_len
        if self.K < 1:
            raise ValueError(f"val region too small: {self.n_val} tokens")
        self.n_batches = (self.K + batch_size - 1) // batch_size
        self.supervised_tokens = self.K * (seq_len - 1)

    def batches(self, device: str = "cuda"):
        """Yield (x, y) batches in fixed order covering all K windows exactly once."""
        for b in range(self.n_batches):
            k0 = b * self.batch_size
            k1 = min(k0 + self.batch_size, self.K)
            starts = self.val_start + np.arange(k0, k1, dtype=np.int64) * self.seq_len
            w = self.store.windows(starts, self.seq_len)
            windows = torch.from_numpy(w).to(dtype=torch.long, device=device)
            yield window_to_xy(windows)

    def val_bytes_per_token(self, tokenizer) -> float:
        """bytes-per-token of the covered val tokens (for bits/byte; plan §3.4).

        Detokenizes the K windows once with the reference tokenizer and measures
        UTF-8 bytes. Deterministic.
        """
        total_bytes = 0
        total_tokens = 0
        for b in range(self.n_batches):
            k0 = b * self.batch_size
            k1 = min(k0 + self.batch_size, self.K)
            starts = self.val_start + np.arange(k0, k1, dtype=np.int64) * self.seq_len
            w = self.store.windows(starts, self.seq_len)
            for row in w:
                text = tokenizer.decode(row.tolist())
                total_bytes += len(text.encode("utf-8"))
                total_tokens += len(row)
        return total_bytes / total_tokens
