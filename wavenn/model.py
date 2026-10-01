"""The full Wave Neural Network (docs/02 §7).

    tokens ─► Embed ─► [WNNBlock ×L] ─► final norm ─► cleanup readout ─► logits

The readout cleans the final hidden state against the tied vocabulary codebook
``V = Eᵀ`` and returns next-token logits ``β Vᵀ r`` (docs/01 eq 4.2). This module is the
forward-pass contract of docs/02; the backprop-free learning rule (docs/03) attaches in
wavenn/ripple.py, and an autograd BPTT twin of the *same* forward lives in
wavenn/rlm.py::bptt_loss (trained by wavenn/train.py::train_bptt).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from .block import RMSNorm, WNNBlock


@dataclass
class WNNConfig:
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    D: int | None = None  # phasor dim (defaults to d_model)
    d_v: int | None = None  # value dim  (defaults to d_model)
    P: int | None = None  # knowledge patterns per block (defaults to 2*d_model)
    beta: float = 4.0  # read inverse temperature (docs/01 §6; ≠ β_n)
    n_max: int = 8  # cleanup settle cap
    rope_base: float = 10000.0
    delta_rule_heads: bool = True
    tie_readout: bool = True  # V = Eᵀ

    def param_estimate(self) -> int:
        d, L, K = self.d_model, self.n_layers, self.vocab_size
        P = self.P or 2 * d
        per_block = 4 * d * d + 2 * d * P  # W_{q,k,v,o} + M_{key,val}
        return per_block * L + d * K  # + tied embedding


class WNN(nn.Module):
    def __init__(self, cfg: WNNConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                WNNBlock(
                    cfg.d_model,
                    H=cfg.n_heads,
                    D=cfg.D,
                    d_v=cfg.d_v,
                    P=cfg.P,
                    beta=cfg.beta,
                    n_max=cfg.n_max,
                    rope_base=cfg.rope_base,
                    delta_rule_heads=cfg.delta_rule_heads,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = RMSNorm(cfg.d_model)
        if cfg.tie_readout:
            self.V = None  # use embed.weight.T
        else:
            self.V = nn.Parameter(torch.randn(cfg.d_model, cfg.vocab_size) / cfg.d_model**0.5)
        # small init so the initial read temperature β·V is well-scaled
        nn.init.normal_(self.embed.weight, std=1.0 / cfg.d_model**0.5)

    def readout_matrix(self) -> Tensor:
        """The vocabulary codebook V:(d_model, K)."""
        return self.embed.weight.t() if self.cfg.tie_readout else self.V

    def forward(self, tokens: Tensor, positions: Tensor | None = None) -> Tensor:
        """tokens:(B,T) int → logits:(B,T,K). Causal by construction (recurrent scan)."""
        B, T = tokens.shape
        assert tokens.dtype in (torch.long, torch.int), "tokens must be integer ids"
        x = self.embed(tokens)  # (B,T,d_model)
        for blk in self.blocks:
            x = blk(x, positions)
        x = self.final_norm(x)
        logits = self.cfg.beta * (x @ self.readout_matrix())  # (B,T,K)
        return logits
