"""WNN block — two associative memories on a bundling bus (docs/02 §3).

    x  ─► [Contextual Memory] ─(bundle)─► x' ─► [Knowledge Memory] ─(bundle)─► x''

The Contextual Memory is the attention role (docs/02 §4); the Knowledge Memory is the
FFN role, implemented as a finite-temperature modern-Hopfield cleanup against a learned
pattern dictionary (docs/02 §5, docs/01 §6). Both sublayers are pre-normalized and their
outputs are *bundled* (added) onto the shared residual bus (docs/02 §6, mutation M7).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from . import cleanup as C
from .memory import ContextualMemory


class RMSNorm(nn.Module):
    """Real RMS normalization for the (real) bundling bus (docs/02 §6)."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps) * self.weight


class KnowledgeMemory(nn.Module):
    """FFN-role Hopfield cleanup: settle the input against a learned codebook (docs/02 §5)."""

    def __init__(self, d_model: int, P: int, beta: float = 4.0, n_max: int = 8, eta: float = 0.3):
        super().__init__()
        self.d_model, self.P, self.beta = d_model, P, beta
        self.n_max, self.eta = n_max, eta
        # M_key, M_val: (d_model, P). Init small so σ_max(M_key) starts in the safe regime.
        scale = 1.0 / (d_model**0.5)
        self.M_key = nn.Parameter(torch.randn(d_model, P) * scale)
        self.M_val = nn.Parameter(torch.randn(d_model, P) * scale)

    def forward(self, h: Tensor) -> Tensor:
        B, T, d = h.shape
        flat = h.reshape(B * T, d)
        r_star, _, _ = C.settle(
            flat, self.M_key, self.M_val, self.beta, n_max=self.n_max, eta=self.eta
        )
        return r_star.reshape(B, T, d)


class WNNBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        H: int = 8,
        D: int | None = None,
        d_v: int | None = None,
        P: int | None = None,
        beta: float = 4.0,
        n_max: int = 8,
        rope_base: float = 10000.0,
        delta_rule_heads: bool = True,
    ):
        super().__init__()
        P = P or 2 * d_model
        self.norm1 = RMSNorm(d_model)
        self.contextual = ContextualMemory(
            d_model, D=D, d_v=d_v, H=H, rope_base=rope_base, delta_rule_heads=delta_rule_heads
        )
        self.norm2 = RMSNorm(d_model)
        self.knowledge = KnowledgeMemory(d_model, P, beta=beta, n_max=n_max)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        x = x + self.contextual(self.norm1(x), positions)  # attention role, bundled onto bus
        x = x + self.knowledge(self.norm2(x))  # FFN role, bundled onto bus
        return x
