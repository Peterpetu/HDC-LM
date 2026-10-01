"""Contextual Memory — the attention replacement (docs/02 §4, docs/01 §5).

A multi-timescale holographic key–value memory. The state update is a *linear*
recurrence (docs/02 §1, commitment 5) — this is what makes the eligibility trace exact
(docs/01 §9); no nonlinearity may be inserted between S_{t-1} and S_t.

    write:  S_t = γ S_{t-1} + k_t (v_t - α v̂_t)^⊤     (bind + bundle; optional delta-rule)
            z_t = γ z_{t-1} + k_t
    read:   c_t = Re(q_t^† S_t) / (Re(q_t^† z_t) + ε)

Keys/queries are unit phasors carrying positional phase (ρ^t, docs/01 §11); values and
the read content c_t are real. This module provides a correctness-first reference scan
(a token loop) used by the tests and the learning rule; the chunked-scan fast path
(docs/04 §4) is added in the throughput stage and validated against this reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from . import algebra as A

_EPS = 1e-6


def multitimescale_gammas(H: int, gamma_min: float = 0.0, gamma_max: float = 0.999) -> Tensor:
    """Log-spaced decays covering distinct effective contexts M_eff=(1+γ)/(1-γ) (docs/02 §4.2).

    Head 0 is the shortest memory (γ≈γ_min, ~1 token), head H-1 the longest (γ≈γ_max).
    Returns the raw ``log_gamma`` such that ``γ_h = sigmoid(log_gamma_h)``.
    """
    if H == 1:
        gammas = torch.tensor([0.9])
    else:
        one_minus = torch.logspace(
            start=torch.log10(torch.tensor(1.0 - gamma_min + 1e-4)).item(),
            end=torch.log10(torch.tensor(1.0 - gamma_max)).item(),
            steps=H,
        )
        gammas = (1.0 - one_minus).clamp(1e-4, 1 - 1e-4)
    # invert sigmoid: log_gamma = logit(gamma)
    return torch.log(gammas) - torch.log1p(-gammas)


@dataclass
class ScanState:
    """Carry for the recurrent scan (current-only; O(1) in T, docs/02 §10)."""

    S: Tensor  # (B, H, d_k, d_vh) complex — associative memory
    z: Tensor  # (B, H, d_k)       complex — normalizer


def scan_reference(
    k: Tensor,
    q: Tensor,
    v: Tensor,
    gamma: Tensor,
    *,
    delta_alpha: Tensor | None = None,
    state: ScanState | None = None,
    return_state: bool = False,
):
    """Reference (token-loop) contextual-memory scan — the correctness ground truth.

    Args:
        k, q: complex ``(B, T, H, d_k)`` unit phasors (already positionally bound).
        v:    real    ``(B, T, H, d_vh)`` values.
        gamma: real ``(H,)`` per-head decays in [0, 1).
        delta_alpha: optional real ``(H,)`` in [0,1]; delta-rule strength per head
                     (docs/01 §5.3). ``None`` or 0 ⇒ pure accumulation.
        state: optional initial :class:`ScanState` (for chunked/continued scans).
        return_state: also return the final :class:`ScanState`.

    Returns:
        c: real ``(B, T, H, d_vh)`` per-step read content. (+ final state if requested).
    """
    B, T, H, d_k = k.shape
    d_vh = v.shape[-1]
    dev = k.device
    cdtype = k.dtype  # complex64 or complex128 — follow the input precision
    if state is None:
        S = torch.zeros(B, H, d_k, d_vh, dtype=cdtype, device=dev)
        z = torch.zeros(B, H, d_k, dtype=cdtype, device=dev)
        den = torch.zeros(1, H, 1, device=dev)
    else:
        S, z = state.S, state.z
        den = getattr(state, "den", torch.zeros(1, H, 1, device=dev))
    g = gamma.view(1, H, 1, 1)
    gz = gamma.view(1, H, 1)
    vc = v.to(cdtype)
    out = torch.empty(B, T, H, d_vh, device=dev)
    for t in range(T):
        k_t = k[:, t]  # (B,H,d_k)
        q_t = q[:, t]
        v_t = vc[:, t]  # (B,H,d_vh) complex
        if delta_alpha is not None:
            # delta-rule: remove the value stored at this key (read S_{t-1} with den_{t-1})
            num = torch.einsum("bhk,bhkv->bhv", k_t.conj(), S).real
            v_hat = (num / (den + _EPS)).to(cdtype)
            v_write = v_t - delta_alpha.view(1, H, 1).to(cdtype) * v_hat
        else:
            v_write = v_t
        S = g * S + torch.einsum("bhk,bhv->bhkv", k_t, v_write)
        z = gz * z + k_t
        den = gz * den + d_k  # POSITIVE count normalizer (docs/01 §5) — stable, signed-safe
        num = torch.einsum("bhk,bhkv->bhv", q_t.conj(), S).real  # Re(q^† S)
        out[:, t] = num / den
    if return_state:
        st = ScanState(S=S, z=z)
        st.den = den
        return out, st
    return out


class ContextualMemory(nn.Module):
    """The attention-role sublayer (docs/02 §4): project → phase+position → scan → read."""

    def __init__(
        self,
        d_model: int,
        *,
        D: int | None = None,
        d_v: int | None = None,
        H: int = 8,
        rope_base: float = 10000.0,
        delta_rule_heads: bool = True,
    ):
        super().__init__()
        D = D or d_model
        d_v = d_v or d_model
        assert D % H == 0 and d_v % H == 0, "D and d_v must be divisible by H"
        self.d_model, self.D, self.d_v, self.H = d_model, D, d_v, H
        self.d_k, self.d_vh = D // H, d_v // H

        self.W_q = nn.Linear(d_model, D, bias=False)
        self.W_k = nn.Linear(d_model, D, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        self.W_o = nn.Linear(d_v, d_model, bias=False)
        self.log_gamma = nn.Parameter(multitimescale_gammas(H))

        # delta-rule on the middle third of the timescale spectrum (docs/03 §9 default)
        alpha = torch.zeros(H)
        if delta_rule_heads and H >= 3:
            alpha[H // 3 : 2 * H // 3] = 1.0
        self.register_buffer("delta_alpha", alpha)

        omega = A.default_frequencies(self.d_k, base=rope_base)
        self.register_buffer("omega", omega)  # (d_k,)

    def gammas(self) -> Tensor:
        return torch.sigmoid(self.log_gamma)

    def project(self, x: Tensor, positions: Tensor):
        """x:(B,T,d_model) → phased+positioned k,q (complex (B,T,H,d_k)), v (real (B,T,H,d_vh))."""
        B, T, _ = x.shape
        q = A.phi(self.W_q(x)).view(B, T, self.H, self.d_k)
        k = A.phi(self.W_k(x)).view(B, T, self.H, self.d_k)
        v = self.W_v(x).view(B, T, self.H, self.d_vh)
        # positional binding ρ^t on both q and k (relative position, docs/01 §11)
        p = A.positional_phasor(positions.to(x.dtype), self.omega)  # (T, d_k)
        p = p.view(1, T, 1, self.d_k)
        return k * p, q * p, v

    def read_content(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        """Return the raw contextual read content c:(B,T,d_v) (pre-W_o, concatenated heads)."""
        B, T, _ = x.shape
        if positions is None:
            positions = torch.arange(T, device=x.device)
        k, q, v = self.project(x, positions)
        c = scan_reference(k, q, v, self.gammas(), delta_alpha=self.delta_alpha)
        return c.reshape(B, T, self.d_v)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        """Contextual sublayer output in model space: W_o(c):(B,T,d_model)."""
        c = self.read_content(x, positions)
        return self.W_o(c)
