"""Recurrent O(1)-state sampler for the RIPPLE LM (docs/04 §7, docs/06 §3).

Generation maintains one running memory state (S, z) per block and processes tokens one
at a time — cost is CONSTANT in context length (no growing KV cache), the structural
long-context advantage of the architecture (docs/06 §3). This is the same recurrence as
the training forward (wavenn/rlm.py, wavenn/ripple.py), unrolled online.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from . import algebra as A
from . import cleanup as C
from .rlm import RippleLM, _rmsnorm, _unit_cols


class _BlockState:
    """Per-block running memory (docs/02 §4)."""

    def __init__(self, B: int, H: int, d_k: int, d_vh: int, device, dtype=torch.complex64):
        self.S = torch.zeros(B, H, d_k, d_vh, dtype=dtype, device=device)
        self.z = torch.zeros(B, H, d_k, dtype=dtype, device=device)  # for delta-rule only
        self.den = torch.zeros(1, H, 1, device=device)  # positive count normalizer (docs/01 §5)


@torch.no_grad()
def _step(model: RippleLM, token: Tensor, pos: int, states: list[_BlockState]) -> Tensor:
    """Advance one position through all blocks; return next-token logits (B, K)."""
    cfg = model.cfg
    H, d_k = cfg.n_heads, model.d_k
    d_vh = model.embed.weight.shape[1] // H
    bus = model.embed(token)  # (B, d_model)
    V = model.V()
    omega = model.omega
    pvec = A.positional_phasor(torch.tensor(float(pos), device=bus.device), omega)  # (d_k,)
    da = model.delta_alpha if cfg.delta_rule else None
    last_r = None
    for ell in range(cfg.n_layers):
        st = states[ell]
        a, _, _ = _rmsnorm(bus.unsqueeze(1), model.norm_gain[ell])
        a = a.squeeze(1)  # (B, d_model)
        gam = model.gammas(ell)
        B = a.shape[0]
        k = (A.phi((a @ model.Wk[ell].t()).view(B, H, d_k)) * pvec)  # (B,H,d_k)
        q = (A.phi((a @ model.Wq[ell].t()).view(B, H, d_k)) * pvec)
        v = (a @ model.Wv[ell].t()).view(B, H, d_vh).to(k.dtype)
        g = gam.view(1, H, 1, 1)
        gz = gam.view(1, H, 1)
        if da is not None and float(da.abs().sum()) > 0:
            num_k = torch.einsum("bhk,bhkv->bhv", k.conj(), st.S).real
            v = v - da.view(1, H, 1).to(k.dtype) * (num_k / (st.den + 1e-6)).to(k.dtype)
        st.S = g * st.S + torch.einsum("bhk,bhv->bhkv", k, v)
        st.z = gz * st.z + k
        st.den = gz * st.den + d_k  # count normalizer den_t = γ den_{t-1} + d_k
        num = torch.einsum("bhk,bhkv->bhv", q.conj(), st.S).real
        c = (num / st.den).reshape(B, cfg.n_heads * d_vh)
        M = _unit_cols(model.M[ell])
        r, _, _ = C.settle(c, M, M, cfg.beta, n_max=cfg.n_settle, eta=cfg.eta, adaptive=False)
        bus = bus + r
        last_r = r
    return cfg.beta_readout * (last_r @ V)  # readout temperature (docs/01 §6.2, F3)


@torch.no_grad()
def generate(
    model: RippleLM,
    prompt: Tensor,
    n_new: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    seed: int | None = None,
) -> Tensor:
    """Autoregressively extend ``prompt`` (B, T0) by ``n_new`` tokens, O(1) state per step.

    Returns the full (B, T0+n_new) token tensor. Deterministic given ``seed``.
    """
    model.eval()
    device = prompt.device
    B, T0 = prompt.shape
    H, d_k = model.cfg.n_heads, model.d_k
    d_vh = model.embed.weight.shape[1] // H
    states = [_BlockState(B, H, d_k, d_vh, device) for _ in range(model.cfg.n_layers)]
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    out = prompt.clone()
    logits = None
    # feed the prompt (build state); keep the last logits to predict token T0
    for t in range(T0):
        logits = _step(model, out[:, t], t, states)
    for i in range(n_new):
        logits = logits / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k, dim=-1)
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        if greedy:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=gen).squeeze(-1)
        out = torch.cat([out, nxt.unsqueeze(1)], dim=1)
        logits = _step(model, nxt, T0 + i, states)
    return out
