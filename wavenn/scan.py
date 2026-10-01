"""S6 chunked kernels — throughput forms of the contextual read and its gradients.

Implements docs/10-s6-protocol.md §6 (plan §M.1/§M.2). Two kernels:

  * :func:`chunked_contextual_read` — the forward causal FHRR linear-attention read
    computed in chunks of ``L_c`` positions (intra-chunk masked matmuls + an
    inter-chunk carried state) instead of the per-token Python loop of
    ``ripple.contextual_read``. Mathematically identical (finite-sum reordering).

  * :func:`adjoint_grads` — ALL projection-parameter gradients (gWk, gWq, gWv,
    g_logγ), and the input pullback g_x, from two chunked scans: the forward state
    scan (query path) and the reverse-time adjoint scans Q, Q̇ (value/key/decay
    paths). This is the §M.2 adjoint reordering: with the deterministic count
    normalizer den_t (independent of q, k) and pure accumulation, every trace-form
    gradient of docs/01 §9 is an exact reordering of

        Q_i = conj(q_i)(g_i/den_i)ᵀ + γ·Q_{i+1}          (reverse recurrence)
        Q̇_i = Q_{i+1} + γ·Q̇_{i+1}                        (decay-path adjoint)

        g^v_i  = Re(Σ_j k_{i,j} Q_i[j,·])                 ⇒ gWv = Σ_i g^v_i x_iᵀ
        g^{ak}_i = −Im(k_i ⊙ (Q_i v_i))                   ⇒ gWk = Σ_i g^{ak}_i x_iᵀ
        g^{aq}_t = Im(conj(q_t) ⊙ (S_t g_t))/den_t        ⇒ gWq = Σ_t g^{aq}_t x_tᵀ
        Σ_t g_tᵀ∂c_t/∂γ = Σ_i Re(Σ_j k_{i,j}(Q̇_i v_i)_j) − Σ_t (g_tᵀc_t)·ḋen_t/den_t

    (chain to log_gamma via γ(1−γ)). NO Tkv/Gv trace is materialized: the
    O(d_k·d_vh·d_in) tensor of ``projection_grads(key_full=True)`` is mathematically
    redundant with the adjoint (same finite double sum, summation order exchanged).

Numerical prescription (binding; docs/10 §6): intra-chunk decay weights
Γ[t,i] = γ^{t−i} are computed by DIRECT POWERS (bounded exponent ≤ L_c−1) and the
inter-chunk carry is multiplicative (γ^{L_c}). The factored form γ^t·γ^{−i} is
PROHIBITED — γ^{−i} overflows for small learned γ. Underflow of γ^{t−i} to 0 for
short-timescale heads is the mathematically correct limit (benign). den_t uses the
float64 closed form d_k(1−γ^{t+1})/(1−γ).

Validity guards (runtime-asserted): the adjoint identities hold ONLY under
(i) pure accumulation (no delta-rule write — v_i independent of past state) and
(ii) the deterministic count normalizer. Violations raise; they never silently
mis-compute (docs/10 §6).

Scope (docs/10 §6): docs/01 §9's "online, O(D·d_in)" statement describes the
forward-trace form, which remains in ``ripple.projection_grads`` as the
hardware-locality reference and the equality ground truth (tests/test_chunked_scan).
These kernels compute the SAME exact gradient by a reverse-time batch sweep — the
computation L1 routing already performs. C2 (no autograd over model parameters) is
untouched: everything here is explicit forward arithmetic.

Certification: tests/test_chunked_scan.py — float64 relative error ≤ 1e-9 and fp32
relative ≤ 1e-4 against BOTH oracles (the projection_grads trace loop AND the
contextual_input_grad adjoint loop) and against autograd; test matrix covers
γ ∈ {1e-4, 0.9, 0.999} heads, T non-divisible by L_c spanning ≥ 3 chunks, B > 1,
and zero/masked g rows.
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import algebra as A

__all__ = ["chunked_contextual_read", "adjoint_grads", "closed_form_den"]


def _cdtype(x: Tensor):
    return torch.complex128 if x.dtype == torch.float64 else torch.complex64


def closed_form_den(gamma: Tensor, T: int, d_k: int, dtype) -> tuple[Tensor, Tensor]:
    """(den, dden) per (T, H): den_t = d_k·Σ_{s≤t} γ^s, ḋen_t = d_k·Σ_{s≤t} s·γ^{s−1}.

    Computed in float64 as cumulative sums of DIRECT powers (γ^s = exp(s·lnγ)) —
    no cancellation-prone (1−γ)-division (which loses ~2 digits at γ = 0.999), cast
    to ``dtype``. Matches the recurrences den_t = γ·den_{t−1}+d_k and
    ḋen_t = den_{t−1}+γ·ḋen_{t−1} to accumulation-order roundoff (validated in
    tests/test_chunked_scan.py).
    """
    g = gamma.to(torch.float64).clamp(1e-12, 1 - 1e-12).unsqueeze(0)  # (1,H)
    s = torch.arange(T, dtype=torch.float64, device=gamma.device).unsqueeze(-1)  # (T,1)
    p = g**s  # γ^s, (T,H) — direct powers, bounded exponents
    den = d_k * torch.cumsum(p, dim=0)
    # ḋen: d/dγ of d_k·Σ γ^s = d_k·Σ s·γ^{s−1}
    pm1 = g ** (s - 1).clamp(min=0)
    dden = d_k * torch.cumsum(s * pm1, dim=0)
    return den.to(dtype), dden.to(dtype)


def _decay_masks(gamma: Tensor, Lc: int, dtype):
    """Γ[h,t,i] = γ_h^{t−i} (i ≤ t, else 0) and Γ̇[h,t,i] = (t−i)γ_h^{t−i−1} (i < t).

    DIRECT POWERS with bounded exponents 0…Lc−1 (the prohibited factored form
    γ^t·γ^{−i} is never used). (H, Lc, Lc), real dtype.
    """
    g = gamma.to(torch.float64)
    idx = torch.arange(Lc, device=gamma.device, dtype=torch.float64)
    delta = idx.view(Lc, 1) - idx.view(1, Lc)  # t − i
    causal = delta >= 0
    d = delta.clamp(min=0)
    G = g.view(-1, 1, 1) ** d.view(1, Lc, Lc)
    G = torch.where(causal.view(1, Lc, Lc), G, torch.zeros_like(G))
    strict = delta >= 1
    dm1 = (delta - 1).clamp(min=0)
    Gd = d.view(1, Lc, Lc) * (g.view(-1, 1, 1) ** dm1.view(1, Lc, Lc))
    Gd = torch.where(strict.view(1, Lc, Lc), Gd, torch.zeros_like(Gd))
    return G.to(dtype), Gd.to(dtype)


def _project(x: Tensor, Wk: Tensor, Wq: Tensor, Wv: Tensor, omega: Tensor, H: int):
    """Phases and values, (B,H,T,·) layout. Same math as ripple.contextual_read."""
    B, T, _ = x.shape
    D, d_v = Wk.shape[0], Wv.shape[0]
    d_k, d_vh = D // H, d_v // H
    pos = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(-1) * omega
    ak = (x @ Wk.t()).view(B, T, H, d_k) + pos.view(1, T, 1, d_k)
    aq = (x @ Wq.t()).view(B, T, H, d_k) + pos.view(1, T, 1, d_k)
    k = A.phi(ak).permute(0, 2, 1, 3)  # (B,H,T,d_k)
    q = A.phi(aq).permute(0, 2, 1, 3)
    v = (x @ Wv.t()).view(B, T, H, d_vh).permute(0, 2, 1, 3)  # (B,H,T,d_vh) real
    return k, q, v, d_k, d_vh


@torch.no_grad()
def chunked_contextual_read(
    x: Tensor,
    Wk: Tensor,
    Wq: Tensor,
    Wv: Tensor,
    omega: Tensor,
    gamma: Tensor,
    H: int,
    delta_alpha: Tensor | None = None,
    return_qkv: bool = False,
    L_c: int = 128,
):
    """Chunked forward read c:(B,T,d_v) ≡ ripple.contextual_read (pure accumulation).

    Guards (docs/10 §6): pure accumulation only — the delta-rule write couples v_t to
    past state, invalidating the chunk decomposition (and the §M.2 adjoint).
    """
    if delta_alpha is not None and float(delta_alpha.abs().sum()) > 0:
        raise ValueError("chunked kernels require the pure-accumulation regime "
                         "(delta_rule off); see docs/10 §6 runtime guards")
    B, T, _ = x.shape
    cdtype = _cdtype(x)
    k, q, v, d_k, d_vh = _project(x, Wk, Wq, Wv, omega, H)
    d_v = H * d_vh
    den, _ = closed_form_den(gamma, T, d_k, x.dtype)  # (T,H)

    S = torch.zeros(B, H, d_k, d_vh, dtype=cdtype, device=x.device)
    out = torch.empty(B, H, T, d_vh, dtype=x.dtype, device=x.device)
    g = gamma.to(x.dtype)

    for t0 in range(0, T, L_c):
        t1 = min(t0 + L_c, T)
        Lc = t1 - t0
        kc, qc, vc = k[:, :, t0:t1], q[:, :, t0:t1], v[:, :, t0:t1]
        G, _ = _decay_masks(gamma, Lc, x.dtype)  # (H,Lc,Lc)
        # intra-chunk: scores[t,i] = Re(q_t† k_i); num += (scores ⊙ Γ) @ v
        scores = torch.einsum("bhtd,bhid->bhti", qc.conj(), kc).real
        num = torch.einsum("bhti,bhiv->bhtv", scores * G.unsqueeze(0), vc)
        # inter-chunk: γ^{t−t0+1} · Re(q_t† S_carry)
        powt = g.view(1, H, 1) ** torch.arange(1, Lc + 1, device=x.device, dtype=x.dtype).view(1, 1, Lc)
        carry = torch.einsum("bhtd,bhdv->bhtv", qc.conj(), S).real
        num = num + powt.unsqueeze(-1) * carry
        out[:, :, t0:t1] = num / den[t0:t1].t().view(1, H, Lc, 1)
        # carry update: S ← γ^{Lc} S + Σ_i γ^{t1−1−i} k_i v_iᵀ  (direct powers)
        powk = g.view(1, H, 1) ** torch.arange(Lc - 1, -1, -1, device=x.device, dtype=x.dtype).view(1, 1, Lc)
        S = (g.view(1, H, 1, 1) ** Lc) * S + torch.einsum(
            "bhid,bhiv->bhdv", kc * powk.unsqueeze(-1).to(cdtype), vc.to(cdtype)
        )

    c = out.permute(0, 2, 1, 3).reshape(B, T, d_v)
    if return_qkv:
        # (B,T,H,·) layout, matching ripple.contextual_read(return_qkv=True)
        return c, q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    return c


@torch.no_grad()
def adjoint_grads(
    x: Tensor,
    Wk: Tensor,
    Wq: Tensor,
    Wv: Tensor,
    omega: Tensor,
    gamma: Tensor,
    H: int,
    g_content: Tensor,
    *,
    c: Tensor | None = None,
    want_gamma: bool = True,
    want_input: bool = True,
    proj: tuple[Tensor, Tensor, Tensor] | None = None,
    delta_alpha: Tensor | None = None,
    L_c: int = 128,
):
    """All projection gradients + input pullback via the §M.2 adjoint reordering.

    Args:
        g_content: (B,T,d_v) content cotangent ∂L/∂c (from the stack-EP settle).
        c:         (B,T,d_v) the forward read — required iff want_gamma (the decay
                   path's −Σ(g·c)·ḋen/den term); pass the cached forward c.
        proj:      optional (Pq, Pk, Pv) matrices for the input pullback projection
                   (Site B). Default = (Wq, Wk, Wv) — the exact L1 pullback; pass
                   feedback matrices for the transport-free variant.
        L_c:       chunk length.

    Returns ((gWk, gWq, gWv), g_x, g_log_gamma); g_x None if not want_input,
    g_log_gamma None if not want_gamma. Shapes match ripple.projection_grads /
    contextual_input_grad exactly.

    Guards (docs/10 §6): pure accumulation only; den is the deterministic count
    normalizer by construction (computed internally, independent of q/k).
    """
    if delta_alpha is not None and float(delta_alpha.abs().sum()) > 0:
        raise ValueError("adjoint_grads requires the pure-accumulation regime "
                         "(delta_rule off); see docs/10 §6 runtime guards")
    if want_gamma and c is None:
        raise ValueError("want_gamma requires the forward read c (ḋen term)")

    B, T, d_in = x.shape
    cdtype = _cdtype(x)
    k, q, v, d_k, d_vh = _project(x, Wk, Wq, Wv, omega, H)
    D, d_v = H * d_k, H * d_vh
    den, dden = closed_form_den(gamma, T, d_k, x.dtype)  # (T,H)
    g = gamma.to(x.dtype)
    gh = g_content.view(B, T, H, d_vh).permute(0, 2, 1, 3)  # (B,H,T,d_vh)
    ght = gh / den.t().view(1, H, T, 1)  # g̃_t = g_t/den_t

    g_aq = torch.empty(B, H, T, d_k, dtype=x.dtype, device=x.device)
    g_ak = torch.empty(B, H, T, d_k, dtype=x.dtype, device=x.device)
    g_v = torch.empty(B, H, T, d_vh, dtype=x.dtype, device=x.device)
    ggam1 = torch.zeros(H, dtype=x.dtype, device=x.device) if want_gamma else None

    # ---- forward chunk scan: query path g_aq (needs S_t at every t) ----
    S = torch.zeros(B, H, d_k, d_vh, dtype=cdtype, device=x.device)
    for t0 in range(0, T, L_c):
        t1 = min(t0 + L_c, T)
        Lc = t1 - t0
        kc, qc, vc, gc = k[:, :, t0:t1], q[:, :, t0:t1], v[:, :, t0:t1], gh[:, :, t0:t1]
        G, _ = _decay_masks(gamma, Lc, x.dtype)
        # (S_t g_t) intra: Σ_{i≤t} γ^{t−i} k_i (v_i·g_t)
        b1 = torch.einsum("bhtv,bhiv->bhti", gc, vc)  # (g_t·v_i), real
        Sg = torch.einsum("bhti,bhij->bhtj", (b1 * G.unsqueeze(0)).to(cdtype), kc)
        powt = g.view(1, H, 1) ** torch.arange(1, Lc + 1, device=x.device, dtype=x.dtype).view(1, 1, Lc)
        Sg = Sg + powt.unsqueeze(-1).to(cdtype) * torch.einsum(
            "bhjm,bhtm->bhtj", S, gc.to(cdtype)
        )
        g_aq[:, :, t0:t1] = (qc.conj() * Sg).imag / den[t0:t1].t().view(1, H, Lc, 1)
        powk = g.view(1, H, 1) ** torch.arange(Lc - 1, -1, -1, device=x.device, dtype=x.dtype).view(1, 1, Lc)
        S = (g.view(1, H, 1, 1) ** Lc) * S + torch.einsum(
            "bhid,bhiv->bhdv", kc * powk.unsqueeze(-1).to(cdtype), vc.to(cdtype)
        )

    # ---- reverse chunk scan: adjoints Q (value/key) and Q̇ (decay) ----
    Qn = torch.zeros(B, H, d_k, d_vh, dtype=cdtype, device=x.device)   # Q_{t1+1}
    Qdn = torch.zeros(B, H, d_k, d_vh, dtype=cdtype, device=x.device)  # Q̇_{t1+1}
    starts = list(range(0, T, L_c))
    for t0 in reversed(starts):
        t1 = min(t0 + L_c, T)
        Lc = t1 - t0
        kc, qc, vc = k[:, :, t0:t1], q[:, :, t0:t1], v[:, :, t0:t1]
        gtc = ght[:, :, t0:t1]  # g̃, (B,H,Lc,d_vh)
        G, Gd = _decay_masks(gamma, Lc, x.dtype)
        Gt = G.transpose(-1, -2).unsqueeze(0)    # Γᵀ[i,t] = γ^{t−i} (t ≥ i)
        Gdt = Gd.transpose(-1, -2).unsqueeze(0)  # Γ̇ᵀ[i,t] = (t−i)γ^{t−i−1} (t > i)
        # scores'[i,t] = Re(k_i† q_t)... = Re(conj(q_t)·k_i) = scoresᵀ (Re symmetric)
        scoresT = torch.einsum("bhid,bhtd->bhit", kc, qc.conj()).real
        # per-position carry powers γ^{t1+1−i} = γ^{Lc−local_i}, and γ^{t1−i}
        gpow1 = g.view(1, H, 1) ** torch.arange(Lc, 0, -1, device=x.device, dtype=x.dtype).view(1, 1, Lc)
        gpow0 = g.view(1, H, 1) ** torch.arange(Lc - 1, -1, -1, device=x.device, dtype=x.dtype).view(1, 1, Lc)
        # -- value path: g_v_i = Re(Σ_j k_{i,j} Q_i[j,·]) --
        gv = torch.einsum("bhit,bhtv->bhiv", scoresT * Gt, gtc)
        gv = gv + gpow1.unsqueeze(-1) * torch.einsum("bhij,bhjv->bhiv", kc, Qn).real
        g_v[:, :, t0:t1] = gv
        # -- key path: g_ak_i = −Im(k_i ⊙ (Q_i v_i)) --
        b2 = torch.einsum("bhtv,bhiv->bhit", gtc, vc)  # (g̃_t·v_i), (B,H,Lc_i,Lc_t)
        Qv = torch.einsum("bhit,bhtj->bhij", (b2 * Gt).to(cdtype), qc.conj())
        Qnv = torch.einsum("bhjm,bhim->bhij", Qn, vc.to(cdtype))  # (Q_next v_i)
        Qv = Qv + gpow1.unsqueeze(-1).to(cdtype) * Qnv
        g_ak[:, :, t0:t1] = -(kc * Qv).imag
        # -- decay path: Σ_i Re(Σ_j k_{i,j}(Q̇_i v_i)_j) --
        if want_gamma:
            Qdv = torch.einsum("bhit,bhtj->bhij", (b2 * Gdt).to(cdtype), qc.conj())
            Qdnv = torch.einsum("bhjm,bhim->bhij", Qdn, vc.to(cdtype))
            Qdv = Qdv + gpow1.unsqueeze(-1).to(cdtype) * Qdnv \
                      + (gpow0 * torch.arange(Lc, 0, -1, device=x.device, dtype=x.dtype).view(1, 1, Lc)).unsqueeze(-1).to(cdtype) * Qnv
            ggam1 += torch.einsum("bhij->h", (kc * Qdv).real)
        # -- carry updates (order: Q̇ first — it needs the OLD Qn) --
        u = torch.einsum("bhtj,bhtv->bhjv", qc.conj(), gtc.to(cdtype))  # unweighted; reweighted below
        # Q̇_{t0} = Σ_{t>t0} (t−t0)γ^{t−t0−1} u_t + γ^{Lc} Q̇n + Lc·γ^{Lc−1} Qn
        wdot = (torch.arange(Lc, device=x.device, dtype=x.dtype).view(1, 1, Lc)
                * g.view(1, H, 1) ** (torch.arange(Lc, device=x.device, dtype=x.dtype) - 1).clamp(min=0).view(1, 1, Lc))
        wdot[:, :, 0] = 0.0
        Qdn = torch.einsum("bhtj,bhtv->bhjv", (qc.conj() * wdot.unsqueeze(-1).to(cdtype)), gtc.to(cdtype)) \
            + (g.view(1, H, 1, 1) ** Lc) * Qdn \
            + Lc * (g.view(1, H, 1, 1) ** (Lc - 1)) * Qn
        # Q_{t0} = Σ_{t≥t0} γ^{t−t0} u_t + γ^{Lc} Qn
        wq = g.view(1, H, 1) ** torch.arange(Lc, device=x.device, dtype=x.dtype).view(1, 1, Lc)
        Qn = torch.einsum("bhtj,bhtv->bhjv", (qc.conj() * wq.unsqueeze(-1).to(cdtype)), gtc.to(cdtype)) \
            + (g.view(1, H, 1, 1) ** Lc) * Qn

    # ---- assemble parameter gradients (outer products with the input) ----
    xf = x.reshape(B * T, d_in)
    gWk = g_ak.permute(0, 2, 1, 3).reshape(B * T, D).t() @ xf
    gWq = g_aq.permute(0, 2, 1, 3).reshape(B * T, D).t() @ xf
    gWv = g_v.permute(0, 2, 1, 3).reshape(B * T, d_v).t() @ xf

    g_x = None
    if want_input:
        Pq, Pk, Pv = proj if proj is not None else (Wq, Wk, Wv)
        g_x = (
            g_aq.permute(0, 2, 1, 3).reshape(B, T, D) @ Pq
            + g_ak.permute(0, 2, 1, 3).reshape(B, T, D) @ Pk
            + g_v.permute(0, 2, 1, 3).reshape(B, T, d_v) @ Pv
        )

    g_log_gamma = None
    if want_gamma:
        ch = c.view(B, T, H, d_vh).permute(0, 2, 1, 3)
        gc_dot = torch.einsum("bhtv,bhtv->bht", gh, ch)  # (g_t·c_t)
        ggam2 = torch.einsum("bht,th->h", gc_dot, (dden / den))
        g_log_gamma = (ggam1 - ggam2) * (g * (1 - g))

    return (gWk, gWq, gWv), g_x, g_log_gamma
