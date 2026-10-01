"""RIPPLE — the backpropagation-free learning rule (docs/03).

Resonance-Induced Plasticity with Phase-Local Eligibility. This module assembles the
per-parameter gradient locally, with **no** reverse-mode autodiff over model parameters
(constitutional requirement C2, docs/00 §1):

    * the EP content teaching signal  δ_t = (r⁺ − r⁻)/(2β_n)            (docs/01 §7, eq 7.3)
    * the two-path codebook gradient (read path + readout path)          (docs/01 §8)
    * exact eligibility traces for W_v, W_q, W_k through the linear
      recurrence — forward, online, O(D·d_in) memory                     (docs/01 §9)

The functions here are validated against autograd (BPTT twin) in
tests/test_gradient_fidelity.py, tests/test_two_path_codebook.py, tests/test_traces.py —
the S3 kill gate (docs/05 P1). Autograd appears ONLY in those tests and in the BPTT twin
(wavenn/rlm.py::bptt_loss + wavenn/train.py::train_bptt), never here.

Sign convention: every ``*_grad`` returns an estimate of ∂C/∂θ (the LOSS gradient), so a
gradient-descent step is ``θ -= lr · grad``. The EP content signal δ satisfies
∂C/∂c = −δ (docs/01 eq 7.2); ``content_gradient`` returns ∂C/∂c directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from . import algebra as A
from . import cleanup as C

_EPS = 1e-6


# ======================================================================================
# Contextual read (functional; differentiable so tests can autograd the ground truth)
# ======================================================================================
def contextual_read(
    x: Tensor,
    Wk: Tensor,
    Wq: Tensor,
    Wv: Tensor,
    omega: Tensor,
    gamma: Tensor,
    H: int,
    delta_alpha: Tensor | None = None,
    return_qkv: bool = False,
):
    """Forward contextual read c:(B,T,d_v) from projections (docs/01 §5).

    x:(B,T,d_in); Wk,Wq:(D,d_in); Wv:(d_v,d_in); omega:(d_k,); gamma:(H,). Differentiable
    — the BPTT twin autographs through it.

    delta_alpha:(H,) in [0,1] enables the delta-rule write on those heads (docs/01 §5.3,
    docs/02 §4.1): the value currently stored at the key is removed before writing the new
    one — the Widrow–Hoff/DeltaNet correction, which improves recall (P6). ``None`` or 0 ⇒
    pure accumulation (the exact-trace regime, docs/03 §4).

    ``return_qkv`` also returns the cached forward activations ``(c, q, k, v)`` — the phasors
    q,k:(B,T,H,d_k) and values v:(B,T,H,d_vh) — so the transport-free L1-FA pullback
    (``contextual_input_grad_fa``) can read the network STATE instead of recomputing it from
    the forward weights (plan Stage 1b). ``v`` is the pre-delta value; supported only in the
    pure-accumulation regime (delta_alpha None/0), which is where L1/L1-FA operate.
    """
    B, T, _ = x.shape
    D = Wk.shape[0]
    d_v = Wv.shape[0]
    d_k, d_vh = D // H, d_v // H
    cdtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
    pos = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(-1) * omega  # (T,d_k)
    ak = (x @ Wk.t()).view(B, T, H, d_k) + pos.view(1, T, 1, d_k)
    aq = (x @ Wq.t()).view(B, T, H, d_k) + pos.view(1, T, 1, d_k)
    k = A.phi(ak)
    q = A.phi(aq)
    v = (x @ Wv.t()).view(B, T, H, d_vh)
    g = gamma.view(1, H, 1, 1)
    use_delta = delta_alpha is not None and float(delta_alpha.abs().sum()) > 0
    if use_delta:
        alpha = delta_alpha.view(1, H, 1)
    S = torch.zeros(B, H, d_k, d_vh, dtype=cdtype, device=x.device)
    # POSITIVE count normalizer den_t = γ den_{t-1} + d_k (docs/01 §5): deterministic and
    # bounded away from 0, so c is a stable weighted average — FHRR similarities Re(q^†k) are
    # SIGNED, so the query-dependent normalizer Re(q^†z) can go negative and explode the read.
    den = torch.zeros(1, H, 1, device=x.device)
    gh = gamma.view(1, H, 1)
    outs = []
    for t in range(T):
        v_t = v[:, t].to(cdtype)
        if use_delta:  # remove the value stored at k_t (read S_{t-1} with normalizer den_{t-1})
            num_k = torch.einsum("bhk,bhkv->bhv", k[:, t].conj(), S).real
            v_hat = (num_k / (den + _EPS)).to(cdtype)  # den still holds den_{t-1} here
            v_t = v_t - alpha.to(cdtype) * v_hat
        S = g * S + torch.einsum("bhk,bhv->bhkv", k[:, t], v_t)
        den = gh * den + d_k
        num = torch.einsum("bhk,bhkv->bhv", q[:, t].conj(), S).real
        outs.append((num / den).reshape(B, d_v))
    c = torch.stack(outs, dim=1)
    if return_qkv:
        if use_delta:
            raise ValueError("return_qkv is only defined in the pure-accumulation regime")
        return c, q, k, v  # cached forward activations (network state) for the L1-FA pullback
    return c


# ======================================================================================
# Exact eligibility-trace gradients for W_k, W_q, W_v  (docs/01 §9; validated vs autograd)
# ======================================================================================
def projection_grads(
    x: Tensor,
    Wk: Tensor,
    Wq: Tensor,
    Wv: Tensor,
    omega: Tensor,
    gamma: Tensor,
    H: int,
    g_content: Tensor,
    *,
    key_full: bool = True,
    return_input_grad: bool = False,
    return_gamma: bool = False,
    delta_alpha: Tensor | None = None,
):
    """Return (gWk, gWq, gWv) = ∂L/∂{Wk,Wq,Wv} for L = <g_content, c>, via exact traces.

    g_content:(B,T,d_v) is the content-space cotangent ∂L/∂c (from EP: ∂C/∂c = −δ_t).
    Implements the value/query/key pathways of docs/01 §9 & Appendix 03-A. Exact because
    the recurrence is linear in S — verified equal to autograd in tests/test_traces.py.

    key_full=True uses the exact key eligibility trace (needs the O(d_k·d_vh·d_in) tensor
    ``Tkv``; used by the tests and small-scale exact runs). key_full=False uses the
    cheap current-step key attribution (S-A0 for the key path, docs/03 §4.3 / whitepaper
    S-A0) — no large trace, suitable for training at scale. The value path is always the
    exact trace (cheap: O(d_k·d_in)); the query path is always current-step (exact).

    return_input_grad=True also returns g_x:(B,T,d_in) — the CURRENT-STEP input cotangent
    (query + key + value current-step pathways), used to give the embedding a local
    gradient (docs/03 §4.4, S-A0 for the input).
    """
    B, T, d_in = x.shape
    D = Wk.shape[0]
    d_v = Wv.shape[0]
    d_k, d_vh = D // H, d_v // H
    dev = x.device
    cdtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64

    pos = torch.arange(T, device=dev, dtype=x.dtype).unsqueeze(-1) * omega
    ak = (x @ Wk.t()).view(B, T, H, d_k) + pos.view(1, T, 1, d_k)
    aq = (x @ Wq.t()).view(B, T, H, d_k) + pos.view(1, T, 1, d_k)
    k = A.phi(ak)
    q = A.phi(aq)
    v = (x @ Wv.t()).view(B, T, H, d_vh)
    gh = g_content.view(B, T, H, d_vh)
    use_delta = delta_alpha is not None and float(delta_alpha.abs().sum()) > 0
    alpha = delta_alpha.view(1, H, 1).to(cdtype) if use_delta else None

    gvw = gamma.view(1, H, 1, 1)
    gzw = gamma.view(1, H, 1)

    S = torch.zeros(B, H, d_k, d_vh, dtype=cdtype, device=dev)
    z = torch.zeros(B, H, d_k, dtype=cdtype, device=dev)  # kept only for the delta-rule read
    den = torch.zeros(1, H, 1, device=dev)  # deterministic count normalizer den_t=γden+d_k
    Gv = torch.zeros(B, H, d_k, d_in, dtype=cdtype, device=dev)  # docs/01 eq 9.4
    Tkv = (
        torch.zeros(B, H, d_k, d_vh, d_in, dtype=cdtype, device=dev) if key_full else None
    )  # exact key value-trace (large); None ⇒ current-step key path

    gWk = torch.zeros(H, d_k, d_in, device=dev)
    gWq = torch.zeros(H, d_k, d_in, device=dev)
    gWv = torch.zeros(H, d_vh, d_in, device=dev)
    g_x = torch.zeros(B, T, d_in, device=dev) if return_input_grad else None
    # decay-path traces (docs/03 §4.5): dotS_t = S_{t-1}+γ dotS_{t-1}, dotden_t = den_{t-1}+γ dotden
    dotS = torch.zeros_like(S) if return_gamma else None
    dotden = torch.zeros(1, H, 1, device=dev) if return_gamma else None
    ggamma = torch.zeros(H, device=dev) if return_gamma else None

    for t in range(T):
        k_t, q_t = k[:, t], q[:, t]  # (B,H,d_k)
        v_t = v[:, t].to(cdtype)  # (B,H,d_vh)
        x_t = x[:, t]  # (B,d_in)
        gt = gh[:, t]  # (B,H,d_vh)

        if return_gamma:  # decay traces use S_{t-1}, den_{t-1} (before this step's write)
            dotS = S + gvw * dotS
            dotden = den + gzw * dotden

        if use_delta:  # DeltaNet write (docs/01 §5.3); direct pathway learned, feedback dropped
            num_k = torch.einsum("bhk,bhkv->bhv", k_t.conj(), S).real
            v_t = v_t - alpha * (num_k / (den + _EPS)).to(cdtype)

        # advance states and traces (all linear recurrences, docs/01 §9)
        S = gvw * S + torch.einsum("bhk,bhv->bhkv", k_t, v_t)
        z = gzw * z + k_t
        den = gzw * den + d_k  # POSITIVE deterministic normalizer (signed-safe, docs/01 §5)
        Gv = gvw * Gv + torch.einsum("bhk,bn->bhkn", k_t, x_t.to(cdtype))
        if key_full:
            Tkv = gvw.unsqueeze(-1) * Tkv + torch.einsum(
                "bhk,bhv,bn->bhkvn", k_t, v_t, x_t.to(cdtype)
            )

        num = torch.einsum("bhk,bhkv->bhv", q_t.conj(), S).real
        c_t = num / den  # (B,H,d_vh); den (1,H,1) broadcasts

        # --- decay path (docs/03 §4.5): ∂c/∂γ = [Re(q^†dotS) − c·dotden]/den ---
        if return_gamma:
            qdotS = torch.einsum("bhk,bhkv->bhv", q_t.conj(), dotS).real  # (B,H,d_vh)
            dcdg = (qdotS - c_t * dotden) / den  # ∂c/∂γ (B,H,d_vh)
            ggamma += torch.einsum("bhv,bhv->h", gt, dcdg)

        # --- value path (docs/01 §9.1): gWv += g_t ⊗ Re(q^† Gv)/den ---
        qGv = torch.einsum("bhk,bhkn->bhn", q_t.conj(), Gv).real / den  # (B,H,d_in)
        gWv += torch.einsum("bhv,bhn->hvn", gt, qGv)

        # --- query path (den deterministic ⇒ only the numerator term): Im(q̄ gS)/den ---
        gS = torch.einsum("bhv,bhkv->bhk", gt.to(cdtype), S)  # Σ_m g_m S[:, :, k, m]
        gq = (q_t.conj() * gS).imag / den  # (B,H,d_k)
        gWq += torch.einsum("bhk,bn->hkn", gq, x_t)

        # --- key path (den deterministic ⇒ only the numerator term: −Im(q̄ gT)/den) ---
        if key_full:
            gT = torch.einsum("bhv,bhkvn->bhkn", gt.to(cdtype), Tkv)  # Σ_m g_m Tkv (B,H,d_k,d_in)
            gk_full = -((q_t.conj().unsqueeze(-1) * gT).imag) / den.unsqueeze(-1)
            gWk += gk_full.sum(dim=0)
        else:
            # current-step attribution (S-A0): only the i=t term, no large trace
            im_qk = (q_t.conj() * k_t).imag  # Im(conj(q_t) k_t)  (B,H,d_k)
            gv_dot = torch.einsum("bhv,bhv->bh", gt, v_t.real).unsqueeze(-1)  # g_t·v_t (B,H,1)
            gk_cur = -im_qk * gv_dot / den  # (B,H,d_k)
            gWk += torch.einsum("bhk,bn->hkn", gk_cur, x_t)

        # --- current-step input cotangent (docs/03 §4.4, for the embedding) ---
        if return_input_grad:
            g_x[:, t] += gq.reshape(B, D) @ Wq  # query pathway
            # value pathway (i=t): ∂c_t/∂v_t = a_{t,t} = Re(q^†k)/den
            a_tt = (q_t.conj() * k_t).real.sum(-1, keepdim=True) / den.squeeze(0)  # (B,H,1)
            g_x[:, t] += (gt * a_tt).reshape(B, d_v) @ Wv
            # key pathway (i=t)
            im_qk = (q_t.conj() * k_t).imag
            gv_dot = torch.einsum("bhv,bhv->bh", gt, v_t.real).unsqueeze(-1)
            g_x[:, t] += (-im_qk * gv_dot / den).reshape(B, D) @ Wk

    base = (gWk.reshape(D, d_in), gWq.reshape(D, d_in), gWv.reshape(d_v, d_in))
    extras = []
    if return_input_grad:
        extras.append(g_x)
    if return_gamma:
        # chain through γ = σ(log_gamma): ∂L/∂log_gamma = (∂L/∂γ)·γ(1-γ)  (docs/03 §4.5)
        extras.append(ggamma * (gamma * (1 - gamma)))
    if extras:
        return (base, *extras)
    return base


# ======================================================================================
# Exact input pullback through the contextual read (docs/03 §7, for L1 stack routing)
# ======================================================================================
def contextual_input_grad(
    x: Tensor,
    Wk: Tensor,
    Wq: Tensor,
    Wv: Tensor,
    omega: Tensor,
    gamma: Tensor,
    H: int,
    g_content: Tensor,
) -> Tensor:
    """Exact input cotangent g_x = ∂L/∂x for L = ⟨g_content, c⟩ (pure-accumulation regime).

    Unlike ``projection_grads(return_input_grad=True)`` — which uses the cheap current-step
    (S-A0) attribution for the key/value input paths — this returns the **full** input
    pullback, including the temporal cross-terms (how x_t affects *future* reads c_{t'>t}
    through the linear state S). Exact because the recurrence is linear in S; the temporal
    sum is done by one reverse-time adjoint (docs/03 §7, docs/01 §9):

        Q_t = conj(q_t) ⊗ (g_t/den_t) + γ Q_{t+1}          (reverse recurrence, Q_T = 0)
        ∂L/∂v_t   = Re( Σ_j k_t[j] Q_t[j,·] )              (value path, all t'≥t)
        ∂L/∂ak_t  = − Im( k_t ⊙ (Q_t · v_t) )             (key path, phase domain)
        ∂L/∂aq_t  = Im( conj(q_t) ⊙ (S_t · g_t) ) / den_t  (query path, current step)

    then g_x_t = ∂L/∂aq_t @ Wq + ∂L/∂ak_t @ Wk + ∂L/∂v_t @ Wv. Validated vs autograd in
    tests/test_stack_ep_fidelity.py. Count normalizer den_t is deterministic (independent of
    x) so it contributes no input path. Pure-accumulation only (no delta-rule): the delta
    write couples v_t to past state and is excluded from the exact-trace regime (docs/03 §4.6).
    """
    B, T, d_in = x.shape
    D = Wk.shape[0]
    d_v = Wv.shape[0]
    d_k, d_vh = D // H, d_v // H
    dev = x.device
    cdtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64

    pos = torch.arange(T, device=dev, dtype=x.dtype).unsqueeze(-1) * omega  # (T,d_k)
    ak = (x @ Wk.t()).view(B, T, H, d_k) + pos.view(1, T, 1, d_k)
    aq = (x @ Wq.t()).view(B, T, H, d_k) + pos.view(1, T, 1, d_k)
    k = A.phi(ak)  # (B,T,H,d_k)
    q = A.phi(aq)
    v = (x @ Wv.t()).view(B, T, H, d_vh).to(cdtype)
    g = g_content.view(B, T, H, d_vh).to(cdtype)
    gh = gamma.view(1, H, 1)

    # --- forward pass: state S_t, deterministic den_t, and the current-step query path ---
    S = torch.zeros(B, H, d_k, d_vh, dtype=cdtype, device=dev)
    den = torch.zeros(1, H, 1, device=dev, dtype=x.dtype)
    g_aq = torch.zeros(B, T, H, d_k, device=dev, dtype=x.dtype)
    for t in range(T):
        S = gh.view(1, H, 1, 1) * S + torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t])
        den = gh * den + d_k
        gS = torch.einsum("bhv,bhkv->bhk", g[:, t], S)  # Σ_m g_t[m] S[k,m]  (B,H,d_k)
        g_aq[:, t] = (q[:, t].conj() * gS).imag / den  # query path (current step, exact)

    # --- reverse pass: adjoint Q_t gives the exact value + key (future) input paths ---
    Q = torch.zeros(B, H, d_k, d_vh, dtype=cdtype, device=dev)
    g_x = torch.zeros(B, T, d_in, device=dev, dtype=x.dtype)
    den_list = []  # deterministic den_t, precomputed for the reverse sweep
    den = torch.zeros(1, H, 1, device=dev, dtype=x.dtype)
    for t in range(T):
        den = gh * den + d_k
        den_list.append(den.clone())
    for t in reversed(range(T)):
        Q = torch.einsum("bhk,bhv->bhkv", q[:, t].conj(), g[:, t] / den_list[t]) + gh.view(
            1, H, 1, 1
        ) * Q
        g_v_t = torch.einsum("bhk,bhkv->bhv", k[:, t], Q).real  # ∂L/∂v_t (B,H,d_vh)
        Qv = torch.einsum("bhkv,bhv->bhk", Q, v[:, t])  # (Q·v_t)  (B,H,d_k) complex
        g_ak_t = -(k[:, t] * Qv).imag  # ∂L/∂ak_t (B,H,d_k)
        g_x[:, t] = (
            g_aq[:, t].reshape(B, D) @ Wq
            + g_ak_t.reshape(B, D) @ Wk
            + g_v_t.reshape(B, d_v) @ Wv
        )
    return g_x


def contextual_input_grad_fa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gamma: Tensor,
    H: int,
    g_content: Tensor,
    Bk: Tensor,
    Bq: Tensor,
    Bv: Tensor,
) -> Tensor:
    """Transport-free (feedback-alignment) input pullback for L1-FA (docs/03 §7, plan Stage 1).

    Identical to ``contextual_input_grad`` EXCEPT: (i) it reads the **cached forward
    activations** q,k,v (the network STATE, passed in) instead of recomputing them from the
    forward weights, and (ii) it projects the adjoint back to input space through the supplied
    feedback matrices ``Bk,Bq,Bv`` instead of the forward-weight transpose. Because this
    function's signature contains **no forward weights** ``Wk/Wq/Wv``, the error projection is
    weight-transport-free *by construction* — that is the property tests/test_transport_free.py
    verifies structurally (perturbing the forward weights cannot change this output).

    The reverse-time adjoint ``Q_t`` (which carries the temporal credit) is preserved exactly;
    only the final input-space projection (Site B) is randomized. The caller supplies the
    *effective* feedback ``B_eff = (1-λ)·W + λ·B``: at λ=0 ``B_eff=W`` reproduces the exact
    pullback (a plumbing check), at λ=1 it is fully transport-free (the headline point).

    q,k:(B,T,H,d_k); v:(B,T,H,d_vh) real; g_content:(B,T,d_v); Bk,Bq:(D,d_in); Bv:(d_v,d_in).
    ``den_t`` is recomputed from ``gamma`` (deterministic decay time-constant — dynamics, not a
    transported weight). Pure-accumulation regime only.
    """
    B, T, _, d_k = q.shape
    d_vh = v.shape[-1]
    D, d_v, d_in = H * d_k, H * d_vh, Bq.shape[1]
    dev = q.device
    rdtype = g_content.dtype
    cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64
    vc = v.to(cdtype)
    g = g_content.view(B, T, H, d_vh).to(cdtype)
    gh = gamma.view(1, H, 1)

    # --- forward: state S_t, deterministic den_t, current-step query path (activations only) ---
    S = torch.zeros(B, H, d_k, d_vh, dtype=cdtype, device=dev)
    den = torch.zeros(1, H, 1, device=dev, dtype=rdtype)
    g_aq = torch.zeros(B, T, H, d_k, device=dev, dtype=rdtype)
    for t in range(T):
        S = gh.view(1, H, 1, 1) * S + torch.einsum("bhk,bhv->bhkv", k[:, t], vc[:, t])
        den = gh * den + d_k
        gS = torch.einsum("bhv,bhkv->bhk", g[:, t], S)
        g_aq[:, t] = (q[:, t].conj() * gS).imag / den

    # --- reverse: adjoint Q_t → value/key input cotangents, projected via FEEDBACK B (Site B) ---
    Q = torch.zeros(B, H, d_k, d_vh, dtype=cdtype, device=dev)
    g_x = torch.zeros(B, T, d_in, device=dev, dtype=rdtype)
    den_list = []
    den = torch.zeros(1, H, 1, device=dev, dtype=rdtype)
    for t in range(T):
        den = gh * den + d_k
        den_list.append(den.clone())
    for t in reversed(range(T)):
        Q = torch.einsum("bhk,bhv->bhkv", q[:, t].conj(), g[:, t] / den_list[t]) + gh.view(
            1, H, 1, 1
        ) * Q
        g_v_t = torch.einsum("bhk,bhkv->bhv", k[:, t], Q).real
        Qv = torch.einsum("bhkv,bhv->bhk", Q, vc[:, t])
        g_ak_t = -(k[:, t] * Qv).imag
        g_x[:, t] = (
            g_aq[:, t].reshape(B, D) @ Bq  # transport-free feedback projection (was @ Wq)
            + g_ak_t.reshape(B, D) @ Bk
            + g_v_t.reshape(B, d_v) @ Bv
        )
    return g_x


# ======================================================================================
# Warm-read EP: content signal δ_t and the two-path codebook gradient (docs/01 §7, §8)
# ======================================================================================
@dataclass
class WarmReadEP:
    r_free: Tensor  # (B,d_v) free-phase fixed point
    r_star: Tensor  # readout state (== r_free for the plain readout)
    p: Tensor  # (B,K) readout distribution softmax(β Vᵀ r*)
    delta: Tensor  # (B,d_v) EP content signal (r⁺−r⁻)/(2β_n)
    grad_c: Tensor  # (B,d_v) ∂C/∂c = −δ
    grad_M: Tensor  # (d_v,P) read-path gradient ∂C/∂M for the SETTLE codebook M
    grad_V_readout: Tensor  # (d_v,K) readout-path gradient ∂C/∂V for the readout codebook
    n_iter: int = 0  # settle iterations (adaptive-compute signal, docs/02 §5.1)

    @property
    def grad_V_read(self) -> Tensor:
        """Alias: when the settle codebook IS the readout codebook (tied RWR), the read
        path acts on V. Kept for the two-path regression test (docs/01 §8)."""
        return self.grad_M


def warm_read_ep(
    c: Tensor,
    M: Tensor,
    V: Tensor | None = None,
    beta: float = 4.0,
    targets: Tensor | None = None,
    beta_n: float = 0.05,
    *,
    n_settle: int = 40,
    eta: float = 0.3,
    adaptive: bool = False,
    tol: float = 1e-4,
    beta_readout: float | None = None,
    valid: Tensor | None = None,
) -> WarmReadEP:
    """Per-step EP on the warm read: settle against codebook ``M``, read out against ``V``.

    * Separate case (``V`` != ``M``): the knowledge memory ``M`` (symmetric Hopfield,
      docs/02 §5) shapes the settle and gets a single **read-path** gradient; the vocab
      codebook ``V`` appears only in the readout softmax and gets a single **readout-path**
      gradient. Each is exact (docs/01 §7,§8).
    * Tied case (``V is None`` ⇒ ``V = M``): the whitepaper RWR (settle and readout against
      the same codebook). The read path and readout path then BOTH act on that codebook and
      must be summed — the two-path correction (docs/01 §8), locked by
      tests/test_two_path_codebook.py.

    c:(B,d_v), M:(d_v,P), V:(d_v,K), targets:(B,) int.
    """
    if V is None:
        V = M
    b_ro = beta if beta_readout is None else beta_readout  # readout temperature (F3)
    # ignore-index masking: positions with an invalid target (e.g. -100) contribute NO loss
    # gradient — the content signal and codebook grads are zeroed there, but the token still
    # participates in the running state/context (masked next-token / MQAR training).
    safe_targets = targets.clamp(min=0)  # dummy target for masked rows (contribution zeroed)
    # free phase (β_n = 0), reused as warm start; nudge uses the readout codebook V @ β_ro
    r0, _, n0 = C.settle(c, M, M, beta, n_max=n_settle, eta=eta, adaptive=adaptive, tol=tol)
    rp, _, _ = C.settle(
        c, M, M, beta, n_max=n_settle, eta=eta, adaptive=False,
        target=safe_targets, beta_n=+beta_n, r_init=r0, readout=V, beta_readout=b_ro,
    )
    rn, _, _ = C.settle(
        c, M, M, beta, n_max=n_settle, eta=eta, adaptive=False,
        target=safe_targets, beta_n=-beta_n, r_init=r0, readout=V, beta_readout=b_ro,
    )
    delta = (rp - rn) / (2 * beta_n)  # docs/01 eq 7.3
    grad_c = -delta  # ∂C/∂c = −δ  (docs/01 eq 7.2)

    if valid is not None:
        vmask = valid.to(grad_c.dtype).view(-1, 1)  # (B,1)
        grad_c = grad_c * vmask
        rp = rp * vmask
        rn = rn * vmask
        r0m = r0 * vmask
    else:
        r0m = r0

    # read-path gradient on the SETTLE codebook M (M shapes r*): use the settle
    # distributions p_M = softmax(β Mᵀ r) at the nudged fixed points (docs/01 eq 8.1)
    pMp = torch.softmax(beta * (rp @ M), dim=-1)
    pMn = torch.softmax(beta * (rn @ M), dim=-1)
    grad_M = -(torch.einsum("bv,bp->vp", rp, pMp) - torch.einsum("bv,bp->vp", rn, pMn)) / (
        2 * beta_n
    )

    # readout-path gradient on V (V only in the final softmax): ∂C/∂V_k = β_ro r*(p_k − 1{k=y})
    p = torch.softmax(b_ro * (r0 @ V), dim=-1)  # (B,K)
    onehot = F.one_hot(safe_targets, num_classes=V.shape[1]).to(p.dtype)
    grad_V_readout = b_ro * torch.einsum("bv,bk->vk", r0m, (p - onehot))

    return WarmReadEP(
        r_free=r0, r_star=r0, p=p, delta=delta, grad_c=grad_c,
        grad_M=grad_M, grad_V_readout=grad_V_readout, n_iter=n0,
    )


# ======================================================================================
# Stack EP (L1): per-block content signal with a top-down cotangent (docs/03 §7)
# ======================================================================================
def stack_read_ep(
    c: Tensor,
    M: Tensor,
    V: Tensor,
    beta: float,
    targets: Tensor | None,
    beta_n: float,
    *,
    r_free: Tensor,
    g_top: Tensor | None = None,
    n_settle: int = 40,
    eta: float = 0.3,
    beta_readout: float | None = None,
    valid: Tensor | None = None,
):
    """L1 stack-EP per-block signal (docs/03 §7). Returns (grad_c, grad_M, grad_V_readout, p).

    Extends the per-step EP (``warm_read_ep``) with inter-block credit:

    * ``g_top``:(B,d_v) is the top-down content cotangent ∂L/∂r_ℓ routed down the residual
      bus from the blocks above. It enters the EP cost as the linear term ⟨g_top, r⟩, so the
      ± nudged settles gain the constant force −g_top (see ``cleanup.settle``). Then
      δ=(r⁺−r⁻)/2β_n estimates −∂C_total/∂c and ``grad_c = −δ`` is the content cotangent that
      already carries the credit from every block above (EP identity 7.1 with the augmented
      cost). The read-path ``grad_M`` picks up the same top-down credit automatically through
      the nudged fixed points r⁺,r⁻.
    * Masking (``valid``): unlike L0, ``grad_c`` is **not** zeroed at ignore_index rows — a
      masked position's r still influences downstream *valid* positions through the temporal
      state, so its top-down credit is a real part of the BPTT gradient. Only the LOCAL
      deep-supervision readout is masked: the readout nudge (``readout_scale``) and
      ``grad_V_readout`` apply at valid rows only.

    ``r_free`` is the free-phase fixed point from the forward pass (reused as the warm start
    and the readout base). ``targets=None`` ⇒ no local readout (final-only mode): the block
    is trained purely by the top-down cotangent, ``grad_V_readout=0``.
    """
    b_ro = beta if beta_readout is None else beta_readout
    rscale = None if valid is None else valid.to(c.dtype).view(-1, 1)
    tgt = None if targets is None else targets.clamp(min=0)
    kw = dict(
        n_max=n_settle, eta=eta, adaptive=False, r_init=r_free, readout=V,
        beta_readout=b_ro, g_top=g_top, readout_scale=rscale,
    )
    rp, _, _ = C.settle(c, M, M, beta, target=tgt, beta_n=+beta_n, **kw)
    rn, _, _ = C.settle(c, M, M, beta, target=tgt, beta_n=-beta_n, **kw)
    delta = (rp - rn) / (2 * beta_n)
    grad_c = -delta  # ∂C_total/∂c, carries inter-block credit; NOT masked (docs/03 §7)

    # read-path codebook gradient on the settle codebook M (docs/01 §8); the nudged fixed
    # points already include the top-down force, so this is the total ∂C_total/∂M.
    pMp = torch.softmax(beta * (rp @ M), dim=-1)
    pMn = torch.softmax(beta * (rn @ M), dim=-1)
    grad_M = -(torch.einsum("bv,bp->vp", rp, pMp) - torch.einsum("bv,bp->vp", rn, pMn)) / (
        2 * beta_n
    )

    # readout-path gradient on V (local deep-supervision softmax), masked to valid rows
    if targets is None:
        p = torch.softmax(b_ro * (r_free @ V), dim=-1)
        grad_V_readout = torch.zeros_like(V)
    else:
        p = torch.softmax(b_ro * (r_free @ V), dim=-1)
        onehot = F.one_hot(tgt, num_classes=V.shape[1]).to(p.dtype)
        r_free_m = r_free if rscale is None else r_free * rscale  # zero at masked rows
        grad_V_readout = b_ro * torch.einsum("bv,bk->vk", r_free_m, (p - onehot))
    return grad_c, grad_M, grad_V_readout, p


# ======================================================================================
# Three-factor neuromodulation and local Adam (docs/03 §5, §6)
# ======================================================================================
def surprise_gate(nll: Tensor, running_mean: float, g0: float, g1: float, lo: float, hi: float):
    """Global scalar modulator μ_t from prediction surprise (docs/03 §5, eq 5.2)."""
    mu = (g0 + g1 * (nll.mean().item() - running_mean))
    return max(lo, min(hi, mu))


class LocalAdam:
    """Adam arithmetic on locally-computed gradients — no autograd graph (docs/03 §6).

    Permitted under C2: the moments are element-wise functions of the local Δθ, never a
    backward pass. Operates in-place on a dict of named parameter tensors.
    """

    def __init__(self, params: dict[str, Tensor], lr=3e-3, betas=(0.9, 0.99), eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in params.items()}
        self.t = 0

    # -- checkpoint/resume support (S6 plan A.4): the moments ARE optimizer state --
    def state_dict(self) -> dict:
        return {"lr": self.lr, "t": self.t,
                "m": {k: v.detach().cpu() for k, v in self.m.items()},
                "v": {k: v.detach().cpu() for k, v in self.v.items()}}

    def load_state_dict(self, sd: dict) -> None:
        self.lr = float(sd["lr"])
        self.t = int(sd["t"])
        for k in self.m:
            self.m[k].copy_(sd["m"][k].to(self.m[k].device))
            self.v[k].copy_(sd["v"][k].to(self.v[k].device))

    @torch.no_grad()
    def step(self, grads: dict[str, Tensor], clip: float | None = 1.0):
        self.t += 1
        bc1 = 1 - self.b1**self.t
        bc2 = 1 - self.b2**self.t
        if clip is not None:  # global-norm clip on the assembled local gradient
            total = torch.sqrt(
                sum((g.detach() ** 2).sum() for g in grads.values() if g is not None) + 1e-12
            )
            scale = min(1.0, clip / (total.item() + 1e-12))
        else:
            scale = 1.0
        for k, g in grads.items():
            if g is None:
                continue
            g = g * scale
            self.m[k].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[k].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            mhat = self.m[k] / bc1
            vhat = self.v[k] / bc2
            self.params[k].add_(-self.lr * mhat / (vhat.sqrt() + self.eps))
