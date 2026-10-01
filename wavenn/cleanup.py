"""Knowledge Memory — the FFN replacement (docs/02 §5, docs/01 §6).

A finite-temperature modern-Hopfield cleanup: the read content ``c`` is denoised by
settling on the energy

    E(r) = ½‖r‖² − ⟨c, r⟩ − (1/β) lse(β M_keyᵀ r)      (docs/01 eq 6.1)

whose gradient flow ``dr/dτ = −∂E/∂r = −r + c + M_val·softmax(β M_keyᵀ r)`` converges to
``r* = c + M_val·softmax(β M_keyᵀ r*)`` (docs/01 eq 6.3). With ``M_key = M_val`` this is
the exact §6 energy; separate key/value codebooks generalize it to attention-over-stored-
patterns (Ramsauer et al. 2021).

Convergence & uniqueness are guaranteed in the safe regime ``β < 2/σ_max(M_key)²``
(docs/01 §6.2). The settle supports an optional external nudge force for equilibrium
propagation (docs/03 §2); the free read uses no nudge.
"""

from __future__ import annotations

import torch
from torch import Tensor


def safe_beta_limit(M_key: Tensor) -> float:
    """Upper bound ``2/σ_max(M_key)²`` for a guaranteed-unique read (docs/01 eq 6.5)."""
    # M_key: (d_v, P). σ_max is the largest singular value.
    sig = torch.linalg.matrix_norm(M_key, ord=2).item()
    return 2.0 / (sig * sig + 1e-12)


def settle(
    c: Tensor,
    M_key: Tensor,
    M_val: Tensor,
    beta: float,
    *,
    n_max: int = 8,
    eta: float = 0.3,
    tol: float = 1e-4,
    target: Tensor | None = None,
    beta_n: float = 0.0,
    r_init: Tensor | None = None,
    adaptive: bool = True,
    readout: Tensor | None = None,
    beta_readout: float | None = None,
    g_top: Tensor | None = None,
    readout_scale: Tensor | None = None,
):
    """Settle the Warm Read to its fixed point (docs/01 §6, docs/02 §5.1).

    Args:
        c:      real ``(B, d_v)`` read content (external bias).
        M_key:  real ``(d_v, P)`` cleanup keys (stored patterns).
        M_val:  real ``(d_v, P)`` cleanup values.
        beta:   read inverse temperature (docs/01 §6; keep < safe_beta_limit).
        n_max:  hard cap on iterations (adaptive stop usually 1–3, docs/02 §5.1).
        eta:    gradient-flow step size.
        tol:    convergence tolerance on ‖Δr‖ (adaptive stop).
        target: optional int ``(B,)`` target ids for an EP nudge (docs/03 §2).
        beta_n: nudge strength; ``+`` clamps toward the target, ``−`` away (symmetric).
        r_init: optional warm start (e.g. the free-phase r⁻), else ``r = c``.
        adaptive: stop early when ‖Δr‖ < tol.
        readout: optional separate readout codebook R:(d_v,K) for the EP nudge. The nudge
                 force is the cross-entropy gradient through softmax(β Rᵀ r) toward the
                 target. Defaults to ``M_key`` (the whitepaper's tied RWR); pass the vocab
                 codebook V when the knowledge memory M and the readout differ (docs/02 §5).
        g_top:  optional ``(B, d_v)`` top-down content cotangent ∂L/∂r from blocks *above*
                (L1 stack mode, docs/03 §7). It enters the EP cost as the linear term
                ⟨g_top, r⟩, so the nudge force gains a constant ``−g_top`` (derivation:
                F(r)=E(r)+β_n(CE_local+⟨g_top,r⟩) ⇒ dr += β_n(readout_force − g_top)). The
                resulting δ=(r⁺−r⁻)/2β_n then estimates −∂C_total/∂c, i.e. the content
                cotangent that already carries inter-block credit. ``None`` ⇒ pure L0/local.
        readout_scale: optional ``(B, 1)`` per-row weight on the *readout* (local-target)
                nudge only — used in L1 with a masked objective (MQAR): the local
                deep-supervision readout applies only at valid positions (scale 1), while
                the top-down ``g_top`` credit still flows at masked positions (scale 0 here,
                g_top unaffected). ``None`` ⇒ full weight.

    Returns:
        (r_star, p, n_iters) where p = softmax(β M_keyᵀ r_star) over the P patterns,
        and n_iters is the number of steps actually taken (adaptive-compute signal).
    """
    r = c.clone() if r_init is None else r_init.clone()
    RC = M_key if readout is None else readout
    b_ro = beta if beta_readout is None else beta_readout  # readout temperature (docs/01 §6.2 F3)
    n_used = 0
    for _ in range(n_max):
        n_used += 1
        a = torch.softmax(beta * (r @ M_key), dim=-1)  # (B, P) — SETTLE temperature β
        dr = -r + c + a @ M_val.t()  # (B, d_v)
        if beta_n != 0.0 and (target is not None or g_top is not None):
            # symmetric EP nudge: total force = readout_force − g_top (docs/03 §7).
            nudge = torch.zeros_like(r)
            if target is not None:
                # readout force −∂CE/∂r = β_ro(R_y − R·softmax(β_ro Rᵀr))
                p_ro = torch.softmax(b_ro * (r @ RC), dim=-1)
                onehot = torch.zeros_like(p_ro)
                onehot.scatter_(1, target.view(-1, 1), 1.0)
                rf = b_ro * ((onehot - p_ro) @ RC.t())
                if readout_scale is not None:  # mask local readout at ignore_index rows (L1)
                    rf = rf * readout_scale
                nudge = nudge + rf
            if g_top is not None:
                # top-down cotangent enters as the linear cost ⟨g_top,r⟩ ⇒ force −g_top
                nudge = nudge - g_top
            dr = dr + beta_n * nudge
        r_new = r + eta * dr
        step = (r_new - r).norm(dim=-1).max().item()
        r = r_new
        if adaptive and step < tol:
            break
    a = torch.softmax(beta * (r @ M_key), dim=-1)
    return r, a, n_used


def readout_logits(r_star: Tensor, V: Tensor, beta: float) -> Tensor:
    """Next-token / pattern logits ``β Vᵀ r*`` (docs/01 §6.1; whitepaper eq 4.2). V:(d_v,K) → (B,K)."""
    return beta * (r_star @ V)


# ======================================================================================
# S6 settle upgrades (docs/10 §6, plan §M.3/§M.4/§M.4a) — additive; settle() unchanged
# ======================================================================================
def settle_free_perrow(
    c: Tensor,
    M_key: Tensor,
    M_val: Tensor,
    beta: float,
    *,
    n_max: int,
    eta: float,
    tol: float,
    r_init: Tensor | None = None,
):
    """Free-phase settle with PER-ROW adaptive stopping (plan §M.3).

    Each row relaxes independently (the settle is row-separable); row i stops when
    its own ‖Δr_i‖ < tol (or at n_max) and is frozen thereafter. Returns
    (r, a, n_row) where n_row:(N,) long is each row's exit iteration — the SHARED
    iteration schedule the ± nudged phases must reuse (§M.4a symmetric stopping).
    """
    r = c.clone() if r_init is None else r_init.clone()
    N = r.shape[0]
    n_row = torch.full((N,), n_max, dtype=torch.long, device=r.device)
    active_idx = torch.arange(N, device=r.device)
    for it in range(n_max):
        ra = r[active_idx]
        a = torch.softmax(beta * (ra @ M_key), dim=-1)
        r_new = ra + eta * (-ra + c[active_idx] + a @ M_val.t())
        step = (r_new - ra).norm(dim=-1)
        r[active_idx] = r_new
        done = step < tol
        if done.any():
            n_row[active_idx[done]] = it + 1
            active_idx = active_idx[~done]
            if active_idx.numel() == 0:
                break
    a_full = torch.softmax(beta * (r @ M_key), dim=-1)
    return r, a_full, n_row


def settle_nudged_pair(
    c: Tensor,
    M_key: Tensor,
    M_val: Tensor,
    beta: float,
    *,
    r_free: Tensor,
    beta_n: float,
    n_row: Tensor,
    eta: float,
    const_force: Tensor | None = None,
    live_force=None,
):
    """The ± nudged settles under ONE shared per-row iteration schedule (§M.4a).

    Both signs start at ``r_free`` and run EXACTLY ``n_row[i]`` iterations for row i
    — identical schedules by construction, so the finite-iteration truncation error
    is symmetric and cancels in δ=(r⁺−r⁻)/2β_n at leading order (per-sign adaptive
    stopping would leave an O(tol/β_n) asymmetric bias the symmetric nudge cannot
    cancel; docs/10 §6).

    Force modes (exactly one):
      * ``const_force`` (N, d_v): the FROZEN force (§M.4) — the local-CE readout
        force linearized at r_free (masked rows already zeroed) minus g_top; applied
        as dr += sign·β_n·const_force every iteration. The S6 production mode.
      * ``live_force``: callable r -> (N, d_v) recomputing the force from the
        current r each iteration (the legacy ``settle`` behaviour, for equivalence
        tests and the frozen-vs-live fidelity comparison).

    Returns (r_plus, r_minus).
    """
    assert (const_force is None) != (live_force is None), "exactly one force mode"
    out = []
    n_total = int(n_row.max().item()) if n_row.numel() else 0
    for sign in (+1.0, -1.0):
        r = r_free.clone()
        active_idx = torch.arange(r.shape[0], device=r.device)
        for it in range(n_total):
            active_idx = active_idx[n_row[active_idx] > it]
            if active_idx.numel() == 0:
                break
            ra = r[active_idx]
            a = torch.softmax(beta * (ra @ M_key), dim=-1)
            dr = -ra + c[active_idx] + a @ M_val.t()
            if const_force is not None:
                f = const_force[active_idx]
            else:
                f = live_force(ra, active_idx)
            r[active_idx] = ra + eta * (dr + sign * beta_n * f)
        out.append(r)
    return out[0], out[1]


def frozen_readout_terms(
    r_free: Tensor,
    V: Tensor,
    beta_ro: float,
    targets: Tensor | None,
    valid: Tensor | None,
    *,
    row_chunk: int = 2048,
):
    """Row-chunked vocab-facing terms at the free fixed point (plan §M.3/§M.4).

    One vocab matmul family per block per step (instead of per settle iteration):
      * force:  −∂CE/∂r |_{r_free} = β_ro (V_y − V·softmax(β_ro Vᵀ r_free)) per row,
                zeroed at invalid rows (the local-readout mask; g_top is separate);
      * grad_V_readout: β_ro Σ_rows r_free (p − onehot)ᵀ over valid rows (docs/01 §8b);
      * nll_sum over valid rows (the per-block CE numerator for the gate/telemetry).

    Peak temporaries ≤ 3×(row_chunk × K) fp32 (§M.3; ~1.2 GB at 2048×50304).
    targets=None (final-only lower blocks): force=0, grad_V=0, nll_rows=0.
    Returns (force, grad_V_readout, nll_rows, n_valid) — nll_rows:(N,) is the
    per-row NLL (0 at invalid rows; free by-product of the force computation, used
    for the per-block mean CE and the replay surprise priorities).
    """
    N, d_v = r_free.shape
    K = V.shape[1]
    force = torch.zeros_like(r_free)
    gV = torch.zeros(d_v, K, dtype=r_free.dtype, device=r_free.device)
    nll_rows = torch.zeros(N, dtype=r_free.dtype, device=r_free.device)
    if targets is None:
        return force, gV, nll_rows, 0
    n_valid = 0
    tgt = targets.clamp(min=0)
    vmask = None if valid is None else valid
    for s in range(0, N, row_chunk):
        e = min(s + row_chunk, N)
        r_c = r_free[s:e]
        logits = beta_ro * (r_c @ V)  # (n, K)
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        t_c = tgt[s:e]
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, t_c.view(-1, 1), 1.0)
        f_c = beta_ro * ((onehot - p) @ V.t())
        row_nll = -logp.gather(1, t_c.view(-1, 1)).squeeze(1)
        if vmask is not None:
            vm = vmask[s:e].to(r_c.dtype).view(-1, 1)
            f_c = f_c * vm
            r_m = r_c * vm
            row_nll = row_nll * vm.squeeze(1)
            n_valid += int(vmask[s:e].sum().item())
        else:
            r_m = r_c
            n_valid += r_c.shape[0]
        force[s:e] = f_c
        nll_rows[s:e] = row_nll
        gV += beta_ro * torch.einsum("bv,bk->vk", r_m, (p - onehot))
        del logits, logp, p, onehot, f_c
    return force, gV, nll_rows, n_valid
