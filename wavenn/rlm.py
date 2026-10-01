"""RIPPLE Language Model — the full trainable stacked architecture (docs/02, docs/03).

Each block (docs/02 §3):
    a   = RMSNorm(bus)
    c   = ContextualMemory(a)          # multi-timescale + delta-rule + positional (docs/02 §4)
    r   = KnowledgeMemory.settle(c)     # symmetric modern-Hopfield cleanup vs codebook M (§5)
    p   = softmax(β Vᵀ r)               # readout vs the tied vocab codebook V
    bus = bus + r                       # bundle onto the residual bus (§6, M7)

The SAME forward trains two ways so P2/P3 are a pure learning-rule comparison (docs/05 §3):
  * ``bptt_loss``  — differentiable; the autograd BPTT twin (baseline).
  * ``ripple_grads`` — backprop-free RIPPLE (mode L0), assembled from: the EP content
    signal (docs/01 §7), the read-path knowledge-codebook update and readout-path vocab
    update (docs/01 §8), the exact value trace + cheap key/query attribution (docs/01 §9),
    the learned-timescale decay trace (docs/03 §4.5), and three-factor neuromodulation
    (docs/03 §5). NO autograd over model parameters (C2).

All six core mechanisms of docs/02–03 are present: multi-timescale memory, delta-rule
writes, separate knowledge memory, learned timescales, three-factor gating, and adaptive
settling (exposed for the M4 diagnostic; training uses fixed iterations for a fair twin
comparison). Depth-credit mode L1 (stack-EP) is the research extension in docs/03 §7.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import algebra as A
from . import cleanup as C
from . import ripple as R
from . import scan as SC
from .memory import multitimescale_gammas


@dataclass
class RLMConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    P: int | None = None  # knowledge patterns per block (defaults to 2*d_model)
    # SETTLE temperature β: kept modest so the knowledge-memory read is single-valued and
    # converges (the EP precondition, docs/01 §6.2). Decoupled from the readout (F3).
    beta: float = 1.0
    beta_readout: float = 6.0  # READOUT temperature β_ro: free to be sharp (F3)
    beta_n: float = 0.05  # EP nudge (docs/03); ≠ beta
    n_settle: int = 30  # enough to converge the read (EP precondition)
    eta: float = 0.3
    rope_base: float = 10000.0
    gamma_min: float = 0.0
    gamma_max: float = 0.99
    # Faithful defaults: pure-accumulation heads (exact traces) + exact key trace. The
    # delta-rule (forward recall, M6) and the cheap S-A0 key path have local gradients
    # that do NOT track BPTT (verified) — they are P6/P4 ABLATIONS, not the trained rule.
    delta_rule: bool = False
    key_full: bool = True
    deep_supervision: bool = True
    # depth-credit mode (docs/03 §7): "L0" pure-local (ripple_grads); "L1" stack-EP with
    # inter-block credit via the residual-bus adjoint (ripple_grads_l1); "L1FA" transport-free
    # feedback-alignment routing (ripple_grads_l1fa). All backprop-free (C2).
    depth_mode: str = "L0"
    # L1-FA transport-free routing (docs/03 §7, plan Stage 1). Fixed random feedback matrices
    # Bk/Bq/Bv replace the forward-weight transpose in the inter-block input pullback. Drawn
    # from a DEDICATED generator (feedback_seed) off the main RNG stream, so the L0/L1/BPTT
    # triplet stays byte-identical while the feedback draw is swept independently. feedback_lambda
    # is the dose-response knob: B_eff = (1-λ)·W + λ·B, so λ=0 ≡ L1-exact, λ=1 ≡ fully
    # transport-free. feedback_scale multiplies the random matrices (init-scale robustness).
    feedback_seed: int = 1234
    feedback_scale: float = 1.0
    feedback_lambda: float = 1.0
    feedback_shuffle: bool = False  # control: B = shuffled Wᵀ (matched statistics null)
    # L1-VF: space-time Vector-Field EP (docs/09, plan Stage 2). depth_mode="L1VF" routes credit
    # via a JOINT relaxation instead of a backward sweep. vf_variant: "VF-ST-adjoint" (temporal
    # credit via the reverse-time Q_t, the tractable rule), "VF-ST-forward" (Q_t disabled — the
    # OP3(b) forward-only probe), "2A-control" (no CR re-run). vf_rerun toggles the space-time
    # cycle (re-run CR in the nudged phase); vf_n_outer/vf_relax/vf_tol control the outer
    # fixed-point loop; feedback_gain λ_fb bounds the loop gain (docs/09 §4 convergence).
    vf_variant: str = "VF-ST-adjoint"
    vf_rerun: bool = True
    vf_n_outer: int = 12
    vf_relax: float = 0.5     # under-relaxation η_o
    vf_tol: float = 1e-4
    feedback_gain: float = 1.0  # λ_fb
    # S6 fast-path kernels (docs/10 §6; used by ripple_grads_l1_fast only — the legacy
    # paths are bit-unchanged). scan_Lc: chunk length for the §M.1/§M.2 kernels.
    # row_chunk: row-chunking of vocab-facing settle terms (§M.3). settle_tol: the
    # free-phase per-row adaptive tolerance; the ± nudged phases reuse the free
    # phase's per-row iteration schedule (§M.4a symmetric stopping). frozen_force:
    # linearize the local-CE readout force at r_free (§M.4; one vocab matmul per
    # block per step). fixed_schedule=True forces n_row=n_settle everywhere
    # (with frozen_force=False this reproduces the legacy nudged settles exactly —
    # the equivalence-test mode). cbar_init: optional Cbar initialization (ln K for
    # S6 — prevents the μ=gate_hi burn-in pin; None = legacy zeros).
    scan_Lc: int = 128
    row_chunk: int = 2048
    settle_tol: float = 1e-4
    frozen_force: bool = True
    fixed_schedule: bool = False
    cbar_init: float | None = None
    # three-factor neuromodulation (docs/03 §5)
    gate_g0: float = 1.0
    gate_g1: float = 0.5
    gate_lo: float = 0.1
    gate_hi: float = 3.0
    gate_ema: float = 0.01

    def resolved_P(self) -> int:
        return self.P if self.P is not None else 2 * self.d_model


def _rmsnorm(x: Tensor, gain: Tensor, eps: float = 1e-6):
    """Return (xn = normed*gain, normed, inv_rms) for the VJP (docs/02 §6)."""
    s = x.pow(2).mean(-1, keepdim=True) + eps
    inv = torch.rsqrt(s)
    normed = x * inv
    return normed * gain, normed, inv


def _rmsnorm_vjp(d_normed: Tensor, x: Tensor, inv_rms: Tensor):
    """∂L/∂x given ∂L/∂normed (docs/03 §4.4 input pullback). d = width."""
    d = x.shape[-1]
    s = 1.0 / (inv_rms.squeeze(-1) ** 2)
    dot = (d_normed * x).sum(-1)
    return inv_rms * (d_normed - x * (dot / (d * s)).unsqueeze(-1))


def _unit_cols(M: Tensor) -> Tensor:
    """Unit-norm columns — VSA codewords are unit magnitude (docs/01 §2); bounds σ_max(M)
    so the Hopfield read stays in the stable regime as the codebook trains (docs/01 §6.2)."""
    return F.normalize(M, p=2, dim=0)


def _unit_cols_vjp(g: Tensor, raw: Tensor, dim: int) -> Tensor:
    """Backprop-free vector–Jacobian product of the normalize(·, dim) map (F4).

    Given ``g = ∂C/∂(raw/‖raw‖)`` (gradient w.r.t. the unit-normalized tensor), return
    ``∂C/∂raw = (g − r̂ (r̂·g)) / ‖raw‖`` along ``dim``. This is the closed-form Jacobian
    of L2 normalization — required so RIPPLE's codebook gradients match what the BPTT twin
    computes through ``F.normalize`` (otherwise the P2/P3 comparison is biased). Closed
    form, no autograd (C2)."""
    norm = raw.norm(dim=dim, keepdim=True).clamp_min(1e-12)
    rhat = raw / norm
    return (g - rhat * (rhat * g).sum(dim=dim, keepdim=True)) / norm


class RippleLM(nn.Module):
    def __init__(self, cfg: RLMConfig):
        super().__init__()
        self.cfg = cfg
        d, L, H = cfg.d_model, cfg.n_layers, cfg.n_heads
        assert d % H == 0
        self.d_k = d // H
        P = cfg.resolved_P()
        self.embed = nn.Embedding(cfg.vocab_size, d)
        nn.init.normal_(self.embed.weight, std=1.0 / d**0.5)
        self.Wk = nn.ParameterList([nn.Parameter(torch.randn(d, d) / d**0.5) for _ in range(L)])
        self.Wq = nn.ParameterList([nn.Parameter(torch.randn(d, d) / d**0.5) for _ in range(L)])
        self.Wv = nn.ParameterList([nn.Parameter(torch.randn(d, d) / d**0.5) for _ in range(L)])
        self.M = nn.ParameterList([nn.Parameter(torch.randn(d, P) / d**0.5) for _ in range(L)])
        self.norm_gain = nn.ParameterList([nn.Parameter(torch.ones(d)) for _ in range(L)])
        self.log_gamma = nn.ParameterList(
            [nn.Parameter(multitimescale_gammas(H, cfg.gamma_min, cfg.gamma_max)) for _ in range(L)]
        )
        # delta-rule on the middle third of the timescale spectrum (docs/02 §4.1)
        alpha = torch.zeros(H)
        if cfg.delta_rule and H >= 3:
            alpha[H // 3 : 2 * H // 3] = 1.0
        self.register_buffer("delta_alpha", alpha)
        self.register_buffer("omega", A.default_frequencies(self.d_k, base=cfg.rope_base))
        # running surprise mean (docs/03 §5); cbar_init=ln K for S6 (docs/10 §4 — the
        # μ=gate_hi burn-in guard), None = legacy zeros (default behavior unchanged)
        cbar0 = 0.0 if cfg.cbar_init is None else float(cfg.cbar_init)
        self.register_buffer("Cbar", torch.full((), cbar0))
        # L1-FA fixed random feedback matrices Bk/Bq/Bv (docs/03 §7, plan Stage 1). Drawn from
        # a DEDICATED generator so the forward-weight draw order (and thus the L0/L1/BPTT
        # triplet) is byte-identical whether or not these exist. Shapes match Wk/Wq/Wv = (d,d);
        # they replace the forward-weight transpose in the inter-block input pullback (Site B).
        fg = torch.Generator().manual_seed(cfg.feedback_seed)
        s = cfg.feedback_scale / d**0.5
        self.register_buffer("Bk", torch.randn(L, d, d, generator=fg) * s)
        self.register_buffer("Bq", torch.randn(L, d, d, generator=fg) * s)
        self.register_buffer("Bv", torch.randn(L, d, d, generator=fg) * s)

    def V(self) -> Tensor:
        return _unit_cols(self.embed.weight).t()  # (d_model, K), unit-norm columns

    def gammas(self, ell: int) -> Tensor:
        return torch.sigmoid(self.log_gamma[ell])

    def M_used(self, ell: int) -> Tensor:
        return _unit_cols(self.M[ell])

    # ---- shared differentiable forward (BPTT twin) --------------------------------
    def forward_blocks(self, tokens: Tensor):
        cfg = self.cfg
        x = self.embed(tokens)
        V = self.V()
        block_logits = []
        da = self.delta_alpha if cfg.delta_rule else None
        for ell in range(cfg.n_layers):
            a, _, _ = _rmsnorm(x, self.norm_gain[ell])
            c = R.contextual_read(
                a, self.Wk[ell], self.Wq[ell], self.Wv[ell], self.omega, self.gammas(ell),
                cfg.n_heads, da,
            )
            B, T, dm = c.shape
            M = self.M_used(ell)
            r, _, _ = C.settle(
                c.reshape(B * T, dm), M, M, cfg.beta, n_max=cfg.n_settle, eta=cfg.eta,
                adaptive=False,
            )
            r = r.reshape(B, T, dm)
            block_logits.append(cfg.beta_readout * (r @ V))  # readout temperature (F3)
            x = x + r
        return block_logits

    def bptt_loss(self, tokens: Tensor, targets: Tensor) -> Tensor:
        block_logits = self.forward_blocks(tokens)
        K = self.cfg.vocab_size
        if self.cfg.deep_supervision:
            return sum(
                F.cross_entropy(lg.reshape(-1, K), targets.reshape(-1)) for lg in block_logits
            ) / len(block_logits)
        return F.cross_entropy(block_logits[-1].reshape(-1, K), targets.reshape(-1))

    @torch.no_grad()
    def logits(self, tokens: Tensor) -> Tensor:
        return self.forward_blocks(tokens)[-1]


# ======================================================================================
# RIPPLE (L0): backprop-free gradient assembly — NO autograd over model params (C2)
# ======================================================================================
@torch.no_grad()
def ripple_grads(model: RippleLM, tokens: Tensor, targets: Tensor, *,
                 ignore_index: int = -100, clip_norm: float | None = 1.0):
    """Return (mean_nll, grads) for the L0 local rule with all core mechanisms (docs/03).

    ``ignore_index`` positions contribute no loss gradient (masked next-token / MQAR).
    ``clip_norm`` global-norm-clips the assembled RAW gradient BEFORE the three-factor gate
    μ multiplies it, so the gate's novelty (μ>1) and protection (μ<1) modulation are both
    preserved (audit N5); pass the clipped result to the optimizer with clip=None."""
    cfg = model.cfg
    H, K = cfg.n_heads, cfg.vocab_size
    x = model.embed(tokens)
    B, T, dm = x.shape
    tflat = targets.reshape(B * T)
    valid = (tflat != ignore_index)
    V = model.V()
    da = model.delta_alpha if cfg.delta_rule else None

    grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    gV = torch.zeros_like(V)
    total_nll = 0.0

    for ell in range(cfg.n_layers):
        a, normed, inv = _rmsnorm(x, model.norm_gain[ell])
        c = R.contextual_read(
            a, model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega, model.gammas(ell), H, da
        )
        cflat = c.reshape(B * T, dm)
        M = model.M_used(ell)
        ep = R.warm_read_ep(
            cflat, M, V, cfg.beta, tflat, cfg.beta_n, n_settle=cfg.n_settle, eta=cfg.eta,
            beta_readout=cfg.beta_readout, valid=valid,
        )
        g_content = ep.grad_c.reshape(B, T, dm)

        # the decay (log_gamma) trace is exact ONLY for the pure-linear recurrence; under
        # the delta-rule it is invalid (docs/03 §4.6), so request it only when delta is off.
        want_gamma = da is None
        (gWk, gWq, gWv), g_a, *rest = R.projection_grads(
            a, model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega, model.gammas(ell), H,
            g_content, key_full=cfg.key_full, return_input_grad=True, return_gamma=want_gamma,
            delta_alpha=da,
        )
        grads[f"Wk.{ell}"] += gWk
        grads[f"Wq.{ell}"] += gWq
        grads[f"Wv.{ell}"] += gWv
        # read-path knowledge-codebook gradient, with the unit-norm Jacobian (F4)
        grads[f"M.{ell}"] += _unit_cols_vjp(ep.grad_M, model.M[ell], dim=0)
        if want_gamma:
            grads[f"log_gamma.{ell}"] += rest[0]
        grads[f"norm_gain.{ell}"] += (g_a * normed).reshape(-1, dm).sum(0)
        gV += ep.grad_V_readout

        if ell == 0:  # block-1 input pullback → embedding (docs/03 §4.4)
            d_normed = g_a * model.norm_gain[ell]
            g_x = _rmsnorm_vjp(d_normed, x, inv)
            grads["embed.weight"].index_add_(0, tokens.reshape(-1), g_x.reshape(-1, dm))

        # real per-block next-token NLL from the logits (F1: NOT cross_entropy(ep.p, …),
        # which would double-softmax and compress the surprise signal to ~0). Masked
        # positions are excluded via ignore_index.
        logits = cfg.beta_readout * (ep.r_star @ V)
        total_nll += F.cross_entropy(logits, tflat, ignore_index=ignore_index).item()
        x = x + ep.r_star.reshape(B, T, dm)  # bus update (bundle); detached (L0)

    # tied readout gradient through the unit-norm embedding Jacobian (F4, docs/01 §8).
    # dim=0: V()=_unit_cols(E).t() normalizes E over dim=0, so the VJP is along dim=0
    # (dim=1 is a 55%-error bug found by the L1 exactness check, tests/test_tied_readout_grad).
    grads["embed.weight"] += _unit_cols_vjp(gV.t(), model.embed.weight, dim=0)

    mean_nll = total_nll / cfg.n_layers
    # match the BPTT twin's mean-of-means loss scale so lr is comparable (F8): the local
    # grads sum over B·T tokens and L blocks; the twin averages over them.
    n_valid = max(1, int(valid.sum().item()))
    scale = 1.0 / (cfg.n_layers * n_valid)
    for k in grads:
        grads[k].mul_(scale)
    # clip the RAW gradient FIRST (audit N5), so the gate's up/down modulation survives
    cs = 1.0
    if clip_norm is not None:
        total = (sum((g.detach() ** 2).sum() for g in grads.values()) + 1e-12) ** 0.5
        cs = min(1.0, clip_norm / (float(total) + 1e-12))
    # three-factor neuromodulation (docs/03 §5): one global scalar μ per step, applied AFTER clip
    mu = R.surprise_gate(
        torch.tensor(mean_nll), model.Cbar.item(), cfg.gate_g0, cfg.gate_g1, cfg.gate_lo, cfg.gate_hi
    )
    for k in grads:
        grads[k].mul_(cs * mu)
    model.Cbar.mul_(1 - cfg.gate_ema).add_(cfg.gate_ema * mean_nll)
    return mean_nll, grads


# ======================================================================================
# RIPPLE (L1): stack-EP — inter-block credit via the residual-bus adjoint (docs/03 §7)
# ======================================================================================
@torch.no_grad()
def ripple_grads_l1(model: RippleLM, tokens: Tensor, targets: Tensor, *,
                    ignore_index: int = -100, clip_norm: float | None = 1.0):
    """L1 stack-EP gradient (docs/03 §7). Closes the L0 depth-credit gap (docs/07 R1).

    Same per-block machinery as ``ripple_grads`` (EP content signal + exact projection
    traces + two-path codebook), but each block's content cotangent now includes the credit
    routed *down the residual bus* from every block above. Because the bus is additive
    (``bus_{ℓ+1}=bus_ℓ+r_ℓ``), the exact cotangent propagates by accumulation: the top-down
    ``∂L/∂r_ℓ`` is fed into block ℓ's EP settle as a linear force (``stack_read_ep``), and the
    resulting input-pullback ``g_a`` is pushed back onto the bus (``_rmsnorm_vjp``) for the
    next lower block. This is the exact gradient of the deep-supervision loss (validated vs
    the 2-layer BPTT twin in tests/test_stack_ep_fidelity.py), computed with NO autograd over
    model parameters (C2). It uses inter-block weight transposes for the ``g_a`` routing —
    so, unlike the within-block EP, it is not weight-transport-free; it is the exact-credit
    reference that isolates the depth-credit question (docs/07). The transport-free
    Vector-Field variant (Scellier et al. 2018) is the follow-on (docs/03 §7, L1-VF).

    ``cfg.deep_supervision`` selects the loss: True ⇒ every block has a local readout target
    (matches ``bptt_loss`` deep-supervision); False ⇒ only the top block is nudged toward the
    target and lower blocks are trained purely by top-down credit (the pure stack-EP test).
    """
    cfg = model.cfg
    H, K = cfg.n_heads, cfg.vocab_size
    x = model.embed(tokens)
    B, T, dm = x.shape
    tflat = targets.reshape(B * T)
    valid = (tflat != ignore_index)
    V = model.V()
    da = model.delta_alpha if cfg.delta_rule else None
    L = cfg.n_layers

    grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    gV = torch.zeros_like(V)

    # ---- forward pass: build the bus, store per-block (a, normed, inv, c, r_free) ----
    buses = [x]  # buses[ℓ] = input bus to block ℓ; buses[0] = embedding output
    a_list, normed_list, inv_list, c_list, r_list = [], [], [], [], []
    total_nll = 0.0
    for ell in range(L):
        a, normed, inv = _rmsnorm(buses[ell], model.norm_gain[ell])
        c = R.contextual_read(
            a, model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega, model.gammas(ell), H, da
        )
        cflat = c.reshape(B * T, dm)
        M = model.M_used(ell)
        r0, _, _ = C.settle(cflat, M, M, cfg.beta, n_max=cfg.n_settle, eta=cfg.eta, adaptive=False)
        a_list.append(a); normed_list.append(normed); inv_list.append(inv)
        c_list.append(cflat); r_list.append(r0)
        logits = cfg.beta_readout * (r0 @ V)
        total_nll += F.cross_entropy(logits, tflat, ignore_index=ignore_index).item()
        buses.append(buses[ell] + r0.reshape(B, T, dm))

    # ---- backward sweep top→bottom: route the bus cotangent, per-block EP with g_top ----
    # bus_cotan := ∂L/∂bus_{ℓ+1}; starts at 0 above the top block (nothing above nudges it).
    bus_cotan = torch.zeros(B, T, dm, dtype=x.dtype, device=x.device)
    want_gamma = da is None
    for ell in reversed(range(L)):
        M = model.M_used(ell)
        # ∂L/∂r_ℓ from above = ∂L/∂bus_{ℓ+1} (since bus_{ℓ+1}=bus_ℓ+r_ℓ). None at the top.
        g_top = None if ell == L - 1 else bus_cotan.reshape(B * T, dm)
        # deep supervision: every block has a local target; else only the top block.
        blk_targets = tflat if (cfg.deep_supervision or ell == L - 1) else None
        gc, gM, gVro, _ = R.stack_read_ep(
            c_list[ell], M, V, cfg.beta, blk_targets, cfg.beta_n, r_free=r_list[ell],
            g_top=g_top, n_settle=cfg.n_settle, eta=cfg.eta, beta_readout=cfg.beta_readout,
            valid=valid,
        )
        g_content = gc.reshape(B, T, dm)
        # exact projection PARAMETER grads via forward traces (docs/01 §9); the input
        # pullback for inter-block routing is the EXACT reverse-time adjoint (docs/03 §7,
        # ``contextual_input_grad``) in the pure-accumulation regime — the current-step
        # (S-A0) g_a would drop the temporal cross-terms that carry the recall credit.
        if da is None:
            res = R.projection_grads(
                a_list[ell], model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega,
                model.gammas(ell), H, g_content, key_full=cfg.key_full,
                return_input_grad=False, return_gamma=want_gamma, delta_alpha=da,
            )
            (gWk, gWq, gWv), ggamma = res if want_gamma else (res, None)
            g_a = R.contextual_input_grad(
                a_list[ell], model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega,
                model.gammas(ell), H, g_content,
            )
        else:  # delta-rule: exact input trace not defined (docs/03 §4.6) — current-step g_a
            (gWk, gWq, gWv), g_a, *rest = R.projection_grads(
                a_list[ell], model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega,
                model.gammas(ell), H, g_content, key_full=cfg.key_full,
                return_input_grad=True, return_gamma=want_gamma, delta_alpha=da,
            )
            ggamma = rest[0] if want_gamma else None
        grads[f"Wk.{ell}"] += gWk
        grads[f"Wq.{ell}"] += gWq
        grads[f"Wv.{ell}"] += gWv
        grads[f"M.{ell}"] += _unit_cols_vjp(gM, model.M[ell], dim=0)
        if want_gamma:
            grads[f"log_gamma.{ell}"] += ggamma
        grads[f"norm_gain.{ell}"] += (g_a * normed_list[ell]).reshape(-1, dm).sum(0)
        gV += gVro
        # propagate the bus cotangent down: ∂L/∂bus_ℓ = ∂L/∂bus_{ℓ+1} (residual identity)
        #                                    + rmsnorm_vjp(g_a·gain) (through block ℓ)
        d_normed = g_a * model.norm_gain[ell]
        bus_cotan = bus_cotan + _rmsnorm_vjp(d_normed, buses[ell], inv_list[ell])

    # embedding receives the full bus cotangent at bus_0 (all depth credit, docs/03 §7)
    grads["embed.weight"].index_add_(0, tokens.reshape(-1), bus_cotan.reshape(-1, dm))
    # tied readout path through the unit-norm embedding Jacobian (F4); dim=0 matches the
    # forward V()=_unit_cols(E).t() (tests/test_tied_readout_grad).
    grads["embed.weight"] += _unit_cols_vjp(gV.t(), model.embed.weight, dim=0)

    mean_nll = total_nll / L
    # match bptt_loss scale (F8): deep supervision averages over blocks (1/L); final-only
    # has a single readout. Both average the CE over valid tokens.
    n_valid = max(1, int(valid.sum().item()))
    scale = 1.0 / (n_valid * (L if cfg.deep_supervision else 1))
    for k in grads:
        grads[k].mul_(scale)
    cs = 1.0
    if clip_norm is not None:  # clip RAW gradient BEFORE the gate (audit N5)
        total = (sum((g.detach() ** 2).sum() for g in grads.values()) + 1e-12) ** 0.5
        cs = min(1.0, clip_norm / (float(total) + 1e-12))
    mu = R.surprise_gate(
        torch.tensor(mean_nll), model.Cbar.item(), cfg.gate_g0, cfg.gate_g1, cfg.gate_lo, cfg.gate_hi
    )
    for k in grads:
        grads[k].mul_(cs * mu)
    model.Cbar.mul_(1 - cfg.gate_ema).add_(cfg.gate_ema * mean_nll)
    return mean_nll, grads


# ======================================================================================
# RIPPLE (L1) — S6 FAST PATH: chunked kernels + adjoint reordering + settle upgrades
# ======================================================================================
@torch.no_grad()
def ripple_grads_l1_fast(model: RippleLM, tokens: Tensor, targets: Tensor, *,
                         ignore_index: int = -100, clip_norm: float | None = 1.0,
                         raw: bool = False, want_rows_nll: bool = False):
    """The S6 throughput form of ``ripple_grads_l1`` (docs/10 §6; plan Phase A).

    Identical rule, reorganized computation:
      * forward reads via the chunked scan (§M.1, ``scan.chunked_contextual_read``);
      * free settle with per-row adaptive stopping (§M.3) — its per-row iteration
        counts become the SHARED schedule of the ± nudged settles (§M.4a symmetric
        stopping, ``cleanup.settle_nudged_pair``);
      * the local-CE readout force FROZEN at r_free (§M.4 linearization; one
        row-chunked vocab-matmul family per block per step,
        ``cleanup.frozen_readout_terms``) — set ``cfg.frozen_force=False`` +
        ``cfg.fixed_schedule=True`` to reproduce the legacy nudged settles exactly
        (the equivalence-test mode, tests/test_settle_rowchunk.py);
      * ALL projection/decay gradients + the exact input pullback from the §M.2
        adjoint reordering (``scan.adjoint_grads``) — no Tkv/Gv materialization.

    ``raw=True`` returns ``(mean_nll, grads, n_valid)`` with the gradient scaled by
    1/(L_sup·n_valid) but clip/gate/Cbar DEFERRED — the §M.6 accumulation interface.
    ``raw=False`` applies the legacy tail (clip → gate μ → Cbar EMA), matching
    ``ripple_grads_l1``'s semantics. C2: no autograd over model parameters.
    """
    cfg = model.cfg
    if cfg.delta_rule:
        raise ValueError("the S6 fast path requires pure accumulation "
                         "(delta_rule=False); docs/10 §6 runtime guards")
    H, K, L = cfg.n_heads, cfg.vocab_size, cfg.n_layers
    x = model.embed(tokens)
    B, T, dm = x.shape
    tflat = targets.reshape(B * T)
    valid = (tflat != ignore_index)
    V = model.V()
    grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    gV = torch.zeros_like(V)
    tol = 0.0 if cfg.fixed_schedule else cfg.settle_tol

    # ---- forward pass: chunked reads, per-row free settles, frozen vocab terms ----
    buses = [x]
    a_l, nm_l, inv_l, c_l, r_l, nrow_l, force_l, gVro_l = [], [], [], [], [], [], [], []
    total_nll = 0.0
    for ell in range(L):
        a, normed, inv = _rmsnorm(buses[ell], model.norm_gain[ell])
        c = SC.chunked_contextual_read(
            a, model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega,
            model.gammas(ell), H, L_c=cfg.scan_Lc,
        )
        cflat = c.reshape(B * T, dm)
        M = model.M_used(ell)
        r0, _aM, n_row = C.settle_free_perrow(
            cflat, M, M, cfg.beta, n_max=cfg.n_settle, eta=cfg.eta, tol=tol,
        )
        blk_targets = tflat if (cfg.deep_supervision or ell == L - 1) else None
        force_ro, gVro, nll_rows, n_val_blk = C.frozen_readout_terms(
            r0, V, cfg.beta_readout, blk_targets, valid, row_chunk=cfg.row_chunk,
        )
        if blk_targets is not None and n_val_blk > 0:
            total_nll += float(nll_rows.sum().item()) / n_val_blk
        if ell == L - 1:
            top_nll_rows = nll_rows  # per-row NLL of the model output (replay priorities)
        a_l.append(a); nm_l.append(normed); inv_l.append(inv); c_l.append(c)
        r_l.append(r0); nrow_l.append(n_row); force_l.append(force_ro); gVro_l.append(gVro)
        buses.append(buses[ell] + r0.reshape(B, T, dm))

    # ---- backward sweep top→bottom: frozen-force ± pairs + adjoint gradients ----
    bus_cotan = torch.zeros(B, T, dm, dtype=x.dtype, device=x.device)
    rscale = valid.to(x.dtype).view(-1, 1)
    for ell in reversed(range(L)):
        M = model.M_used(ell)
        cflat = c_l[ell].reshape(B * T, dm)
        g_top = None if ell == L - 1 else bus_cotan.reshape(B * T, dm)
        # total constant force: frozen local readout force (already masked) − g_top
        const = force_l[ell] if g_top is None else force_l[ell] - g_top
        if cfg.frozen_force:
            rp, rn = C.settle_nudged_pair(
                cflat, M, M, cfg.beta, r_free=r_l[ell], beta_n=cfg.beta_n,
                n_row=nrow_l[ell], eta=cfg.eta, const_force=const,
            )
        else:  # live-force mode (legacy-faithful; equivalence tests / A.5 fidelity)
            blk_targets = tflat if (cfg.deep_supervision or ell == L - 1) else None

            def live(ra, idx, _tgt=blk_targets, _gt=g_top):
                f = torch.zeros_like(ra)
                if _tgt is not None:
                    p_ro = torch.softmax(cfg.beta_readout * (ra @ V), dim=-1)
                    onehot = torch.zeros_like(p_ro)
                    onehot.scatter_(1, _tgt[idx].clamp(min=0).view(-1, 1), 1.0)
                    f = cfg.beta_readout * ((onehot - p_ro) @ V.t()) * rscale[idx]
                if _gt is not None:
                    f = f - _gt[idx]
                return f

            rp, rn = C.settle_nudged_pair(
                cflat, M, M, cfg.beta, r_free=r_l[ell], beta_n=cfg.beta_n,
                n_row=nrow_l[ell], eta=cfg.eta, live_force=live,
            )
        delta = (rp - rn) / (2 * cfg.beta_n)
        g_content = (-delta).reshape(B, T, dm)  # NOT masked (L1 semantics, docs/03 §7)
        pMp = torch.softmax(cfg.beta * (rp @ M), dim=-1)
        pMn = torch.softmax(cfg.beta * (rn @ M), dim=-1)
        gM = -(torch.einsum("bv,bp->vp", rp, pMp)
               - torch.einsum("bv,bp->vp", rn, pMn)) / (2 * cfg.beta_n)
        (gWk, gWq, gWv), g_a, ggamma = SC.adjoint_grads(
            a_l[ell], model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega,
            model.gammas(ell), H, g_content, c=c_l[ell], want_gamma=True,
            want_input=True, L_c=cfg.scan_Lc,
        )
        grads[f"Wk.{ell}"] += gWk
        grads[f"Wq.{ell}"] += gWq
        grads[f"Wv.{ell}"] += gWv
        grads[f"M.{ell}"] += _unit_cols_vjp(gM, model.M[ell], dim=0)
        grads[f"log_gamma.{ell}"] += ggamma
        grads[f"norm_gain.{ell}"] += (g_a * nm_l[ell]).reshape(-1, dm).sum(0)
        gV += gVro_l[ell]
        d_normed = g_a * model.norm_gain[ell]
        bus_cotan = bus_cotan + _rmsnorm_vjp(d_normed, buses[ell], inv_l[ell])

    grads["embed.weight"].index_add_(0, tokens.reshape(-1), bus_cotan.reshape(-1, dm))
    grads["embed.weight"] += _unit_cols_vjp(gV.t(), model.embed.weight, dim=0)

    L_sup = L if cfg.deep_supervision else 1
    mean_nll = total_nll / L_sup
    n_valid = max(1, int(valid.sum().item()))
    scale = 1.0 / (n_valid * L_sup)
    for k in grads:
        grads[k].mul_(scale)
    if raw:  # §M.6 accumulation interface: clip/gate/Cbar deferred to the wrapper
        if want_rows_nll:  # top-block per-row NLL (B,T) — replay surprise priorities
            return mean_nll, grads, n_valid, top_nll_rows.view(B, T)
        return mean_nll, grads, n_valid
    cs = 1.0
    if clip_norm is not None:
        total = (sum((g.detach() ** 2).sum() for g in grads.values()) + 1e-12) ** 0.5
        cs = min(1.0, clip_norm / (float(total) + 1e-12))
    mu = R.surprise_gate(
        torch.tensor(mean_nll), model.Cbar.item(), cfg.gate_g0, cfg.gate_g1,
        cfg.gate_lo, cfg.gate_hi,
    )
    for k in grads:
        grads[k].mul_(cs * mu)
    model.Cbar.mul_(1 - cfg.gate_ema).add_(cfg.gate_ema * mean_nll)
    return mean_nll, grads


# ======================================================================================
# RIPPLE (L1-FA): transport-free depth credit via feedback alignment (docs/03 §7, plan S1)
# ======================================================================================
@torch.no_grad()
def ripple_grads_l1fa(model: RippleLM, tokens: Tensor, targets: Tensor, *,
                      ignore_index: int = -100, clip_norm: float | None = 1.0):
    """L1-FA gradient: L1 stack credit with the inter-block routing made WEIGHT-TRANSPORT-FREE.

    Identical to ``ripple_grads_l1`` except the single step where the top-down cotangent is
    pushed back onto the residual bus. L1-exact routes it with the forward-weight transpose
    (``contextual_input_grad``, Site B); L1-FA routes it with **fixed random feedback matrices**
    ``Bk/Bq/Bv`` (``contextual_input_grad_fa``), reading the **cached forward activations**
    q,k,v — so no forward weight ``Wk/Wq/Wv`` appears in the error projection (transport-free by
    construction; tests/test_transport_free.py). Dose-response knob ``cfg.feedback_lambda``
    interpolates ``B_eff=(1-λ)W+λB``: λ=0 ≡ L1-exact (plumbing check), λ=1 ≡ fully transport-free.

    Everything else — the within-block EP settle (already transport-free), the exact projection
    PARAMETER traces, the two-path codebook, the ``_rmsnorm_vjp`` bus accumulation, the gate,
    the scale — is byte-for-byte the L1 path, so any deviation from L1-exact is attributable to
    the feedback substitution alone (plan Scientific Standards). Pure-accumulation only.
    """
    cfg = model.cfg
    if cfg.delta_rule:
        raise ValueError("L1-FA requires the pure-accumulation regime (delta_rule=False)")
    H, K = cfg.n_heads, cfg.vocab_size
    x = model.embed(tokens)
    B, T, dm = x.shape
    tflat = targets.reshape(B * T)
    valid = (tflat != ignore_index)
    V = model.V()
    L = cfg.n_layers
    lam = cfg.feedback_lambda

    grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    gV = torch.zeros_like(V)

    # ---- forward pass: build the bus, cache per-block (a, normed, inv, c, r_free, q, k, v) ----
    buses = [x]
    a_list, normed_list, inv_list, c_list, r_list, qkv_list = [], [], [], [], [], []
    total_nll = 0.0
    for ell in range(L):
        a, normed, inv = _rmsnorm(buses[ell], model.norm_gain[ell])
        c, q, k, v = R.contextual_read(
            a, model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega, model.gammas(ell), H,
            None, return_qkv=True,
        )
        cflat = c.reshape(B * T, dm)
        M = model.M_used(ell)
        r0, _, _ = C.settle(cflat, M, M, cfg.beta, n_max=cfg.n_settle, eta=cfg.eta, adaptive=False)
        a_list.append(a); normed_list.append(normed); inv_list.append(inv)
        c_list.append(cflat); r_list.append(r0); qkv_list.append((q, k, v))
        logits = cfg.beta_readout * (r0 @ V)
        total_nll += F.cross_entropy(logits, tflat, ignore_index=ignore_index).item()
        buses.append(buses[ell] + r0.reshape(B, T, dm))

    # ---- backward sweep top→bottom: FA-routed bus cotangent ----
    bus_cotan = torch.zeros(B, T, dm, dtype=x.dtype, device=x.device)
    for ell in reversed(range(L)):
        M = model.M_used(ell)
        g_top = None if ell == L - 1 else bus_cotan.reshape(B * T, dm)
        blk_targets = tflat if (cfg.deep_supervision or ell == L - 1) else None
        gc, gM, gVro, _ = R.stack_read_ep(
            c_list[ell], M, V, cfg.beta, blk_targets, cfg.beta_n, r_free=r_list[ell],
            g_top=g_top, n_settle=cfg.n_settle, eta=cfg.eta, beta_readout=cfg.beta_readout,
            valid=valid,
        )
        g_content = gc.reshape(B, T, dm)
        # exact projection PARAMETER grads via forward traces (transport-free already)
        (gWk, gWq, gWv), ggamma = R.projection_grads(
            a_list[ell], model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega,
            model.gammas(ell), H, g_content, key_full=cfg.key_full,
            return_input_grad=False, return_gamma=True, delta_alpha=None,
        )
        # inter-block routing via FEEDBACK matrices (the ONLY change vs L1-exact). B_eff mixes
        # in the forward weight for the dose-response; at λ=1 it is model.B* only (no transport).
        q_e, k_e, v_e = qkv_list[ell]
        Bk_e = (1 - lam) * model.Wk[ell] + lam * model.Bk[ell]
        Bq_e = (1 - lam) * model.Wq[ell] + lam * model.Bq[ell]
        Bv_e = (1 - lam) * model.Wv[ell] + lam * model.Bv[ell]
        g_a = R.contextual_input_grad_fa(
            q_e, k_e, v_e, model.gammas(ell), H, g_content, Bk_e, Bq_e, Bv_e,
        )
        grads[f"Wk.{ell}"] += gWk
        grads[f"Wq.{ell}"] += gWq
        grads[f"Wv.{ell}"] += gWv
        grads[f"M.{ell}"] += _unit_cols_vjp(gM, model.M[ell], dim=0)
        grads[f"log_gamma.{ell}"] += ggamma
        grads[f"norm_gain.{ell}"] += (g_a * normed_list[ell]).reshape(-1, dm).sum(0)
        gV += gVro
        d_normed = g_a * model.norm_gain[ell]
        bus_cotan = bus_cotan + _rmsnorm_vjp(d_normed, buses[ell], inv_list[ell])

    grads["embed.weight"].index_add_(0, tokens.reshape(-1), bus_cotan.reshape(-1, dm))
    grads["embed.weight"] += _unit_cols_vjp(gV.t(), model.embed.weight, dim=0)

    mean_nll = total_nll / L
    n_valid = max(1, int(valid.sum().item()))
    scale = 1.0 / (n_valid * (L if cfg.deep_supervision else 1))
    for k in grads:
        grads[k].mul_(scale)
    cs = 1.0
    if clip_norm is not None:
        total = (sum((g.detach() ** 2).sum() for g in grads.values()) + 1e-12) ** 0.5
        cs = min(1.0, clip_norm / (float(total) + 1e-12))
    mu = R.surprise_gate(
        torch.tensor(mean_nll), model.Cbar.item(), cfg.gate_g0, cfg.gate_g1, cfg.gate_lo, cfg.gate_hi
    )
    for k in grads:
        grads[k].mul_(cs * mu)
    model.Cbar.mul_(1 - cfg.gate_ema).add_(cfg.gate_ema * mean_nll)
    return mean_nll, grads


# ======================================================================================
# RIPPLE (L1-VF): space-time Vector-Field EP — credit via a JOINT relaxation (docs/09)
# ======================================================================================
@torch.no_grad()
def ripple_grads_l1vf(model: RippleLM, tokens: Tensor, targets: Tensor, *,
                      ignore_index: int = -100, clip_norm: float | None = 1.0):
    """L1-VF gradient: inter-block credit via a joint relaxation instead of a backward sweep.

    Two single-sign outer relaxations (docs/09 §3, Formulation A) converge the nudged reads
    r^± per block; the downward force on each block is the F1-normalized (÷β_n) content signal
    of the block above, routed through the transport-free F3 operator (``contextual_input_grad_fa``,
    exact temporal Q_t for VF-ST-adjoint). ``vf_rerun`` re-runs ``contextual_read`` in the nudged
    phase so ``c_{ℓ+1}`` responds to ``r_ℓ`` — the space-time cycle that makes it a genuine
    relaxation (G2). δ_ℓ=(r^+_ℓ−r^-_ℓ)/2β_n → the reused δ→parameter-grad pipeline. No autograd (C2).

    Correctness is gated by G1 (``tests/test_spacetime_ep.py``): with exact coupling (λ=0, B=W,
    λ_fb=1, no re-run) it MUST reproduce L1-exact/BPTT at cos=1.0.
    """
    cfg = model.cfg
    H, K, L = cfg.n_heads, cfg.vocab_size, cfg.n_layers
    lam, lfb = cfg.feedback_lambda, cfg.feedback_gain
    x = model.embed(tokens)
    B, T, dm = x.shape
    tflat = targets.reshape(B * T)
    valid = (tflat != ignore_index)
    rscale = valid.to(x.dtype).view(-1, 1)
    V = model.V()
    grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    gV = torch.zeros_like(V)

    def Beff(ell):
        return ((1 - lam) * model.Wk[ell] + lam * model.Bk[ell],
                (1 - lam) * model.Wq[ell] + lam * model.Bq[ell],
                (1 - lam) * model.Wv[ell] + lam * model.Bv[ell])

    def read_block(ell, bus_ell):
        a, normed, inv = _rmsnorm(bus_ell, model.norm_gain[ell])
        c, q, k, v = R.contextual_read(
            a, model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega, model.gammas(ell), H,
            None, return_qkv=True,
        )
        return a, normed, inv, c.reshape(B * T, dm), (q, k, v)

    # ---- free forward pass (the free equilibrium) ----
    buses = [x]
    a_f, nm_f, inv_f, c_f, r_f, qkv_f = [], [], [], [], [], []
    total_nll = 0.0
    for ell in range(L):
        a, normed, inv, cflat, qkv = read_block(ell, buses[ell])
        M = model.M_used(ell)
        r0, _, _ = C.settle(cflat, M, M, cfg.beta, n_max=cfg.n_settle, eta=cfg.eta, adaptive=False)
        a_f.append(a); nm_f.append(normed); inv_f.append(inv); c_f.append(cflat)
        r_f.append(r0); qkv_f.append(qkv)
        total_nll += F.cross_entropy(cfg.beta_readout * (r0 @ V), tflat, ignore_index=ignore_index).item()
        buses.append(buses[ell] + r0.reshape(B, T, dm))

    # ---- single-sign outer relaxation → nudged reads {r^s_ℓ} ----
    def relax(sign):
        r_s = [r_f[ell].clone() for ell in range(L)]
        c_s = list(c_f)
        qkv_s = list(qkv_f)
        for _ in range(cfg.vf_n_outer):
            r_prev = [r.clone() for r in r_s]
            if cfg.vf_rerun:  # space-time cycle: re-run CR with current nudged reads
                bus = x
                for ell in range(L):
                    _, _, _, c_s[ell], qkv_s[ell] = read_block(ell, bus)
                    bus = bus + r_s[ell].reshape(B, T, dm)
            bus_cotan = torch.zeros(B, T, dm, dtype=x.dtype, device=x.device)
            new_r = [None] * L
            for ell in reversed(range(L)):
                M = model.M_used(ell)
                g_top = None if ell == L - 1 else bus_cotan.reshape(B * T, dm)
                tgt = tflat if (cfg.deep_supervision or ell == L - 1) else None
                r_new, _, _ = C.settle(
                    c_s[ell], M, M, cfg.beta, n_max=cfg.n_settle, eta=cfg.eta, adaptive=False,
                    target=(tgt.clamp(min=0) if tgt is not None else None), beta_n=sign * cfg.beta_n,
                    r_init=r_f[ell], readout=V, beta_readout=cfg.beta_readout, g_top=g_top,
                    readout_scale=(rscale if tgt is not None else None),
                )
                new_r[ell] = r_new
                delta_s = (r_new - r_f[ell]) / (sign * cfg.beta_n)  # F1: one-sided, ÷β_n
                q, k, v = qkv_s[ell]
                Bk_e, Bq_e, Bv_e = Beff(ell)
                g_a = R.contextual_input_grad_fa(
                    q, k, v, model.gammas(ell), H, (-delta_s).reshape(B, T, dm), Bk_e, Bq_e, Bv_e
                )
                bus_cotan = bus_cotan + lfb * _rmsnorm_vjp(g_a * model.norm_gain[ell], buses[ell], inv_f[ell])
            for ell in range(L):
                r_s[ell] = (1 - cfg.vf_relax) * r_s[ell] + cfg.vf_relax * new_r[ell]
            if max((r_s[ell] - r_prev[ell]).norm().item() for ell in range(L)) < cfg.vf_tol:
                break
        return r_s

    rp = relax(+1)
    rn = relax(-1)

    # ---- symmetric δ → reuse the δ→parameter-grad pipeline ----
    bus_cotan = torch.zeros(B, T, dm, dtype=x.dtype, device=x.device)
    for ell in reversed(range(L)):
        M = model.M_used(ell)
        delta = (rp[ell] - rn[ell]) / (2 * cfg.beta_n)
        g_content = (-delta).reshape(B, T, dm)
        pMp = torch.softmax(cfg.beta * (rp[ell] @ M), dim=-1)
        pMn = torch.softmax(cfg.beta * (rn[ell] @ M), dim=-1)
        gM = -(torch.einsum("bv,bp->vp", rp[ell], pMp) - torch.einsum("bv,bp->vp", rn[ell], pMn)) / (2 * cfg.beta_n)
        tgt = tflat if (cfg.deep_supervision or ell == L - 1) else None
        if tgt is not None:
            p = torch.softmax(cfg.beta_readout * (r_f[ell] @ V), dim=-1)
            oh = F.one_hot(tgt.clamp(min=0), num_classes=V.shape[1]).to(p.dtype)
            gVro = cfg.beta_readout * torch.einsum("bv,bk->vk", r_f[ell] * rscale, (p - oh))
        else:
            gVro = torch.zeros_like(V)
        (gWk, gWq, gWv), ggamma = R.projection_grads(
            a_f[ell], model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega, model.gammas(ell),
            H, g_content, key_full=cfg.key_full, return_input_grad=False, return_gamma=True,
            delta_alpha=None,
        )
        q, k, v = qkv_f[ell]
        Bk_e, Bq_e, Bv_e = Beff(ell)
        g_a = R.contextual_input_grad_fa(q, k, v, model.gammas(ell), H, g_content, Bk_e, Bq_e, Bv_e)
        grads[f"Wk.{ell}"] += gWk; grads[f"Wq.{ell}"] += gWq; grads[f"Wv.{ell}"] += gWv
        grads[f"M.{ell}"] += _unit_cols_vjp(gM, model.M[ell], dim=0)
        grads[f"log_gamma.{ell}"] += ggamma
        grads[f"norm_gain.{ell}"] += (g_a * nm_f[ell]).reshape(-1, dm).sum(0)
        gV += gVro
        bus_cotan = bus_cotan + lfb * _rmsnorm_vjp(g_a * model.norm_gain[ell], buses[ell], inv_f[ell])

    grads["embed.weight"].index_add_(0, tokens.reshape(-1), bus_cotan.reshape(-1, dm))
    grads["embed.weight"] += _unit_cols_vjp(gV.t(), model.embed.weight, dim=0)

    mean_nll = total_nll / L
    n_valid = max(1, int(valid.sum().item()))
    scale = 1.0 / (n_valid * (L if cfg.deep_supervision else 1))
    for k in grads:
        grads[k].mul_(scale)
    cs = 1.0
    if clip_norm is not None:
        total = (sum((g.detach() ** 2).sum() for g in grads.values()) + 1e-12) ** 0.5
        cs = min(1.0, clip_norm / (float(total) + 1e-12))
    mu = R.surprise_gate(
        torch.tensor(mean_nll), model.Cbar.item(), cfg.gate_g0, cfg.gate_g1, cfg.gate_lo, cfg.gate_hi
    )
    for k in grads:
        grads[k].mul_(cs * mu)
    model.Cbar.mul_(1 - cfg.gate_ema).add_(cfg.gate_ema * mean_nll)
    return mean_nll, grads


@torch.no_grad()
def fast_logits(model: RippleLM, tokens: Tensor) -> Tensor:
    """Final-block logits via the S6 chunked forward (docs/10 §6) — the fast,
    deterministic inference path used by FROZEN-EVAL for the WNN.

    Identical math to ``RippleLM.logits`` (chunked read ≡ loop read,
    tests/test_chunked_scan.py; per-row adaptive settle at tol=cfg.settle_tol is
    deterministic given the weights). Pure accumulation only (guarded in the kernel).
    """
    cfg = model.cfg
    x = model.embed(tokens)
    B, T, dm = x.shape
    V = model.V()
    for ell in range(cfg.n_layers):
        a, _, _ = _rmsnorm(x, model.norm_gain[ell])
        c = SC.chunked_contextual_read(
            a, model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega,
            model.gammas(ell), cfg.n_heads, L_c=cfg.scan_Lc,
        )
        M = model.M_used(ell)
        r0, _, _ = C.settle_free_perrow(
            c.reshape(B * T, dm), M, M, cfg.beta, n_max=cfg.n_settle, eta=cfg.eta,
            tol=cfg.settle_tol,
        )
        x = x + r0.reshape(B, T, dm)
        if ell == cfg.n_layers - 1:
            return cfg.beta_readout * (r0.reshape(B, T, dm) @ V)
    raise AssertionError("unreachable")


@torch.no_grad()
def codebook_utilization(model: RippleLM, tokens: Tensor) -> dict:
    """§M.7 utilization signals per block: normalized read entropy H̄/ln P and the
    pattern participation ratio PR, at the free fixed points (docs/10 §5)."""
    cfg = model.cfg
    x = model.embed(tokens)
    B, T, dm = x.shape
    P = cfg.resolved_P()
    out = {}
    for ell in range(cfg.n_layers):
        a, _, _ = _rmsnorm(x, model.norm_gain[ell])
        c = SC.chunked_contextual_read(
            a, model.Wk[ell], model.Wq[ell], model.Wv[ell], model.omega,
            model.gammas(ell), cfg.n_heads, L_c=cfg.scan_Lc,
        )
        M = model.M_used(ell)
        r0, aM, _ = C.settle_free_perrow(
            c.reshape(B * T, dm), M, M, cfg.beta, n_max=cfg.n_settle, eta=cfg.eta,
            tol=cfg.settle_tol,
        )
        Hbar = float((-(aM * (aM + 1e-30).log()).sum(-1)).mean().item()) / max(1e-12, torch.log(torch.tensor(float(P))).item())
        u = aM.mean(0)  # ū_p
        pr = float((u.sum() ** 2 / (u.pow(2).sum() + 1e-30)).item())
        out[f"util_Hbar_L{ell}"] = Hbar
        out[f"util_PR_L{ell}"] = pr
        x = x + r0.reshape(B, T, dm)
    return out


@torch.no_grad()
def adaptive_settle_stats(model: RippleLM, tokens: Tensor, tol: float = 1e-3):
    """M4 diagnostic (docs/02 §5.1): settle iterations per token vs read entropy.

    Runs the first block's knowledge-memory read with adaptive stopping and returns the
    per-token iteration count and read entropy, to show harder tokens 'think longer'.
    """
    cfg = model.cfg
    x = model.embed(tokens)
    a, _, _ = _rmsnorm(x, model.norm_gain[0])
    da = model.delta_alpha if cfg.delta_rule else None
    c = R.contextual_read(
        a, model.Wk[0], model.Wq[0], model.Wv[0], model.omega, model.gammas(0), cfg.n_heads, da
    )
    B, T, dm = c.shape
    M = model.M_used(0)
    iters, ents = [], []
    for i in range(B * T):
        ci = c.reshape(B * T, dm)[i : i + 1]
        r, p, n = C.settle(ci, M, M, cfg.beta, n_max=cfg.n_settle * 3, eta=cfg.eta,
                           adaptive=True, tol=tol)
        iters.append(n)
        ents.append(-(p * (p + 1e-12).log()).sum().item())
    return torch.tensor(iters, dtype=torch.float), torch.tensor(ents)
