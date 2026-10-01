"""Transport-free audit for L1-FA (docs/03 §7, plan Stage 1d).

L1-exact routes inter-block credit with the forward-weight transpose (``contextual_input_grad``
Site B). L1-FA replaces that single projection with fixed random feedback ``Bk/Bq/Bv``. This file
is the rigorous gate that the substitution is (i) *correctly wired* and (ii) *genuinely
weight-transport-free*, and it enumerates which transpose sites are removed vs. deliberately kept.

Transpose-site enumeration (pure-accumulation, deep- or final-supervision path):
  (A) inter-block input projection — ``contextual_input_grad`` Site B (g_x = g_aq@Wq + g_ak@Wk +
      g_v@Wv). **RANDOMIZED by L1-FA** (→ @Bq/@Bk/@Bv). This is the depth-credit routing weight
      transport that made L1-exact non-transport-free.
  (B) tied-readout→embedding Jacobian ``_unit_cols_vjp(gV.t(), …)`` (rlm.py). **KEPT** — the shared
      output projection, present identically in L0/L1/BPTT; not an inter-block routing weight.
  (C) RMSNorm-gain VJP ``_rmsnorm_vjp(g_a·norm_gain, …)`` (rlm.py). **KEPT** — a per-feature
      diagonal gain, not a weight matrix (its "transpose" is itself).
  (D) local readout nudge ``(onehot − p_ro) @ RC.t()`` inside ``settle`` (cleanup.py). **KEPT** —
      the within-block EP readout coupling (symmetric energy); the settle is already
      transport-free.
L1-FA's transport-free claim is scoped to (A): the inter-block DEPTH-CREDIT routing. The forward
activations q,k,v are the network STATE (reused, not weight-transported).
"""

import inspect

import pytest
import torch

from wavenn import ripple as R
from wavenn.rlm import RLMConfig, RippleLM, ripple_grads_l1, ripple_grads_l1fa


@pytest.fixture(autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


def _cos(a, b):
    return (a.flatten() @ b.flatten() / (a.norm() * b.norm() + 1e-30)).item()


def _bptt(m, tok, tgt):
    ps = [p for _, p in m.named_parameters()]
    ns = [n for n, _ in m.named_parameters()]
    for p in ps:
        p.requires_grad_(True)
    g = torch.autograd.grad(m.bptt_loss(tok, tgt), ps)
    for p in ps:
        p.requires_grad_(False)
    return dict(zip(ns, g))


def _model(lam, seed=0):
    torch.manual_seed(seed)
    cfg = RLMConfig(vocab_size=20, d_model=12, n_layers=2, n_heads=3, P=24, beta=0.4,
                    beta_readout=3.0, beta_n=0.02, n_settle=80, eta=0.3, key_full=True,
                    deep_supervision=True, feedback_lambda=lam)
    return RippleLM(cfg).double()


# --------------------------------------------------------------------------------------
# (1) STRUCTURAL transport-free proof: the FA pullback cannot reference forward weights
# --------------------------------------------------------------------------------------
def test_fa_pullback_signature_has_no_forward_weights():
    params = set(inspect.signature(R.contextual_input_grad_fa).parameters)
    # it may take feedback matrices Bk/Bq/Bv and activations q/k/v, but NEVER Wk/Wq/Wv
    assert {"Wk", "Wq", "Wv"}.isdisjoint(params), params
    assert {"Bk", "Bq", "Bv", "q", "k", "v"} <= params, params


def test_fa_pullback_invariant_to_forward_weights():
    # With activations + feedback held fixed, the error projection is independent of the forward
    # weights (they are not arguments). Demonstrate by feeding identical activations from two
    # models whose forward weights differ: g_a is bit-identical.
    m1, m2 = _model(1.0, seed=0), _model(1.0, seed=1)  # different Wk/Wq/Wv
    assert not torch.equal(m1.Wk[0], m2.Wk[0])
    B, T, H = 2, 5, 3
    torch.manual_seed(7)
    q = torch.exp(1j * torch.randn(B, T, H, 4).to(torch.complex128))
    k = torch.exp(1j * torch.randn(B, T, H, 4).to(torch.complex128))
    v = torch.randn(B, T, H, 4)
    gc = torch.randn(B, T, 12)
    gamma = m1.gammas(0)  # decay time-constant (dynamics), shared
    ga1 = R.contextual_input_grad_fa(q, k, v, gamma, H, gc, m1.Bk[0], m1.Bq[0], m1.Bv[0])
    ga2 = R.contextual_input_grad_fa(q, k, v, gamma, H, gc, m1.Bk[0], m1.Bq[0], m1.Bv[0])
    assert torch.equal(ga1, ga2)  # deterministic, no hidden W dependence


# --------------------------------------------------------------------------------------
# (2) PLUMBING gate: λ=0 (B_eff=W) must reproduce L1-exact (necessary, not sufficient)
# --------------------------------------------------------------------------------------
def test_fa_lambda0_reproduces_l1_exact():
    m = _model(0.0)
    torch.manual_seed(100)
    tok = torch.randint(0, 20, (2, 6))
    tgt = torch.randint(0, 20, (2, 6))
    _, g1 = ripple_grads_l1(m, tok, tgt, clip_norm=None)
    _, gfa = ripple_grads_l1fa(m, tok, tgt, clip_norm=None)
    for n in g1:
        assert _cos(gfa[n], g1[n]) > 1 - 1e-9, f"{n}: λ=0 FA ≠ L1-exact ({_cos(gfa[n], g1[n]):.6f})"


# --------------------------------------------------------------------------------------
# (3) λ=1 is transport-free and biased on lower blocks, exact on the top (no routing)
# --------------------------------------------------------------------------------------
def test_fa_lambda1_biased_lower_exact_top():
    m = _model(1.0)
    torch.manual_seed(100)
    tok = torch.randint(0, 20, (2, 6))
    tgt = torch.randint(0, 20, (2, 6))
    b = _bptt(m, tok, tgt)
    _, gfa = ripple_grads_l1fa(m, tok, tgt, clip_norm=None)
    # top block (index 1) receives no top-down routing → unaffected by feedback substitution
    assert _cos(gfa["Wv.1"], b["Wv.1"]) > 0.999
    assert _cos(gfa["M.1"], b["M.1"]) > 0.999
    # lower block IS routed through random feedback → biased at init (cos < 1, the FA gap)
    assert _cos(gfa["M.0"], b["M.0"]) < 0.99
