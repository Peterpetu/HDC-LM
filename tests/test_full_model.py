"""Tests for the full-model mechanisms added in S3b (docs/02–03).

Covers: delta-rule forward correctness, separate knowledge-memory / vocab-readout EP
fidelity (M read-path and V readout-path vs autograd), the three-factor gate, and that
the full ripple_grads produces finite, nonzero gradients for EVERY parameter (no dead /
un-learned tensor — the completeness check the user asked for).
"""

import pytest
import torch

from wavenn import cleanup as C
from wavenn import ripple as R

torch.manual_seed(0)


@pytest.fixture(autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)


def _cos(a, b):
    return (a.flatten() @ b.flatten() / (a.norm() * b.norm() + 1e-30)).item()


# --------------------------------------------------------------------------------------
# Delta-rule forward correctness (docs/01 §5.3, docs/02 §4.1)
# --------------------------------------------------------------------------------------
def test_delta_rule_forward_matches_manual():
    from wavenn import algebra as A

    B, T, d_in, D, d_v, H = 2, 5, 8, 8, 8, 2
    x = torch.randn(B, T, d_in) * 0.5
    Wk = torch.randn(D, d_in) * 0.5
    Wq = torch.randn(D, d_in) * 0.5
    Wv = torch.randn(d_v, d_in) * 0.5
    omega = torch.linspace(0.1, 1.0, D // H)
    gamma = torch.tensor([0.9, 0.7])
    alpha = torch.tensor([0.0, 1.0])  # head 0 pure, head 1 delta-rule
    d_k, d_vh = D // H, d_v // H

    c = R.contextual_read(x, Wk, Wq, Wv, omega, gamma, H, alpha)

    # manual delta recurrence
    pos = torch.arange(T).unsqueeze(-1) * omega
    k = A.phi((x @ Wk.t()).view(B, T, H, d_k) + pos.view(1, T, 1, d_k))
    q = A.phi((x @ Wq.t()).view(B, T, H, d_k) + pos.view(1, T, 1, d_k))
    v = (x @ Wv.t()).view(B, T, H, d_vh)
    S = torch.zeros(B, H, d_k, d_vh, dtype=k.dtype)
    den = torch.zeros(1, H, 1)  # positive count normalizer den_t = γ den_{t-1} + d_k
    out = torch.zeros(B, T, H, d_vh)
    for t in range(T):
        vt = v[:, t].to(k.dtype)
        num_k = torch.einsum("bhk,bhkv->bhv", k[:, t].conj(), S).real
        v_hat = (num_k / (den + 1e-6)).to(k.dtype)  # den holds den_{t-1}
        vt = vt - alpha.view(1, H, 1).to(k.dtype) * v_hat
        S = gamma.view(1, H, 1, 1) * S + torch.einsum("bhk,bhv->bhkv", k[:, t], vt)
        den = gamma.view(1, H, 1) * den + d_k
        num = torch.einsum("bhk,bhkv->bhv", q[:, t].conj(), S).real
        out[:, t] = num / den
    assert torch.allclose(c, out.reshape(B, T, d_v), atol=1e-6)


def test_delta_rule_differs_from_pure():
    x = torch.randn(2, 5, 8) * 0.5
    Wk, Wq, Wv = (torch.randn(8, 8) * 0.5 for _ in range(3))
    omega = torch.linspace(0.1, 1.0, 4)
    gamma = torch.tensor([0.9, 0.7])
    c_pure = R.contextual_read(x, Wk, Wq, Wv, omega, gamma, 2, torch.tensor([0.0, 0.0]))
    c_delta = R.contextual_read(x, Wk, Wq, Wv, omega, gamma, 2, torch.tensor([0.0, 1.0]))
    assert not torch.allclose(c_pure, c_delta)  # delta actually changes the read


# --------------------------------------------------------------------------------------
# Separate knowledge-memory (M) and vocab-readout (V) EP fidelity (docs/02 §5, docs/01 §7-8)
# --------------------------------------------------------------------------------------
def test_separate_codebook_ep_matches_autograd():
    B, d_v, P, K = 8, 12, 20, 16
    c = torch.randn(B, d_v) * 0.5
    M = torch.randn(d_v, P) * (0.5 / d_v**0.5)
    V = torch.randn(d_v, K) * (0.5 / d_v**0.5)
    beta = 0.5 * C.safe_beta_limit(M)
    targets = torch.randint(0, K, (B,))
    beta_n, n = 0.01, 80

    Mg = M.clone().requires_grad_(True)
    Vg = V.clone().requires_grad_(True)
    r, _, _ = C.settle(c, Mg, Mg, beta, n_max=n, eta=0.3, adaptive=False)
    Closs = torch.nn.functional.cross_entropy(beta * (r @ Vg), targets, reduction="sum")
    gM_auto, gV_auto = torch.autograd.grad(Closs, [Mg, Vg])

    ep = R.warm_read_ep(c, M, V, beta=beta, targets=targets, beta_n=beta_n, n_settle=n)
    assert _cos(ep.grad_M, gM_auto) > 0.99, f"grad_M cos={_cos(ep.grad_M, gM_auto):.4f}"
    assert _cos(ep.grad_V_readout, gV_auto) > 0.999, (
        f"grad_V cos={_cos(ep.grad_V_readout, gV_auto):.4f}"
    )
    # EP content signal recovers ∂C/∂c too
    cc = c.clone().requires_grad_(True)
    r2, _, _ = C.settle(cc, Mg.detach(), Mg.detach(), beta, n_max=n, eta=0.3, adaptive=False)
    Closs2 = torch.nn.functional.cross_entropy(beta * (r2 @ V), targets, reduction="sum")
    (gc_auto,) = torch.autograd.grad(Closs2, [cc])
    assert _cos(ep.grad_c, gc_auto) > 0.99


# --------------------------------------------------------------------------------------
# Three-factor gate (docs/03 §5)
# --------------------------------------------------------------------------------------
def test_surprise_gate():
    # at the running mean, μ = g0; above it μ rises; clipped to [lo,hi]
    assert abs(R.surprise_gate(torch.tensor(2.0), 2.0, 1.0, 0.5, 0.1, 3.0) - 1.0) < 1e-9
    hi = R.surprise_gate(torch.tensor(5.0), 2.0, 1.0, 0.5, 0.1, 3.0)
    lo = R.surprise_gate(torch.tensor(0.0), 2.0, 1.0, 0.5, 0.1, 3.0)
    assert hi > 1.0 and lo < 1.0
    assert R.surprise_gate(torch.tensor(100.0), 2.0, 1.0, 0.5, 0.1, 3.0) == 3.0  # clipped hi
    assert R.surprise_gate(torch.tensor(-100.0), 2.0, 1.0, 0.5, 0.1, 3.0) == 0.1  # clipped lo


# --------------------------------------------------------------------------------------
# Full ripple_grads: every parameter gets a finite, nonzero gradient (completeness)
# --------------------------------------------------------------------------------------
def test_ignore_index_masking():
    # masked (ignore_index) positions must contribute NO loss gradient (masked next-token /
    # MQAR support): all-masked ⇒ zero gradient; masking a position changes the gradient.
    from wavenn.rlm import RLMConfig, RippleLM, ripple_grads

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    cfg = RLMConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, P=64, beta=1.0, n_settle=8)
    model = RippleLM(cfg)
    x = torch.randint(0, 64, (3, 12))
    y = torch.randint(0, 64, (3, 12))

    _, g_full = ripple_grads(model, x, y, clip_norm=None)
    y_masked = y.clone()
    y_masked[:, 5] = -100  # mask position 5 in every sequence
    _, g_masked = ripple_grads(model, x, y_masked, clip_norm=None)
    # removing position 5 from the loss must change the assembled gradient
    assert not torch.allclose(g_full["Wv.0"], g_masked["Wv.0"], atol=1e-9)

    # masking everything ⇒ exactly zero gradient (no supervised positions)
    _, g_zero = ripple_grads(model, x, torch.full((3, 12), -100), clip_norm=None)
    for n, g in g_zero.items():
        assert g.abs().max().item() < 1e-9, f"{n}: nonzero gradient with all positions masked"


def test_full_ripple_grads_all_params_learn():
    from wavenn.rlm import RLMConfig, RippleLM, ripple_grads

    torch.set_default_dtype(torch.float32)
    cfg = RLMConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, P=48, beta=3.0, n_settle=8)
    model = RippleLM(cfg)
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    nll, grads = ripple_grads(model, x, y)
    param_names = {n for n, _ in model.named_parameters()}
    assert set(grads) == param_names, "grads must cover every parameter"
    for n, g in grads.items():
        assert torch.isfinite(g).all(), f"non-finite grad {n}"
        assert g.abs().sum().item() > 0, f"DEAD gradient (no learning signal) for {n}"
