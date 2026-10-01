"""Gradient-fidelity kill gate (docs/05 P1) — the program's decisive test.

Two levels:
  (a) the EP content signal δ_t recovers ∂C/∂c_t as β_n→0 (whitepaper D.2, cos→1);
  (b) the FULL single-layer RIPPLE update (EP δ → exact traces for W_{k,q,v},
      two-path codebook for V) has cosine > 0.9 with the truncated-BPTT gradient, and
      →1 as β_n→0.

If (b) fails after the isolated pieces (test_traces, test_two_path_codebook) pass, the
local rule does not track the gradient and the program's central mechanism is refuted
(docs/05 §5, P1 kill criterion).
"""

import torch
import torch.nn.functional as F

from wavenn import cleanup as C
from wavenn import ripple as R

torch.manual_seed(0)
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _use_float64():
    # tight autograd-comparison tolerance needs double precision; conftest
    # restores the default dtype after each test so forward tests stay float32.
    torch.set_default_dtype(torch.float64)



def _cos(a, b):
    return (a.flatten() @ b.flatten() / (a.norm() * b.norm() + 1e-30)).item()


# --------------------------------------------------------------------------------------
# (a) EP content signal recovers ∂C/∂c  (whitepaper D.2, docs/01 §7)
# --------------------------------------------------------------------------------------
def test_ep_content_signal_recovers_gradient():
    B, d_v, K = 12, 16, 24
    c = (torch.randn(B, d_v) * 0.5).requires_grad_(True)
    V = torch.randn(d_v, K) * (0.5 / d_v**0.5)
    beta = 0.5 * C.safe_beta_limit(V)
    targets = torch.randint(0, K, (B,))

    r, _, _ = C.settle(c, V, V, beta, n_max=80, eta=0.3, adaptive=False)
    Closs = F.cross_entropy(beta * (r @ V), targets, reduction="sum")
    (grad_c_auto,) = torch.autograd.grad(Closs, [c])

    prev = None
    for bn in (0.2, 0.05, 0.01):
        ep = R.warm_read_ep(c.detach(), V, beta=beta, targets=targets, beta_n=bn, n_settle=80)
        cos = _cos(ep.grad_c, grad_c_auto)  # grad_c = ∂C/∂c = −δ
        assert cos > 0.9, f"beta_n={bn}: cos={cos:.4f}"
        if prev is not None:
            assert cos >= prev - 1e-6  # improves (or holds) as β_n shrinks
        prev = cos
    assert prev > 0.999  # →1 in the small-nudge limit


# --------------------------------------------------------------------------------------
# (b) Full single-layer RIPPLE vs truncated BPTT  (the kill gate)
# --------------------------------------------------------------------------------------
def _build(B=3, T=6, d_in=12, D=12, d_v=12, H=3, K=20):
    x = torch.randn(B, T, d_in) * 0.5
    Wk = torch.randn(D, d_in) * 0.4
    Wq = torch.randn(D, d_in) * 0.4
    Wv = torch.randn(d_v, d_in) * 0.4
    V = torch.randn(d_v, K) * (0.5 / d_v**0.5)
    omega = torch.linspace(0.1, 1.0, D // H)
    gamma = torch.linspace(0.5, 0.9, H)
    y = torch.randint(0, K, (B, T))
    return x, Wk, Wq, Wv, V, omega, gamma, H, y


def _bptt_grads(x, Wk, Wq, Wv, V, omega, gamma, H, y, beta, n_settle):
    Wk = Wk.clone().requires_grad_(True)
    Wq = Wq.clone().requires_grad_(True)
    Wv = Wv.clone().requires_grad_(True)
    V = V.clone().requires_grad_(True)
    c = R.contextual_read(x, Wk, Wq, Wv, omega, gamma, H)  # (B,T,d_v)
    B, T, d_v = c.shape
    cflat = c.reshape(B * T, d_v)
    r, _, _ = C.settle(cflat, V, V, beta, n_max=n_settle, eta=0.3, adaptive=False)
    Closs = F.cross_entropy(beta * (r @ V), y.reshape(B * T), reduction="sum")
    return torch.autograd.grad(Closs, [Wk, Wq, Wv, V])


def _ripple_grads(x, Wk, Wq, Wv, V, omega, gamma, H, y, beta, beta_n, n_settle):
    B, T, _ = x.shape
    c = R.contextual_read(x, Wk, Wq, Wv, omega, gamma, H)
    d_v = V.shape[0]
    cflat = c.reshape(B * T, d_v)
    ep = R.warm_read_ep(cflat, V, beta=beta, targets=y.reshape(B * T), beta_n=beta_n, n_settle=n_settle)
    g_content = ep.grad_c.reshape(B, T, d_v)  # ∂C/∂c = −δ
    gWk, gWq, gWv = R.projection_grads(x, Wk, Wq, Wv, omega, gamma, H, g_content)
    gV = ep.grad_V_read + ep.grad_V_readout
    return gWk, gWq, gWv, gV


def _ratio(a, b):
    return (a.norm() / (b.norm() + 1e-30)).item()


def test_full_ripple_matches_bptt_kill_gate():
    x, Wk, Wq, Wv, V, omega, gamma, H, y = _build()
    beta = 0.5 * C.safe_beta_limit(V)
    n_settle = 80
    auto = _bptt_grads(x, Wk, Wq, Wv, V, omega, gamma, H, y, beta, n_settle)
    names = ["Wk", "Wq", "Wv", "V"]

    overalls = []
    for beta_n in (0.1, 0.03, 0.01):  # assert EVERY β_n, and that cos improves toward 1
        rip = _ripple_grads(x, Wk, Wq, Wv, V, omega, gamma, H, y, beta, beta_n, n_settle)
        per = {n: _cos(r, a) for n, r, a in zip(names, rip, auto)}
        cat_r = torch.cat([g.flatten() for g in rip])
        cat_a = torch.cat([g.flatten() for g in auto])
        overall = _cos(cat_r, cat_a)
        for n, r, a in zip(names, rip, auto):
            # P1 kill criterion: direction AND magnitude both match (docs/05 P1, H3)
            assert per[n] > 0.9, f"{n} cos={per[n]:.4f} at beta_n={beta_n}"
            assert 0.5 < _ratio(r, a) < 2.0, f"{n} ratio={_ratio(r, a):.3f} at beta_n={beta_n}"
        overalls.append(overall)
    # monotone improvement toward 1 as the nudge shrinks (the β_n→0 limit)
    assert overalls[-1] >= overalls[0] - 1e-6
    assert overalls[-1] > 0.99, f"overall cos={overalls[-1]:.4f} at smallest nudge"
