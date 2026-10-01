"""Exact eligibility-trace tests (docs/01 §9) — the enabling theorem.

Verifies that the forward, O(D·d_in) eligibility traces for W_v, W_q, W_k compute the
SAME gradient as reverse-mode autograd through the linear recurrence, to floating-point
tolerance. This is the hard test behind the claim that local sequence credit assignment
is exact (docs/01 §9 Theorem, docs/05 P1). If it fails, the traces have a bug (typically
a sign in the phase-domain query/key paths).
"""

import torch

from wavenn import ripple as R

torch.manual_seed(0)
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _use_float64():
    # tight autograd-comparison tolerance needs double precision; conftest
    # restores the default dtype after each test so forward tests stay float32.
    torch.set_default_dtype(torch.float64)



def _setup(B=2, T=6, d_in=10, D=12, d_v=8, H=2):
    x = torch.randn(B, T, d_in)
    Wk = torch.randn(D, d_in) * 0.5
    Wq = torch.randn(D, d_in) * 0.5
    Wv = torch.randn(d_v, d_in) * 0.5
    omega = torch.linspace(0.1, 1.0, D // H)
    gamma = torch.linspace(0.5, 0.95, H)  # one decay per head
    g = torch.randn(B, T, d_v)  # arbitrary content-space cotangent
    return x, Wk, Wq, Wv, omega, gamma, H, g


def _autograd_grads(x, Wk, Wq, Wv, omega, gamma, H, g):
    Wk = Wk.clone().requires_grad_(True)
    Wq = Wq.clone().requires_grad_(True)
    Wv = Wv.clone().requires_grad_(True)
    c = R.contextual_read(x, Wk, Wq, Wv, omega, gamma, H)
    L = (g * c).sum()
    gWk, gWq, gWv = torch.autograd.grad(L, [Wk, Wq, Wv])
    return gWk, gWq, gWv


def _cos(a, b):
    return (a.flatten() @ b.flatten() / (a.norm() * b.norm() + 1e-30)).item()


def test_value_trace_matches_autograd():
    x, Wk, Wq, Wv, omega, gamma, H, g = _setup()
    _, _, gWv_auto = _autograd_grads(x, Wk, Wq, Wv, omega, gamma, H, g)
    _, _, gWv = R.projection_grads(x, Wk, Wq, Wv, omega, gamma, H, g)
    assert torch.allclose(gWv, gWv_auto.to(gWv.dtype), atol=1e-8, rtol=1e-5)


def test_query_trace_matches_autograd():
    x, Wk, Wq, Wv, omega, gamma, H, g = _setup()
    _, gWq_auto, _ = _autograd_grads(x, Wk, Wq, Wv, omega, gamma, H, g)
    _, gWq, _ = R.projection_grads(x, Wk, Wq, Wv, omega, gamma, H, g)
    assert torch.allclose(gWq, gWq_auto.to(gWq.dtype), atol=1e-8, rtol=1e-5), (
        f"cos={_cos(gWq, gWq_auto):.6f}"
    )


def test_key_trace_matches_autograd():
    x, Wk, Wq, Wv, omega, gamma, H, g = _setup()
    gWk_auto, _, _ = _autograd_grads(x, Wk, Wq, Wv, omega, gamma, H, g)
    gWk, _, _ = R.projection_grads(x, Wk, Wq, Wv, omega, gamma, H, g)
    assert torch.allclose(gWk, gWk_auto.to(gWk.dtype), atol=1e-8, rtol=1e-5), (
        f"cos={_cos(gWk, gWk_auto):.6f}"
    )


def test_all_projection_traces_together():
    x, Wk, Wq, Wv, omega, gamma, H, g = _setup(B=3, T=8, d_in=16, D=16, d_v=12, H=4)
    gWk_a, gWq_a, gWv_a = _autograd_grads(x, Wk, Wq, Wv, omega, gamma, H, g)
    gWk, gWq, gWv = R.projection_grads(x, Wk, Wq, Wv, omega, gamma, H, g)
    for got, ref, name in [(gWk, gWk_a, "Wk"), (gWq, gWq_a, "Wq"), (gWv, gWv_a, "Wv")]:
        assert _cos(got, ref.to(got.dtype)) > 0.999999, f"{name} cos={_cos(got, ref):.6f}"
        assert torch.allclose(got, ref.to(got.dtype), atol=1e-7, rtol=1e-4), name


def test_decay_trace_matches_autograd():
    # learned-timescale gradient (docs/03 §4.5): dL/d log_gamma via the dotS/dotz traces
    x, Wk, Wq, Wv, omega, gamma, H, g = _setup()
    log_gamma = (torch.log(gamma) - torch.log1p(-gamma)).clone().requires_grad_(True)
    gam = torch.sigmoid(log_gamma)
    c = R.contextual_read(x, Wk, Wq, Wv, omega, gam, H)
    L = (g * c).sum()
    (glog_auto,) = torch.autograd.grad(L, [log_gamma])
    (_, _, _), _g_x, glog = R.projection_grads(
        x, Wk, Wq, Wv, omega, gamma, H, g, return_input_grad=True, return_gamma=True
    )
    assert torch.allclose(glog, glog_auto.to(glog.dtype), atol=1e-7, rtol=1e-4), (
        f"cos={_cos(glog, glog_auto):.6f}"
    )
