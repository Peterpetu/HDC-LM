"""Regression: the tied-readout → embedding Jacobian (docs/01 §8, F4).

The vocabulary codebook is tied to the embedding: ``V() = _unit_cols(E).t()`` where
``_unit_cols`` normalizes over dim=0. The backprop-free readout gradient must push ∂L/∂V back
to E through the *matching* L2-normalization Jacobian (dim=0). A dim mismatch (the original
dim=1) silently biases the embedding's readout gradient by ~55% — it slips past
test_gradient_fidelity because that test treats V as a free parameter, never the tie. This
locks the tie against autograd. Found by the L1 stack-EP exactness check.
"""

import torch

from wavenn.rlm import _unit_cols, _unit_cols_vjp


def _cos(a, b):
    return (a.flatten() @ b.flatten() / (a.norm() * b.norm() + 1e-30)).item()


def test_tied_readout_embedding_jacobian_matches_autograd():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    K, d = 11, 7
    E = torch.randn(K, d, requires_grad=True)
    V = _unit_cols(E).t()  # (d, K) — exactly what RippleLM.V() computes
    gV = torch.randn(d, K)
    L = (gV * V).sum()
    (gE_auto,) = torch.autograd.grad(L, [E])

    gE = _unit_cols_vjp(gV.t(), E.detach(), dim=0)  # the shipped path
    assert _cos(gE, gE_auto) > 1 - 1e-10
    assert (gE - gE_auto).norm() / gE_auto.norm() < 1e-8

    # guard against a silent regression to the wrong axis
    gE_wrong = _unit_cols_vjp(gV.t(), E.detach(), dim=1)
    assert _cos(gE_wrong, gE_auto) < 0.95
    torch.set_default_dtype(torch.float32)
