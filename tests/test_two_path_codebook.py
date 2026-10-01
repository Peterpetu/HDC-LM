"""Two-path codebook regression (docs/01 §8, whitepaper D.3) — locks the correction.

The codebook V enters the per-step loss by two paths: the read path (V shapes the
settled r*) and the readout path (V appears explicitly in softmax(β Vᵀ r*)). Read-only
trains in the WRONG direction (whitepaper D.3: cos ≈ 0.74, loss rises); read + readout
is exact (cos = 1.0). This test is a permanent CI gate — if it ever regresses, the
learning rule is silently broken.
"""

import torch

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


def _autograd_grad_V(c, V, beta, targets, n_settle):
    V = V.clone().requires_grad_(True)
    r, _, _ = C.settle(c, V, V, beta, n_max=n_settle, eta=0.3, adaptive=False)
    logits = beta * (r @ V)
    Closs = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
    (gV,) = torch.autograd.grad(Closs, [V])
    return gV


def test_two_path_matches_autograd_and_read_only_does_not():
    B, d_v, K = 8, 12, 20
    c = torch.randn(B, d_v) * 0.5
    V = torch.randn(d_v, K) * (0.5 / d_v**0.5)
    beta = 0.5 * C.safe_beta_limit(V)  # safe regime ⇒ unique read
    targets = torch.randint(0, K, (B,))
    beta_n = 0.01
    n_settle = 80

    ep = R.warm_read_ep(c, V, beta=beta, targets=targets, beta_n=beta_n, n_settle=n_settle)
    grad_auto = _autograd_grad_V(c, V, beta, targets, n_settle)

    cos_read_only = _cos(ep.grad_V_read, grad_auto)
    cos_both = _cos(ep.grad_V_read + ep.grad_V_readout, grad_auto)

    # read + readout is exact; read-only is materially worse (whitepaper D.3)
    assert cos_both > 0.99, f"read+readout cos={cos_both:.4f}"
    assert cos_read_only < 0.8, f"read-only cos={cos_read_only:.4f} (should be far from 1)"
    assert cos_both > cos_read_only + 0.1


def _loss(c, V, beta, targets, n_settle):
    r, _, _ = C.settle(c, V, V, beta, n_max=n_settle, eta=0.3, adaptive=False)
    return torch.nn.functional.cross_entropy(beta * (r @ V), targets, reduction="mean").item()


def test_read_only_update_is_broken_vs_two_path():
    # First-order check (robust): the two-path update is the exact gradient, so a small
    # descent step reduces the loss; the read-only update is badly misaligned (cos ≈ 0.2),
    # so the same-size step reduces the loss far less. This is the operational form of
    # whitepaper D.3 without depending on the divergence knife-edge.
    B, d_v, K = 16, 12, 24
    torch.manual_seed(3)
    c = torch.randn(B, d_v) * 0.5
    V = torch.randn(d_v, K) * (0.5 / d_v**0.5)
    beta = 0.5 * C.safe_beta_limit(V)
    targets = torch.randint(0, K, (B,))
    lr, n_settle = 0.02, 60

    L0 = _loss(c, V, beta, targets, n_settle)
    ep = R.warm_read_ep(c, V, beta=beta, targets=targets, beta_n=0.01, n_settle=n_settle)
    g_both = ep.grad_V_read + ep.grad_V_readout
    g_read = ep.grad_V_read
    L_both = _loss(c, V - lr * g_both, beta, targets, n_settle)
    L_read = _loss(c, V - lr * g_read, beta, targets, n_settle)
    assert L_both < L0, f"two-path (exact grad) must descend ({L0:.4f}→{L_both:.4f})"
    assert (L0 - L_both) > 3 * (L0 - L_read), (
        f"two-path reduction {L0 - L_both:.4f} should dominate read-only {L0 - L_read:.4f}"
    )


def test_readout_path_alone_is_the_dominant_direction():
    # sanity: the explicit readout path is the larger contributor to the true gradient
    B, d_v, K = 6, 10, 16
    c = torch.randn(B, d_v) * 0.5
    V = torch.randn(d_v, K) * (0.5 / d_v**0.5)
    beta = 0.5 * C.safe_beta_limit(V)
    targets = torch.randint(0, K, (B,))
    ep = R.warm_read_ep(c, V, beta=beta, targets=targets, beta_n=0.01, n_settle=80)
    grad_auto = _autograd_grad_V(c, V, beta, targets, 80)
    assert _cos(ep.grad_V_readout, grad_auto) > _cos(ep.grad_V_read, grad_auto)
