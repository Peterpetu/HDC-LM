"""Fixed-point tests for the Warm Read (docs/01 §6.2, resolves OP1).

Verifies the settle converges, that in the safe regime β < 2/σ_max(M_key)² the fixed
point is unique (same r* from different initializations), and that safe_beta_limit is a
correct convexity threshold.
"""

import torch

from wavenn import cleanup as C

torch.manual_seed(0)


def _codebook(d_v: int, P: int, scale: float) -> torch.Tensor:
    return torch.randn(d_v, P) * scale


def test_settle_converges():
    d_v, P, B = 32, 64, 8
    M = _codebook(d_v, P, scale=0.5 / d_v**0.5)
    beta = min(4.0, 0.5 * C.safe_beta_limit(M))
    c = torch.randn(B, d_v)
    r, _, n = C.settle(c, M, M, beta=beta, n_max=100, tol=1e-6)
    # residual of the fixed-point equation r* = c + M softmax(β Mᵀ r*)
    a = torch.softmax(beta * (r @ M), dim=-1)
    resid = (r - (c + a @ M.t())).norm(dim=-1).max().item()
    assert resid < 1e-3
    assert n < 100  # adaptive stop kicked in


def test_fixed_point_is_unique_in_safe_regime():
    # from very different initializations, the safe-regime read lands in the same basin
    d_v, P, B = 24, 40, 4
    M = _codebook(d_v, P, scale=0.5 / d_v**0.5)
    beta = 0.5 * C.safe_beta_limit(M)  # strictly inside the safe regime
    c = torch.randn(B, d_v)
    r1, _, _ = C.settle(c, M, M, beta=beta, n_max=200, tol=1e-7, r_init=c)
    r2, _, _ = C.settle(c, M, M, beta=beta, n_max=200, tol=1e-7, r_init=c + 5 * torch.randn_like(c))
    r3, _, _ = C.settle(c, M, M, beta=beta, n_max=200, tol=1e-7, r_init=-3 * torch.randn_like(c))
    assert torch.allclose(r1, r2, atol=1e-3)
    assert torch.allclose(r1, r3, atol=1e-3)


def test_safe_beta_limit_value():
    d_v, P = 16, 20
    M = _codebook(d_v, P, scale=1.0)
    sig = torch.linalg.matrix_norm(M, ord=2).item()
    assert abs(C.safe_beta_limit(M) - 2.0 / sig**2) < 1e-4


def test_convergence_across_beta_grid():
    # docs/01 §6.2: verify convergence across a β-grid and that the safe-limit predicts
    # the boundary — below it the read is uniquely convergent from any init.
    d_v, P, B = 24, 40, 4
    M = _codebook(d_v, P, scale=0.5 / d_v**0.5)
    limit = C.safe_beta_limit(M)
    c = torch.randn(B, d_v)
    for frac in (0.1, 0.3, 0.6, 0.9):  # β strictly inside the safe regime
        beta = frac * limit
        r1, _, _ = C.settle(c, M, M, beta, n_max=300, tol=1e-7, r_init=c)
        r2, _, _ = C.settle(c, M, M, beta, n_max=300, tol=1e-7, r_init=c + 5 * torch.randn_like(c))
        assert torch.allclose(r1, r2, atol=1e-3), f"β={frac}·limit: non-unique in safe regime"


def test_strong_content_clamp_restores_uniqueness_above_limit():
    # docs/01 §6.2(b): the safe bound 2/σ_max² is SUFFICIENT not necessary. Above it the
    # bare energy can be multi-basin, but a strong content clamp c makes the read effectively
    # convex around the retrieved value, restoring a single-valued fixed point. This is the
    # regime the trained model operates in (β_settle above the conservative bound).
    d_v, P, B = 24, 40, 3
    M = torch.nn.functional.normalize(_codebook(d_v, P, 1.0), dim=0)  # unit columns (as trained)
    limit = C.safe_beta_limit(M)  # ≈ small (σ_max ≥ 1)
    beta = 3.0 * limit  # deliberately ABOVE the sufficient bound
    strong_c = 6.0 * torch.randn(B, d_v)  # strong content clamp
    r1, _, _ = C.settle(strong_c, M, M, beta, n_max=400, tol=1e-8, r_init=strong_c)
    r2, _, _ = C.settle(strong_c, M, M, beta, n_max=400, tol=1e-8,
                        r_init=strong_c + 3 * torch.randn_like(strong_c))
    # with a strong clamp, distinct inits still converge together (basin selected by c)
    assert torch.allclose(r1, r2, atol=1e-2), "strong clamp should select a single basin above limit"


def test_convexity_below_limit():
    # below the limit the Hopfield energy Hessian I − β V Σ(p) Vᵀ is PD at the fixed point
    d_v, P = 20, 30
    M = _codebook(d_v, P, scale=0.5)
    beta = 0.5 * C.safe_beta_limit(M)
    c = torch.randn(1, d_v)
    r, p, _ = C.settle(c, M, M, beta=beta, n_max=200, tol=1e-7)
    p = p.squeeze(0)
    Sigma = torch.diag(p) - torch.outer(p, p)  # covariance of the categorical
    H = torch.eye(d_v) - beta * (M @ Sigma @ M.t())
    eigs = torch.linalg.eigvalsh(H)
    assert eigs.min().item() > 0  # strictly convex ⇒ unique minimum
