"""Mathematical-identity tests for the FHRR phasor algebra (docs/01 §1, §12).

These lock the load-bearing algebraic properties: binding inverts, binding distributes
over bundling, permutation is order-sensitive and unitary, similarity behaves, and the
FHRR feature map is unit magnitude. Every test names the spec property it verifies.
"""

import math

import pytest
import torch

from wavenn import algebra as A

torch.manual_seed(0)


# --------------------------------------------------------------------------------------
# phi: unit magnitude (docs/01 §1, Appendix A)
# --------------------------------------------------------------------------------------
def test_phi_is_unit_magnitude():
    u = torch.randn(4, 512)
    z = A.phi(u)
    assert z.is_complex()
    assert torch.allclose(z.abs(), torch.ones_like(z.abs()), atol=1e-5)


def test_phi_encodes_angle():
    u = torch.empty(1000).uniform_(-math.pi + 1e-3, math.pi - 1e-3)
    z = A.phi(u)
    # angle recovers u on the principal branch
    assert torch.allclose(z.angle(), u, atol=1e-5)


def test_phi_rejects_complex_input():
    with pytest.raises(TypeError):
        A.phi(torch.randn(4, 8, dtype=torch.complex64))


# --------------------------------------------------------------------------------------
# Binding inverts (docs/01 §1.1): unbind(bind(a,b), b) == a for unit b
# --------------------------------------------------------------------------------------
def test_binding_inverse_identity():
    a = A.random_phasors(8, 1024)
    b = A.random_phasors(8, 1024)
    recovered = A.unbind(A.bind(a, b), b)
    assert torch.allclose(recovered, a, atol=1e-5)


def test_binding_commutes():
    a = A.random_phasors(4, 256)
    b = A.random_phasors(4, 256)
    assert torch.allclose(A.bind(a, b), A.bind(b, a), atol=1e-6)


def test_binding_distributes_over_bundling():
    # (a + b) ⊗ c  ==  a⊗c + b⊗c     (docs/01 §1.1 — the load-bearing property)
    a = A.random_phasors(256)
    b = A.random_phasors(256)
    c = A.random_phasors(256)
    lhs = A.bind(a + b, c)
    rhs = A.bind(a, c) + A.bind(b, c)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_bound_pair_is_dissimilar_to_factors():
    # E[sim(a⊗b, a)] ≈ 0 : a bound pair is unlike its factors (docs/01 §1.1)
    D = 4096
    a = A.random_phasors(D)
    b = A.random_phasors(D)
    ab = A.bind(a, b)
    assert abs(A.similarity(ab, a).item()) < 5.0 / math.sqrt(D)


# --------------------------------------------------------------------------------------
# Bundling is order-invariant (docs/01 §1.2) — WHY order needs rho/bind
# --------------------------------------------------------------------------------------
def test_bundling_is_permutation_invariant():
    xs = A.random_phasors(5, 512)
    perm = torch.randperm(5)
    assert torch.allclose(A.bundle(xs, dim=0), A.bundle(xs[perm], dim=0), atol=1e-5)


def test_bundle_resembles_summands():
    # sim(bundle, summand) > 0 (docs/01 §1.2)
    D = 4096
    xs = A.random_phasors(4, D)
    s = A.bundle(xs, dim=0)
    sims = torch.stack([A.similarity(s, xs[i]) for i in range(4)])
    assert (sims > 0).all()


# --------------------------------------------------------------------------------------
# Permutation: unitary, order-sensitive, relative position (docs/01 §1.3, §11)
# --------------------------------------------------------------------------------------
def test_permutation_is_invertible():
    D = 1024
    omega = A.default_frequencies(D)
    z = A.random_phasors(D)
    assert torch.allclose(A.rho(A.rho(z, 3, omega), -3, omega), z, atol=1e-5)


def test_permutation_is_norm_preserving():
    D = 1024
    omega = A.default_frequencies(D)
    z = A.random_phasors(D)
    assert torch.allclose(z.abs(), A.rho(z, 7, omega).abs(), atol=1e-6)


def test_permutation_is_order_sensitive():
    # rho^i(t) != rho^j(t) for i != j  (docs/01 §1.3, "generic omega"): swapping
    # positions changes the encoding. Generic (random) base frequencies make rho a
    # fully decorrelating permutation. (The RoPE schedule, default_frequencies, rotates
    # only slowly at low frequencies — its relative-position property is tested below.)
    D = 2048
    omega = torch.empty(D).uniform_(-math.pi, math.pi)
    t0 = A.random_phasors(D)
    t1 = A.random_phasors(D)
    ctx_forward = A.rho(t0, 0, omega) + A.rho(t1, 1, omega)
    ctx_reversed = A.rho(t1, 0, omega) + A.rho(t0, 1, omega)
    cos = A.hermitian_inner(ctx_forward, ctx_reversed).real / (
        ctx_forward.abs().square().sum().sqrt() * ctx_reversed.abs().square().sum().sqrt()
    )
    # two distinct orderings of independent tokens are near-orthogonal (distinguishable)
    assert cos.item() < 0.5


def test_relative_position_property():
    # <rho^m q, rho^n k> depends only on n - m  (docs/01 §1.3, RoPE relative position)
    D = 2048
    omega = A.default_frequencies(D)
    q = A.random_phasors(D)
    k = A.random_phasors(D)
    s_02 = A.hermitian_inner(A.rho(q, 0, omega), A.rho(k, 2, omega))
    s_35 = A.hermitian_inner(A.rho(q, 3, omega), A.rho(k, 5, omega))
    assert torch.allclose(s_02, s_35, atol=1e-4)


# --------------------------------------------------------------------------------------
# Similarity + cleanup (docs/01 §1.4)
# --------------------------------------------------------------------------------------
def test_similarity_self_is_one():
    a = A.random_phasors(8, 1024)
    s = A.similarity(a, a)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_similarity_random_is_near_zero_with_1_over_sqrt_D_scale():
    D = 4096
    a = A.random_phasors(2000, D)
    b = A.random_phasors(2000, D)
    sims = A.similarity(a, b)
    assert sims.abs().mean().item() < 5.0 / math.sqrt(D)
    # empirical std should track 1/sqrt(2D)-ish (real part of unit-variance/D)
    assert 0.3 / math.sqrt(D) < sims.std().item() < 3.0 / math.sqrt(D)


def test_hard_cleanup_recovers_codeword():
    D, K = 2048, 64
    codebook = A.random_phasors(K, D)
    idx = 17
    # noisy version of codeword 17: add a bit of another random phasor
    noisy = codebook[idx] + 0.3 * A.random_phasors(D)
    assert A.cleanup(noisy, codebook).item() == idx


def _cos(a, b):
    return (
        A.hermitian_inner(a, b).real / (a.abs().square().sum().sqrt() * b.abs().square().sum().sqrt())
    ).item()


def test_soft_cleanup_moves_toward_codeword():
    # normalized cosine to the clean codeword must increase after cleanup (denoising)
    D, K = 2048, 32
    codebook = A.random_phasors(K, D)
    idx = 5
    noisy = codebook[idx] + 0.5 * A.random_phasors(D)
    recon = A.cleanup(noisy, codebook, soft=True, beta=20.0)
    assert _cos(recon, codebook[idx]) > _cos(noisy, codebook[idx])


# --------------------------------------------------------------------------------------
# Normalization and real/complex round-trips
# --------------------------------------------------------------------------------------
def test_magnitude_rms_norm_unit_rms():
    x = torch.randn(4, 512, dtype=torch.complex64) * 5.0
    y = A.magnitude_rms_norm(x)
    rms = torch.sqrt((y.real**2 + y.imag**2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_realpair_roundtrip():
    z = A.random_phasors(4, 256)
    assert torch.allclose(A.from_realpair(A.to_realpair(z)), z, atol=1e-6)
