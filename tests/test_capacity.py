"""Capacity / interference tests — the √(D/M) wall (docs/01 §4).

Verifies the retrieval-SNR theorem empirically:
    * bundling: signal = D, noise power E[|N|^2] = (M-1)·D, so SNR = √(D/(M-1)).
    * key–value: the normalized read recovers the queried value, with interference
      shrinking as √(M/D).
These are the quantitative guarantees the memory design relies on (docs/01 §12).
"""

import math

import torch

from wavenn import algebra as A

torch.manual_seed(1)


def _bundle_noise_power(D: int, M: int, trials: int) -> tuple[float, float]:
    """Return (mean signal, mean |noise|^2) for querying a bundle of M random phasors."""
    signal = 0.0
    noise_power = 0.0
    for _ in range(trials):
        cb = A.random_phasors(M, D)  # (M, D)
        s = A.bundle(cb, dim=0)  # superposition
        inner = A.hermitian_inner(cb[0], s)  # <c_1, s> = D + N
        signal += inner.real.item()
        noise = inner - D  # N = sum_{i>=2} <c_1, c_i>
        noise_power += (noise.conj() * noise).real.item()
    return signal / trials, noise_power / trials


def test_bundling_signal_is_D():
    D, M, trials = 1024, 20, 200
    signal, _ = _bundle_noise_power(D, M, trials)
    assert abs(signal - D) < 0.02 * D  # signal is deterministic = D


def test_bundling_noise_power_matches_theory():
    # E[|N|^2] = (M-1) D  (docs/01 §4.1)
    D, trials = 1024, 400
    for M in (5, 10, 40):
        _, npow = _bundle_noise_power(D, M, trials)
        theory = (M - 1) * D
        rel = abs(npow - theory) / theory
        assert rel < 0.15, f"M={M}: measured {npow:.0f} vs theory {theory} (rel {rel:.3f})"


def test_bundling_snr_scaling():
    # SNR = signal/√E[|N|^2] = √(D/(M-1))  (docs/01 eq 4.1)
    D, M, trials = 2048, 25, 400
    signal, npow = _bundle_noise_power(D, M, trials)
    snr = signal / math.sqrt(npow)
    theory = math.sqrt(D / (M - 1))
    assert abs(snr - theory) / theory < 0.15


def test_snr_improves_with_dimension():
    # doubling D multiplies SNR by √2 at fixed M (blessing of dimensionality)
    M, trials = 20, 300
    s1, n1 = _bundle_noise_power(512, M, trials)
    s2, n2 = _bundle_noise_power(2048, M, trials)
    snr1 = s1 / math.sqrt(n1)
    snr2 = s2 / math.sqrt(n2)
    assert 1.7 < snr2 / snr1 < 2.3  # theory: √(2048/512) = 2


def test_key_value_read_recovers_value():
    # S = sum_i k_i v_i^T ; read q=k_1 -> Re(q^† S)/D ≈ v_1 up to O(√(M/D)) interference
    D, d_v, M = 4096, 8, 12
    keys = A.random_phasors(M, D)
    vals = torch.randn(M, d_v)
    # S: (D, d_v) complex
    S = torch.einsum("md,mv->dv", keys, vals.to(keys.dtype))
    q = keys[0]
    read = torch.einsum("d,dv->v", q.conj(), S).real / D  # (d_v,)
    err = (read - vals[0]).norm() / vals[0].norm()
    assert err < 0.3  # interference ~ √(M/D) = √(12/4096) ≈ 0.054 per component


def test_effective_context_participation_number():
    # M_eff = (1+γ)/(1-γ) (docs/01 eq 4.2) is the participation number of a decayed
    # superposition with weights w_i = γ^{t-i}: N = (Σ w_i)² / Σ w_i². Verify the closed
    # form matches the formula (this is the quantity that sets the effective context).
    for gamma in (0.5, 0.8, 0.9, 0.99):
        T = 2000  # γ^T ≈ 0 so the finite sum equals the infinite-series limit
        w = torch.tensor([gamma**k for k in range(T)], dtype=torch.float64)
        n_eff = (w.sum() ** 2 / (w**2).sum()).item()
        n_eff_theory = (1 + gamma) / (1 - gamma)
        assert abs(n_eff - n_eff_theory) / n_eff_theory < 1e-3, (
            f"γ={gamma}: participation {n_eff:.3f} vs (1+γ)/(1-γ) {n_eff_theory:.3f}"
        )


def test_key_value_interference_shrinks_with_D():
    d_v, M, trials = 4, 10, 100
    def mean_err(D):
        tot = 0.0
        for _ in range(trials):
            keys = A.random_phasors(M, D)
            vals = torch.randn(M, d_v)
            S = torch.einsum("md,mv->dv", keys, vals.to(keys.dtype))
            read = torch.einsum("d,dv->v", keys[0].conj(), S).real / D
            tot += ((read - vals[0]).norm() / vals[0].norm()).item()
        return tot / trials
    e_small = mean_err(512)
    e_large = mean_err(4096)
    # error ~ √(M/D): 8x more D -> ~√8 ≈ 2.8x smaller error
    assert e_large < e_small
    assert e_small / e_large > 2.0
