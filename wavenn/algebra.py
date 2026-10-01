"""FHRR phasor algebra — the Vector-Symbolic primitives (docs/01 §1).

Implements the four operations the architecture is built on:

    binding      (bind / unbind)   — complex elementwise product / conjugate product
    bundling     (bundle)          — complex sum (superposition)
    permutation  (rho / positional)— unitary phase rotation (RoPE-as-primitive, docs/01 §11)
    similarity + cleanup            — normalized Hermitian real part; nearest-codeword

Every function documents its spec section, dimensionality, and the property it must
satisfy (verified in tests/test_algebra.py, tests/test_capacity.py).

Conventions (docs/01 §0):
    * Phasors live in C^D, realized as torch.complex64.
    * Atomic codewords are UNIT magnitude; states (bundles) carry amplitude (docs/01 §2).
    * The last tensor dimension is the hypervector dimension D unless stated otherwise.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "phi",
    "bind",
    "unbind",
    "bundle",
    "positional_phasor",
    "rho",
    "similarity",
    "hermitian_inner",
    "cleanup",
    "magnitude_rms_norm",
    "random_phasors",
    "to_realpair",
    "from_realpair",
]

# Numerical floor used wherever a magnitude appears in a denominator (docs/04 §6).
_EPS = 1e-6


# --------------------------------------------------------------------------------------
# The FHRR feature map  (docs/01 §1, Appendix A: phi(u)_j = e^{i u_j}, |phi|=1)
# --------------------------------------------------------------------------------------
def phi(u: Tensor) -> Tensor:
    """Map a real projection to unit-magnitude phasors: ``phi(u)_j = exp(i u_j)``.

    Args:
        u: real tensor ``(..., D)`` — the phase angles (e.g. ``W_k x``).

    Returns:
        complex tensor ``(..., D)`` with every component on the unit circle.

    Property (tested): ``abs(phi(u)) == 1`` and ``angle(phi(u)) == wrap(u)``.
    """
    if u.is_complex():
        raise TypeError("phi expects a real tensor of phase angles, got complex")
    # exp(i u) = cos u + i sin u, computed in the input's real dtype then viewed complex.
    return torch.polar(torch.ones_like(u), u)


# --------------------------------------------------------------------------------------
# Binding  (docs/01 §1.1): componentwise complex product; unbinding by conjugate.
# --------------------------------------------------------------------------------------
def bind(a: Tensor, b: Tensor) -> Tensor:
    """Binding ``(a ⊗ b)_j = a_j b_j`` (docs/01 §1.1).

    For unit phasors this is phase addition. Commutative, associative, distributes over
    :func:`bundle`. Both inputs complex ``(..., D)``.
    """
    _assert_complex(a, "bind:a")
    _assert_complex(b, "bind:b")
    return a * b


def unbind(a: Tensor, b: Tensor) -> Tensor:
    """Unbinding ``a ⊗ b^{-1}``. For unit phasors ``b^{-1}_j = conj(b_j)`` (docs/01 §1.1).

    Property (tested): ``unbind(bind(a, b), b) == a`` for unit-magnitude ``b``.
    """
    _assert_complex(a, "unbind:a")
    _assert_complex(b, "unbind:b")
    return a * b.conj()


# --------------------------------------------------------------------------------------
# Bundling  (docs/01 §1.2): superposition by complex sum. NOT unit magnitude.
# --------------------------------------------------------------------------------------
def bundle(x: Tensor, dim: int = -2) -> Tensor:
    """Bundle a stack of phasors by summation over ``dim`` (docs/01 §1.2).

    The result carries amplitude (the superposition weights live in the magnitudes),
    which is exactly why the *state* leaves the unit torus (docs/01 §2). Order-invariant
    by construction — order must come from :func:`rho`/:func:`bind`, not bundling.
    """
    _assert_complex(x, "bundle")
    return x.sum(dim=dim)


# --------------------------------------------------------------------------------------
# Permutation / position  (docs/01 §1.3, §11): unitary phase rotation = RoPE primitive.
# --------------------------------------------------------------------------------------
def positional_phasor(positions: Tensor, omega: Tensor) -> Tensor:
    """Positional phasor ``p_t = phi(t * omega)`` (docs/01 §3, §11).

    Args:
        positions: real tensor ``(...,)`` of positions ``t`` (may be integer-valued).
        omega:     real tensor ``(D,)`` of base frequencies.

    Returns:
        complex ``(..., D)`` with ``p_t[j] = exp(i * t * omega_j)``.
    """
    positions = positions.to(omega.dtype)
    angles = positions.unsqueeze(-1) * omega  # (..., D)
    return phi(angles)


def rho(z: Tensor, k: int | Tensor, omega: Tensor) -> Tensor:
    """Apply the ``k``-fold permutation ``rho^k(z)_j = z_j e^{i k omega_j}`` (docs/01 §1.3).

    Unitary and invertible (``rho^{-k} rho^k = id``); order-sensitive for generic omega.
    ``k`` may be an int (same shift for all) or a tensor broadcastable to ``z[..., 0]``.
    """
    _assert_complex(z, "rho:z")
    if isinstance(k, int):
        k = z.new_tensor(float(k)).real
    p = positional_phasor(k if k.ndim else k.reshape(()), omega)
    return z * p


def default_frequencies(D: int, base: float = 10000.0, device=None, dtype=torch.float32) -> Tensor:
    """RoPE-style log-spaced base frequencies ``omega_j`` (docs/01 §11; Su et al. 2021).

    ``omega_j = base^{-2 floor(j/2)/D}`` reproduces the rotary schedule so that WNN's
    positional binding subsumes RoPE as its real-domain shadow.
    """
    idx = torch.arange(D, device=device, dtype=dtype)
    return base ** (-2.0 * torch.floor(idx / 2) / D)


# --------------------------------------------------------------------------------------
# Similarity and cleanup  (docs/01 §1.4)
# --------------------------------------------------------------------------------------
def hermitian_inner(a: Tensor, b: Tensor, dim: int = -1) -> Tensor:
    """Hermitian inner product ``<a, b> = sum_j conj(a_j) b_j`` (complex, docs/01 §0)."""
    _assert_complex(a, "hermitian_inner:a")
    _assert_complex(b, "hermitian_inner:b")
    return (a.conj() * b).sum(dim=dim)


def similarity(a: Tensor, b: Tensor, dim: int = -1) -> Tensor:
    """Cosine-like similarity ``(1/D) Re<a, b>`` (docs/01 §1.4).

    For unit phasors this is the mean phase coherence ``(1/D) sum_j cos(alpha_j - beta_j)``
    in ``[-1, 1]``. Returns a real tensor with the ``dim`` axis reduced.

    Property (tested): ``similarity(a, a) == 1`` for unit ``a``; ``E[similarity]≈0`` for
    independent random phasors, with std ``~1/sqrt(D)`` (blessing of dimensionality).
    """
    D = a.shape[dim]
    return hermitian_inner(a, b, dim=dim).real / D


def cleanup(y: Tensor, codebook: Tensor, *, soft: bool = False, beta: float = 1.0) -> Tensor:
    """Nearest-codeword cleanup (docs/01 §1.4).

    Args:
        y:        query phasor(s) ``(..., D)``.
        codebook: ``(K, D)`` stored codewords.
        soft:     if False, return hard ``argmax`` index/indices; if True, return the
                  softmax-weighted reconstruction ``sum_k softmax(beta*sim)_k codebook_k``.
        beta:     inverse temperature for the soft read.

    Returns:
        hard: integer index tensor ``(...,)``. soft: complex reconstruction ``(..., D)``.

    This is the discrete/one-shot ancestor of the modern-Hopfield settling read
    implemented in wavenn/cleanup.py (docs/01 §6); kept here for the algebra tests.
    """
    _assert_complex(y, "cleanup:y")
    _assert_complex(codebook, "cleanup:codebook")
    D = y.shape[-1]
    # sims: (..., K) = Re< y, codebook_k > / D
    sims = torch.einsum("...d,kd->...k", y.conj(), codebook).real / D
    if not soft:
        return sims.argmax(dim=-1)
    w = torch.softmax(beta * sims, dim=-1)  # (..., K)
    return torch.einsum("...k,kd->...d", w.to(codebook.dtype), codebook)


# --------------------------------------------------------------------------------------
# Magnitude normalization for the bundling bus  (docs/02 §6)
# --------------------------------------------------------------------------------------
def magnitude_rms_norm(x: Tensor, weight: Tensor | None = None, dim: int = -1) -> Tensor:
    """Phasor-native RMS normalization: ``x / sqrt(mean_j |x_j|^2)`` (docs/02 §6).

    Keeps the bundling-bus magnitude inside the SNR budget (docs/01 §4) — a correctness
    requirement, not a convenience. Works for real or complex ``x``. ``weight`` (real,
    shape ``(D,)``) is an optional per-component gain.
    """
    if x.is_complex():
        ms = (x.real**2 + x.imag**2).mean(dim=dim, keepdim=True)
    else:
        ms = (x**2).mean(dim=dim, keepdim=True)
    out = x / torch.sqrt(ms + _EPS)
    if weight is not None:
        out = out * weight
    return out


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def random_phasors(*shape: int, device=None, generator: torch.Generator | None = None) -> Tensor:
    """i.i.d. random unit phasors with phases ``~ U[-pi, pi)`` (docs/01 §1)."""
    theta = torch.empty(*shape, device=device).uniform_(-math.pi, math.pi, generator=generator)
    return phi(theta)


def to_realpair(z: Tensor) -> Tensor:
    """Complex ``(..., D)`` -> real ``(..., 2D)`` interleaved [re, im] (docs/01 §0)."""
    _assert_complex(z, "to_realpair")
    return torch.view_as_real(z).flatten(-2)


def from_realpair(x: Tensor) -> Tensor:
    """Inverse of :func:`to_realpair`: real ``(..., 2D)`` -> complex ``(..., D)``."""
    return torch.view_as_complex(x.unflatten(-1, (-1, 2)).contiguous())


def _assert_complex(x: Tensor, who: str) -> None:
    if not x.is_complex():
        raise TypeError(f"{who} expects a complex tensor, got dtype {x.dtype}")
