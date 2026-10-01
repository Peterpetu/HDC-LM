"""S6 kernel certificates (docs/10 §6, plan §M.1/§M.2 / [P-cert]).

Certifies the chunked kernels in wavenn/scan.py against THREE independent oracles:
  O1  the forward-trace loops (ripple.projection_grads key_full=True + the exact
      reverse adjoint ripple.contextual_input_grad) — themselves autograd-validated;
  O2  reverse-mode autograd through ripple.contextual_read (the independent ground
      truth, including d/dγ);
  O3  a test-local re-implementation of the per-token adjoint recurrence (written
      here, independently of wavenn/scan.py) for the key/value paths.

Test matrix (binding, docs/10 §6): heads with γ ∈ {1e-4, 0.9, 0.999}; T NOT a
multiple of L_c, spanning ≥3 chunks; B > 1; zero/masked g rows. Tolerances:
float64 relative ≤ 1e-9; fp32 relative ≤ 1e-4; fp32 no-NaN at γ→1e-4 over T=1024.
Runtime guards must raise on the delta-rule.
"""

import pytest
import torch

from wavenn import scan
from wavenn.algebra import default_frequencies
from wavenn.ripple import contextual_input_grad, contextual_read, projection_grads

# heads deliberately span the γ extremes (short/medium/long timescales)
GAMMAS = [1e-4, 0.9, 0.999]
H = 3
D_IN = 12          # d_model; d_k = d_vh = 4
B, T, L_C = 2, 50, 16  # 50 = 3·16 + 2 → 4 chunks, last partial (≥3 chunks, T∤L_c)


def _setup(dtype):
    torch.manual_seed(0)
    gamma = torch.tensor(GAMMAS, dtype=dtype)
    x = torch.randn(B, T, D_IN, dtype=dtype)
    Wk = torch.randn(D_IN, D_IN, dtype=dtype) / D_IN**0.5
    Wq = torch.randn(D_IN, D_IN, dtype=dtype) / D_IN**0.5
    Wv = torch.randn(D_IN, D_IN, dtype=dtype) / D_IN**0.5
    omega = default_frequencies(D_IN // H, dtype=dtype)
    g_content = torch.randn(B, T, D_IN, dtype=dtype)
    g_content[0, 7] = 0.0     # zero row (masked position)
    g_content[1, T - 1] = 0.0  # masked last position (the training convention)
    return x, Wk, Wq, Wv, omega, gamma, g_content


def _rel(a, b):
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def _local_adjoint_oracle(x, Wk, Wq, Wv, omega, gamma, g_content):
    """O3: independent per-token adjoint (test-local; NOT scan.py's chunk math).

    Q_i = conj(q_i)(g_i/den_i)ᵀ + γ Q_{i+1};  g_v = Re(kᵀQ);  g_ak = −Im(k⊙(Qv)).
    """
    Bx, Tx, _ = x.shape
    d_k = Wk.shape[0] // H
    d_vh = Wv.shape[0] // H
    cdtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
    pos = torch.arange(Tx, dtype=x.dtype).unsqueeze(-1) * omega
    k = torch.polar(torch.ones(Bx, Tx, H, d_k, dtype=x.dtype),
                    (x @ Wk.t()).view(Bx, Tx, H, d_k) + pos.view(1, Tx, 1, d_k))
    q = torch.polar(torch.ones(Bx, Tx, H, d_k, dtype=x.dtype),
                    (x @ Wq.t()).view(Bx, Tx, H, d_k) + pos.view(1, Tx, 1, d_k))
    v = (x @ Wv.t()).view(Bx, Tx, H, d_vh).to(cdtype)
    g = g_content.view(Bx, Tx, H, d_vh).to(cdtype)
    den = torch.zeros(H, dtype=x.dtype)
    dens = []
    for t in range(Tx):
        den = gamma * den + d_k
        dens.append(den.clone())
    Q = torch.zeros(Bx, H, d_k, d_vh, dtype=cdtype)
    gWk = torch.zeros(H * d_k, D_IN, dtype=x.dtype)
    gWv = torch.zeros(H * d_vh, D_IN, dtype=x.dtype)
    for t in reversed(range(Tx)):
        Q = torch.einsum("bhk,bhv->bhkv", q[:, t].conj(), g[:, t] / dens[t].view(1, H, 1).to(cdtype)) \
            + gamma.view(1, H, 1, 1).to(cdtype) * Q
        g_v_t = torch.einsum("bhk,bhkv->bhv", k[:, t], Q).real
        g_ak_t = -(k[:, t] * torch.einsum("bhkv,bhv->bhk", Q, v[:, t])).imag
        gWk += g_ak_t.reshape(Bx, H * d_k).t().to(x.dtype) @ x[:, t]
        gWv += g_v_t.reshape(Bx, H * d_vh).t() @ x[:, t]
    return gWk, gWv


def _autograd_oracle(x, Wk, Wq, Wv, omega, gamma, g_content):
    """O2: reverse-mode autograd through the reference forward read."""
    Wk = Wk.clone().requires_grad_(True)
    Wq = Wq.clone().requires_grad_(True)
    Wv = Wv.clone().requires_grad_(True)
    gm = gamma.clone().requires_grad_(True)
    xx = x.clone().requires_grad_(True)
    c = contextual_read(xx, Wk, Wq, Wv, omega, gm, H)
    loss = (g_content * c).sum()
    gk, gq, gv, gg, gx = torch.autograd.grad(loss, (Wk, Wq, Wv, gm, xx))
    return gk, gq, gv, gg, gx, c.detach()


@pytest.fixture(autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


def test_chunked_forward_matches_loop_f64():
    x, Wk, Wq, Wv, omega, gamma, _ = _setup(torch.float64)
    c_loop = contextual_read(x, Wk, Wq, Wv, omega, gamma, H)
    c_chunk = scan.chunked_contextual_read(x, Wk, Wq, Wv, omega, gamma, H, L_c=L_C)
    assert _rel(c_chunk, c_loop) <= 1e-9, _rel(c_chunk, c_loop)
    # qkv cache layout matches the loop's
    c2, q2, k2, v2 = scan.chunked_contextual_read(
        x, Wk, Wq, Wv, omega, gamma, H, return_qkv=True, L_c=L_C)
    c1, q1, k1, v1 = contextual_read(x, Wk, Wq, Wv, omega, gamma, H, return_qkv=True)
    for a, b in ((c2, c1), (q2, q1), (k2, k1), (v2, v1)):
        assert _rel(a.resolve_conj() if a.is_complex() else a,
                    b.resolve_conj() if b.is_complex() else b) <= 1e-9


def test_closed_form_den_matches_recurrence():
    gamma = torch.tensor(GAMMAS, dtype=torch.float64)
    d_k = 4
    den_cf, dden_cf = scan.closed_form_den(gamma, T, d_k, torch.float64)
    den = torch.zeros(H, dtype=torch.float64)
    dden = torch.zeros(H, dtype=torch.float64)
    for t in range(T):
        dden = den + gamma * dden  # ḋen_t = den_{t−1} + γ ḋen_{t−1}
        den = gamma * den + d_k
        assert _rel(den_cf[t], den) <= 1e-12
        assert _rel(dden_cf[t], dden) <= 1e-11


def test_adjoint_grads_vs_all_oracles_f64():
    x, Wk, Wq, Wv, omega, gamma, g_content = _setup(torch.float64)
    c = contextual_read(x, Wk, Wq, Wv, omega, gamma, H)
    (gWk, gWq, gWv), g_x, g_lg = scan.adjoint_grads(
        x, Wk, Wq, Wv, omega, gamma, H, g_content, c=c, L_c=L_C)

    # O1a: the forward-trace loop (exact key trace)
    (tWk, tWq, tWv), t_gamma = projection_grads(
        x, Wk, Wq, Wv, omega, gamma, H, g_content, key_full=True,
        return_input_grad=False, return_gamma=True)
    assert _rel(gWk, tWk) <= 1e-9, f"gWk vs trace: {_rel(gWk, tWk):.2e}"
    assert _rel(gWq, tWq) <= 1e-9, f"gWq vs trace: {_rel(gWq, tWq):.2e}"
    assert _rel(gWv, tWv) <= 1e-9, f"gWv vs trace: {_rel(gWv, tWv):.2e}"
    # NOTE: projection_grads' log_gamma output includes the σ chain γ(1−γ) — as does ours.
    assert _rel(g_lg, t_gamma) <= 1e-9, f"g_logγ vs trace: {_rel(g_lg, t_gamma):.2e}"

    # O1b: the exact reverse-adjoint loop (input pullback cross-oracle)
    gx_loop = contextual_input_grad(x, Wk, Wq, Wv, omega, gamma, H, g_content)
    assert _rel(g_x, gx_loop) <= 1e-9, f"g_x vs adjoint loop: {_rel(g_x, gx_loop):.2e}"

    # O2: autograd ground truth (params, γ-chain removed for raw dγ comparison, input)
    aWk, aWq, aWv, a_gamma, a_gx, _ = _autograd_oracle(x, Wk, Wq, Wv, omega, gamma, g_content)
    assert _rel(gWk, aWk) <= 1e-9
    assert _rel(gWq, aWq) <= 1e-9
    assert _rel(gWv, aWv) <= 1e-9
    assert _rel(g_x, a_gx) <= 1e-9
    # autograd differentiates wrt γ directly; ours returns d/d log_gamma = dγ · γ(1−γ)
    assert _rel(g_lg, a_gamma * gamma * (1 - gamma)) <= 1e-9

    # O3: the test-local adjoint re-implementation (key + value paths)
    oWk, oWv = _local_adjoint_oracle(x, Wk, Wq, Wv, omega, gamma, g_content)
    assert _rel(gWk, oWk) <= 1e-9
    assert _rel(gWv, oWv) <= 1e-9


def test_adjoint_grads_fa_projection():
    """Site-B projection matrices route the input pullback (transport-free option)."""
    x, Wk, Wq, Wv, omega, gamma, g_content = _setup(torch.float64)
    gen = torch.Generator().manual_seed(7)
    Bq = torch.randn(D_IN, D_IN, generator=gen, dtype=torch.float64) / D_IN**0.5
    Bk = torch.randn(D_IN, D_IN, generator=gen, dtype=torch.float64) / D_IN**0.5
    Bv = torch.randn(D_IN, D_IN, generator=gen, dtype=torch.float64) / D_IN**0.5
    c = contextual_read(x, Wk, Wq, Wv, omega, gamma, H)
    _, g_x_fa, _ = scan.adjoint_grads(
        x, Wk, Wq, Wv, omega, gamma, H, g_content, c=c, proj=(Bq, Bk, Bv), L_c=L_C)
    # oracle: the loop pullback per-position quantities projected through B.
    # contextual_input_grad_fa reads cached activations; equivalently project the
    # W-based pullback difference — here we verify vs the FA loop directly.
    from wavenn.ripple import contextual_input_grad_fa
    _, q, kk, vv = contextual_read(x, Wk, Wq, Wv, omega, gamma, H, return_qkv=True)
    gx_loop = contextual_input_grad_fa(q, kk, vv, gamma, H, g_content, Bk, Bq, Bv)
    assert _rel(g_x_fa, gx_loop) <= 1e-9


def test_fp32_tolerance_and_no_nan():
    torch.set_default_dtype(torch.float32)
    x, Wk, Wq, Wv, omega, gamma, g_content = _setup(torch.float32)
    c_loop = contextual_read(x, Wk, Wq, Wv, omega, gamma, H)
    c_chunk = scan.chunked_contextual_read(x, Wk, Wq, Wv, omega, gamma, H, L_c=L_C)
    assert _rel(c_chunk, c_loop) <= 1e-4
    (gWk, gWq, gWv), g_x, g_lg = scan.adjoint_grads(
        x, Wk, Wq, Wv, omega, gamma, H, g_content, c=c_loop, L_c=L_C)
    (tWk, tWq, tWv), t_gamma = projection_grads(
        x, Wk, Wq, Wv, omega, gamma, H, g_content, key_full=True,
        return_input_grad=False, return_gamma=True)
    for a, b in ((gWk, tWk), (gWq, tWq), (gWv, tWv), (g_lg, t_gamma)):
        assert _rel(a, b) <= 1e-4, _rel(a, b)
    # long-T no-NaN at the short-γ extreme (underflow-to-zero must be benign)
    torch.manual_seed(1)
    T_long = 1024
    xl = torch.randn(1, T_long, D_IN) * 0.1
    gl = torch.randn(1, T_long, D_IN)
    cl = scan.chunked_contextual_read(xl, Wk, Wq, Wv, omega, gamma, H, L_c=128)
    (a1, a2, a3), gx, glg = scan.adjoint_grads(
        xl, Wk, Wq, Wv, omega, gamma, H, gl, c=cl, L_c=128)
    for t in (cl, a1, a2, a3, gx, glg):
        assert torch.isfinite(t).all()


def test_runtime_guards_delta_rule():
    x, Wk, Wq, Wv, omega, gamma, g_content = _setup(torch.float64)
    da = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="pure-accumulation"):
        scan.chunked_contextual_read(x, Wk, Wq, Wv, omega, gamma, H, delta_alpha=da)
    with pytest.raises(ValueError, match="pure-accumulation"):
        scan.adjoint_grads(x, Wk, Wq, Wv, omega, gamma, H, g_content, c=x, delta_alpha=da)
    with pytest.raises(ValueError, match="want_gamma"):
        scan.adjoint_grads(x, Wk, Wq, Wv, omega, gamma, H, g_content, c=None)
