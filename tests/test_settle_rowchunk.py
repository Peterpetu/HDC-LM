"""S6 settle-upgrade certificates (docs/10 §6; plan §M.3/§M.4a + Phase A.3).

Locks four properties of the S6 fast path:
  1. FULL-ASSEMBLY EQUIVALENCE: ``ripple_grads_l1_fast`` in legacy-faithful mode
     (fixed_schedule=True + frozen_force=False — fixed n_settle, live readout force)
     reproduces ``ripple_grads_l1`` on every parameter to float64 rel ≤ 1e-9. This
     certifies the entire reorganized pipeline (chunked read, per-row settle
     machinery, row-chunked vocab terms, adjoint gradients, tail) against the
     committed, autograd-validated legacy assembly.
  2. ROW-CHUNK EXACTNESS (§M.3): the frozen-force path is invariant to row_chunk.
  3. SYMMETRIC STOPPING (§M.4a): the ± nudged settles execute the IDENTICAL per-row
     iteration schedule (asserted via an instrumented force callback).
  4. Per-row free settle with tol=0 ≡ the legacy fixed-n settle.
"""

import pytest
import torch

from wavenn import cleanup as C
from wavenn.rlm import RLMConfig, RippleLM, ripple_grads_l1, ripple_grads_l1_fast


@pytest.fixture(autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


def _cfg(**over):
    base = dict(
        vocab_size=20, d_model=12, n_layers=2, n_heads=3, P=24, beta=0.4,
        beta_readout=3.0, beta_n=0.02, n_settle=25, eta=0.3, key_full=True,
        deep_supervision=True, depth_mode="L1", scan_Lc=4, row_chunk=7,
    )
    base.update(over)
    return RLMConfig(**base)


def _batch(seed=100):
    torch.manual_seed(seed)
    tok = torch.randint(0, 20, (2, 10))
    tgt = torch.randint(0, 20, (2, 10))
    tgt[0, 3] = -100  # masked rows exercise the valid/rscale path
    tgt[1, 9] = -100
    return tok, tgt


def _rel(a, b):
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def test_fast_legacy_mode_equals_ripple_grads_l1():
    tok, tgt = _batch()
    torch.manual_seed(0)
    m1 = RippleLM(_cfg(fixed_schedule=True, frozen_force=False)).double()
    torch.manual_seed(0)
    m2 = RippleLM(_cfg(fixed_schedule=True, frozen_force=False)).double()
    nll_l, g_l = ripple_grads_l1(m1, tok, tgt, clip_norm=None)
    nll_f, g_f = ripple_grads_l1_fast(m2, tok, tgt, clip_norm=None)
    assert abs(nll_l - nll_f) <= 1e-9 * max(1.0, abs(nll_l))
    for n in g_l:
        r = _rel(g_f[n], g_l[n])
        assert r <= 1e-9, f"{n}: fast(legacy-mode) vs legacy rel={r:.2e}"


def test_fast_legacy_mode_final_only_grads_match():
    # final-only: lower blocks trained purely by routed credit. Gradient equality
    # still holds; the logged mean_nll differs by design (legacy logs every block's
    # readout NLL, the fast path only supervised blocks') — documented, not asserted.
    tok, tgt = _batch(7)
    torch.manual_seed(0)
    m1 = RippleLM(_cfg(deep_supervision=False, fixed_schedule=True, frozen_force=False)).double()
    torch.manual_seed(0)
    m2 = RippleLM(_cfg(deep_supervision=False, fixed_schedule=True, frozen_force=False)).double()
    # neutralize the gate difference caused by the different logged NLL definitions:
    # compare RAW grads (clip/gate deferred), scaled identically.
    _, g_l = ripple_grads_l1(m1, tok, tgt, clip_norm=None)
    mu_l = 1.0  # legacy applied gate with clip_norm=None ⇒ grads *= mu; undo via Cbar
    _, g_f, _ = ripple_grads_l1_fast(m2, tok, tgt, clip_norm=None, raw=True)
    # legacy applied its μ gate; recompute it from m1's telemetry to unscale.
    # Simpler and exact: both models started with Cbar=0 ⇒ μ_legacy = clip(g0+g1·nll, lo, hi).
    # Instead of reconstructing μ, compare DIRECTIONS (cosine == 1 to 1e-12) and
    # per-parameter norm RATIOS being constant across parameters.
    ratios = []
    for n in g_l:
        a, b = g_f[n].flatten(), g_l[n].flatten()
        if b.norm() < 1e-18:
            assert a.norm() < 1e-15, f"{n}: zero-vs-nonzero"
            continue
        cos = float(a @ b / (a.norm() * b.norm()))
        assert cos > 1 - 1e-10, f"{n}: cos={cos}"
        ratios.append(float(a.norm() / b.norm()))
    ratios = torch.tensor(ratios)
    assert (ratios.max() - ratios.min()) / ratios.mean() < 1e-9  # one global scalar (the gate)


def test_frozen_force_row_chunk_invariance():
    tok, tgt = _batch(3)
    grads = []
    for rc in (3, 10_000):
        torch.manual_seed(0)
        m = RippleLM(_cfg(row_chunk=rc)).double()
        _, g, _ = ripple_grads_l1_fast(m, tok, tgt, clip_norm=None, raw=True)
        grads.append(g)
    for n in grads[0]:
        r = _rel(grads[0][n], grads[1][n])
        assert r <= 1e-12, f"{n}: row-chunk changed the gradient (rel={r:.2e})"


def test_symmetric_stopping_identical_schedules():
    torch.manual_seed(0)
    N, d, P = 16, 6, 11
    c = torch.randn(N, d)
    M = torch.nn.functional.normalize(torch.randn(d, P), dim=0)
    r0, _, n_row = C.settle_free_perrow(c, M, M, 0.5, n_max=12, eta=0.3, tol=1e-4)
    assert n_row.min() >= 1 and n_row.max() <= 12
    seen = {+1: [], -1: []}
    sign_holder = [+1]

    def probe(ra, idx):
        seen[sign_holder[0]].append(idx.clone())
        return torch.zeros_like(ra)

    # settle_nudged_pair runs +1 fully, then −1: flip the recorder when the first
    # sign has executed its n_total iterations.
    n_total = int(n_row.max())
    calls = []

    def probe2(ra, idx):
        calls.append(idx.clone())
        return torch.zeros_like(ra)

    C.settle_nudged_pair(c, M, M, 0.5, r_free=r0, beta_n=0.02, n_row=n_row,
                         eta=0.3, live_force=probe2)
    per_sign = len(calls) // 2
    assert len(calls) == 2 * per_sign
    for a, b in zip(calls[:per_sign], calls[per_sign:]):
        assert torch.equal(a, b), "± phases visited different row sets (§M.4a violation)"
    # and the schedule follows n_row: iteration it visits exactly rows with n_row > it
    it = 0
    for idx in calls[:per_sign]:
        expect = (n_row > it).nonzero().squeeze(-1)
        assert torch.equal(idx.sort().values, expect.sort().values)
        it += 1


def test_perrow_tol0_equals_fixed_settle():
    torch.manual_seed(1)
    N, d, P = 9, 6, 11
    c = torch.randn(N, d)
    M = torch.nn.functional.normalize(torch.randn(d, P), dim=0)
    r_fix, a_fix, _ = C.settle(c, M, M, 0.5, n_max=15, eta=0.3, adaptive=False)
    r_pr, a_pr, n_row = C.settle_free_perrow(c, M, M, 0.5, n_max=15, eta=0.3, tol=0.0)
    assert torch.equal(n_row, torch.full_like(n_row, 15))
    assert _rel(r_pr, r_fix) <= 1e-12 and _rel(a_pr, a_fix) <= 1e-12
