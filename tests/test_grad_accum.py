"""S6 gradient-accumulation certificates (docs/10 §6 M.6; plan A.4).

Locks: (1) the token-weighted accumulation of raw micro-gradients equals the
single-call union-batch gradient exactly (the §M.6 identity, float64 ≤1e-9 — the
batch dimension is independent end-to-end); (2) raw=False equals raw=True + the
manually-applied legacy tail (clip → gate μ → Cbar EMA) — i.e. the deferred tail is
exactly the legacy tail; (3) accumulate_step's optimizer update equals the manual
union-batch update (parameters bit-compared).
"""

import pytest
import torch

from wavenn.ripple import LocalAdam, surprise_gate
from wavenn.rlm import RLMConfig, RippleLM, ripple_grads_l1_fast
from wavenn.train import accumulate_step


@pytest.fixture(autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


def _cfg(**over):
    base = dict(
        vocab_size=20, d_model=12, n_layers=2, n_heads=3, P=24, beta=0.4,
        beta_readout=3.0, beta_n=0.02, n_settle=12, eta=0.3, key_full=True,
        deep_supervision=True, depth_mode="L1", scan_Lc=4, row_chunk=5,
    )
    base.update(over)
    return RLMConfig(**base)


def _batches(seed=11):
    torch.manual_seed(seed)
    xs, ys = [], []
    for b in range(3):  # 3 micro-batches of B=2, T=8; unequal valid counts
        x = torch.randint(0, 20, (2, 8))
        y = torch.randint(0, 20, (2, 8))
        y[0, b] = -100  # different masking per micro ⇒ unequal n_valid weights
        xs.append(x)
        ys.append(y)
    return xs, ys


def _rel(a, b):
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def test_accumulation_equals_union_batch():
    xs, ys = _batches()
    torch.manual_seed(0)
    m = RippleLM(_cfg()).double()
    # union batch: concatenate along the (independent) batch dimension
    xu, yu = torch.cat(xs, 0), torch.cat(ys, 0)
    _, g_union, n_union = ripple_grads_l1_fast(m, xu, yu, clip_norm=None, raw=True)
    # micro accumulation (token-weighted average of scaled raw grads)
    acc, n_tot = None, 0
    for x, y in zip(xs, ys):
        _, g_mu, n_mu = ripple_grads_l1_fast(m, x, y, clip_norm=None, raw=True)
        if acc is None:
            acc = {k: g_mu[k] * float(n_mu) for k in g_mu}
        else:
            for k in acc:
                acc[k].add_(g_mu[k], alpha=float(n_mu))
        n_tot += n_mu
    assert n_tot == n_union
    for k in acc:
        acc[k].div_(n_tot)
        r = _rel(acc[k], g_union[k])
        assert r <= 1e-9, f"{k}: accumulation vs union rel={r:.2e}"


def test_raw_plus_tail_equals_nonraw():
    xs, ys = _batches(5)
    x, y = xs[0], ys[0]
    torch.manual_seed(0)
    m1 = RippleLM(_cfg()).double()
    torch.manual_seed(0)
    m2 = RippleLM(_cfg()).double()
    cfg = m1.cfg
    nll_r, g_r, _ = ripple_grads_l1_fast(m1, x, y, clip_norm=None, raw=True)
    # manual legacy tail on the raw grads (clip → gate → Cbar EMA)
    clip = 1.0
    total = (sum((g.detach() ** 2).sum() for g in g_r.values()) + 1e-12) ** 0.5
    cs = min(1.0, clip / (float(total) + 1e-12))
    mu = surprise_gate(torch.tensor(nll_r), m1.Cbar.item(), cfg.gate_g0, cfg.gate_g1,
                       cfg.gate_lo, cfg.gate_hi)
    for k in g_r:
        g_r[k].mul_(cs * mu)
    m1.Cbar.mul_(1 - cfg.gate_ema).add_(cfg.gate_ema * nll_r)
    nll_n, g_n = ripple_grads_l1_fast(m2, x, y, clip_norm=clip)
    assert abs(nll_r - nll_n) <= 1e-12
    assert torch.equal(m1.Cbar, m2.Cbar)
    for k in g_r:
        assert _rel(g_r[k], g_n[k]) <= 1e-12, k


def test_accumulate_step_equals_manual_union_update():
    xs, ys = _batches(9)
    xu, yu = torch.cat(xs, 0), torch.cat(ys, 0)
    torch.manual_seed(0)
    m1 = RippleLM(_cfg()).double()
    torch.manual_seed(0)
    m2 = RippleLM(_cfg()).double()
    o1 = LocalAdam(dict(m1.named_parameters()), lr=1e-3)
    o2 = LocalAdam(dict(m2.named_parameters()), lr=1e-3)
    # path 1: the accumulation wrapper over the micros
    accumulate_step(m1, ripple_grads_l1_fast,
                    o1, list(zip(xs, ys)), clip_norm=1.0)
    # path 2: single union-batch call with the legacy tail + step
    nll, g = ripple_grads_l1_fast(m2, xu, yu, clip_norm=1.0)
    o2.step(g, clip=None)
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        r = _rel(p1, p2)
        assert r <= 1e-9, f"{n1}: wrapper vs union update rel={r:.2e}"
    assert abs(m1.Cbar.item() - m2.Cbar.item()) <= 1e-12
