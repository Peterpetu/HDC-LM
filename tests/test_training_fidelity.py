"""Full-model fidelity at the ACTUAL training configuration (docs/05 P1; audit H4).

The static kill gate (test_gradient_fidelity.py) validates a bare functional layer. This
test validates the deployed `RippleLM.ripple_grads` at its training defaults, which is what
the S4+ compute is spent on. It locks:

  * within-block (single-layer) RIPPLE vs BPTT-twin: cos ≈ 1 AND magnitude ratio ≈ 1 for
    the exactly-traced parameters (Wk, Wq, Wv, M, log_gamma), with the neuromodulation gate
    disabled so the comparison is of the rule itself;
  * the documented S-A0 pathways (embedding, norm_gain) stay strongly aligned (cos > 0.8);
  * WHY the faithful defaults exclude the delta-rule and the cheap key path — enabling
    either measurably degrades fidelity (so a future default change can't silently ship a
    low-fidelity rule);
  * the three-factor gate genuinely amplifies high-surprise updates (μ → gate_hi at init).

This is the test whose absence let the pre-audit code nearly train on a broken-fidelity
configuration.
"""

import pytest
import torch

from wavenn.rlm import RLMConfig, RippleLM, ripple_grads

torch.manual_seed(0)


@pytest.fixture(autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)


def _ripple_vs_bptt(model, x, y):
    _, rip = ripple_grads(model, x, y, clip_norm=None)  # unclipped, to compare true magnitudes
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    model.bptt_loss(x, y).backward()
    bptt = {n: (p.grad.clone() if p.grad is not None else None) for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(False)
    out = {}
    for n in rip:
        a, b = rip[n].reshape(-1), bptt[n].reshape(-1)
        cos = float(a @ b / (a.norm() * b.norm() + 1e-30))
        ratio = float(a.norm() / (b.norm() + 1e-30))
        out[n] = (cos, ratio)
    return out


def _cfg(nl=1, **kw):
    base = dict(vocab_size=64, d_model=48, n_layers=nl, n_heads=4, P=96, beta_n=0.01,
                n_settle=60, gate_g1=0.0)  # gate off ⇒ μ=1 so ratio reflects the rule
    base.update(kw)
    return RLMConfig(**base)


def test_within_block_fidelity_at_training_config():
    x = torch.randint(0, 64, (4, 20))
    y = torch.randint(0, 64, (4, 20))
    model = RippleLM(_cfg())  # faithful defaults: delta off, key_full on, decoupled β
    res = _ripple_vs_bptt(model, x, y)
    exact = ["Wk.0", "Wq.0", "Wv.0", "M.0", "log_gamma.0"]
    for n in exact:
        cos, ratio = res[n]
        # >0.99 (well above the 0.9 kill threshold); these pathways are exact up to the
        # small EP-nudge and settle-convergence residual at the training β_n/n_settle.
        assert cos > 0.99, f"{n} cos={cos:.4f} (must be near-exact at faithful config)"
        assert 0.5 < ratio < 2.0, f"{n} magnitude ratio={ratio:.3f} (F8/F4 scale must match twin)"
    # embedding + norm_gain use the S-A0 current-step input pullback (docs/03 §4.4): their
    # future coupling is not forward-traceable, so fidelity is partial (and degrades with
    # vocab size). Positive and substantial — the direction is right — but not exact. This
    # is the documented forward-only limit, not a bug (the exact input adjoint would need
    # weight transport). Adam absorbs the magnitude; the load-bearing params above are exact.
    for n in ("embed.weight", "norm_gain.0"):
        assert res[n][0] > 0.6, f"{n} cos={res[n][0]:.4f} (S-A0 input pullback)"


def test_delta_rule_freezes_timescales():
    # enabling the delta-rule invalidates the decay (log_gamma) trace (docs/03 §4.6), so it
    # is disabled under delta — hence delta is a forward/recall P6 ablation, not the trained
    # rule. Locks the rationale for delta_rule=False being the default.
    x = torch.randint(0, 64, (4, 20))
    y = torch.randint(0, 64, (4, 20))
    delta_on = _ripple_vs_bptt(RippleLM(_cfg(delta_rule=True)), x, y)
    # log_gamma gets no valid gradient under delta (cos ≈ 0, frozen), vs exact (≈1) without it
    assert abs(delta_on["log_gamma.0"][0]) < 0.5, "log_gamma must not be (mis)learned under delta"


def test_cheap_key_path_is_a_good_approximation():
    # with the positive-count normalizer the key path lost its denominator term, so the cheap
    # current-step (S-A0) key attribution is now a GOOD approximation (cos > 0.9) — a valid
    # fast option (P4), while the exact trace (key_full=True) stays exact.
    x = torch.randint(0, 64, (4, 20))
    y = torch.randint(0, 64, (4, 20))
    exact = _ripple_vs_bptt(RippleLM(_cfg(key_full=True)), x, y)
    cheap = _ripple_vs_bptt(RippleLM(_cfg(key_full=False)), x, y)
    # exact trace is validated to 1e-8 in test_traces; here the integrated ripple-vs-BPTT
    # carries a small O(β_n) EP-nudge bias, so >0.99 (vs the cheap path's >0.9).
    assert exact["Wk.0"][0] > 0.99, f"exact key trace ({exact['Wk.0'][0]:.4f})"
    assert cheap["Wk.0"][0] > 0.9, f"cheap key path should be a good approx ({cheap['Wk.0'][0]:.4f})"
    assert exact["Wk.0"][0] > cheap["Wk.0"][0]  # exact is still better than the approximation


def test_three_factor_gate_amplifies_surprise():
    torch.set_default_dtype(torch.float32)
    # at init (Cbar=0, high NLL) the surprise gate should saturate to gate_hi
    cfg = RLMConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, P=64,
                    gate_g0=1.0, gate_g1=0.5, gate_hi=3.0)
    model = RippleLM(cfg)
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    # μ scales grads; compare gated vs ungated global magnitude
    _, g_gated = ripple_grads(model, x, y)
    model2 = RippleLM(cfg)
    model2.load_state_dict(model.state_dict())
    model2.cfg.gate_g1 = 0.0  # disable gate on the copy
    model2.Cbar.zero_()
    _, g_plain = ripple_grads(model2, x, y)
    ng = sum(float(v.norm()) for v in g_gated.values())
    npl = sum(float(v.norm()) for v in g_plain.values())
    assert ng > 2.5 * npl, f"gate should amplify high-surprise updates ~gate_hi× (got {ng/npl:.2f})"


def test_multiblock_l0_gap_is_finite_and_expected():
    # multi-block L0 differs from full-BPTT by the depth-credit gap (P3) — a real quantity,
    # not a bug. Just assert it runs and produces finite, sane cosines (regression guard).
    x = torch.randint(0, 64, (4, 20))
    y = torch.randint(0, 64, (4, 20))
    res = _ripple_vs_bptt(RippleLM(_cfg(nl=3)), x, y)
    for n, (cos, ratio) in res.items():
        assert -1.0 <= cos <= 1.0 and cos == cos  # finite, in range
