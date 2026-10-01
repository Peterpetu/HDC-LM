"""L1 stack-EP fidelity (docs/03 §7, docs/07 R1) — does inter-block credit reach lower
blocks, and how close is it to the 2-layer BPTT twin?

The L0 depth-credit gap (docs/07): a lower block's parameters get gradient only from that
block's own readout, never from how its output helped the blocks above. L1 routes the credit
down the additive residual bus (``ripple_grads_l1``). This test measures, per parameter, the
cosine of the L0 and L1 updates against the exact 2-layer BPTT gradient:

  (a) top block: L1 ≈ BPTT (it is the readout block, no routing needed);
  (b) lower block, deep supervision ON: L1 cosine ≥ L0 cosine (credit is added, P3);
  (c) lower block, deep supervision OFF (final-only): the lower block is trained *purely* by
      top-down credit — this isolates the routing fidelity (L0 would be ~0 correlation there).

Run with ``-s`` to print the per-parameter cosine table.
"""

import pytest
import torch

from wavenn.rlm import RLMConfig, RippleLM, ripple_grads, ripple_grads_l1


@pytest.fixture(autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)


def _cos(a, b):
    return (a.flatten() @ b.flatten() / (a.norm() * b.norm() + 1e-30)).item()


def _bptt_grads(model, tokens, targets):
    names = [n for n, _ in model.named_parameters()]
    ps = [p for _, p in model.named_parameters()]
    for p in ps:
        p.requires_grad_(True)
    loss = model.bptt_loss(tokens, targets)
    gs = torch.autograd.grad(loss, ps)
    for p in ps:
        p.requires_grad_(False)
    return {n: g for n, g in zip(names, gs)}


def _build(deep=True, beta=0.4, seed=0, L=2):
    torch.manual_seed(seed)
    cfg = RLMConfig(
        vocab_size=20, d_model=12, n_layers=L, n_heads=3, P=24,
        beta=beta, beta_readout=3.0, beta_n=0.02, n_settle=80, eta=0.3,
        key_full=True, delta_rule=False, deep_supervision=deep,
    )
    return RippleLM(cfg).double(), cfg


def _grads(deep, seed=0, B=2, T=6):
    model, cfg = _build(deep=deep, seed=seed)
    torch.manual_seed(100 + seed)
    tokens = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    bptt = _bptt_grads(model, tokens, targets)
    _, g0 = ripple_grads(model, tokens, targets, clip_norm=None)
    _, g1 = ripple_grads_l1(model, tokens, targets, clip_norm=None)
    return bptt, g0, g1


def _table(bptt, g0, g1, names):
    rows = []
    for n in names:
        rows.append((n, _cos(g0[n], bptt[n]), _cos(g1[n], bptt[n])))
    return rows


def test_l1_top_block_matches_bptt():
    bptt, g0, g1 = _grads(deep=True)
    # top block (index 1) only affects the top readout; L1 must match BPTT there
    for n in ["Wv.1", "M.1", "Wk.1", "Wq.1"]:
        assert _cos(g1[n], bptt[n]) > 0.999, f"{n}: L1 cos {_cos(g1[n], bptt[n]):.4f}"


def test_l1_lower_block_exact_and_beats_l0_deep():
    # with the exact reverse-time input routing, L1 = BPTT exactly, and strictly beats L0.
    bptt, g0, g1 = _grads(deep=True)
    rows = _table(bptt, g0, g1, ["Wk.0", "Wq.0", "Wv.0", "M.0", "log_gamma.0", "norm_gain.0",
                                 "embed.weight"])
    print("\n[deep supervision] lower-block cos vs BPTT   L0      L1")
    for n, c0, c1 in rows:
        print(f"  {n:14s} {c0:+.4f}  {c1:+.4f}")
    for n, c0, c1 in rows:
        assert c1 >= c0 - 1e-3, f"{n}: L1 {c1:.4f} < L0 {c0:.4f}"
    # exact match for the projection/codebook params (the L0 depth-credit gap, closed)
    for n in ["Wk.0", "Wq.0", "Wv.0", "M.0", "log_gamma.0", "norm_gain.0"]:
        assert _cos(g1[n], bptt[n]) > 0.999, f"{n}: L1 not exact ({_cos(g1[n], bptt[n]):.5f})"


def test_l1_lower_block_pure_topdown_final_only():
    # deep supervision OFF: block 0 has NO local readout — trained purely by routed credit.
    # L0 there is ~0 (or negative) correlation; L1 recovers the exact gradient.
    bptt, g0, g1 = _grads(deep=False)
    rows = _table(bptt, g0, g1, ["Wk.0", "Wq.0", "Wv.0", "M.0"])
    print("\n[final-only] lower-block cos vs BPTT (pure top-down)   L0      L1")
    for n, c0, c1 in rows:
        print(f"  {n:14s} {c0:+.4f}  {c1:+.4f}")
    for n, c0, c1 in rows:
        assert c1 > 0.999, f"{n}: L1 pure-top-down cos {c1:.5f} (routing not exact)"


if __name__ == "__main__":
    for deep in (True, False):
        bptt, g0, g1 = _grads(deep=deep)
        tag = "deep" if deep else "final-only"
        names = ["Wk.0", "Wq.0", "Wv.0", "M.0", "log_gamma.0", "norm_gain.0",
                 "Wv.1", "M.1", "embed.weight"]
        print(f"\n=== {tag} : cos vs 2-layer BPTT ===")
        print(f"{'param':14s}  {'L0':>8s}  {'L1':>8s}")
        for n, c0, c1 in _table(bptt, g0, g1, names):
            print(f"{n:14s}  {c0:+.4f}  {c1:+.4f}")
