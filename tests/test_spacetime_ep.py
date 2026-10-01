"""L1-VF space-time EP gates (docs/09, plan Stage 2). G1 plumbing + the G1/G2 obstruction.

These lock the two decisive findings:
  G1  — the joint-relaxation plumbing is CORRECT: with exact coupling (λ=0 ⇒ B=W, λ_fb=1) and NO
        space-time cycle (vf_rerun=False, the acyclic/triangular structure), ripple_grads_l1vf
        reproduces L1-exact/BPTT at cos=1.0 on every parameter. This validates F1 (÷β_n), F3 (the
        content-cotangent-through-CR_adj operator), and the δ→parameter-grad reuse.
  G1/G2 tension (the OP3(b) obstruction, docs/09 §3-4) — turning ON the genuine space-time cycle
        (vf_rerun=True) makes the relaxation move (a real equilibrium, G2) BUT corrupts the gradient
        (G1 fails badly, even the top block), because re-running the CAUSAL contextual_read in the
        nudged phase feeds nudge-induced lower-block deviations UP into every c_ℓ — a spurious
        forward-coupling absent from the true ∂C/∂c. For a causal feedforward stack, a spurious-free
        cyclic EP relaxation does not exist; the correct transport-free credit necessarily reduces to
        the acyclic reverse-time structure (= L1-FA). Recorded as a *result*, not a code failure.
"""

import pytest
import torch

from wavenn.rlm import RLMConfig, RippleLM, ripple_grads_l1, ripple_grads_l1vf


@pytest.fixture(autouse=True)
def _f64():
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


def _cos(a, b):
    return (a.flatten() @ b.flatten() / (a.norm() * b.norm() + 1e-30)).item()


def _bptt(m, tok, tgt):
    ps = [p for _, p in m.named_parameters()]
    ns = [n for n, _ in m.named_parameters()]
    for p in ps:
        p.requires_grad_(True)
    g = torch.autograd.grad(m.bptt_loss(tok, tgt), ps)
    for p in ps:
        p.requires_grad_(False)
    return dict(zip(ns, g))


def _model(rerun, *, deep=True, lfb=1.0, n_outer=15, tol=1e-6):
    torch.manual_seed(0)
    cfg = RLMConfig(vocab_size=20, d_model=12, n_layers=2, n_heads=3, P=24, beta=0.4,
                    beta_readout=3.0, beta_n=0.02, n_settle=80, eta=0.3, key_full=True,
                    deep_supervision=deep, feedback_lambda=0.0, feedback_gain=lfb,
                    vf_variant="VF-ST-adjoint", vf_rerun=rerun, vf_n_outer=n_outer, vf_relax=0.5,
                    vf_tol=tol)
    return RippleLM(cfg).double()


def _grads(m):
    torch.manual_seed(100)
    tok = torch.randint(0, 20, (2, 6))
    tgt = torch.randint(0, 20, (2, 6))
    b = _bptt(m, tok, tgt)
    _, gvf = ripple_grads_l1vf(m, tok, tgt, clip_norm=None)
    _, g1 = ripple_grads_l1(m, tok, tgt, clip_norm=None)
    return b, g1, gvf


ALL = ["Wk.0", "Wq.0", "Wv.0", "M.0", "log_gamma.0", "norm_gain.0", "embed.weight",
       "Wv.1", "M.1"]


def test_g1_acyclic_reproduces_l1_exact():
    # G1: exact coupling + no space-time cycle ⇒ the joint relaxation IS L1-exact (cos=1.0 to fp),
    # and matches BPTT to the shared EP finite-β_n bias level (same as L1-exact itself).
    b, g1, gvf = _grads(_model(rerun=False))
    for n in ALL:
        assert _cos(gvf[n], g1[n]) > 1 - 1e-4, f"{n}: VF(no-cycle) ≠ L1-exact ({_cos(gvf[n], g1[n]):.6f})"
        assert _cos(gvf[n], b[n]) > 0.999, f"{n}: VF vs BPTT ({_cos(gvf[n], b[n]):.6f})"


def test_obstruction_converged_cyclic_corrupts_topblock():
    # OP3(b), the CLEAN demonstration (independent-verification-corrected). At feedback_gain=0 the
    # loop is triangular ⇒ PROVABLY CONVERGENT (no positive feedback, no divergence confound). Yet
    # the space-time cycle still corrupts the TOP block: re-running the causal contextual_read in the
    # nudged phase feeds block-0's nudge-induced deviation UP into c_1 — a spurious O(1) forward-
    # coupling absent from the true ∂C/∂c. The acyclic version is exactly correct. This is a
    # converged-but-wrong equilibrium, not a transient of a divergent iteration.
    b_a, _, g_a = _grads(_model(rerun=False, lfb=0.0, n_outer=60, tol=1e-8))  # acyclic
    b_c, _, g_c = _grads(_model(rerun=True,  lfb=0.0, n_outer=60, tol=1e-8))  # cyclic (converged)
    for n in ["M.1", "Wv.1", "Wk.1"]:
        assert _cos(g_a[n], b_a[n]) > 0.999, f"acyclic top block should be exact: {n}"
        assert _cos(g_c[n], b_c[n]) < 0.95, f"cyclic top block should be corrupted (O(1)): {n} cos={_cos(g_c[n], b_c[n]):.4f}"


def test_obstruction_mechanism_finalonly_recovers():
    # The falsifiable mechanism proof: the spurious term is block-0's LOCAL nudge fed up. Removing it
    # (final-only supervision ⇒ no lower local nudge) makes the cyclic top block recover to cos=1.0.
    b, _, g = _grads(_model(rerun=True, deep=False, lfb=0.0, n_outer=60, tol=1e-8))
    for n in ["M.1", "Wv.1", "Wk.1"]:
        assert _cos(g[n], b[n]) > 0.999, f"final-only should remove the fed-up nudge: {n} cos={_cos(g[n], b[n]):.4f}"


def test_acyclic_l1vf_equals_l1fa():
    # Locks the G5 corollary: at full transport-free coupling (λ=1) the ACYCLIC L1-VF IS L1-FA — so
    # "L1-VF (no cycle) reproduces L1-FA's MQAR ceiling" is a proven equivalence, not a fresh run.
    torch.manual_seed(0)
    cfg = RLMConfig(vocab_size=20, d_model=12, n_layers=2, n_heads=3, P=24, beta=0.4,
                    beta_readout=3.0, beta_n=0.02, n_settle=80, eta=0.3, key_full=True,
                    deep_supervision=True, feedback_lambda=1.0, feedback_gain=1.0,
                    vf_variant="VF-ST-adjoint", vf_rerun=False, vf_n_outer=40, vf_relax=0.5, vf_tol=1e-8)
    m = RippleLM(cfg).double()
    from wavenn.rlm import ripple_grads_l1fa
    torch.manual_seed(100)
    tok = torch.randint(0, 20, (2, 6))
    tgt = torch.randint(0, 20, (2, 6))
    _, gvf = ripple_grads_l1vf(m, tok, tgt, clip_norm=None)
    _, gfa = ripple_grads_l1fa(m, tok, tgt, clip_norm=None)
    for n in ALL:
        assert _cos(gvf[n], gfa[n]) > 1 - 1e-3, f"{n}: acyclic L1-VF ≠ L1-FA ({_cos(gvf[n], gfa[n]):.5f})"
