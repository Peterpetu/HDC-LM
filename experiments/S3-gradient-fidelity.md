# S3 — Gradient-fidelity kill gate (P1): PASS

**Date:** 2026-07-15. **Env:** PyTorch 2.7.1+cu118, Python 3.11.2, float64 (CPU) for the
autograd comparisons. Seeds fixed (`torch.manual_seed(0)`). Reproduce: `pytest tests/`.

## Result

The RIPPLE local update reproduces the truncated-BPTT gradient. All three isolated
mechanisms and the integrated rule match reverse-mode autograd:

| Test | Claim (docs) | Measured |
|---|---|---|
| `test_traces` | exact eligibility traces for W_k,W_q,W_v = BPTT (docs/01 §9) | `allclose` to `1e-7`, cos > 0.999999 |
| `test_two_path_codebook` | read+readout codebook update is exact; read-only is not (docs/01 §8, D.3) | read+readout cos = **1.0000**; read-only cos ≈ 0.22 |
| `test_gradient_fidelity` (a) | EP δ_t recovers ∂C/∂c as β_n→0 (docs/01 §7, D.2) | cos → **1.000** (0.2→0.003) |
| `test_gradient_fidelity` (b) | full single-layer RIPPLE vs BPTT, cos > 0.9 (**P1**) | cos = **1.00000** for W_k,W_q,W_v,V at all β_n∈[0.003,0.2] |

Full P1 table (single-layer, safe-regime β, symmetric nudge, converged settle):

```
beta_n=0.2 : Wk=+1.00000 Wq=+1.00000 Wv=+1.00000 V=+1.00000 | overall=+1.000000
beta_n=0.1 : Wk=+1.00000 Wq=+1.00000 Wv=+1.00000 V=+1.00000 | overall=+1.000000
beta_n=0.03: Wk=+1.00000 Wq=+1.00000 Wv=+1.00000 V=+1.00000 | overall=+1.000000
beta_n=0.01: Wk=+1.00000 Wq=+1.00000 Wv=+1.00000 V=+1.00000 | overall=+1.000000
```

## Interpretation

- The kill criterion (cos > 0.9) is met with large margin — the update is essentially
  exact, not merely aligned. Two reasons: (i) the eligibility traces are *exact* (linear
  recurrence, docs/01 §9), so the only approximation is the EP nudge bias; (ii) in the
  safe regime (β < 2/σ_max(V)²) the read has a unique fixed point and the symmetric
  nudge cancels the leading O(β_n) bias, so EP recovers the implicit-function gradient
  to high order.
- This validates, at the single-layer / full-sequence level, what whitepaper Appendix D
  showed only per-step. The program's central mechanism is not refuted.

## Scope / caveats (honest)

- Verified for **pure-accumulation** heads (delta_alpha = 0); delta-rule heads make S a
  nonlinear function of past reads, so their exact-trace treatment is separate (S4/S6).
- Single block. **Depth credit (L1 stack-EP) is not tested here** — it is measured by
  loss gap in S4 (P2/P3), not gradient cosine, because inter-block credit flows through
  EP relaxation, not the forward trace (docs/03 §7).
- Embedding gradient uses current-step attribution (S-A0 for the input); its future
  coupling is not captured by the forward trace and is a documented design choice.

## Next

S4: train WNN-L0/L1 on TinyStories vs the BPTT twin; measure the val-loss gap (P2) and
the depth-credit recovery (P3).
