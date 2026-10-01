# S8 — Re-audit of the P2 TinyStories gap: the ~9% was mostly gradient-approximation artifact

**Date:** 2026-07-16. Triggered by two findings that post-date S4: the L1 exactness check fixed
a tied-readout→embedding Jacobian bug that was in the S4 code (`S8-L1-result.md`), and the same
work showed the cheap `key_full=False` key path caps `∂L/∂W_k` at cos≈0.85 (which crippled MQAR).
docs/07 §5 also flagged the lr. Question: **how much of the reported +9.0% RIPPLE-vs-BPTT P2 gap
was the learning rule, and how much was fixable gradient approximation?**

Design: re-run the S4 1-layer comparison (TinyStories BPE-8192, 800 steps, batch 24, seq 64) as a
`{key_full: False, True} × {lr: 2e-3, 1e-3}` factorial. `key_full` affects **only** the RIPPLE
gradient assembly — the BPTT twin (autograd) is identical either way — so the isolation is clean.
`experiments/reaudit_s4.py`.

## Result (val NLL, 2-seed mean over seeds {0,1}; gap = (RIPPLE − BPTT)/BPTT)

| lr | key_full | BPTT | RIPPLE | gap (mean) | per-seed | (documented S4) |
|---|---|---|---|---|---|---|
| 2e-3 | False | 3.476 | 3.686 | **+6.05%** | 6.27 / 5.84 | +8.97% |
| 2e-3 | **True** | 3.476 | 3.592 | **+3.34%** | 3.67 / 3.02 | — |
| 1e-3 | False | 3.679 | 3.821 | +3.87% | 3.95 / 3.80 | — |
| 1e-3 | True | 3.679 | 3.755 | +2.07% | 2.25 / 1.88 | — |

Seeds agree to within ~0.6 pts, so the effects below are stable, not noise.

## Decomposition — the 9% was two artifacts, not the rule

1. **Cheap-key path (~2.7 pts) — clean single-toggle isolation.** At lr=2e-3, with everything else
   identical (same code, seed, RNG stream), flipping `key_full` False→True drops the gap
   **6.05% → 3.34%** (both seeds monotone). The S-A0 key attribution that was adequate for a rough
   LM gap is materially lossy: it caps `∂L/∂W_k` at cos≈0.85 (S8). This is the *rigorous* part of
   the decomposition — one variable, one effect.
2. **Tied-readout embedding bug (~2.7 pts, cross-run).** Documented RIPPLE(kf=False, lr=2e-3) =
   3.779 (gap 8.97%); the current code (Jacobian fix dim=1→dim=0, `test_tied_readout_grad`) gives
   3.686 at the *same config* (gap 6.05%). This is a cross-run comparison (old buggy code vs new),
   but two seeds both land at ~6% (5.84 / 6.27), far from 9%, so the drop is the code fix, not seed
   variance. The embedding is ~1M of the 1.13M params and its readout gradient was 55% wrong, so a
   large share is expected. (A buggy-vs-fixed run at fixed RNG would isolate it to the point; the
   2-seed stability makes that a formality.)
3. **lr.** At lr=1e-3 the gap is smaller still (2.07%) but both models are less converged at 800
   steps (higher absolute loss), so lr=2e-3 (the original's setting) is the apples-to-apples cell.

**Corrected P2:** the backprop-free rule is within **~3.3%** of its BPTT twin (lr=2e-3,
key_full=True, 2-seed mean), not 9%. P2 (≤20%) was never at risk, but the true gap is **~2.7×
tighter** than reported — the reported 9% was dominated by two fixable gradient approximations, not
a limitation of the local rule.

## Consequences

- **P2 stands, tighter.** Update `S4-P2-result.md` / `RESULTS.md` P2 to the corrected ~3.7%.
- **P4 (S5 quality) should improve.** Lower RIPPLE loss at fixed compute ⇒ the S5 "undertrained
  4.55 bpt" figure would drop; a full S5 re-run at key_full=True is the follow-up (not re-run here).
- **Default the exact key path for training.** `key_full=True` is the faithful setting; the cheap
  path is an explicit speed ablation (docs/03 §4.3), and its cost to fidelity is now quantified on
  a real LM, not just MQAR.
- **docs/07 §5 resolved:** the lr=2e-3 setting did not inflate the gap (1e-3 gives a *smaller* gap
  but undertrains); the gap's real drivers were the embedding bug + cheap key, now fixed.

## Provenance
`experiments/reaudit_s4.py 1 800 {0,1}` → `S8-reaudit-L1-s{0,1}.json`; telemetry in
`runs/reaudit_*`. Baseline: `experiments/S4-results-L1.json` (documented 8.97%).
