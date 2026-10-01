# S4 — P2 result: the backprop-free rule trains competitively

> **⚠️ RE-AUDIT (2026-07-16, `experiments/S8-reaudit-result.md`).** The **+9.0%** gap below was
> measured with `key_full=False` (cheap key) and a tied-readout→embedding Jacobian bug later found
> and fixed. Re-running the factorial at the corrected config: the true gap is **~3.3%** (2-seed
> mean, lr=2e-3, `key_full=True`), ~2.7× tighter than reported — the 9% was dominated by two
> fixable gradient approximations (the embedding bug + the cheap-key path), not the learning rule.
> The original analysis is kept below unedited as the record. **Use `key_full=True` for training.**

---

**Date:** 2026-07-15. Config: `RippleLM` d_model=128, n_heads=4, **n_layers=1**, β=1.0
(settle), β_readout=6.0, β_n=0.05, n_settle=20, key_full=False (S-A0 key, cos>0.9),
delta_rule=False. TinyStories (BPE 8192), 800 steps, batch 24, seq_len 64, lr 2e-3 +
30-step warmup, seed 0. Same architecture / data order / seed for both learners; the only
difference is the learning rule (docs/05 §3). Full telemetry in `runs/{ripple,bptt}_L1/`.

## Result — **P2 PASSES**

| Learner | final val NLL | bits/token |
|---|---|---|
| BPTT twin (autograd) | 3.468 | 5.00 |
| **RIPPLE (backprop-free)** | **3.779** | **5.45** |

**Gap: +9.0%** — well within the P2 criterion (≤ 20%, docs/05).

At single layer, L0 = the full model (no cross-block credit to drop), so this is a *direct*
learning-rule comparison: **a local, backprop-free rule (EP content signal + exact
value/decay traces + S-A0 key + two-path codebook + three-factor gate) trains a WNN to
within 9% of its backpropagation twin's validation loss on a real language task.** This is
the core S4 claim (docs/00 §1, contribution 1), and it holds.

## Notes

- Both trained stably to convergence after the S4a denominator fix; no divergence.
- Throughput: BPTT 81 s, RIPPLE 159 s for 800 steps (RIPPLE ~2× — the two-phase settles).
  Both O(1)-state; RIPPLE keeps no autograd graph.
- Online P1 fidelity tracked during training ended at cos_overall ≈ 0.62 (single-layer;
  dragged by the S-A0 embedding path ~0.76 — the exact core params stay ≈1.0). The 9% loss
  gap despite imperfect embedding credit shows the rule is robust to that approximation.
- Absolute loss (5.0–5.5 bpt) is a single-layer / short-run figure, not SOTA; depth, more
  steps, and the S5 tuning lower it. P2 is about the *gap*, and the gap is small.

## Depth comparison (P2 robustness + P3 note)

| depth | RIPPLE | BPTT twin | gap | P2 |
|---|---|---|---|---|
| n_layers=1 | 3.779 | 3.468 | **+9.0%** | ✅ |
| n_layers=3 | 3.923 | 3.583 | **+9.5%** | ✅ |

**P2 holds robustly across depth** — the local-rule gap is a stable ~9%, so RIPPLE tracks
backprop regardless of depth. Separately, depth does **not** help *either* learner at 800
steps (both slightly worse at L3): the deeper models are undertrained (more params, same
step budget). A clean **P3 depth-credit gap** therefore needs a longer run where depth
first helps the BPTT twin (so there is a gain for L0 to fail to capture); at 800 steps
there is no depth gain to attribute. This is an honest limitation of the current budget,
not a rule failure — recorded rather than glossed.

## Next (systematic)

- **S5 (quality + P4):** longer RIPPLE run with periodic generation samples (now logged
  concisely, `gen_every`); grade coherence.
- **P3 (clean):** longer run so depth helps BPTT, then measure the L0 gap (motivates L1).
- Ablations: exact-key path (key_full=True), delta-rule (P6).
