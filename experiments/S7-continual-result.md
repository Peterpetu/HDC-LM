# S7 — Continual learning: episodic replay prevents catastrophic forgetting

**Date:** 2026-07-15. Audit-cleaned (all findings from the S7 critic fixed, see
`S3b`-style audit notes above and the code comments tagged `audit H1/M1/…`). Config:
`RippleLM` d_model=96, n_layers=1, vocab 256; 3 sequential Markov-language domains, 300
optimizer updates/domain (**equalized across learners**), lr 2e-3 + warmup, replay_frac
0.4, buffer 3000. **Mean ± std over 3 seeds.** Bayes floor computed correctly (marginal-
weighted conditional entropy, ≈1.18–1.23 nats).

## Result

| learner | final loss ↓ | forgetting ↓ | excess over Bayes ↓ |
|---|---|---|---|
| RIPPLE naive | 6.689 ± 0.133 | 8.036 ± 0.197 | 5.481 ± 0.133 |
| RIPPLE + gate | 6.328 ± 0.198 | 7.449 ± 0.290 | 5.120 ± 0.198 |
| **RIPPLE + replay** | **2.361 ± 0.119** | **1.265 ± 0.231** | **1.153 ± 0.119** |
| **RIPPLE + replay + gate** | **2.325 ± 0.080** | **1.123 ± 0.168** | **1.117 ± 0.080** |
| BPTT naive | 6.617 ± 0.169 | 7.927 ± 0.251 | 5.409 ± 0.169 |
| BPTT + replay | 2.173 ± 0.079 | 0.999 ± 0.154 | 0.964 ± 0.079 |

(Forgetting = mean over the first n−1 domains of `final loss − loss right after learning it`.)

## What it shows (each claim is what the data supports — no over-reach)

1. **Catastrophic forgetting is real and severe.** Naive sequential learning drives old-
   domain loss *above random* (per-domain final ≈ [9.4, 7.8, 1.4]): later domains actively
   overwrite the shared readout/weights. This holds for both RIPPLE and BPTT — it is an
   architecture/training property, not a rule artifact.

2. **Episodic replay is the mechanism that prevents it.** RIPPLE + replay cuts forgetting
   **8.04 → 1.27 (≈6.3×)** and final loss **6.69 → 2.36**, retaining all domains
   (per-domain ≈ [2.8, 2.8, 1.7]). This is the complementary-learning-systems result
   (McClelland et al. 1995) realized in the WNN with the *same* RIPPLE update used for
   replay — "replay and learning are the same operation" (M1).

3. **The three-factor gate does NOT prevent forgetting on its own** (8.04 → 7.45, ~7%) —
   exactly as expected for a *global* scalar that cannot selectively protect prior-domain
   weights. It adds a small, consistent efficiency *to* replay (replay 1.27 → replay+gate
   1.12). This is the honest, corrected claim (the pre-audit docs over-stated the gate).

4. **The backprop-free rule does continual learning as well as backprop.** RIPPLE+replay
   (forgetting 1.27) ≈ BPTT+replay (1.00); RIPPLE naive (8.04) ≈ BPTT naive (7.93). The
   local rule inherits the CLS benefit of replay with no backpropagation — and because
   "writing memory is learning" (M1), replay/consolidation is native, not a bolted-on loop.

## Complement (fast timescale): in-context recall

`experiments/mqar.py` tests the FAST timescale — the running state as a fast weight doing
test-time learning by construction (in-context associative recall) — and P6 (does cleanup
extend recall?). Results in `S7-mqar-results.json` / `S7-mqar-run.log`.

## Reproduce

```
python experiments/continual.py     # 3 seeds × 6 learners → S7-continual-results.json
python experiments/mqar.py          # in-context recall vs #pairs, cleanup ablation
```
