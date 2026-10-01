# S7 — In-context recall (MQAR) and P6: a weak/negative result, reported honestly

> **⚠️ CORRECTION (superseded by `docs/07-architectural-analysis.md`).** The diagnosis in
> §3 below — "count-normalizer stability↔recall trade-off, not depth" — is **WRONG**. It was
> made at a broken operating point (lr=3e-3, which caps *even a transformer* at 0.39). A
> controlled root-cause analysis at the correct lr (1e-3) shows: (1) the forward architecture
> solves MQAR (WNN-BPTT-2L = 0.997 = transformer); (2) the count normalizer is fine (`qz`, the
> one I hypothesized would help, is the failure at 0.12); (3) the real cause is the **L0
> depth-credit gap** — the backprop-free RIPPLE-L0 model is stuck at ~0.12–0.24 at depth≥2
> (0.242 at N=4, 0.127 at N=8) while its BPTT twin reaches ~1.0 (0.996 / 0.998). See
> `docs/07-architectural-analysis.md`. The original
> analysis is kept below, unedited, as a record of the confounded conclusion.

---


**Date:** 2026-07-15. `experiments/mqar.py`, masked query-only objective (audit M3),
d_model=128, n_heads=4, keys/values ∈ disjoint 64-token bands, 2000 steps, seed 0.

## Result (1-layer)

In-context associative recall accuracy vs number of in-context pairs N:

| config | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|
| cleanup on (n_settle=15) | 0.193 | 0.136 | 0.100 | 0.071 |
| cleanup off (n_settle=1) | 0.186 | 0.139 | 0.094 | 0.070 |
| cleanup + delta-rule | 0.187 | 0.140 | 0.094 | 0.059 |

(chance = 1/64 ≈ 0.016)

## Honest reading

1. **The effective-context wall is real and visible.** Recall degrades monotonically with
   N (0.19 → 0.07), consistent with the √(D/M) binding-interference bound (docs/01 §4.3).
   Above chance throughout, so the fast-weight state *does* do in-context binding/recall.
2. **P6 is NOT supported in this setup (negative result).** The finite-temperature Hopfield
   cleanup (and the delta-rule) leave recall statistically unchanged — cleanup on ≈ off ≈
   delta at every N. The conjecture "cleanup denoises interference and extends effective
   context" does not hold here. Recorded as a negative result (project rules), not buried.
3. **Absolute recall is weak** (0.19 at only 4 pairs). I tested the obvious hypothesis —
   that it needs a 2-layer induction head (Olsson et al. 2022) — and it is **ruled out**:

   | | N=4 | N=8 | N=16 |
   |---|---|---|---|
   | 1 layer | 0.187 | 0.142 | 0.102 |
   | 2 layers | 0.233 | 0.145 | 0.104 |

   Depth barely moves it. The root cause is more fundamental — **the positive count
   normalizer** (`den_t = γden+d_k`, introduced in S4a to stop the signed denominator from
   exploding). It makes the read a *weighted average*: querying `q=k_j` gives
   `c ≈ (d_k·v_j + Σ_{i≠j}Re(q^†k_i)v_i) / (N·d_k) = v_j/N + noise`, with recall SNR
   `≈ √(d_k/N)` (d_k=32 per head ⇒ √(32/8)=2.0 at N=8, √(32/16)=1.4 at N=16 — marginal).
   The *old* query-dependent normalizer `Re(q^†z)` self-normalized toward the matched key
   (`c ≈ v_j`, sharp) — but that is exactly the denominator that went negative and blew up
   LM training. **So this is a genuine stability↔recall trade-off**: the normalizer that
   makes the model a good *language model* (stable, coherent generation, P2) blunts the
   sharp *associative lookup* that MQAR rewards. (The forward read still recalls perfectly
   at large D — `test_forward` passes at d_v=48, D=1024 — so it is a low-per-head-dimension +
   averaging effect, not a broken mechanism.)

## Consequence for the long-context plan

- The O(1)-state *structural* long-context advantage is unaffected (it is inherent).
- Whether the WNN can *use* long context (recall far-back bindings) is gated by the wall,
  and — on this evidence — the cleanup mechanism does not push the wall. If the 2-layer
  induction head restores strong recall, P6 should be re-tested there (interference bites
  harder at high N, where cleanup could still help). If not, the honest conclusion is that
  the fixed-width state's effective context is set by (D, γ) alone, and long context needs
  the hierarchical/episodic route (docs/06 §3), not cleanup.

This is exactly the kind of falsification the program is built to surface: a clean
prediction (P6) met with a clean negative, which redirects the long-context effort.
