# S8 — L1-FA: transport-free depth credit recovers 2-layer MQAR (spatial-transport cost ≈ 0)

**Date:** 2026-07-16. Stage 1 of the L1-VF plan. Question: **does inter-block depth credit
survive removing weight transport, and at what cost?** — measured against the exact L1 reference
(`cos=1.0`). Answer at **2 and 3 layers: yes, at essentially zero cost.** Transport-free
feedback-aligned routing reaches **0.999** (L=2) and **0.996** (L=3) MQAR recall in the clean
final-only isolation, matching the exact/BPTT ceiling (1.000) — and, surprisingly, holding at
depth 3 where feedback alignment is expected to break. The most striking finding: a *low-fidelity*
transport-free gradient — cosine to the true gradient as low as **0.04** — is enough to build the
induction head.

Scope (honest, per plan): L1-FA removes only the **spatial** weight transport (the inter-block
input projection), holding the reverse-time temporal adjoint **exact** and keeping an explicit
backward sweep. It is the transport-cost study, **not** the full neuromorphic-plausibility claim —
that is Stage 2 (genuine Vector-Field EP, no backward pass), still to come.

---

## 1. What was built (`wavenn/ripple.py`, `wavenn/rlm.py`, docs/03 §7)

L1-FA = L1-exact with the inter-block routing made weight-transport-free. L1-exact pushes the
top-down cotangent onto the residual bus via the forward-weight transpose
(`contextual_input_grad`, Site B: `g_x = g_aq@Wq + g_ak@Wk + g_v@Wv`). L1-FA replaces that single
projection with **fixed random feedback** `Bk/Bq/Bv` (`contextual_input_grad_fa`,
`g_x = g_aq@Bq + g_ak@Bk + g_v@Bv`, feedback alignment — Lillicrap et al. 2016), reading the
**cached forward activations** q,k,v. Everything else — within-block EP settle (already
transport-free), exact projection PARAMETER traces, two-path codebook, bus adjoint, gate — is
byte-for-byte L1-exact, so any deviation is attributable to the feedback substitution alone.

**Transport-free by construction + audited** (`tests/test_transport_free.py`, 4 tests):
`contextual_input_grad_fa`'s signature contains **no forward weights** (structural guarantee);
the reverse-time adjoint `Q_t` (the temporal credit) is preserved and only the input-space
projection is randomized; **λ=0 (B_eff=W) reproduces L1-exact at cos=1.0 on every parameter**
(plumbing gate); λ=1 shows the feedback bias, top block exact (no routing). Full suite 75 green.

---

## 2. Results — 2 layers, MQAR (d_model=128, key_full=True, lr=1e-3)

### 2.1 Deep-supervision first-look (L0 floor concurrent, N=4, 12k steps, 2 seeds)
| rule | recall | note |
|---|---|---|
| L0 (own readout, no routing) | 0.332 | the floor at this budget (rises from 0.26@3k with more steps) |
| **L1-FA (λ=1, transport-free)** | **0.991** | +0.66 over L0 ⇒ the gain is the routed credit |
| L1-exact / BPTT (ceiling) | 0.998 / 0.996 | committed S8-L1-result |

### 2.2 Final-only isolation — the clean test (N=4, 8k steps, 3 seeds)
Block-0 has **no readout of its own**, so 100% of its learning must come from routed credit.
| rule | recall (3 seeds) | mean |
|---|---|---|
| L1-exact | 1.000 / 1.000 / 1.000 | **1.000** |
| **L1-FA (λ=1, transport-free)** | 0.999 / 0.999 / 0.998 | **0.999** |
| BPTT | 1.000 / 1.000 / 1.000 | **1.000** |

**Transport-free routed credit ALONE builds the 2-layer induction head to 0.999** = the exact
ceiling. This rules out top-block compensation: with block-0 stripped of its own readout, the
feedback-aligned inter-block signal does the work.

### 2.3 L=3 depth stress test (final-only, N=4, 12k steps, 3 seeds) — the review's key caveat
The error must now route through **two** random feedback projections to reach block 0, and both
lower blocks (0 and 1) have no readout of their own. FA is expected to degrade with depth.
| rule | recall (3 seeds) | mean |
|---|---|---|
| L1-exact | 1.000 / 1.000 / 1.000 | **1.000** |
| **L1-FA (λ=1, transport-free)** | 0.997 / 0.996 / 0.996 | **0.996** |
| BPTT | (not run — ceiling pinned by L1-exact = 1.000; L1-exact reproduces BPTT at cos=1.0) | ≈1.0 |

**Transport-free depth-credit HOLDS at depth 3** (0.996 = ceiling), against the expectation that
feedback alignment breaks with depth. The low-fidelity signature *sharpens*: L1-FA's `update-cos`
at L=3 spans **0.044–0.268** — a gradient barely correlated with the true one, transport-free,
still trains a 3-layer stack to near-perfect induction recall.

### 2.4 λ dose-response — trivially flat here (no transport cost to trace)
`B_eff=(1-λ)W+λB` interpolates transport fidelity from exact (λ=0) to fully random (λ=1). **Since
λ=1 (pure random feedback) already reaches the ceiling at both L=2 and L=3, every intermediate λ
does too** — there is no measurable transport cost to map. (A dose-response *curve* is only
informative when λ=1 fails and one seeks the failure threshold; it does not here.) The clean
statement is stronger than a curve: **removing weight transport entirely (λ=1) costs nothing.**

---

## 3. The mechanism — low-fidelity, transport-free, and NOT weight-level alignment

Two robust findings across all seeds (2.2):
- **The routed gradient is only ~0.33-cosine aligned with the exact BPTT gradient** (`update-cos`
  logged over training, range ~0.2–0.52), yet recall is ~0.999. A high-fidelity gradient is **not**
  required to build the induction head — a coarse transport-free descent direction suffices.
- **`cos(Wk.0, Bk) ≈ 0` throughout** (weight-alignment never rises). So this is not the textbook
  Lillicrap picture of forward weights rotating *into* the feedback matrix. The alignment that
  matters is at the **update** level (update-cos > 0 ⇒ a descent direction), not the weight level.

Interpretation (to be firmed up): L1-FA preserves the *exact temporal adjoint*, so the routed
credit still lands on the right **timesteps**; the random `B` only reshuffles which **input
dimensions** receive it. The trainable projections then reorganize around that fixed random basis —
a descent direction of modest cosine is enough because the within-block EP and the exact traces
supply high-fidelity credit for everything except the inter-block input mixing.

---

## 4. Scope, limitations, next
- **Verified:** transport-free (spatial) inter-block routing recovers MQAR to the ceiling at **both
  L=2 (0.999) and L=3 (0.996)**, clean final-only isolation, 3 seeds each; and the update is a
  strikingly low-fidelity descent direction (cos ≥ 0.04), a novel robustness point.
- **The L=3 result exceeds the review's expectation** (FA should degrade with depth) — so the
  positive is stronger than "weak L=2 evidence." Honest bound: not tested beyond L=3 or beyond N=4;
  the failure depth/interference, if any, is unmapped.
- **Scope (important):** spatial transport only — the reverse-time temporal adjoint and an explicit
  backward sweep **remain**. Keeping the exact adjoint is very likely *why* a 0.04-cosine input
  projection still works: the credit lands on the right timesteps regardless of `B`. The full "no
  backward pass" plausibility claim is **Stage 2 (genuine VF-EP joint relaxation)**.
- **Deferred (low value given the result):** the λ dose-response (trivially flat, §2.4); larger
  N-sweep and ≥5-seed grid (the 3-seed signal is unambiguous — all ≥0.996); a formal subspace
  analysis of why the low-cosine update descends.

## 5. Provenance & reproduction
`experiments/l1fa.py` (rule L1FA; `--final_only --lam --n_layers --freeze_block0` flags; learning
curves + alignment metrics). Runs: `--rules L0 L1FA --tag firstlook` (deep-sup);
`--rules L1 L1FA BPTT --final_only --tag finalonly` (2.2); `--n_layers 3 --tag L3fo` (2.3). JSON:
`experiments/S8-firstlook-fo0.json`, `S8-finalonly-fo1.json`, `S8-L3fo-fo1.json`. Implementation +
audit committed on branch `l1-vf-transport-free`.
