# S8 — L1-VF: the space-time relaxation is obstructed by causality (OP3(b) resolved)

**Date:** 2026-07-17. Stage 2 of the L1-VF plan (`docs/09`, `docs/03` §7). Question: can inter-block
depth credit be assigned by a genuine **joint relaxation** — removing *both* the weight transpose *and*
the explicit backward pass (the paradigm's "no backward pass" promise, `docs/00` §M2) — for the WNN's
*causal* stack? This is the repo's flagged open question **OP3(b)** (`docs/06` §5 / whitepaper OP3):
*does the causal (triangular) dependence admit a valid EP nudge, or does causality force an anticausal
component that reduces to a backward pass?*

**Answer (independently verified): the space-time cyclic relaxation is genuinely obstructed by
causality.** A spurious-free cyclic EP relaxation that computes the correct gradient does not exist for
this stack; the correct transport-free credit is exactly the **acyclic reverse-time structure = L1-FA**
(Stage 1). So the transport-free depth-credit *capability* is delivered by L1-FA (L=2 0.999, L=3
0.996); the genuine *no-backward-pass* realization for a causal stack does not add capability — it is
obstructed, exactly as the design review's T1/T2 predicted.

---

## 1. What was built (`wavenn/rlm.py::ripple_grads_l1vf`, docs/09)

A genuine joint-relaxation rule: two single-sign outer fixed-point loops converge the nudged reads
`r^±`; each block's downward force is the block-above's **F1-normalized** (÷β_n) content signal δ_{ℓ+1},
routed through the **F3** operator (`contextual_input_grad_fa`: exact temporal `Q_t`, transport-free
spatial `B`), accumulated down the additive bus; `δ_ℓ=(r^+_ℓ−r^-_ℓ)/2β_n` → the reused δ→grad pipeline.
`vf_rerun` toggles the **space-time cycle** — re-running the causal `contextual_read` in the nudged
phase so `c_{ℓ+1}` responds to `r_ℓ`. No autograd over parameters (C2).

## 2. Result — gradient level (2-layer, float64; `tests/test_spacetime_ep.py`)

- **G1 — plumbing correct.** Acyclic (`vf_rerun=False`) + exact coupling (λ=0 ⇒ B=W, λ_fb=1)
  reproduces L1-exact to floating point and autograd BPTT to the shared **O(β_n²)** EP bias. The F1
  ÷β_n normalization is load-bearing (without it the rule collapses to L0).
- **The cyclic relaxation is obstructed.** The clean, **provably convergent** demonstration
  (`feedback_gain=0` ⇒ triangular loop, no positive-feedback divergence): the space-time cycle still
  corrupts the **top block** (`M.1` cos **0.763**, converged, **O(1)** in β_n) while the acyclic
  version is exactly correct (cos 1.000).
- **Mechanism — a falsifiable control.** The spurious term is block-0's *local* nudge fed **up**
  through the causal re-read into `c_1` (`δ_1 = J_s1 f_1 + J_s1 K_{10} J_s0 f_0`; the second term is
  spurious, absent from `∂C/∂c`). It vanishes iff there is no lower local nudge: **final-only
  supervision recovers the top block to cos 1.000 exactly.** Prediction holds precisely.

## 3. Independent verification (Phase D — the 100%-confidence gate)

Three independent reviews (spec-only; did not write the code) verified from scratch:
- **V2 invariants — PASS.** C2 (no autograd over params; `@torch.no_grad()`); transport-free (error
  projection only via `B`, signature-level guarantee); genuine-relaxation-not-sweep (acyclic is
  bit-identical at n_outer 1 vs 15 = a sweep; cyclic iterates to a distinct fixed point).
- **V3 math — PASS.** Independently re-derived δ_ℓ and F1 from the EP identity; confirmed F1 is
  load-bearing (raw force → L0 at cos 1.0, ~0% credit); exact-coupling correctness VF = L1-exact to fp,
  = BPTT to the shared O(β_n²) bias (1−cos ∝ β_n⁴).
- **V1 obstruction adjudication — VERDICT H1 (genuine obstruction), not an implementation artifact.**
  Five independent lines: the corruption is O(1) in β_n (rules out finite-difference/one-sided-δ
  artifacts); it persists at a **converged** fixed point (`feedback_gain=0`, stable across n_outer
  12–400); the mechanism is proven by the final-only recovery to 1.0; every code suspect (symmetric
  force, qkv caching, tighter convergence) ruled out; no spurious-free stable cyclic relaxation exists
  across swept γ, λ_fb, η_o, n_outer.
  - **Verification-caught correction:** my original headline numbers came from the `feedback_gain=1`
    config where the cyclic loop additionally **diverges** (positive feedback) — transient, not a
    clean measurement. The decisive result is the stable `feedback_gain=0` demonstration above, which
    is *cleaner* than the first commit. (Fixed: `commit 311b50d`.)

## 4. Capability level (G5) — the obstruction confirmed on MQAR recall

The gradient-level obstruction manifests as complete capability failure. Final-only MQAR (N=4, 2 seeds,
key_full=True, `experiments/l1vf_capability.py`):

| rule | recall (2 seeds) | mean |
|---|---|---|
| L1-FA (transport-free acyclic) | 0.996 / 0.997 | **0.997** |
| **L1-VF acyclic** (= L1-FA) | 0.991 / 0.998 | **0.995** |
| **L1-VF cyclic** (genuine relaxation) | 0.015 / 0.008 | **0.012** (= chance 1/64) |
| BPTT | 1.000 / 1.000 | **1.000** |

- **Acyclic L1-VF = L1-FA at the capability level too** (0.995 ≈ 0.997) — the equivalence proven at the
  gradient level (`test_acyclic_l1vf_equals_l1fa`) holds through training to the ceiling.
- **The genuine cyclic relaxation collapses to chance** (0.012 ≈ 1/64): the O(1) spurious
  forward-coupling corrupts the gradient enough that the induction head cannot be learned *at all* —
  not merely "degraded," but no better than random. This is the recall-level face of OP3(b).

## 5. Conclusion — OP3(b) resolved

For a **causal feed-forward** stack, a spurious-free *cyclic* EP relaxation that computes the correct
gradient **does not exist**: re-running the causal read to propagate the nudge forward-in-time
necessarily injects an O(1) spurious forward-coupling. Carrying the true **anticausal** (query→source)
credit requires the reverse-time adjoint `Q_t` — i.e. the genuine "no-backward-pass" relaxation
**reduces to L1-FA** (the design review's T2). The transport-free depth-credit capability is delivered
by **Stage 1 (L1-FA)**; the space-time relaxation adds the mechanism question and answers it in the
negative. This closes `docs/06` §5 / whitepaper OP3(b): **causality obstructs a genuine forward-only
EP relaxation on this stack.**

**Scope/honesty.** L=2, T small; every claim measured against the exact L1-exact reference; the
`feedback_gain=1` divergence is documented as a symptom (not a separable bug) and the clean result uses
the convergent `feedback_gain=0` regime.

## 6. Provenance
`wavenn/rlm.py::ripple_grads_l1vf`; `docs/09-spacetime-ep.md`; `tests/test_spacetime_ep.py`
(G1 + converged-obstruction + mechanism, 3 tests; full suite 78 green). Independent verification
reports (V1/V2/V3) archived from the Phase-D reviews. Commits: `1a86335` (core), `311b50d`
(verification-corrected demonstration).
