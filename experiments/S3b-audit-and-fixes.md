# S3b — Pre-training faithfulness audit and fixes

**Date:** 2026-07-15. Gate: before spending S4 training compute, a three-pass independent audit
(math faithfulness, code completeness + C2 compliance, test adequacy) verified the full
model against the spec (`docs/00–03`) and the project rules. The audit was decisive — it caught
that the deployed *training* configuration had collapsed gradient fidelity, which would
have wasted compute on a rule that does not track the gradient. This document records the
findings and the fixes, so the training runs are interpretable and reproducible.

## What the audit confirmed (no action needed)

- **C2 (backprop-free) holds:** zero `loss.backward()` / autograd over model parameters in
  the RIPPLE path; autograd appears only in the BPTT twin (`rlm.py::bptt_loss` +
  `train.py::train_bptt`) and in tests. Independently reproduced by a probe running
  `ripple_grads` under `torch.set_grad_enabled(False)` with no `.grad` populated.
- **No stubs / placeholders.** All six core mechanisms present and real.
- **The implemented math is correct.** Every load-bearing sign in `ripple.py` matches
  autograd in float64 (traces, EP signal, two-path codebook, decay trace).

## Critical findings and fixes

| ID | Finding | Fix |
|---|---|---|
| **F1** | `rlm.py` fed probabilities `ep.p` into `F.cross_entropy` → double softmax; the logged train-NLL, the surprise gate μ, and `Cbar` were all distorted (μ stuck ≈ g0, so neuromodulation was silently off). | Compute the real NLL from logits `β_ro·(r*@V)`. μ now responds to surprise (verified: μ→gate_hi at init). |
| **C1 / F3** | At the *training* config (delta on, cheap key path, β=4, 12 settle steps) the RIPPLE update's per-parameter cosine vs BPTT was **−1.0 to 0.44** (log_gamma anti-aligned under delta-rule); β=4 is ~10× above the safe-uniqueness limit so the EP precondition (unique, converged read) was violated. | (a) **Faithful defaults:** `delta_rule=False` (delta becomes a forward/recall P6 ablation — its exact gradient is the DeltaNet-trace refinement), `key_full=True` (exact key trace; `key_full=False` is the P4 S-A0 scaling ablation). (b) **Decoupled temperatures:** settle `β=1.0` (modest → converged/single-valued read, EP-valid), readout `β_readout=6.0` (free to be sharp). (c) `n_settle=25` (converged). Result: within-block fidelity now cos≈1.0 for Wk/Wq/Wv/M/log_gamma, ~0.76–0.91 for the S-A0 embedding/norm-gain paths. |
| **F4 / H1** | RIPPLE's `M` and tied-`V` gradients were taken w.r.t. the unit-normalized codebooks but applied to the raw parameters, skipping the `F.normalize` Jacobian — a systematic bias vs the BPTT twin (which differentiates through it). | Added `_unit_cols_vjp` (closed-form `(g − r̂(r̂·g))/‖r‖`, no autograd). `M` cos went 0.97→1.00. |
| **F8** | RIPPLE summed per-token/per-block gradients while the twin averaged (a B·T·L× scale gap) — breaking lr comparability for P2/P3. | RIPPLE grads scaled by `1/(n_layers·B·T)` to match the twin's mean-of-means loss. |
| **F2** | The docs (`01` eq 9.6, `03` §4.2a) had the query-path phase sign **flipped**; the code was correct. | Docs corrected (query numerator term is `+Im`, verified against autograd). |
| **F5, F6, F7, F9** | Docs `cos→−1` vs `+1` inconsistency; T1 "one-step" proof gap; `memory.py` hardcoded complex64 in the delta branch; misc citation slips. | `memory.py` dtype fixed; docs corrected (sign convention unified, T1 reworded as a corner-case correspondence, citations fixed). |

## New/strengthened tests (the coverage gaps the audit found)

- `tests/test_training_fidelity.py` — **the test whose absence let the pre-audit code
  nearly train on a broken rule.** Locks within-block RIPPLE-vs-BPTT cosine ≈1 AND
  magnitude ratio ≈1 for the exact params at the *training* config; documents that enabling
  delta-rule or the cheap key path degrades fidelity (so defaults can't silently regress);
  checks the three-factor gate amplifies surprise.
- `tests/test_optimizer.py` — LocalAdam convergence, bias correction, global-norm clip.
- `test_two_path_codebook.py` — added a first-order loss-descent check (two-path descends,
  read-only barely moves) beyond the cosine bound.
- `test_gradient_fidelity.py` — asserts every β_n (not just the last), adds magnitude-ratio
  assertions (H3).
- `test_capacity.py` — added the `M_eff=(1+γ)/(1−γ)` participation-number identity.
- `test_fixed_point.py` — β-grid convergence + the strong-content-clamp characterization
  that validates the above-limit operating regime (docs/01 §6.2(b)).

All 62 tests pass.

## Honest residual approximations (documented, not hidden)

- **Embedding / norm_gain use S-A0 current-step input pullback** (docs/03 §4.4): fidelity
  ~0.76–0.91, degrading with vocab size, because the input's future coupling is not
  forward-traceable (the exact adjoint needs weight transport). The core trainable params
  are exact; Adam absorbs the embedding magnitude.
- **Operating β is above the sufficient safe bound** but the read is single-valued in
  practice via a strong content clamp (validated: `test_fixed_point.py`); telemetry logs the
  safe-β margin (`codebook_health`) every eval.
- **Delta-rule and the cheap key path are ablations**, not the trained rule (their local
  gradients don't track BPTT); measured in P4/P6.

## Takeaway

The audit gate did exactly its job: it converted "it should work" into a measured,
regression-locked fidelity guarantee at the actual deployed configuration, and forced the
approximations to be either fixed (F1/F4/F8) or honestly scoped and documented (embedding
S-A0, delta/key ablations). Only then did training start.
