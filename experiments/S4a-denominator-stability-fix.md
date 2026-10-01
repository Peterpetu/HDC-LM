# S4a — Training exposed an architectural instability; fixed by a positive normalizer

**Date:** 2026-07-15. This is a negative-then-positive result: attempting the S4 training
run (P2) surfaced a real architectural flaw that all the S1–S3 tests had missed, because
the tests exercise the read at *initialization* while the flaw only manifests as the
projections grow during training. The fix is a one-line change with a favorable side
effect. Recorded per the project rules (negative results are results; run the code).

## Symptom

Both the RIPPLE model and its BPTT twin trained **unstably**: validation NLL oscillated
7–11 (never converging), training loss spiked to 13.8 (> ln|V| = worse than random). The
instability was in the *architecture* (the twin diverged too), not the learning rule.

## Diagnosis (instrumented training)

Instrumenting the read denominator and logits during training:

```
step  50 den_min -177.65  sigmaM 4.20  logit_max 269.4
step 250 den_min -196.49  sigmaM 4.76  logit_max 837.3
```

The read `c_t = Re(q_t^† S_t) / Re(q_t^† z_t)` has a **signed denominator**. Unlike
positive-feature-map linear attention (where `φ ≥ 0` ⇒ denominator > 0), the FHRR feature
map `φ(u) = e^{iu}` gives **signed** similarities `Re(q^†k) ∈ [−D, D]`. At initialization
the projections are small, so all phasors are near `1` and `den ≈ +165` (fine — which is
why every S1–S3 test passed). As `W_k, W_q` grow during training, the phasors spread,
`Re(q^†z)` crosses zero and goes negative (`den_min = −196`), and `c` explodes → logits
hit 837 → the loss spikes and training diverges.

## Fix

Replace the query-dependent normalizer with a **positive, deterministic count normalizer**
```
den_t = γ·den_{t-1} + d_k       (per head; always > 0)
c_t   = Re(q_t^† S_t) / den_t   (a bounded weighted average)
```
This preserves the **linearity of the state recurrence** (required for the exact
eligibility trace, docs/01 §9) — `den` no longer depends on `q` or `k`, so it is a
deterministic scalar per head. Applied in `contextual_read`, `scan_reference`, and the
delta-rule `v̂` read.

**Result:** training is stable and converges. BPTT twin, 500 steps, small config:
```
step  50 val_nll 5.86  logit_max 8.2
step 300 val_nll 4.06  logit_max 16.8
step 500 val_nll 3.99  logit_max 15.1   (bounded logits; smooth descent)
```
RIPPLE (backprop-free) likewise: val_nll 6.25 → 4.99 in 200 steps.

## Favorable side effect

Because `den` is now deterministic, the **key and query eligibility traces lose their
denominator terms** (docs/01 §9 re-derivation), which:
- simplifies `projection_grads` (validated equal to autograd in `test_traces.py`), and
- makes the **cheap current-step (S-A0) key path** a *good* approximation: `cos(Wk)` rose
  from 0.44 (old normalizer) to **0.918** (> the 0.9 kill threshold). So `key_full=False`
  is now a viable fast default (no `O(d_k·d_vh·d_in)` `Tkv` trace), with `key_full=True`
  retained as the exact option. This is a strict improvement in the cost/fidelity trade.

## Consequences for the spec / tests

- The decay (`log_gamma`) trace gains a `dotden` term (`den` now depends on `γ`); still
  exact vs autograd (`test_traces.py::test_decay_trace_matches_autograd`).
- `test_forward` manual recurrences and the delta-forward test updated to the count
  normalizer; recall test made deterministic (`d_v=48` for clean value separation).
- `test_training_fidelity` updated: cheap key path is now a *good approximation* (was a
  degradation); delta-rule freezes `log_gamma` (decay trace disabled under delta) — the
  documented reason delta stays a P6 forward/recall ablation.
- All 63 tests pass; single-layer within-block fidelity: Wk/Wv/M/log_gamma ≈ 1.0,
  embed ≈ 0.76 (documented S-A0 input pullback).

## Lesson

Init-time tests are necessary but not sufficient; an online diagnostic that runs *during*
training (`diagnostics.ep_bptt_cosine`, telemetry `codebook_health`) is what surfaces
growth-dependent failures. The signed-denominator flaw is the kind of thing only training
reveals — the audit gate got the rule right, and running the rule got the architecture right.
