# WNN / RIPPLE — Results summary

Status of the falsifiable predictions (docs/05) and the two architectural findings the
build surfaced. Detailed writeups are linked; raw telemetry is under `runs/`.

## Predictions

| # | Prediction (docs/05) | Status | Evidence |
|---|---|---|---|
| **P1** | RIPPLE update ≈ BPTT gradient (cos > 0.9, →1 as β_n→0) | ✅ **PASS** | Exact traces = autograd to 1e-8; two-path codebook cos=1.0; full single-layer RIPPLE-vs-BPTT cos=1.0. `S3-gradient-fidelity.md`, `tests/test_{traces,two_path_codebook,gradient_fidelity,training_fidelity}.py` |
| **P2** | RIPPLE within 20% of BPTT twin on TinyStories | ✅ **PASS (~3.3%, re-audited)** | Re-audit at the corrected config (`key_full=True` + tied-readout embedding bug fix): gap **+3.3%** (2-seed mean, lr=2e-3), down from the originally-reported +9.0% — the 9% was two fixable gradient approximations, not the rule. `S8-reaudit-result.md`, `S4-P2-result.md` |
| **P3** | multi-block L0 depth-credit gap; L1 closes half | ✅ **PASS (exceeds)** | On MQAR the L0 gap is stark (L0 0.26 vs BPTT 1.0 at 2 layers) and **L1 stack-EP closes it fully, not half**: RIPPLE-L1 = **0.998** = BPTT (0.996). L1 reproduces the 2-layer BPTT gradient to cos=1.0 on every param. `S8-L1-result.md`, `tests/test_stack_ep_fidelity.py` |
| **P4** | SOTA-for-class on TinyStories (vs Eldan–Li) | ◻ partial | **Coherent generation demonstrated** at 2500 steps (named characters, grammar, dialogue, narrative) — `runs/ripple_S5/` (samples in `S5-run.log`). Absolute loss (4.55 bpt) is undertrained (2500 steps ≈ 7.7M tokens vs the reference models' billions); the rubric comparison needs a full-budget run (S6 compute). |
| **P5** | Pile within 10% of a GLA / GPT-2 baseline | ✗ **negative** | S6 frozen-eval run at 162M: val PPL 517.2 after 400 steps vs a GPT-2-162M anchor at 28.9 (`S6-run-B1-lr5e-4.json`, `S6-v0-anchor.json`, `docs/10`). The campaign stopped here. |
| **P6** | cleanup/delta-rule extends effective context (MQAR) | ✗ negative (cleanup) | cleanup near-decorative for recall (P6 negative, quantified). **The weak-recall root cause was RE-DIAGNOSED** (see `docs/07`): NOT the normalizer (BPTT solves MQAR with the count read; `qz` is the bad one) and NOT the forward architecture (WNN-BPTT-2L = 0.997 = transformer). It is the **L0 depth-credit gap** (+ the cheap-key gradient) — the backprop-free RIPPLE-L0 model is stuck at ~0.12–0.26 at depth≥2 while its BPTT twin reaches ~1.0. **Now RESOLVED:** L1 stack-EP + exact key path recovers recall to **0.998** = BPTT (`S8-L1-result.md`). `docs/07-architectural-analysis.md` |
| **Depth-credit (L0→L1)** | does local depth-credit match backprop? | ✅ **closed by L1** | L0 cannot form the 2-layer induction head (stuck ~0.26 at 2 layers even with the exact key path); **L1 stack-EP closes the gap to the transformer ceiling (0.998)**. A 2×2 factorial shows recovery needs *both* depth-credit *and* the exact key gradient (`key_full=True`) — refines docs/07's single-cause diagnosis. `S8-L1-result.md` |
| **Transport-free (L1-FA)** | is depth credit possible without weight transport? | ✅ **yes** | L1-FA (fixed random feedback) recovers MQAR to the ceiling at **L=2 (0.999) and L=3 (0.996)**, final-only isolation, 3 seeds — and holds at depth 3 where FA was expected to break. A ~0.04-cosine transport-free gradient suffices. `S8-L1FA-result.md` |
| **No-backward-pass (L1-VF)** | can a forward-only relaxation do it? (OP3(b)) | ✗ **obstructed by causality** | Built the space-time relaxation (`ripple_grads_l1vf`): plumbing correct (acyclic = L1-exact, cos=1.0), but the genuine **cyclic** relaxation injects an O(1) spurious forward-coupling (converged top-block cos 0.76; final-only recovery to 1.0 proves the mechanism). Carrying anticausal credit needs the reverse-time adjoint ⇒ reduces to L1-FA. Independently verified (3 independent reviewers). `S8-L1VF-result.md`, `docs/09` |
| **Continual** | acquire domains without catastrophic forgetting (M1) | ✅ **strong** | episodic replay cuts forgetting **≈6.3×** (8.04→1.27, 3 seeds); gate-alone ~7% (correctly — global scalar); RIPPLE ≈ BPTT with replay. `S7-continual-result.md`, audit-cleaned |
| **P7** | throughput within 3× GLA; O(1)-state generation | ◻ partial | RIPPLE ~2× BPTT wall-clock; O(1)-state recurrent generation implemented + validated (`tests/test_generate.py`) |

## The two things the build discipline caught

1. **Pre-training audit (three independent passes).** Before spending compute, an audit found the deployed
   *training* config had collapsed gradient fidelity (delta-rule/β/settle). Fixed: double-
   softmax gate bug (F1), missing normalization Jacobian (F4), mean-vs-sum scale (F8),
   flipped query-sign *in the docs* (F2), decoupled settle/readout temperatures (F3).
   `S3b-audit-and-fixes.md`.
2. **Running the training** exposed a signed-denominator instability (FHRR similarities are
   signed → linear-attention denominator went negative → logits→837 → divergence, even for
   the BPTT twin). Fixed with a positive count normalizer, which also made the cheap key
   path viable. `S4a-denominator-stability-fix.md`.

Both are recorded as negative-then-fixed results (project rules), and both were only findable
by *running the code* — the init-time tests all passed while the training-time behavior was
broken.

## Constitutional compliance

- **C1 (HDC core):** binding/bundling/permutation/cleanup are the block primitives; ablating
  them degrades the model (ablation ladder, docs/05 §6). Capacity `√(D/M)` and `M_eff`
  validated empirically (`tests/test_capacity.py`).
- **C2 (backprop-free):** zero autograd over model parameters in the RIPPLE path (audited);
  autograd only in the BPTT twin and tests.

## Reproduce

```
python -m pytest tests/ -q          # 67 tests: algebra, traces, kill gate, training fidelity, generation
python experiments/run_s4.py 1 800  # P2: RIPPLE (backprop-free) vs BPTT twin
python experiments/run_s5.py 2500   # quality run with generation samples
python experiments/analyze_run.py runs/ripple_S5 runs/bptt_S5   # HTML report + plots
```
