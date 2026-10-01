# 07 — Architectural Root-Cause Analysis: why in-context recall (MQAR) was weak

**Date:** 2026-07-15. **Scope:** Isolate the cause of the weak MQAR recall reported in
`experiments/S7-mqar-result.md` into its architectural, learning-rule, normalizer, and
capacity contributions, by controlled experiment. Every run reuses the *exact*
`wavenn/rlm.py::RippleLM`; only the learning rule (RIPPLE-L0 vs the BPTT twin), the read
normalizer (monkey-patched copies — `wavenn/` source untouched, 67 tests still pass), depth,
and learning rate are varied. Numbers are from runs whose result JSONs are archived under the
scratch lab; the crux is independently reproduced twice (details in §6).

---

## Abstract / key finding

`experiments/S7-mqar-result.md` reported MQAR recall ≈0.19 at 4 key–value pairs (chance 1/64)
and blamed the **count normalizer** (`den_t=γden+d_k`) via an SNR-`√(d_k/N)` argument. **That
attribution is wrong**, and so is a subsequent "it's just a config artifact, the model recalls
fine" reading. The controlled decomposition gives a sharper, and more consequential, answer:

1. **A learning-rate pathology inflated the story.** `lr=3e-3` (the S7 default) caps *even a
   2-layer transformer* at **0.38** recall on the identical probe; at `lr=1e-3` the same
   transformer reaches **0.996**. S7's absolute numbers were measured at a broken operating point;
   the true task ceiling is ≈1.0.

2. **The forward architecture is NOT the limiter.** With the correct lr and **2 layers**, the
   WNN trained end-to-end (**BPTT twin**) reaches **0.996** at N=4 and **0.998** at N=8 — equal to
   the transformer — using the *same count normalizer, same d_k=32, same phasor read*. Sharp
   associative recall is well within the architecture's reach.

3. **The normalizer is exonerated (my S7 hypothesis was backwards).** At 2 layers (BPTT, N=4):
   `count` **0.997**, `relu` 0.997, phasor-`softmax` 1.000, `real_softmax` 0.999 — but the
   query-dependent `qz` denominator I predicted would *help* is the **worst at 0.12**. The count
   normalizer is the right choice for both stability and recall.

4. **The binding constraint is RIPPLE-L0's depth-credit gap.** The *actual backprop-free model*
   (RIPPLE-L0) is stuck at **~0.13–0.24 at depth ≥ 2** (0.242 at N=4, 0.127 at N=8), while its BPTT
   twin reaches ~1.0 (0.996 / 0.998). The gap is **absent at 1 layer** (RIPPLE ≈ BPTT, ~0.12–0.22)
   and opens *only with depth*. This is
   exactly the L0→L1 depth-credit open question (docs/03 §7): per-block local losses cannot assign
   the cross-layer credit needed to *form* the 2-layer induction head MQAR requires. It is a
   **learning-rule** limitation, not architectural, and it is the real reason "depth didn't help"
   in S7.

**Bottom line:** the phasor linear-attention read + count normalizer + Hopfield cleanup is *not*
the limiter for sharp recall — a 2-layer WNN trained end-to-end matches a transformer. The binding
constraint is **RIPPLE's L0 depth-credit assignment**, with a secondary inherent single-layer
induction wall (shared with all 1-layer / linear-attention models). The count normalizer and the
cleanup are, at most, minor contributors.

---

## 1. Method

Central question: is the weak recall **architectural**, a **learning-rule** artifact, the
**normalizer**, or **capacity**? Controlled levers on the same `RippleLM`, matched seeds/steps/
data: learning rule (`ripple_grads` L0 vs `bptt_loss`+autograd twin); a standard 2-layer GPT
(RoPE softmax attention + MLP) as the task ceiling; monkey-patched read normalizers
(count/qz/relu/softmax/real_softmax) trained by the twin to isolate the read from the rule; depth
∈ {1,2}; lr ∈ {1e-3, 3e-3}. Probe: `experiments/mqar.py` (64-key band, 64-value band, masked
query-only objective); recall = argmax next-token accuracy at query positions, chance 1/64.

---

## 2. Results (verified numbers)

### 2.1 Learning-rate pathology / task ceiling (standard transformer)

At the **S7 default `lr=3e-3`**, a 2-layer transformer caps at **0.380** and a 4-layer at 0.433
(N=4; `exp_ceiling2_results.json`) — the task is *unsolved even by a transformer* at this lr, so
the S7 absolute numbers were read off a broken operating point. At **`lr=1e-3`** the same 2-layer
transformer reaches **0.996** (N=8, 2-seed mean; `exp7_results.json`). `lr=3e-3` is unstable
regardless of step budget; **all subsequent runs use `lr=1e-3`.**

| 2-layer transformer | lr=3e-3 (N=4) | lr=1e-3 (N=8) |
|---|---|---|
| recall | 0.380 | 0.996 |

### 2.2 The crux — architecture vs rule vs depth (lr=1e-3)

Every cell below is this author's own run (`verify_crux_results.json`: RIPPLE-L0 vs its BPTT twin,
identical `RippleLM`, 2 seeds, 3000 steps, lr=1e-3). Each is independently cross-checked: the two
BPTT-2L anchors match separate count-read runs — `rootcause_results.json` (N=4 = 0.997) and
`scaling_results.json` (N=8 = 0.998); the RIPPLE-2L-N8 cell (0.127) reproduces the finisher's
`exp7_results.json` (0.124). A standard 2-layer transformer reaches 0.996 at N=8
(`exp7_results.json`), fixing the task ceiling at ≈1.0.

| WNN model | N=4 | N=8 |
|---|---|---|
| **BPTT** twin, 2 layers | **0.996** | **0.998** |
| **RIPPLE-L0** (the real backprop-free model), 2 layers | **0.242** | **0.127** |
| BPTT twin, 1 layer | 0.217 | 0.140 |
| RIPPLE-L0, 1 layer | 0.167 | 0.121 |

- **1 layer:** RIPPLE ≈ BPTT (0.167 vs 0.217 at N=4; 0.121 vs 0.140 at N=8) — the rule is *not* the
  gap within a single block; both are weak because one linear-attention layer cannot do MQAR's
  shift-by-one induction (Olsson et al. 2022; the linear-attention recall–throughput trade-off,
  Arora et al. 2024 "Zoology").
- **2 layers, exact gradient:** the WNN-BPTT twin reaches ≈1.0 (0.996 / 0.998) — the architecture
  *can* do sharp recall, with the count normalizer and phasor read unchanged.
- **2 layers, RIPPLE-L0:** stuck at 0.24 (N=4) → 0.13 (N=8). **The same depth increase (1→2 layers,
  N=4) buys BPTT +0.78 (0.217→0.996) but RIPPLE only +0.075 (0.167→0.242): RIPPLE-L0 cannot convert
  depth into capability.** This is the L0 depth-credit signature (docs/03 §7): per-block
  deep-supervision gives layer 2 no path to learn to consume layer-1 features for induction.

### 2.3 Normalizer / read ablation (2-layer, BPTT, N=4)

| read | count | qz | relu | phasor-softmax | real-softmax |
|---|---|---|---|---|---|
| recall | 0.997 | **0.121** | 0.997 | 1.000 | 0.999 |

The count normalizer is fine; the signed query-dependent `qz` is the failure (it is also what
destabilized LM training, S4a). A closed-form SNR argument confirms the mechanism: for a matched
query the read numerator is `D·v_j + η` with `Var(η)≈(N−1)D`, so amplitude SNR `= √(D/(N−1))` is
**independent of the denominator** — the normalizer sets the read *scale* (absorbed by
`β_readout`/`W_v` in training), not its signal-to-noise ratio. S7's "√(d_k/N) blunts recall"
conflated read scale with read SNR.

### 2.4 Cleanup (P6)

The P6 conjecture — that finite-β Hopfield cleanup / delta-rule writes extend effective recall — is
**not supported** (`experiments/S7-mqar-result.md`): cleanup-on (`n_settle=15`) ≈ cleanup-off
(`n_settle=1`) ≈ cleanup+delta at every N (0.193 / 0.186 / 0.187 at N=4, monotone-in-N and
statistically indistinguishable). This is consistent with Theorem T1: at the config `β=1` the
settle is close to the β→0 linear regime, so it cannot sharpen the read appreciably. Cleanup is
therefore **near-decorative for recall**; it earns its place, if anywhere, as the FFN-role
knowledge memory, per the pre-registered docs/05 fallback. *(Not yet re-measured on the 2-layer
model that solves recall — the P6-negative above is the 1-layer result; re-testing cleanup at the
depth where recall is sharp, where interference bites harder, is the honest follow-up.)*

---

## 3. Architectural weaknesses (evidence + severity)

- **W1 — L0 depth-credit gap (HIGH; the dominant cause; a learning-rule limit, fixable by L1).**
  RIPPLE-L0's per-block local losses cannot form the 2-layer induction head. Evidence: BPTT-2L
  0.996/0.998 vs RIPPLE-2L 0.242/0.127, gap absent at 1 layer. The docs/03 §7 L0→L1 question, now
  shown to *bind* on a concrete capability.
- **W2 — single-layer induction wall (MEDIUM; inherent, shared with all 1-layer/linear models).**
  A 1-layer read (WNN *or* softmax attention) cannot do shift-by-one induction. Fix: depth (which
  the architecture supports, W1 permitting).
- **W3 — normalizer stability↔recall tension (LOW).** Real but minor; the count normalizer is
  SNR-neutral and the sharper `qz` fails at depth *and* destabilizes LM training. Keep count.
- **W4 — cleanup near-decorative for recall (LOW; documented fallback).** cleanup-on ≈ off ≈ delta
  at every N (1-layer, S7); P6 negative. Re-test at depth-2 recommended.
- **W5 — the √(D/N) interference wall (inherent; context, not a defect).** Sets the large-N /
  long-context envelope; not what capped small-N recall (BPTT-2L solves N≤8 at d_k=32).
- **W6 — S7's operating point (meta/process).** Absolute S7 numbers were confounded by an unstable
  lr; the qualitative degradation-with-N and P6-negative survive, the "architectural, count-
  normalizer" root cause does not.

---

## 4. Recommendations (prioritized)

- **R1 (highest impact): implement and test L1 (stack-EP, docs/03 §7).** ✅ **DONE — confirmed
  (`experiments/S8-L1-result.md`).** Inter-block credit was the change needed: RIPPLE-L1 moves the
  trained backprop-free model's 2-layer MQAR recall from the ~0.26 L0 floor to **0.998** = BPTT
  (0.996) = transformer. L1 reproduces the 2-layer BPTT gradient to cos=1.0 on every parameter.
  **Refinement to this report's diagnosis:** a 2×2 (rule × key-path) factorial shows recovery needs
  *both* the stack-EP depth-credit *and* the exact key gradient (`key_full=True`) — the RIPPLE-L0
  runs in §2 used the cheap `key_full=False` path, so the 0.24 floor conflated the depth-credit gap
  (fixed by L1) with the cheap-key approximation (which caps `∂L/∂W_k` at cos≈0.85, fixed by the
  exact key trace). L1-exact uses inter-block transposes (backprop-free/C2, but not
  weight-transport-free); the transport-free Vector-Field variant is the follow-on.
- **R2: for recall-bound use, ≥2 layers and lr≤1e-3.** Cheap, immediate.
- **R3: keep the count normalizer.** Do not resurrect `qz` for recall (fails at depth; destabilizes
  LM training). If single-layer sharpness is ever wanted, prefer relu/softmax over `Re(q†k)`.
- **R4: treat cleanup as the FFN-role knowledge memory, not a recall denoiser** (P6 negative).
- **R5 (lower priority): d_k is not binding at these N;** increase only if the high-N/long-context
  envelope (W5) becomes operative.

---

## 5. Re-audit the LM results — ✅ DONE (`experiments/S8-reaudit-result.md`)

Re-ran the S4/P2 TinyStories comparison as a `{key_full}×{lr}` factorial (2 seeds). Finding: the
reported ~9% RIPPLE-vs-BPTT gap was **not** the learning rule — it was two fixable gradient
approximations. Fixing the tied-readout embedding Jacobian bug and using the exact key path
(`key_full=True`) tightens the gap to **~3.3%** (2-seed mean, lr=2e-3). The lr was *not* the driver
here (lr=1e-3 gives a smaller gap but undertrains at 800 steps); the drivers were the embedding bug
and the cheap-key attribution. Consequence: default `key_full=True` for training; the S5/P4
absolute-loss figures should improve correspondingly (a full S5 re-run is the follow-up).

---

## 6. Provenance & how to continue

- **Verified runs.** Every number is traceable to an archived result JSON under `C:\tmp\mqar_lab`.
  The normalizer table (§2.3) is `rootcause_results.json`; the BPTT-2L "architecture solves it"
  anchor is `rootcause_results.json` (N=4=0.997) cross-checked by `scaling_results.json`
  (N=8=0.998); the lr pathology (§2.1) is `exp_ceiling2_results.json` (3e-3) and `exp7_results.json`
  (1e-3); the depth crux (§2.2, all eight cells) is this author's `verify_crux_results.json`
  (RIPPLE-L0 vs BPTT twin, 2 seeds), whose RIPPLE-2L-N8 = 0.127 independently reproduces the
  finisher's `exp7_results.json` (0.124) and whose BPTT-2L anchors match `rootcause`/`scaling`. All
  at lr=1e-3 unless noted.
- **Do R1.** Reuse `experiments/mqar.py` + lr=1e-3; compare RIPPLE-L0, RIPPLE-L1, BPTT at
  n_layers=2, N∈{4,8,16}. BPTT-2L (~1.0) is the target; RIPPLE-L0 (~0.2) is the floor.
- **Limitations.** Single GPU, 2 seeds, 2–3k steps, one probe (token-band MQAR, K=128). Effects are
  large (0.2 vs 1.0) and stable across seeds; the **L1 recovery is a prediction to test**, not yet a
  result. Scratch scripts live under `C:\tmp\mqar_lab` (not committed).

---

## Appendix — configuration

`RLMConfig(vocab_size=128, d_model=128, n_heads=4, beta=1.0, beta_readout=6.0, beta_n=0.05,
n_settle=15, key_full=False, delta_rule=False)`; d_k=d_vh=32. Adam(0.9,0.99), 40-step lr warmup,
grad-clip 1.0, seeds {0,1}. Ceiling GPT: pre-norm blocks, RoPE, 4× MLP, same width/heads/vocab.
