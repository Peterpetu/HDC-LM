# 10 — S6 Protocol: WNN-162M vs GPT-2/NeoX-162M on The Pile (frozen, pre-registered)

**Role:** the canonical, self-contained protocol for the S6 headline experiment
(docs/05 P5 + P7). Fixed **before** any S6 training run; every deviation from it is a
reportable event, not a tweak. The plan of record (with full revision/verification
history: adversarial critic REJECT → rev. 2 → four-way independent verification
APPROVE-WITH-CHANGES → rev. 4) is archived in the session plan file; this document is
the repo's operational copy. Gate results live in `experiments/S6-v0-anchor.json` and
subsequent `experiments/S6-*.json`.

---

## 1. The reference run (the comparison anchor)

External, pre-existing, **read-only**: `C:\koodi\NCALM\experiments\GPT-2\
pile_250m_gpt2_20260319_141247\` — a Pythia-160M-clone transformer trained by its
author (custom harness, not the EleutherAI lib) on The Pile.

| Item | Value (verbatim from `config.json` / `analysis.json`) |
|---|---|
| Model | d=768, L=12, H=12 (d_head 64), d_ff 3072, vocab **50304**, untied embeddings, partial RoPE 25% (rotary_dim 16), parallel residuals, dropout 0, LayerNorm ε=1e-5 |
| Params | **162,322,944** = 85,054,464 layers + 2×38,633,472 embeddings + 1,536 final-LN |
| Tokenizer | `EleutherAI/gpt-neox-20b` HF fast (NeoX/Pythia BPE), vocab 50,304, EOS id 0; cached at `C:\koodi\NCALM\hf_cache\` |
| Data | `C:\koodi\NCALM\data_cache\pile_neox50k_tokens_250m.bin` — flat int32 memmap, 250,000,000 tokens of `monology/pile-uncopyrighted` (HF streaming), docs joined by EOS(0); sidecar `.meta.json` |
| Training | T=1024; micro-batch 8 × grad-accum 8 = 65,536 tokens/step; 3,815 steps = **250,019,840 tokens**; AdamW (β 0.9/0.99, ε 1e-8), lr 2e-4 cosine→0.1× (warmup 188), wd 0.1 (decay group only), clip 1.0, AMP, **seed 42** |
| Hardware | RTX 4080 SUPER 16GB (the same GPU as S6); 49.8k tok/s; 1.39 h; peak 6.7 GB |

**Tokenizer deviation, documented:** the original S6 request said "the GPT-2
tokenizer"; the reference actually used the **NeoX tokenizer** (vocab 50304).
Commensurability with the reference (and reuse of its tokenized cache) requires the
NeoX tokenizer; that is what S6 uses.

### 1.1 The anchor caveat and the frozen re-anchor (Gate V0 result)

The reference's own eval was **stochastic**: val batches drawn by `torch.randint` on
the unseeded *global* RNG (`ncalm/data/dataset.py:458-469`), consumed per eval
(`ncalm/trainer.py:413-414`) under autocast. Adjacent end-of-run evals swing
±0.13 nats (σ ≈ 0.05–0.06 from `metrics.jsonl`). Consequently the archived numbers
(final val CE 3.3459 / PPL 28.39; "best" PPL 27.57 @3400) are noisy draws;
"best" is a min-of-noisy-draws, biased low.

**Gate V0 (2026-07-20, `experiments/S6-v0-anchor.json`) re-anchored the reference
checkpoints under FROZEN-EVAL (§2):**

| Checkpoint | Archived (stochastic) | **FROZEN-EVAL anchor** |
|---|---|---|
| `final_model.pt` (step 3815) | CE 3.3459 / PPL 28.39 | **CE\*_ref-final = 3.3638 / PPL 28.90** |
| `best_model.pt` (step 3400) | PPL 27.57 ("best") | CE 3.3868 / PPL 29.57 — *worse than final* |

The frozen protocol confirms the predicted bias: the "best" checkpoint is not best.
**The S6 headline comparison target is CE\*_ref-final = 3.3638 nats (PPL 28.90).**
Archived numbers are reported as historical context only.

V0 also certified: (i) the reference model class loads and **trains** in our env
(1-step forward/backward/AdamW smoke: loss 10.97→9.70); (ii) our evaluator agrees
with the reference-side evaluator (the model's own `forward_train` CE accumulation,
Trainer.evaluate semantics, fp32) on **identical fixed batches** to |Δ| = 1.6e-08
(bound 1e-4).

---

## 2. FROZEN-EVAL — the deterministic evaluation protocol

Implemented in `wavenn/pile_data.py::FrozenEval` + `experiments/eval_pile.py::
frozen_eval_ce`. Definition (exact):

- Data ids[0..N), N = 250,000,000 (the reference cache, memmapped **read-only**).
- Split (ported from `dataset.py:444-447`): train = [0, int(0.995·N)) =
  [0, 248,750,000); **val V = ids[248,750,000 .. 250,000,000)**, |V| = 1,250,000.
- Windows k = 0..K−1, **K = (|V|−1)//1024 = 1,220**, x_k = V[1024k .. 1024k+1024) —
  non-overlapping, fixed order, fully inside V; batched 8 windows/batch → **153
  batches** (last partial).
- Supervision = the reference's within-window shift (`gpt2/model.py:297-301`):
  CE(logits[:, :-1], x[:, 1:]) — **1023 targets/window, 1,248,060 supervised
  tokens**, no cross-window target. Expressed in the WNN interface as
  (x = window, y = shift-left with ignore_index −100 at the last position).
- **fp32** logits and CE (no autocast — deviation from the reference's AMP eval,
  disclosed; both sides are re-anchored under the identical protocol so it cannot
  bias the comparison). CE = token-weighted mean in nats, accumulated as fp64 sum of
  per-batch fp32 sums; **PPL = exp(min(CE, 20))**; bits/byte = CE/ln2 ÷
  (bytes-per-token of V, measured once by detokenizing the K windows).
- Deterministic by construction (no RNG anywhere): repeated evaluation of the same
  checkpoint is bit-identical.
- During training: the full frozen set every 200 steps (~19 evals/run ≈ 24M
  forward-only tokens ≈ 1–2% overhead, budgeted).

## 3. Training-side protocol (both learners)

- **Data/order:** the same memmap; train windows sampled uniformly with starts in
  [0, n_train − 1024 − 1) (the reference's `randint` bound, `dataset.py:460`), from a
  **dedicated seeded generator** (deviation from the reference's global-RNG draw,
  documented; identical distribution). Seed **42**.
- **Token budget:** micro-batch 8 × T=1024 × accum 8 = 65,536 tokens/step; **3,815
  steps = 250,019,840 consumed tokens** (1023 supervised positions per window).
- **Schedule shape:** cosine → 0.1× with 188-step linear warmup (the reference's).
- Divergence/NaN ⇒ halt & diagnose (never silent retry). All outcomes published.

## 4. The WNN-162M configuration (full config; plan §2)

`RLMConfig`: vocab 50304, d_model 768, n_layers 12, n_heads 12, **P = 11,116**
(total params 162,321,552 = within 0.00086% of the reference's 162,322,944; exact
integer match impossible — would need P = 11,116.151), β=1.0, β_ro=6.0, β_n=0.05,
n_settle cap 15 (free phase adaptive tol 1e-4 per-row; **± nudged phases share one
iteration schedule derived from the free phase** — the §M.4a symmetric-stopping
rule), η=0.3, γ logspace with γ_max=0.999 (learned; code default 0.99 overridden —
flagged), delta_rule=False (its gradient does not track BPTT, docs/03 §4.6),
key_full=True (computed Tkv-free via the adjoint reordering, §6), deep_supervision=
True, depth_mode="L1", gate (1.0, 0.5, [0.1,3]) with **Cbar init = ln 50304 =
10.826**, tied unit-norm readout. LocalAdam (0.9/0.99, ε 1e-8); lr selected by the
pre-registered B.1 pilot ({5e-4, 1e-3, 2e-3} × 400 steps, lowest frozen-eval CE,
ties → lower lr).

Parameter-matching consequence (pre-registered): total-matching gives the WNN
**+45.4% non-embedding parameters** (123.7M vs 85.1M; the reference's other 77M is
untied embeddings). A WNN *loss* is robust to this asymmetry; a narrow WNN *win*
would be confounded by it. The write-up reports both matchings (non-embedding match:
P = 6,924).

Headline variants: **A** replay OFF (250M fresh tokens — the matched run);
**B** replay ON (`ReservoirBuffer`, A-Res surprise-prioritized, 3,000 sequences,
replay_frac 0.4; same 3,815 steps ⇒ ~150M fresh + ~100M replayed; pre-registered to
trail A on this i.i.d. stream — its ground is the Phase-E continual-learning
contrast).

## 5. Codebook-utilization monitoring (the dead-codebook guard)

63.1% of the WNN's parameters (102.4M) sit in the 12 Hopfield codebooks at
P = 14.5d — far outside the validated P = 2d regime; at β=1 the settle-logit spread
(~‖r‖/√d) may be ≪ ln P ≈ 9.32 ⇒ near-uniform softmax ⇒ dead codebooks. Metrics, at
the free fixed points:

- normalized read entropy H̄ = E_rows[−Σ_p a_p ln a_p]/ln P (1 = uniform);
- participation ratio PR = (Σ_p ū_p)²/Σ_p ū_p², ū_p = E_rows[a_p];
- cleanup contribution ΔCE = CE(cleanup ablated: r=c readout) − CE(full);
- gradient share ‖grad_M‖/‖grad_Wv‖.

**Dead verdict:** H̄ > 0.98 AND |ΔCE| < 0.01 nats. Measured at three scales:
(i) the B.2 scaled twin (d=256/L=4/P=3,712, 1,000 steps); (ii) the B.1 full-config
pilots (the risk metric worsens ~1.9× from d=256 to d=768 — the twin alone does not
certify); (iii) **live during Phase C** (per-block H̄, PR at every eval) with a halt
rule: dead verdict at C.1 step 200 on ≥10/12 blocks, flat since start ⇒ halt and
present the pre-registered options (non-embedding match P=6,924 / d=1024-H=16-
P=5,945 fallback / nominal-match caveat) — user decides. **Re-gate rule:** any shape
change ⇒ Gate-A throughput, B.1 lr selection, and the feasibility budget are re-done
at the new shape before Phase C.

## 6. Kernel mathematics (chunked scans, adjoint reordering, frozen force)

Canonical derivations live in the plan §M and are reproduced in `wavenn/scan.py`
module docs with the code. Summary of the binding rules:

- **§M.1 chunked scans:** intra-chunk decayed masks Γ[t,i] = γ^{t−i} by **direct
  powers** (bounded exponent ≤ L_c−1) + multiplicative inter-chunk carry γ^{L_c}.
  The factored form γ^t·γ^{−i} is **prohibited** (overflow for small learned γ);
  underflow of γ^{t−i} to 0 is the correct limit (benign).
- **§M.2 adjoint reordering:** all projection/decay gradients are exact finite-sum
  reorderings of the reverse-time adjoint Q_i = conj(q_i)(g_i/den_i)ᵀ + γQ_{i+1}
  (and Q̇_i = Q_{i+1} + γQ̇_{i+1} for the decay path): gWv = Σ_i Re(k_iᵀQ_i)⊗x_i-form,
  gWk = Σ_i (−Im(k_i ⊙ (Q_i v_i)))⊗x_i, gWq current-step, g_logγ via Q̇ — **no
  Tkv/Gv trace materialization**. Valid ONLY under (i) pure accumulation and (ii)
  the deterministic count normalizer — **runtime-asserted**. **Scope restatement:**
  docs/01 §9's "online, O(D·d_in)" claim holds for the forward-trace form; the S6
  kernel computes the *same exact gradient* by a reverse-time batch sweep (as L1
  routing already does). The online form remains the hardware-locality statement;
  C2 (no autograd over parameters) is untouched. Certification: float64 relative
  ≤1e-9 + fp32 rel ≤1e-4 against BOTH oracles (`projection_grads` trace loop AND the
  `contextual_input_grad` cross-oracle) and autograd; test matrix covers γ ∈
  {1e-4, 0.9, 0.999}, T non-divisible by L_c spanning ≥3 chunks, B>1, masked rows.
- **§M.4 frozen readout force:** the ± nudged settles use the local CE **linearized
  at r_free** (a constant force — the same mechanism as the g_top credit, stated as
  a linearization, not an identity). Leading-order exact; O(β_n²) symmetric-nudge
  bias unchanged (independently re-derived by the verification panel's theory
  referee). **§M.4a symmetric stopping (binding):** both signs run one shared
  per-row iteration schedule derived from the free phase; independent per-sign
  adaptive stops would add an O(tol/β_n) asymmetric bias the symmetric nudge cannot
  cancel. Empirical certificate: the A.5 fidelity mini-gate (cosines within 1% of
  the committed fixed-settle values) — the existing cos=1.0 g_top test validates the
  constant-force machinery, not CE linearization.
- **§M.6 gradient accumulation:** token-weighted average of per-micro scaled raw
  gradients ≡ union-batch gradient (exact identity); one clip → one gate μ → one
  LocalAdam step → one Cbar EMA per optimizer step. Legacy single-call path
  regression-locked bit-unchanged.

## 7. P5 / P7 scope statement (pre-registered)

- **P5** (docs/05: "Pile bits/byte within 10% of a matched GLA/GPT-2-class baseline
  at matched budget") is adjudicated against the **Phase-D.2 GLA-class baseline**:
  a 162M-total-matched gated-linear-attention transformer (our chunked
  linear-attention forward + SwiGLU FFN in place of the Hopfield codebook),
  BPTT/AdamW, same data/protocol/seed/tokens. **Pre-registered recipe:** the
  reference AdamW recipe verbatim (lr 2e-4 cosine→0.1×, warmup 188, wd 0.1, β2
  0.99, clip 1.0, AMP) plus a mirrored 3-point lr mini-sweep {1e-4, 2e-4, 4e-4} ×
  400 steps under the B.1 selection rule — fixed now, so the comparator cannot be
  under- or over-tuned post hoc. The softmax reference (§1) is the upper reference.
- **P7** (docs/05: "training throughput within 3× of matched GLA-BPTT") is measured
  against the D.2 baseline's tok/s. **Pre-registered expectation: P7-as-registered
  will likely be MISSED by roughly an order of magnitude** (projected WNN 500–1,200
  tok/s vs expected GLA-BPTT ~8.7–17.4k tok/s ⇒ ~10–30×). Dominant causes are
  structural (3 settle phases, the P=11,116 codebook, vocab-facing EP terms) plus
  kernel maturity. S6 reports the measured ratio with its decomposition; a miss is
  published as a miss, not reframed.
- Headline interpretation bands (vs CE\*_ref-final = 3.3638): ≤ +10% strong
  positive; +10–25% expected band; > +25% negative result — published either way
  with the Phase-D decomposition (rule gap at depth: d=256/L=12 RIPPLE-L1 vs BPTT
  twin, 50M tokens, seeds 42/43; width extrapolation stated as an assumption).

## 8. Verification protocol

Per-phase Layer-1 verification by independent reviewers (IV1 code-vs-plan; IV2 independent
math re-derivation; IV3 empirical reproduction from a clean process; IV4 compliance:
C2 grep, NCALM read-only, reproducibility metadata, guards present) — any FAIL
blocks the branch. After Phase F, the **Layer-2 review board** (full code-vs-plan
sweep; results audit — every write-up number traced and recomputed; claims audit;
adversarial pass) gates the merge. Reports archived under `experiments/verify/`.
