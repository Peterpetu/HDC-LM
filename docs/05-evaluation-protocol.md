# 05 — Evaluation Protocol

**Role:** The falsification instrument. Datasets, baselines, metrics, the seven
predictions `P1–P7` with concrete measurement procedures and kill criteria, the ablation
ladder that tests constitutional requirement C1, and the negative-results policy. No
claim in this program is accepted without a procedure here that could refute it
(project rules).

---

## 1. The framing rule (repeated from `00` §5)

Lead with **the learning rule and the certificate**, never a perplexity SOTA claim vs
transformers. The VSA-for-language representation battle was won by linear attention;
WNN's contributions are (a) a local backprop-free rule that trains it, (b) whatever
quality the finite-`β` cleanup adds, (c) native adaptive compute / continual learning.
Every prediction below is designed so that a *negative* result is informative and
publishable.

---

## 2. Datasets

| Dataset | Role | Tokenizer / `K` |
|---|---|---|
| **TinyStories** (Eldan & Li 2023) | primary trainability + coherence (P2–P4); small enough to iterate on 16 GB | SentencePiece 8192 |
| **The Pile** (subset) / **FineWeb** | scaling + bits-per-byte (P5), commensurable with GLA/GPT-2 | GPT-2 BPE 50257 |
| **OpenWebText (char)** | compression control (optional requential-coding instrument, §8) | bytes 256 |
| **MQAR / associative recall** (Arora et al. 2024) | effective-context stress test (P6, OP2) | synthetic |
| **Synthetic order-sensitive copy/induction** | isolates sequence-credit value (L0 vs L1, and delta-rule) | synthetic |

---

## 3. Baselines (do not skip any)

| Baseline | Purpose |
|---|---|
| **BPTT twin** — identical WNN architecture trained by truncated BPTT (autograd) | the primary reference; isolates *rule* from *architecture*. Everything is measured against it. |
| **GLA / DeltaNet (backprop)** | the field-standard linear-attention LM; external quality/throughput anchor. |
| **Small softmax transformer (nanoGPT-class)** | upper reference for perplexity at these scales. |
| **Eldan–Li TinyStories reference models** | the "SOTA-for-class" bar for P4. |
| **WNN-L0 vs WNN-L1** | isolates depth-credit (`03` §7). |
| **e-prop on the same recurrence** | closest existing local sequence rule; RIPPLE's exact trace should match/beat its approximate one. |
| **Forward-Forward LM** | local-learning sanity floor. |

The BPTT twin shares architecture, init seed, data order, and schedule with the RIPPLE
model so the *only* difference is the learning rule.

---

## 4. Metrics

- **Capability:** validation cross-entropy / bits-per-token; TinyStories GPT-graded
  rubric (grammar, coherence, creativity, plot) per Eldan–Li; The Pile bits-per-byte.
- **Rule fidelity:** `cos(RIPPLE update, BPTT gradient)`; `‖·‖` ratio; per-block and
  full-sequence (P1).
- **Read diagnostics:** fixed-point iterations vs token surprise (adaptive compute,
  M4); read entropy; safe-`β` margin `2/σmax(M_key)² − β` (`01` §6.2).
- **Memory diagnostics:** effective context `M_eff` vs measured recall (P6, `01` §4.3);
  eligibility-trace magnitude vs dependency length (P4).
- **Systems:** training tokens/sec; recurrent generation tokens/sec vs context length;
  peak VRAM.
- **Compression (optional instrument, §8):** requential code length `Σ_t KL(Q_t‖P_t)`.

All runs report seed, config hash, versions (`04` §8).

---

## 5. The seven predictions, procedures, and kill criteria

> **[P1 — gradient fidelity] (KILL GATE, S3)**
> *Claim:* the RIPPLE update direction has `cos > 0.9` with the truncated-BPTT gradient
> as `β_n→0`, at the per-step, full-sequence-trace, and (L1) stack levels.
> *Procedure:* build a small WNN (e.g. `d_model=64, L∈{1,2}, T∈{16,64}`); compute the
> BPTT gradient with autograd on the twin; compute the RIPPLE update; report `cos` and
> `‖ratio‖` for `β_n ∈ {0.2,0.1,0.05,0.02}`. Separately verify the exact trace equals
> autograd through the recurrence to `1e-5` (`test_traces.py`), and the two-path codebook
> `cos=1.0` (`test_two_path_codebook.py`). Per-step is already validated (whitepaper D.2,
> cos→1.000); this extends it to the sequence and stack.
> *Kill:* `cos < 0.9` at the smallest stable `β_n` after the exactness tests pass ⇒
> the local rule does not track the gradient; **the program's central mechanism fails**.
> Publish the negative result with the diagnosis (trace error vs EP bias vs depth-credit).

> **[P2 — trainability]**
> *Claim:* WNN-L0/L1 (5–35 M) reaches validation loss within **20%** of the BPTT twin on
> TinyStories.
> *Procedure:* train RIPPLE-L0, RIPPLE-L1, and the BPTT twin at WNN-S with identical
> arch/seed/data/schedule; compare best val loss and generation coherence.
> *Kill/branch:* L0 gap >20% ⇒ escalate to L1; if L1 also >20% after tuning ⇒ negative
> trainability result — report where the gap comes from via P1 cosine per block and the
> ablations (§6).

> **[P3 — depth credit]**
> *Claim:* WNN-L1 closes **≥ half** the remaining L0↔BPTT val-loss gap.
> *Procedure:* the three curves from P2; report gap-closed fraction per depth `L∈{2,4,8}`.
> *Cannot fail uninformatively:* the number quantifies how much local depth-credit
> recovers — a contribution regardless of value.

> **[P4 — SOTA-for-class on TinyStories]**
> *Claim:* at matched params/tokens, WNN beats the Eldan–Li reference models on val loss
> and the GPT-graded rubric.
> *Procedure:* scale to WNN-S/35 M; tune; grade 100 generations with the Eldan–Li rubric
> vs the reference checkpoints.
> *Miss ⇒* analyze via ablations; report honestly (framing rule §1).

> **[P5 — Pile bits-per-byte]**
> *Claim:* within **10%** of a matched-size GLA/GPT-2-class baseline at matched token
> budget.
> *Procedure:* WNN-M on a Pile subset; compare bpb to GLA-backprop and the transformer.
> *Miss ⇒* diagnose optimization vs representation via the ablation ladder (§6) and P1.

> **[P6 — cleanup extends effective context] (constitutional C1 test)**
> *Claim:* delta-rule + finite-`β` Hopfield cleanup extends MQAR/associative-recall
> effective context vs vanilla linear attention at equal `D` (`01` §4.3 conjecture).
> *Procedure:* MQAR with increasing key–value counts and distances; compare WNN
> (cleanup on) vs WNN (cleanup off = raw read, Theorem-T1 corner) vs GLA at equal `D`.
> Plot recall vs `M_eff`.
> *Kill/branch:* no measurable extension ⇒ cleanup is decorative in the contextual path
> ⇒ **drop it there** (keep the FFN-role knowledge memory), simplify (`00` §4 R5). If
> cleanup adds *nothing anywhere*, C1 is threatened — that is a decisive, reportable
> finding about whether Hopfield cleanup earns its place.

> **[P7 — throughput]**
> *Claim:* training tokens/sec within **3×** of matched GLA-BPTT; recurrent generation
> ≥ transformer tokens/sec at 2k+ context.
> *Procedure:* wall-clock at WNN-S/M on the same GPU; generation sweep over context
> length.
> *Miss ⇒* engineering milestone (`04` §7) before scaling; not a correctness failure.

---

## 6. The ablation ladder (isolates every mechanism; tests C1)

Each ablation removes one mechanism and measures the loss/recall change. A mechanism
that can be removed for free was decorative — a C1 failure to be reported.

| Ablation | Removes | Predicted effect | Constitutional stake |
|---|---|---|---|
| cleanup off (β→∞, 1 step) | knowledge memory read | ↑ loss on recall-bound tokens (P6) | C1: is Hopfield cleanup load-bearing? |
| binding → real concat | phasor binding | ↓ capacity, ↑ interference | C1: is VSA binding load-bearing? |
| single timescale (`H=1` γ) | multi-timescale heads | ↓ long-range recall | M6 |
| delta-rule off | error-correcting write | ↓ recall at high `M_eff` | M6 |
| `μ`-gate off (`g1=0`) | neuromodulation | ↑ instability / interference | M2/M5 continual |
| readout-path off | two-path codebook | **trains backwards** (D.3) | rule correctness |
| trace off (current-step only) | sequence credit | ↓ on long-dependency tasks (P4) | `01` §9 |
| L0 vs L1 vs BPTT | depth credit | quantifies the gap (P2/P3) | `03` §7 |

---

## 7. Reproducibility & negative-results policy

- Every result carries: seed, config hash, tokenizer, dataset split, versions, hardware,
  wall-clock (`04` §8). Another researcher can replicate from the config alone.
- **Negative results are results** (project rules). Each kill/branch
  outcome is written to `experiments/` with what was tried, what was observed, and the
  diagnosis. A failed P1 or P2 is a publishable finding about local learning at scale,
  not a discard.
- Regression locks: `test_gradient_fidelity`, `test_two_path_codebook`, `test_traces`
  are permanent CI gates; once green they must stay green (project rules).

---

## 8. Optional instrument — requential coding (later, `06`)

Once the core validates (S4+), the requential code length `L_req = Σ_t KL(Q_t‖P_t)`
(Qiu et al. 2026) can be computed for the RIPPLE model and its BPTT twin at matched loss
to test whether local learning discovers **equally compressible (hence equally
generalizing)** structure. This is a strictly stronger test than perplexity and a
stretch goal (whitepaper §6, M4/M6). Not on the critical path; adopted verbatim from
Qiu et al.'s estimator if pursued.

---

## 9. Stage gates (summary)

| Stage | Gate | Predictions |
|---|---|---|
| S1 | algebra + capacity tests green | — |
| S2 | 100% associative recall; fixed-point convergence | — |
| **S3** | **exact-trace = autograd; two-path cos=1; RIPPLE cos>0.9** | **P1 (KILL)** |
| S4 | L0/L1 within 20% of BPTT twin | P2, P3 |
| S5 | beats Eldan–Li reference | P4 |
| S6 | Pile bpb within 10%; throughput within 3× | P5, P7 |
| S7 | MQAR context extension; continual-learning demo | P6 |
| S8 | space-time EP; workspace routing; ARC probes | research |
