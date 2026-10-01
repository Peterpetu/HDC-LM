# 00 — Vision, Constitution, and the Axiom Reset

**Project:** Wave Neural Network (WNN)
**Status:** Design specification. No experimental results reported here; every empirical
claim is labelled `[PREDICTION Pn]` and carries a kill criterion (see `05-evaluation-protocol.md`).
**Document role:** This is the root specification. It states the constitutional
requirements, records the first-principles axiom reset that produced the architecture,
documents the alternatives that were considered and rejected (with reasons), and fixes
the honest ceiling of what the program can and cannot claim. Every other document in
`docs/` refers back to a section here.

---

## 0. One-paragraph statement

A language model is a hierarchy of **associative memories at different timescales**.
Inference is *resonant retrieval* from a superposition; learning is *local plasticity*
that writes those memories; and — the load-bearing claim — the two are the **same
physical operation at different decay rates**. The Wave Neural Network makes this
ontology explicit: representations are complex phasors (Fourier Holographic Reduced
Representations), the core operations are the Vector-Symbolic algebra (binding,
bundling, permutation, cleanup), and every parameter is trained by a **local,
backpropagation-free rule** (equilibrium propagation + contrastive Hebbian codebook
updates + exact eligibility traces through linear recurrences + three-factor
neuromodulation). We call the learning rule **RIPPLE** (Resonance-Induced Plasticity
with Phase-Local Eligibility). The bet is not "beat transformers with discrete HDC"
— that bet is known to lose. The bet is that the winning architecture for language is
*already* VSA-shaped (linear attention is VSA with a learned codebook), and that a
**local rule can train it**, unlocking on-chip training, native test-time learning,
and adaptive per-token compute that global backpropagation forecloses.

---

## 1. Constitutional requirements (non-negotiable)

These two properties are **structural and performance-bearing**, not decorative. A
design that satisfies them only cosmetically is rejected. Both are testable (see the
"constitutional test" in each row).

| # | Requirement | Why it is core, not decoration | Constitutional test |
|---|---|---|---|
| **C1** | **HDC / VSA mechanisms at the architectural core.** Binding, bundling, permutation, and cleanup are the primitive operations of every layer; the state is a holographic superposition; representations are FHRR phasors. | The contextual memory *is* a bound key–value superposition read by unbinding (§`02`). The knowledge memory *is* modern-Hopfield cleanup against a codebook. Remove the VSA algebra and there is no model left — not a model with VSA bolted on. | Ablating binding→plain concatenation, or cleanup→identity, must **degrade** loss measurably (`05` P6, and the ablation ladder). If ablation is free, the mechanism was decorative and the design has failed C1. |
| **C2** | **Backpropagation-free learning.** No global reverse-mode autodiff through the network for the model's own training. Credit assignment is local: two-phase EP differences, Hebbian codebook updates, and forward-propagated eligibility traces. | This is the entire research contribution. A locally-trained competitive LM is the first credible route to on-chip LM *training* and native continual learning. | The training path uses **zero** `loss.backward()` over model parameters. Autograd is permitted *only* in (a) baseline "BPTT twins" used as measurement references and (b) finite-difference gradient checks in tests. Any `.backward()` in the WNN training loop is a C2 violation. |

**Interaction of C1 and C2.** The two are not independent constraints that happen to
coexist; they are chosen because they *fit*. VSA operations are algebraically simple
(bind = complex multiply, bundle = add, cleanup = one Hopfield step), which makes the
per-parameter local gradient **closed-form and cheap** — no autodiff graph is needed
because the derivative of each primitive is itself a primitive. The recurrence is
**linear in the state** `S`, which is exactly the condition that makes an eligibility
trace *exact* rather than approximate (contrast e-prop on nonlinear RNNs). C1 is what
*makes C2 tractable*. This is the central design synergy, and it is why "HDC" and
"backprop-free" belong in the same architecture rather than being two orthogonal
wish-list items.

---

## 2. The unifying principle and its grounding in existing models

**Principle.** *Attention, feed-forward layers, and the residual stream are all
associative memories; they differ only in read/write timescale and in whether they are
content-addressed by the current token (attention) or by a fixed learned address space
(FFN). A transformer is a stack of these memories trained by a global rule that ignores
their memory structure. WNN keeps the memory structure and trains each memory locally.*

This is not a metaphor. Each equivalence below is established in the literature and is
re-derived in `01-mathematical-foundations.md`:

| Transformer component | Is literally | Reference |
|---|---|---|
| Linear-attention KV state `S_t = Σ φ(k_i) v_i^⊤` | a **bound superposition** of key→value pairs (VSA binding + bundling) | Schlag, Irie & Schmidhuber, ICML 2021 (fast-weight programmers); Katharopoulos et al., ICML 2020 |
| Attention read `φ(q)^⊤ S_t` | **unbinding** by the query + implicit cleanup | ibid.; Ramsauer et al., ICLR 2021 (Hopfield = attention) |
| Feed-forward network `W_2 σ(W_1 x)` | a **key–value memory**: rows of `W_1` are keys, rows of `W_2` are values | Geva et al., EMNLP 2021 |
| Rotary position embedding (RoPE) | **FHRR positional phase binding**: multiply component `j` by `e^{i θ_j t}` | Su et al., 2021 (RoFormer); identical to Plate 1995 permutation-by-phase |
| Residual stream | a **bundling bus**: successive layers *add* (superpose) their contributions | Elhage et al., 2021 (residual stream as communication channel) |

The transformer is therefore a *shadow* of the WNN ontology — the right operations,
performed in the real domain, addressed implicitly, and trained non-locally. WNN lifts
the shadow into its native form: **complex** (phase carries binding and position),
**explicit** (memories are named objects with read/write APIs), and **locally trained**
(each memory owns its plasticity).

---

## 3. The axiom reset (Transformative Creative Reasoning record)

The architecture was designed by explicit Tier-3 Transformative Creative Reasoning
(Universe of Thoughts). The method: surface the *hidden* rules that every mainstream
LLM obeys but that were never stated as requirements, then mutate them (drop / vary /
add) and explore the transformed space. Donor rules were imported from distant domains
— **holography** (distributed storage, graceful degradation), **hippocampal
neuroscience** (theta–gamma phase coding, three-factor plasticity, complementary
learning systems, sharp-wave-ripple replay), **CDMA spread-spectrum** (near-orthogonal
codes recovered by correlation), **statistical annealing** (temperature as a
denoise/sharpen knob), and **market clearing** (local price signals coordinating a
global allocation without a central optimizer).

The seven mutations below define WNN. Each row: the hidden axiom, the mutation, the
donor domain, and the concrete architectural consequence.

| # | Hidden axiom of mainstream LLMs | Mutation | Donor | Architectural consequence |
|---|---|---|---|---|
| **M1** | Training and inference are disjoint phases; weights freeze at deployment. | **Drop.** Writing a memory *is* learning; there is a continuum of plasticity timescales. | Hippocampal complementary learning systems (McClelland et al. 1995); sharp-wave-ripple replay. | Three coupled timescales: fast state `S_t` (per-token), an episodic buffer (per-episode), slow weights (consolidated by replay). Test-time learning is native, not a fine-tuning afterthought. (`06`) |
| **M2** | Credit assignment must be global reverse-mode backprop. | **Vary** with a local rule from energy-based physics. | Equilibrium propagation on oscillators (Scellier–Bengio 2017; Ernoult et al. 2019); three-factor plasticity (Frémaux–Gerstner 2016). | RIPPLE: two-phase EP settling + local contrastive differences + **exact** eligibility traces + neuromodulatory gate. (`03`) |
| **M3** | The loss touches only the top layer; hidden layers get error second-hand. | **Vary.** Every memory owns a local objective. | Predictive coding (each level predicts the one below, Rao–Ballard 1999); market-local price signals. | Every block has its own vocabulary-cleanup readout and local EP loss; depth-credit is a measured research ladder (L0/L1/L2), never assumed. (`03` §L-ladder) |
| **M4** | Prediction is a single feed-forward shot with fixed compute per token. | **Drop.** Compute per token adapts to difficulty. | Statistical annealing; Hopfield settling to a fixed point. | Per-token iterative cleanup settling; iteration count gated by read entropy → **native adaptive compute** (harder tokens think longer). (`02` §cleanup) |
| **M5** | Position is an additive perturbation of content. | **Vary.** Position is a *unitary rotation* of content — an algebra element. | FHRR permutation (Plate 1995); RoPE is the shadow. | Position enters as componentwise phase rotation `e^{iθ_j t}`; order is protected by the VSA permutation operator, and RoPE falls out as the β-real special case. (`01` §RoPE) |
| **M6** | There is no writeable medium-term memory between the KV-cache and the weights. | **Add** an error-correcting medium-timescale store. | Widrow–Hoff/LMS delta rule; DeltaNet (Yang et al. 2024). | Multi-timescale heads with log-spaced decays `γ_h`; **delta-rule writes** (remove the old value before writing the new) on medium heads; episodic store for the slow path. (`02` §memory) |
| **M7** | The residual stream is inert plumbing. | **Vary.** It is an explicit VSA bundling bus with a capacity budget. | Holography (superposition capacity); CDMA (code-division multiplexing). | Blocks *bundle* (add, with magnitude normalization) into a shared phasor bus whose interference budget `√(D/M)` is tracked (`01` §capacity, OP2). |

**Why this is transformative, not combinational.** No single mutation is novel in
isolation (EP exists; delta rule exists; phase position exists). The transformation is
in the *rule set*: once M1 (no train/infer split) and M2 (local credit) are dropped
together, the design space that opens contains architectures — like WNN — that are
*unreachable* from the standard "differentiable feed-forward stack trained by SGD"
frame. The creative work was recognizing that the VSA algebra (C1) is precisely what
makes the mutated rule set *implementable* (C2), because linear + phase operations have
closed-form local gradients.

---

## 4. Rejected alternatives (documented negative design decisions)

Per the project rules, design choices are defensible on technical grounds, and rejected paths
are recorded, not silently dropped.

| Rejected option | Why it was attractive | Why rejected |
|---|---|---|
| **Pure-phase (torus) state.** Keep the entire running state on the unit torus `z_j = e^{iθ_j}`. | Maximally "neuromorphic"; maps directly to phase oscillators; elegant. | **A pure-phase state cannot hold a weighted superposition** — the weights live in the magnitudes, which the torus fixes to 1 (whitepaper v1.1 Appendix D self-falsification). The *codewords* stay unit-magnitude; the *state* must live in ℂ^D with amplitude. This is a solved constraint, not an open question. (`01` §2.3) |
| **Discrete / bipolar VSA on raw text** with a frozen random codebook. | Simplest possible HDC LM; fully local by construction. | Known ~59-point-class accuracy gap versus learned features (whitepaper §3.2, R5/OP3). Bundling is a *set* operation — order-invariant — so a bundling-only model is at best n-gram quality. Order-sensitivity (M5) and a *learned* codebook (C1 via linear attention) are both mandatory. |
| **Evolutionary / clonal-selection codebook learning** (mutate-and-select codewords). | Genuinely gradient-free; biologically plausible (immune analogy). | No credible path to LM-scale sample efficiency; population methods scale poorly with parameter count. EP gives a *directional* local signal (cos→1 with the true gradient, whitepaper App. D.2) at O(1) population size — strictly better where it applies. |
| **Forward-Forward (Hinton 2022) as the sole rule.** | Purely local, no settling, simple. | Contrastive goodness per layer has not shown competitive language modeling; no natural read/write memory semantics; discards the associative-memory structure that is the whole point (C1). Retained only as a baseline (`05`). |
| **Softmax (quadratic) attention with a local rule.** | Best-known quality; could try to make EP work on it. | Quadratic attention has **no O(1) recurrent state**, so there is no linear recurrence and hence **no exact eligibility trace** — the mechanism that makes C2 tractable evaporates. Linear/holographic memory is not a compromise for speed; it is the structural precondition for local training. |
| **Backprop with a "biologically inspired" flavor** (feedback alignment, target prop). | Closes the accuracy gap fastest. | Violates C2 in spirit and often in letter (still a global backward pass). The program's value is precisely in *not* doing this; a locally-trained result is the publishable object. |

---

## 5. What WNN can and cannot claim (the honest ceiling)

Stated once, plainly, and repeated in `05` and `06`.

**Cannot claim:** "Discrete/holographic VSA beats transformers at language." The
VSA-for-language battle was already won — by linear attention, on GPUs, trained by
backprop. WNN inherits that architecture family; it does not out-represent it.

**Can plausibly claim, each falsifiable:**

1. **A local, backprop-free rule trains a competitive VSA/linear-attention language
   model** to within a stated tolerance of its backprop twin (`05` P2). First credible
   route to on-chip LM *training*.
2. **Finite-temperature cleanup buys measurable quality** over the raw linear-attention
   read on recall-bound tasks (`05` P6), or it does not — and the ablation says which,
   cleanly.
3. **Native adaptive compute and test-time/continual learning** fall out of the
   architecture (M1, M4) rather than being bolted on (`06`).
4. **A limiting-case unification:** standard linear-attention training is the β→∞,
   single-Newton-step corner of RIPPLE (whitepaper Theorem T1; re-derived in `01`).
   This sanity-checks the rule and predicts where finite-β should differ.

**The framing rule (R1 in `05`):** lead with *the learning rule and the certificate*,
never with a perplexity SOTA claim against transformers. Every outcome — including
every negative one — is designed to be informative and is published (`05` §negative
results).

---

## 6. Success, in one table

| Horizon | Concrete success | Where verified |
|---|---|---|
| **Correctness of the algebra** | Binding inverts; permutation is order-sensitive; capacity matches `√(D/M)`. | `tests/test_algebra.py`, `tests/test_capacity.py` |
| **Correctness of the rule** | RIPPLE update cos > 0.9 vs BPTT gradient as `β_n→0`; two-path codebook cos = 1.0; exact trace = autograd to fp tolerance. | `tests/test_gradient_fidelity.py`, `tests/test_two_path_codebook.py`, `tests/test_traces.py` — **kill gate** |
| **It learns language** | WNN-L0/L1 within 20% val-loss of the BPTT twin on TinyStories; coherent generation. | `05` P2/P3, S4 |
| **SOTA-for-class** | Beats Eldan–Li reference models at matched size/tokens on val loss + rubric. | `05` P4, S5 |
| **It scales & is fast** | Pile bits-per-byte within 10% of matched GLA baseline; throughput within 3× of GLA-BPTT; recurrent generation ≥ transformer at long context. | `05` P5/P7, S6 |
| **New capabilities** | Extended effective context from cleanup (P6); working continual/test-time learning demo. | `05` P6, S7, `06` |

---

## 7. Reading order

1. This document (vision, constitution, axioms).
2. `01-mathematical-foundations.md` — the algebra, capacity, energy, fixed points, RoPE derivation. *Everything downstream depends on it.*
3. `02-architecture-spec.md` — the forward pass, shapes, budgets.
4. `03-learning-rule-spec.md` — RIPPLE in full, with every trace derivation.
5. `04-implementation-plan.md` — the code, memory budget, kernels.
6. `05-evaluation-protocol.md` — datasets, baselines, P1–P7, kill criteria.
7. `06-roadmap-scaling-agentic.md` — consolidation, long context, reasoning, production path.
