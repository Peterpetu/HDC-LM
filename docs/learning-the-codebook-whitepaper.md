# Learning the Codebook, in Sequence

## Backpropagation-Free Next-Token Prediction with Vector-Symbolic Recurrence, and Compression as the Yardstick

### A research program for training language models with equilibrium propagation and contrastive Hebbian learning, evaluated by requential coding

**Version 2.0 — July 2026**
**Supersedes v1.1 ("Learning the Codebook"), which targeted classification and factorization. This version reorients the entire program around autoregressive next-token prediction, adds compression-based evaluation, and carries forward v1.1's self-falsification (the torus state-space result) as a solved design constraint.**

**Status: research proposal and prior-art analysis. No experimental results are reported. Every empirical claim about the *current* literature is cited; every claim about the *proposed* work is labelled [PROPOSED], [CONJECTURE], or [PREDICTION].**

---

## Abstract

Vector-Symbolic Architectures (VSA / Hyperdimensional Computing) possess a compositional algebra — binding, bundling, permutation — and a novel inference algorithm (the resonator network), but no learning theory: the atomic codebook is random and frozen, and VSA on raw data is dramatically weaker than VSA on backprop-extracted features. The companion literature of backpropagation-free credit assignment — equilibrium propagation (EP), contrastive Hebbian learning (CHL), predictive coding — has a learning theory but has never been welded to VSA, and has never been shown to scale to a frontier task. This document proposes that weld, and targets it directly at the task that defines frontier AI: **autoregressive next-token prediction**.

The reorientation is not cosmetic. The first version of this program (v1.1) targeted classification, and rested entirely on **bundling** — superposition, the commutative sum of hypervectors. Bundling is a *set* operation: it is permutation-invariant by construction, so `bundle(the, cat, sat) == bundle(sat, cat, the)`. Language is the one domain where order is the entire signal. A bundling-based model is therefore the *wrong shape* for language, and pointing it at TinyStories would test a straw man. **This is the central obstacle, and resolving it is what makes the program interesting.**

The resolution runs through three observations, each established in the literature, whose *conjunction* is unclaimed:

1. **VSA already has a sequence mechanism.** Permutation (`ρ`) protects order: a context is encoded as `Σ_k ρ^k(token_k) ⊗ pos_k`, and next-token prediction becomes an associative-memory read against the vocabulary. This is exactly the read that v1.1's Theorem B1 concerns.

2. **Linear attention *is* VSA in continuous form.** A linear transformer's running state `S_t = Σ_{i≤t} φ(k_i) v_i^⊤` is *bundling of bound key–value pairs*; the outer product `φ(k)v^⊤` is *binding*; the read `φ(q)^⊤ S_t` is *unbinding-and-cleanup*. Linear attention, RWKV, DeltaNet, and state-space models (Mamba/S4) are, up to the choice of feature map and gating, **VSA sequence models running on GPUs at scale** — with learned, not random, codebooks. This is the bridge, and it is why the program can be competitive: the winning architecture already absorbed the VSA principle, exactly as attention absorbed the modern-Hopfield principle.

3. **EP applies to converging recurrences.** Ernoult et al. (2019) proved EP's nudged-phase transient matches backpropagation-through-time in a converging RNN with static input. A linear-attention layer's state recursion has the fixed-point structure EP needs.

From these we derive the program's two novel technical objects:

- **[PROPOSED] The Recurrent Warm Read (§4).** We give an energy-based formulation of a *causal, autoregressive* associative-memory sequence model whose per-step read is a finite-temperature modern-Hopfield retrieval over a learned key–value memory, and we show its supervised next-token loss is trainable by a **local, two-phase, three-factor Hebbian rule in the phase/feature domain** — no backpropagation through time, no weight transport, no stored activations beyond the running state. The derivation (Appendix B) yields the explicit update. We also prove a limiting-case theorem: **standard linear-attention training by truncated BPTT is the β→∞, single-Newton-step limit of EP on this energy** (§4.4, Theorem T1), the sequence-model analogue of v1.1's "OnlineHD is the zero-temperature limit of EP."

- **[PROPOSED] Compression as the yardstick (§6).** Perplexity alone cannot answer "is a backprop-free learner *really* competitive," because a model can hit low training loss by memorizing. We adopt **requential coding** ([Qiu, Finzi, Zheng, Zhang & Wilson, arXiv:2607.11883, 2026](https://arxiv.org/abs/2607.11883)) and the Finzi et al. PAC-Bayes bound it certifies as the program's primary evaluation instrument. The sharp question becomes: *at equal validation loss, does an EP-trained model compress — and therefore provably generalize — as well as its backprop-trained twin?* This is a far stronger falsification target than a perplexity gap, and it directly tests whether local learning discovers the same **learnable structure** (in the paper's sense: description length independent of parameter count and data entropy) that backprop does. We give three numbered predictions on the compression axis (§6.4).

The central obstacle carried over from v1.1 — that learning the codebook can destroy the near-orthogonality the algebra relies on — is *reframed and partly dissolved* for the sequence setting (§3.5, §7 OP2): in linear attention the "codebook" is the learned key/value projections, which the field already trains successfully; the open question is whether a *local* rule can match backprop's learned geometry, measured by both loss and compressibility.

We specify the program (§3–§4), the causal-EP derivation (§4, Appendix B), five open problems (§7), a full experimental protocol with the requential-coding architecture and datasets adopted verbatim from Qiu et al. (§6, §8), reference pseudocode (§9), a staged milestone plan with kill criteria (§10), an explicit failure-mode analysis (§11), a prior-art search log (§12), and a complete linked bibliography (§13).

The single most important caveat is stated up front and repeated: **the version of VSA that can do language competitively already exists and is called linear attention, and it won on GPUs.** The contribution this program can plausibly make is therefore *not* "discrete VSA beats transformers at language" — v1.1's Deep-HDC risk analysis predicts that fails — but "a **local, backprop-free learning rule** can train a competitive linear-attention/VSA language model, and the resulting model is as compressible as a backprop-trained one." If that holds, neuromorphic hardware gets a route to on-chip language-model *training*, which nobody currently has. If it fails, the compression instrument tells us *why* — whether the deficit is in optimization, in representation geometry, or in memorization — which is itself a publishable result.

---

## 0. How to use this document

**One-paragraph version:** read the Abstract, then §3.2 (why bundling fails for language), §3.3 (linear-attention-as-VSA), and §4.2–§4.4 (the Recurrent Warm Read and its EP rule).

**To start coding today:** read §2 (background), §9 (pseudocode), §6/§8 (protocol and infrastructure), then execute Milestone M0 in §10 — reproduce a small backprop-trained linear-attention baseline on TinyStories/FineWeb *and* its requential code length. That is the calibration everything else is measured against. Then M2 (EP-gradient validation) is the kill shot.

**To attack the theory:** §7. OP1 (does the Recurrent Warm Read's fixed point exist and is it unique per step?) and Theorem T1's converse (§4.4) are self-contained.

**To check prior-art claims:** §12 logs what was searched and found. Negative literature findings are weak; treat "not found" as "we could not find it."

**Conventions.** **[ESTABLISHED]** = published, cited. **[PROPOSED]** = original here, unvalidated. **[CONJECTURE]** = believed, unproven. **[PREDICTION Pn]** = numbered falsifiable claim. **[RISK]** = identified failure mode.

**Relationship to v1.1.** v1.1 ("Learning the Codebook") targeted image classification and hypervector factorization. It is superseded but not retracted: its Theorem B1 (OnlineHD as a zero-temperature EP limit), its coherence analysis, and — critically — its **self-falsification of the torus state space** (a pure-phase state cannot hold a superposition; Appendix D of v1.1) are carried forward here as established internal results. Where v1.1 is cited below, it refers to that document.

---

## 1. Motivation

### 1.1 The question this program answers

Two facts sit in tension.

**Fact 1: backprop-free learning has never scaled.** Every local/biologically-motivated credit-assignment scheme — feedback alignment, target propagation, Forward-Forward, equilibrium propagation, predictive coding — matches backpropagation on small problems and falls behind at depth and scale. The canonical negative result is [Bartunov et al., NeurIPS 2018](https://arxiv.org/abs/1807.04587). No backprop alternative has a "language moment."

**Fact 2: the winning language architecture is quietly VSA-shaped.** Linear attention ([Katharopoulos et al., ICML 2020](https://arxiv.org/abs/2006.16236)), RWKV ([Peng et al., 2023](https://arxiv.org/abs/2305.13048)), DeltaNet ([Yang et al., 2024](https://arxiv.org/abs/2406.06484)), and state-space models ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)) all maintain a running associative memory that is, structurally, VSA binding-and-bundling with a learned codebook. The neuromorphic principle already won — on GPUs, trained by backprop.

The program asks the question these facts frame: **can a local, backprop-free rule train the VSA-shaped architecture that already works for language — and produce a model as compressible, hence as generalizing, as its backprop-trained twin?**

If yes: neuromorphic hardware gets language-model *training* (not just inference), because the rule is local and two-phase and maps to oscillator/phasor substrates. If no: the compression instrument (§6) localizes the failure.

### 1.2 Why "just run HDC on TinyStories" is the wrong experiment

The naive test — take v1.1's bundling classifier, point it at next-token prediction — fails for a structural reason, not a tuning reason.

**Bundling is a set operation.** VSA bundling `+` is commutative and associative: the bundle of a sequence is invariant to its order. "the cat sat on the mat" and "the mat sat on the cat" have *identical* bundles. Language is precisely the domain where order carries the signal. A bundling-only next-token model can represent *which* tokens are in the context but not *in what order* — so it is, at best, a bag-of-words / n-gram-quality predictor, which has been beatable since 2018.

v1.1's own **R5 risk** and **Open Problem 3 (Deep HDC)** already predicted this: a single flat bundling layer with a frozen random codebook has no mechanism for hierarchical or ordered composition. Confirming it on TinyStories would spend GPU-hours to reproduce a known gap.

**The fix is native to VSA and is the whole pivot of this document:** permutation for order (§2.1.3), and — for competitiveness — the recognition that *learned* linear attention already implements ordered VSA recurrence (§3.3). The experiment worth running is not "random HDC does language" but "**local learning trains VSA-recurrence to language**," evaluated by compression.

### 1.3 What requential coding adds, and why it is the right yardstick

[Qiu et al. (2026)](https://arxiv.org/abs/2607.11883) is not a training method; it is a **measurement** method, and it is exactly the measurement this program needs.

The problem with judging a backprop-free learner by perplexity alone: a model can reach low training loss by *memorizing*, which does not generalize. What we actually want to know is whether local learning discovers the same **learnable structure** backprop does. Requential coding operationalizes "learnable structure" as a description length that is *independent of parameter count and data entropy* — it charges only the bits by which a student's self-generated training trajectory departs from a teacher's. Plugged into the Finzi et al. PAC-Bayes bound, it certifies generalization for billion-parameter LLMs, and the bound *tightens with scale*.

For this program that gives a falsification target with teeth:

> **At matched validation loss, compute the requential code length of an EP-trained model and its backprop-trained twin. If the EP-trained model is systematically *less* compressible, it is memorizing rather than learning structure — and "competitive perplexity" would be a mirage.**

This is strictly stronger than a perplexity comparison, and it is only possible because Qiu et al. gave us a compressor whose code length reflects *learned information* rather than parameter count. We adopt their architecture, datasets, and code-length estimator verbatim (§6, §8) so the comparison is apples-to-apples with a known-good baseline.

One conceptual bridge, flagged as an analogy not a theorem (§6.5): requential coding's core primitive, **relative entropy coding** (REC), transmits a sample from Q using ~KL(Q‖P) bits by *selecting an index into a shared-randomness proposal stream from P*. VSA cleanup *selects a codeword from a shared random codebook by similarity*. Both pay only for the **departure from what is already shared**. The resemblance is real and worth stating; it is not a formal equivalence, and the document treats it as motivation, not result.

### 1.4 The honest ceiling

Stated once, plainly, and repeated in §11: **the VSA-for-language battle was already won by linear attention, on GPUs, trained by backprop.** This program cannot claim "VSA beats transformers." Its plausible contributions are narrower and still worth having:

1. A **local, backprop-free learning rule** that trains a competitive linear-attention/VSA language model — the first credible route to on-chip LM *training* on neuromorphic hardware.
2. A **compression-certified** demonstration that such a model generalizes as well as its backprop twin (or a diagnosis, via the code length, of exactly where it falls short).
3. A **limiting-case theorem** (T1) placing standard linear-attention training as the zero-temperature limit of the proposed rule — a unification, in the spirit of [Millidge et al. 2022](https://arxiv.org/abs/2206.02629).

---

## 2. Technical background

Self-contained. A reader with standard ML background and no VSA or EP experience can proceed to §3 after this.

### 2.1 Vector-Symbolic Architectures

#### 2.1.1 The algebra

Hypervectors live in a *D*-dimensional space, *D* large (10³–10⁴). Atomic symbols get **i.i.d. random** hypervectors. Three operations:

| Operation | Symbol | Effect | Purpose |
|---|---|---|---|
| **Binding** | `⊗` | output dissimilar to both inputs; invertible | role–filler pairs, key–value |
| **Bundling** | `+` | output similar to all inputs | sets, superposition |
| **Permutation** | `ρ` | output dissimilar to input; invertible | **order, sequence** |

Load-bearing property: **binding distributes over bundling**, `(a+b)⊗c = a⊗c + b⊗c`. This makes it an algebra, and it is what will let a *superposition* of key–value pairs be queried by a single key (§3.3).

#### 2.1.2 FHRR: the phase model

In Fourier Holographic Reduced Representations ([Plate, *IEEE TNN* 6(3):623–641, 1995](https://doi.org/10.1109/72.377968)), a hypervector is `z_j = e^{iθ_j}`, `θ_j ~ Uniform[−π,π)` — a point on the *D*-torus. Binding = phase addition; unbinding = phase subtraction; bundling = sum of unit phasors; similarity = `(1/D)Σ_j cos(θ_j − φ_j)`. **A bank of FHRR hypervectors is a bank of phase oscillators** — the identity that connects VSA to EP-trainable oscillator hardware ([Frady & Sommer, *PNAS* 116(36):18050, 2019](https://doi.org/10.1073/pnas.1902653116); [Renner et al., *Nat. Mach. Intell.* 6:641–652, 2024](https://www.nature.com/articles/s42256-024-00848-0)).

> **Carried from v1.1 (Appendix D):** a *pure-phase* (torus) state cannot represent a superposition, because the superposition weights live in the component *magnitudes*, which the torus fixes to 1. For *sequence memory* this matters: the running state must hold a weighted superposition of past key–value bindings, so the **state lives in ℂ^D or ℝ^D (amplitude + phase), not on the torus**. The *codewords* remain unit-magnitude; the *state* does not. This is a solved design constraint here, not an open question.

#### 2.1.3 Sequences in VSA: permutation and positional binding

The mechanism that makes VSA a *sequence* model, and the reason bundling-alone is not the whole story.

To encode an ordered sequence `(t_0, t_1, …, t_{n-1})` of token-hypervectors, protect position by permutation or positional binding:

```
context  =  Σ_{k}  ρ^k(t_k)          (permutation-based; ρ^k = permute k times)
        or  Σ_{k}  t_k ⊗ p_k          (positional-binding; p_k a fixed position vector)
```

Both break the commutativity of raw bundling: swapping two tokens now changes the result, because `ρ^i(t) ≠ ρ^j(t)` for `i≠j`, and `t⊗p_i ≠ t⊗p_j`. Order is preserved. Retrieval of "the token at position k" is `ρ^{-k}(context)` or `context ⊗ p_k^{-1}`, cleaned up against the vocabulary. This is the classic VSA treatment of sequences ([Plate 1995]; [Frady, Kleyko & Sommer, *Neural Computation* 30(6):1449, 2018](https://doi.org/10.1162/neco_a_01084); [Kanerva, *Cognitive Computation* 1:139, 2009](https://doi.org/10.1007/s12559-009-9009-8)).

**Next-token prediction, VSA-style:** given `context`, predict `t_n` by retrieving the continuation most consistent with the bound structure — an associative-memory read against the vocabulary codebook. Which is exactly the read v1.1's Theorem B1 concerns, now placed in a sequence.

**Capacity.** A superposition of *M* items in *D* dimensions has retrieval SNR ~√(D/M) ([Frady, Kleyko & Sommer 2018](https://doi.org/10.1162/neco_a_01084)). For long contexts this is the binding-interference wall, and it is why naive fixed-width VSA sequence memory degrades with length — a limitation linear attention addresses with *learned* projections and gating (§3.3).

### 2.2 Linear attention and its VSA reading

#### 2.2.1 Softmax vs. linear attention

Standard attention computes, for query `q_t` and keys/values `{k_i, v_i}`:

```
o_t  =  Σ_i  [ softmax(q_t · k_i / √d) ]_i  v_i
```

which requires all pairwise `q·k` — O(n²) time and O(n) memory in the sequence length. **Linear attention** ([Katharopoulos et al., ICML 2020](https://arxiv.org/abs/2006.16236)) replaces `exp(q·k)` with a kernel feature map `φ(·)`:

```
o_t  =  Σ_{i≤t}  [ φ(q_t) · φ(k_i) ]  v_i  /  Z_t
     =  φ(q_t)^⊤ ( Σ_{i≤t} φ(k_i) v_i^⊤ )  /  Z_t
```

Define the **running state** (a *d×d* matrix, or a bound superposition):

```
S_t  =  S_{t-1}  +  φ(k_t) v_t^⊤ ,        z_t = z_{t-1} + φ(k_t),        o_t = (φ(q_t)^⊤ S_t) / (φ(q_t)^⊤ z_t)
```

This is O(n) time, O(1) memory — a *recurrent* form.

#### 2.2.2 The state recursion IS VSA

Read the update `S_t = S_{t-1} + φ(k_t)v_t^⊤` in VSA terms:

| Linear-attention quantity | VSA operation |
|---|---|
| outer product `φ(k_t) v_t^⊤` | **binding** of key to value |
| running sum `S_t = S_{t-1} + …` | **bundling** (superposition of bound pairs) |
| read `φ(q_t)^⊤ S_t` | **unbinding** by the query + implicit cleanup |
| feature map `φ(·)` | the **encoder** into hypervector space |
| the matrix `S_t` | the **associative memory** (item memory) |

**A linear-attention layer is a VSA sequence model with a learned codebook (the `W_k, W_v, W_q` projections and the feature map φ) and a running bound-superposition memory.** DeltaNet ([Yang et al. 2024](https://arxiv.org/abs/2406.06484)) adds a *delta rule* (remove the old value before writing the new) — which is VSA memory *update* rather than pure accumulation, and is exactly the Widrow-Hoff/LMS correction. Gated variants (Mamba, GLA, [Yang et al. GLA 2023](https://arxiv.org/abs/2312.06635)) add forgetting. RWKV ([Peng et al. 2023](https://arxiv.org/abs/2305.13048)) is a closely related linear recurrence.

This is the same "the winning architecture already absorbed the neuromorphic principle" pattern as modern-Hopfield↔attention (§2.4) — and it is the reason a VSA-based language program can be competitive rather than merely biologically cute.

#### 2.2.3 Why this matters for *learning*

The state recursion has a **fixed-point / converging structure** under standard conditions (bounded features, gating < 1). This is the structural precondition for equilibrium propagation (§2.5): EP was proven to match BPTT in exactly such converging recurrences ([Ernoult et al., NeurIPS 2019](https://arxiv.org/abs/1905.13633)). So the bridge is not only representational (linear attention = VSA) but *algorithmic* (linear attention is EP-trainable in principle). Making that concrete is §4.

### 2.3 Modern Hopfield networks and the read operation

Modern (continuous) Hopfield networks ([Ramsauer et al., ICLR 2021](https://arxiv.org/abs/2008.02217)): with stored patterns `C = [C_1,…,C_K]`, energy `E(ξ) = ½‖ξ‖² − lse(β Cᵀξ)` and update `ξ ← C·softmax(βCᵀξ)`. **This update is transformer attention.** It also approximates Kanerva's Sparse Distributed Memory ([Bricken & Pehlevan, NeurIPS 2021](https://arxiv.org/abs/2111.05498)), which *is* VSA's memory model. Dense Associative Memories have exponential capacity ([Krotov & Hopfield, NeurIPS 2016](https://arxiv.org/abs/1606.01164); [Demircigil et al. 2017](https://arxiv.org/abs/1702.01929); [Lucibello & Mézard, *PRL* 132:077301, 2024](https://doi.org/10.1103/PhysRevLett.132.077301)). The 2024 Nobel Prize in Physics (Hopfield & Hinton) canonizes the lineage.

For this program, the modern-Hopfield read is the **per-step primitive** of the Recurrent Warm Read (§4): at each token, retrieve from the running memory at finite temperature β, then predict the next token from the retrieved content.

### 2.4 Equilibrium propagation

#### 2.4.1 The theorem

([Scellier & Bengio, *Front. Comput. Neurosci.* 11:24, 2017](https://www.frontiersin.org/articles/10.3389/fncom.2017.00024/full).) Energy-based model with state `s`, parameters `θ`, input `x` clamped, cost `C(s,y)`. Total energy `F = E(θ,x,s) + βC(s,y)`. Relax to a free equilibrium `s⁰` (β=0) and a nudged equilibrium `s^β` (β small). Then:

```
dC(s⁰,y)/dθ  =  lim_{β→0} (1/β) [ ∂E/∂θ(θ,x,s^β) − ∂E/∂θ(θ,x,s⁰) ]
```

The loss gradient equals a difference of energy-gradients at two equilibria. **No backward pass, no weight transport, no stored activations** — two relaxations of the same physical system and a local difference.

#### 2.4.2 EP on a Hopfield energy IS contrastive Hebbian learning

For a Hopfield energy, `∂E/∂W_ij = −ρ(s_i)ρ(s_j)`, so the EP update is `ΔW_ij ∝ (1/β)[ρ(s_i^β)ρ(s_j^β) − ρ(s_i⁰)ρ(s_j⁰)]` — a contrastive Hebbian rule (nudged Hebbian product minus free Hebbian product). This is the fulcrum: express the language model's per-step read as a Hopfield energy, and a principled local learning rule falls out.

#### 2.4.3 The extension that makes language possible

- **EP matches BPTT in converging RNNs with static input** ([Ernoult, Grollier, Querlioz, Bengio & Scellier, NeurIPS 2019](https://arxiv.org/abs/1905.13633)). This is the theorem that lets EP touch sequence models: a linear-attention state recursion is such a converging system *within each step's read*.
- **EP on physical oscillators**: Ising machines ([Laydevant, Marković & Grollier, *Nat. Commun.* 15:3671, 2024](https://www.nature.com/articles/s41467-024-46879-4)); coupled phase oscillators ([Wang, Wanjura & Marquardt, arXiv:2402.08579, 2024](https://arxiv.org/abs/2402.08579)); oscillator networks to ~98% MNIST ([Rageau & Grollier, *Neuromorph. Comput. Eng.* 5(3):034008, 2025](https://iopscience.iop.org/article/10.1088/2634-4386/adebaa)). These are the substrates a local language-model rule would target.
- **Holomorphic EP** ([Laborieux & Zenke, NeurIPS 2022](https://arxiv.org/abs/2209.00530)) — exact gradients via finite-size oscillations in the **complex** domain, the natural fit for FHRR. **Symmetric/centred nudging** ([Laborieux et al., *Front. Neurosci.* 15:633674, 2021](https://www.frontiersin.org/articles/10.3389/fnins.2021.633674/full)) cancels the O(β) bias.
- **The unification**: predictive coding, EP, and CHL all converge to backprop in the infinitesimal-inference limit ([Millidge, Song, Salvatori, Lukasiewicz & Bogacz, arXiv:2206.02629](https://arxiv.org/abs/2206.02629)).

#### 2.4.4 The open frontier: EP through *time*

Standard EP handles static input. Language is a *sequence*, and its loss is a **sum of causal per-step terms**. Applying EP to an autoregressive loss — where each token's prediction depends on a running state that itself depends on all past tokens — is **unexplored**. This is the genuinely novel and genuinely hard part of the program, and §4 is devoted to it. The two candidate strategies:

- **(S-A) EP-within-step, exact-recurrence-across-steps.** Treat the running state `S_t` as an exact deterministic carry (not relaxed), and apply EP only to the *read* at each step, accumulating local updates. This is the conservative route and connects directly to Ernoult et al. (2019).
- **(S-B) Full space-time EP.** Relax the entire unrolled trajectory to a space-time equilibrium and nudge the whole output sequence. Higher-risk, closer to [Lagrangian/Hamiltonian EP](https://arxiv.org/abs/2506.06248).

### 2.5 Requential coding: compression as a measurement of learning

([Qiu, Finzi, Zheng, Zhang & Wilson, arXiv:2607.11883, 2026](https://arxiv.org/abs/2607.11883).) The evaluation backbone of this program. Summarized precisely because §6 depends on the details.

**Prequential coding** compresses a dataset by the area under its training loss curve: `L_preq = Σ_t log 1/P_t(X_t)`. It pays the full data entropy plus approximation error — it grows with dataset size regardless of how much the model learned.

**Requential coding** changes one thing: the student trains on *self-generated* data, and the code records only which samples a stronger teacher accepts, via **relative entropy coding (REC)**. REC transmits a sample from teacher `Q_t` using ~`KL(Q_t‖P_t)` bits by selecting an index into a shared-randomness proposal stream drawn from the student `P_t`. The total code length is the **cumulative teacher–student KL**:

```
L_req  ≈  Σ_t KL(Q_t ‖ P_t)          (+ negligible universal-code overhead at large batch)
```

This is *independent of parameter count and data entropy* — it charges only the bits where teacher and student disagree, i.e. the **learnable information** the student actually absorbs. Plugged into the Finzi et al. PAC-Bayes bound (Theorem D.1 in Qiu et al.), it certifies generalization for billion-parameter LLMs, tightening with scale.

**Key facts this program uses:**

1. **Evaluation needs no actual transmission.** One replaces REC encode/decode by sampling `X_t` directly from the teacher, and reports `Σ_t KL(Q_t‖P_t)` — a ~2.33× training-FLOP overhead, or 0.33× if teacher checkpoints exist ([Qiu et al. §3.1, Runtime](https://arxiv.org/abs/2607.11883)).
2. **The KL is estimated by Monte Carlo on the batch the student trains on**: since `X_t ~ Q_t`, `log Q_t(X_t) − log P_t(X_t)` is an unbiased estimate of `KL(Q_t‖P_t)` (Qiu et al. Appendix C.1).
3. **Larger models compress to *fewer* bits at fixed loss** — code length per parameter decays as a power law in model size; the generalization gap vanishes with scale (Qiu et al. §4.1–4.2).
4. **Lower-entropy text holds far more learnable structure than higher-entropy images** (Qiu et al. §4.4) — which is *why* language is the right target: it is where compressibility and learnable structure are highest.

**For this program:** requential coding is the instrument that distinguishes "EP-trained model reached low loss by learning structure" (compresses well) from "…by memorizing" (compresses poorly). See §6.

---

## 3. The reorientation: from sets to sequences

### 3.1 What changed from v1.1, and why

v1.1 built everything on **bundling** and targeted **classification** (and factorization via resonators). The Warm HDC classifier read a class prototype from a bundled memory; the resonator factorized a bundled composite. Both are *set-shaped* computations, and both are legitimate — for classification.

Language broke the assumption. The rest of §3 establishes the three-step pivot that makes the program a *sequence* program: (i) bundling alone cannot represent order (§3.2); (ii) permutation/positional binding fixes order, and linear attention is the *learned* version of exactly this (§3.3); (iii) the resulting recurrence is EP-trainable (§3.4). The coherence objection is then reframed for the sequence setting (§3.5).

### 3.2 Why bundling alone fails for language (the obstacle, precisely)

**Claim.** A next-token model whose only sequence operation is bundling is at best n-gram-quality, regardless of *D* or codebook.

**Argument.** Bundling is commutative and associative. For any permutation π of a context,

```
bundle(t_0, …, t_{n-1})  =  bundle(t_{π(0)}, …, t_{π(n-1)}).
```

So the model's context representation is a *function of the multiset of tokens only*. Two contexts with the same token counts but different order are represented identically, hence produce identical next-token distributions. A model that cannot distinguish "the cat sat on the" from "the the sat on cat" cannot capture syntax, long-range agreement, or compositional meaning. The best such a model can do is condition on the bag of recent tokens — a bag-of-words / unigram-ish predictor. Adding short permutation windows recovers local n-grams, but n-grams have been beaten since the neural-LM transition.

**This is v1.1's R5 risk and OP3 (Deep HDC), now made concrete for language.** It is *why* pointing v1.1's classifier at TinyStories tests a straw man, and it is the obstacle the rest of the program is organized to overcome.

**Corollary (the design requirement).** Any VSA language model must carry **order-sensitive structure** in its state — via permutation, positional binding, or a learned order-sensitive recurrence — *and* the state must hold an **amplitude-weighted superposition** (not a pure-phase torus point; §2.1.2, carried from v1.1). Linear attention satisfies both.

### 3.3 The bridge: linear attention is learned VSA sequence memory

§2.2.2 established the correspondence. Here is why it is the crux of *competitiveness*.

The reason discrete VSA is weak on raw data (v1.1: 59-point gap vs. deep features) is that its codebook and encoder are **random and frozen**. Linear attention keeps the *same algebra* — bind key to value, bundle into a running memory, unbind by query — but makes the codebook **learned**: `W_k, W_v, W_q` and the feature map φ are trained. And crucially, the field already trains these successfully to competitive language-modeling performance (GLA, DeltaNet, Mamba, RWKV all reach transformer-competitive perplexity at scale).

So the VSA-for-language question splits cleanly:

- **Representation:** *Can a VSA-shaped recurrence do language?* **Answer: yes, demonstrably — it's called linear attention, and it already works.**
- **Learning:** *Can a **local, backprop-free** rule train that recurrence?* **This is open, and it is the program.**

This is the same split that resolved the Hopfield story: modern Hopfield *is* attention (representation settled), and the interesting residual questions are about learning and mechanism. The program inherits a *working architecture* and asks only whether local learning can drive it — which is exactly why it can be competitive, and exactly why "beat transformers with discrete HDC" is the wrong, and unnecessary, claim.

### 3.4 Why the recurrence is EP-trainable

Two established facts compose:

1. **Linear attention's read is a converging fixed-point computation** at each step (bounded features, gating < 1; §2.2.3). Even the pure accumulation `S_t` is a deterministic carry with a well-defined per-step read.
2. **EP matches BPTT in converging recurrences with static input** ([Ernoult et al. 2019](https://arxiv.org/abs/1905.13633)).

The remaining gap — that language has a *sequence* of causally-linked reads, not a single static relaxation — is precisely §2.4.4's open frontier, and §4 gives the two strategies (S-A within-step, S-B space-time) with a full derivation for S-A.

### 3.5 The coherence objection, reframed for sequences

v1.1's central worry: learning the codebook destroys the near-orthogonality the algebra relies on. For the *sequence/linear-attention* setting this is **largely dissolved**, and it is worth seeing why.

In linear attention the "codebook" is the learned key/value geometry. The field *already* learns it by backprop without collapse — gating, normalization (e.g. the `z_t` denominator, LayerNorm, the DeltaNet delta rule) actively manage interference. So the question "does learning the codebook destroy the algebra?" is empirically answered *in the affirmative-for-backprop*: learned codebooks work better than random ones here, decisively (that is the entire reason linear attention beats fixed-kernel random-feature attention).

The residual, sharper question for *this* program is therefore:

> **Can a *local* rule learn a codebook geometry as good as backprop's — measured not only by loss but by compressibility (§6)?**

This is cleaner than v1.1's version because we are no longer asking whether *any* learning helps (it does), only whether *local* learning matches *global* learning. And the compression yardstick gives it a quantitative answer: if the EP-learned geometry is as compressible at equal loss, it has found equally-generalizing structure. (The coherence/capacity trade-off from v1.1 §2.1.3 still governs the *fixed-width state's* interference wall for long contexts; that reappears as OP2 in §7.)

---

## 4. Program core: the Recurrent Warm Read and its EP rule

This section defines the proposed language model and derives its backprop-free learning rule. It is the technical heart of the document.

### 4.1 Design commitments

From the analysis above:

1. **State in ℂ^D (amplitude + phase), not on the torus** — the running memory must hold a weighted superposition (carried from v1.1 Appendix D).
2. **Order via the recurrence itself** — the state is updated causally, so position is implicit in accumulation order (as in linear attention), optionally reinforced by positional binding.
3. **Finite-temperature associative read** — each step retrieves from the running memory via a modern-Hopfield read at inverse temperature β, making the per-step computation an energy-based model to which EP applies.
4. **Learned codebook** — key/value/query maps are parameters, trained by the local rule.
5. **Local, two-phase learning** — no BPTT, no weight transport; updates are contrastive Hebbian differences of two relaxations.

### 4.2 [PROPOSED] The Recurrent Warm Read (RWR) architecture

#### State and update

At token *t*, maintain a complex associative memory `S_t ∈ ℂ^{D×d_v}` (keys in ℂ^D, values in ℝ^{d_v}) and normalizer `z_t ∈ ℂ^D`:

```
k_t = φ(W_k x_t) ∈ ℂ^D        (key: FHRR feature map, unit-magnitude per component)
v_t = W_v x_t   ∈ ℝ^{d_v}      (value)
q_t = φ(W_q x_t) ∈ ℂ^D        (query)

S_t = γ · S_{t-1} + k_t v_t^⊤              (bind key⊗value, bundle into memory; γ = gate/forget)
z_t = γ · z_{t-1} + k_t
```

`φ(u)_j = exp(i·u_j)` maps a real projection to a unit-magnitude phase — the FHRR encoder. The state `S_t` is **not** unit-magnitude: its entries accumulate, holding the superposition. This is the design commitment (1).

#### The Warm Read (per step, energy-based)

Instead of the one-shot linear-attention read `o_t = q_t^⊤ S_t / (q_t^⊤ z_t)`, define a **finite-temperature associative retrieval**. Let the columns of the *value codebook* `V = [ṽ_1,…,ṽ_K] ∈ ℝ^{d_v×K}` be the output vocabulary embeddings (or a learned value dictionary). The read state `r ∈ ℝ^{d_v}` relaxes on the modern-Hopfield energy conditioned on the current query-retrieved content:

```
c_t   =  Re( q_t^⊤ S_t ) / Re( q_t^⊤ z_t )        ∈ ℝ^{d_v}       (query-retrieved content)

E_read(r ; c_t, V, β)  =  ½‖r‖²  −  Re⟨c_t, r⟩  −  (1/β) lse( β Vᵀ r )               (4.1)
```

Relaxation `dr/dt = −∂E/∂r = −r + c_t + V·softmax(βVᵀr)` converges to `r_t^*`. The next-token distribution is:

```
p_t  =  softmax( β Vᵀ r_t^* )              ∈ Δ^{K-1}                                   (4.2)
```

**Interpretation.** `c_t` is the linear-attention read (the content the query pulls from memory). `E_read` then *cleans it up* against the value/vocabulary codebook at temperature β — a modern-Hopfield retrieval. At β→∞ with one step, `r_t^* → c_t` and `p_t →` the ordinary linear-attention logits (Theorem T1, §4.4). At finite β with relaxation, the read denoises toward the vocabulary manifold, and — this is the point — the whole per-step computation is a **gradient system**, so EP applies.

#### The loss

Standard next-token cross-entropy, summed causally:

```
L  =  Σ_t  −log p_t[x_{t+1}]                                                          (4.3)
```

### 4.3 [PROPOSED] The EP learning rule (strategy S-A: EP-within-step)

We treat the memory carry `S_t, z_t` as an **exact deterministic recurrence** (computed forward, not relaxed), and apply EP to the *per-step read* `E_read`. This is the conservative route, directly licensed by Ernoult et al. (2019) since each read is a converging fixed-point system with static "input" `c_t`.

#### Per-step two-phase relaxation

At step *t*, with `c_t` fixed:

- **Free phase** (β_n = 0): relax `E_read(r; c_t, V, β)` → `r_t^-`, posterior `p_t^- = softmax(βVᵀr_t^-)`.
- **Nudged phase** (β_n > 0): relax `E_read + β_n·C_t` where `C_t(r) = −log softmax(βVᵀr)[x_{t+1}]` → `r_t^+`, posterior `p_t^+`.

#### The updates (derived in Appendix B)

The value/vocabulary codebook `V`:

```
ΔV_k  ∝  (1/β_n) [ p_{t,k}^+ r_t^+  −  p_{t,k}^- r_t^- ]          (contrastive Hebbian, §2.4.2)   (4.4)
```

The key/value/query projections enter through `c_t`, which enters `E_read` as a clamped bias `−Re⟨c_t, r⟩`. The gradient of the per-step energy w.r.t. `c_t` is `∂E/∂c_t = −r`, so EP gives a **local error signal on the content**:

```
δ_t  :=  (1/β_n)( r_t^+ − r_t^- )              ∈ ℝ^{d_v}        (the EP "teaching signal" for step t)   (4.5)
```

This `δ_t` is the two-phase contrastive difference — the local analogue of the backprop gradient `∂L/∂c_t`. The projection updates then follow from how `c_t` depends on `W_k, W_v, W_q` through the (exact, forward) recurrence `S_t, z_t`. Two sub-options:

- **(S-A0) Local-in-time approximation.** Attribute `δ_t` only to the *current* step's key/value/query (ignore the dependence of `S_t` on past projections). This gives fully local, online updates — the cheapest rule, and the one that maps to hardware:

```
ΔW_v  ∝  δ_t · (contribution of v_t to c_t) · x_t^⊤            (Hebbian: error × input)          (4.6a)
ΔW_k, ΔW_q  ∝  (phase-domain contrastive terms) · x_t^⊤        (Appendix B.3)                     (4.6b)
```

- **(S-A1) Eligibility-trace credit assignment.** To capture the dependence of the current read on *past* writes (the actual sequence credit assignment), propagate `δ_t` back through the *linear* recurrence `S_t = γS_{t-1} + k_t v_t^⊤` using an **eligibility trace** — exactly the mechanism of e-prop ([Bellec et al., *Nat. Commun.* 11:3625, 2020](https://www.nature.com/articles/s41467-020-17236-y)). Because the recurrence is *linear* in `S`, the trace is exact and cheap: `∂S_t/∂W = Σ_{i≤t} γ^{t-i} ∂(k_i v_i^⊤)/∂W`. This is the sequence-credit-assignment route, and its exactness (linear recurrence ⇒ exact trace) is a genuine advantage over e-prop's nonlinear-RNN approximation.

> **The structure worth seeing:** (4.5) is a *local, two-phase error* on each step's content; (4.6)/S-A1 turns it into weight updates by an *exact eligibility trace through a linear recurrence*. Nowhere is there backpropagation through time, weight transport, or stored activations beyond `(S_t, z_t)` and the trace. This is implementable online, and — because the read is a phase/Hopfield relaxation — on oscillator/phasor hardware.

### 4.4 [PROPOSED] Theorem T1 — linear attention is the zero-temperature limit

> **Theorem T1 (informal).** *In the limit β→∞ with a single read step and the nudged phase reducing to one Newton step, the Recurrent Warm Read's per-step computation reduces to standard linear attention, and the S-A1 eligibility-trace update reduces to truncated-BPTT training of a linear-attention layer.*

*Sketch (full statement and proof in Appendix B.4).* As β→∞, `softmax(βVᵀr)` → one-hot at the argmax, so the Hopfield cleanup term in (4.1) becomes a hard projection and `r_t^* → c_t` (one step, no denoising). Then `p_t → softmax(βVᵀc_t)` — the ordinary linear-attention logits. The free/nudged difference `δ_t = (r_t^+ − r_t^-)/β_n` converges to the cross-entropy gradient `∂C_t/∂c_t` (standard EP→backprop in the read; [Millidge et al. 2022](https://arxiv.org/abs/2206.02629)). Feeding that through the exact linear-recurrence eligibility trace reproduces the truncated-BPTT gradient of a linear-attention layer. ∎

**Why T1 matters.** It is the sequence-model analogue of v1.1's Theorem B1 ("OnlineHD is the zero-temperature EP limit"). It says the proposed rule is not exotic: standard linear-attention training sits at one end (β→∞, one Newton step, BPTT), and the RWR/EP rule is the finite-temperature, two-phase, local generalization at the other. This (a) sanity-checks the construction, (b) predicts the finite-β version should *differ* from — and potentially improve on — standard training in the same three ways v1.1 identified (soft negatives over all vocabulary, denoised read, updates on correct predictions too), and (c) gives a continuous knob (β, number of read steps, S-A0 vs S-A1) interpolating between "cheap and local" and "exact and global."

### 4.5 Predictions for the program core

> **[PREDICTION P1 — gradient fidelity]** On a small linear-attention LM, the S-A1 EP update direction has cosine similarity > 0.9 with the truncated-BPTT gradient as β_n → 0. *Kills the program if false* — it is the sequence analogue of v1.1's P-A1, and there is no reason it should fail given T1, but it must be checked first.

> **[PREDICTION P2 — trainability]** An RWR language model trained end-to-end by the S-A1 local rule reaches validation perplexity within 20% of an identical-architecture backprop-trained baseline on TinyStories at ~5–20M parameters.

> **[PREDICTION P3 — the finite-β dividend]** At its best β, the RWR/EP model matches or beats the β→∞ (linear-attention) baseline in perplexity, with the gap growing on examples requiring vocabulary-level disambiguation (rare tokens, near-synonyms) — because the finite-temperature read uses soft negatives over the whole vocabulary (T1 corollary).

> **[PREDICTION P4 — S-A0 vs S-A1]** The eligibility-trace rule (S-A1) strictly outperforms the local-in-time rule (S-A0), and the gap grows with the dependency length of the task — quantifying how much *sequence* credit assignment (vs. per-step) actually matters.

---

## 5. (reserved)

*Section number reserved to keep §6 = evaluation aligned with the companion v1.1 numbering in cross-references. Proceed to §6.*

---

## 6. Compression as the yardstick: requential coding for backprop-free LMs

This is the evaluation methodology, and it is what makes "competitive with LLMs" a *testable* claim rather than a vibe. It is built entirely on [Qiu et al. (2026)](https://arxiv.org/abs/2607.11883).

### 6.1 Why perplexity is not enough

A backprop-free learner could reach low training loss by **memorizing** rather than by discovering generalizable structure. Perplexity on a held-out set partly guards against this, but:

- it does not tell you *how much information* the model extracted from data vs. stored idiosyncratically;
- it does not certify a generalization *bound*;
- it cannot separate "the optimizer found simple structure" from "the model has enough capacity to fit."

Requential coding addresses exactly these. Its code length is the **cumulative teacher–student KL**, `Σ_t KL(Q_t‖P_t)`, which is *independent of parameter count and data entropy* and measures the **learnable information the student actually absorbed** (§2.5). A model that memorizes pays more bits for the same loss; a model that learns structure pays fewer.

### 6.2 The core evaluation protocol

For each trained model (EP-trained and its backprop-trained twin), at matched validation loss:

1. **Instantiate teacher and student** with the *same architecture* (RWR or the linear-attention baseline). Per Qiu et al., share width, depth, learning rate, batch size, and init seed so the initial per-step KL is zero.
2. **Run the equivalent stochastic process** (no actual REC): the teacher trains on real tokens; at each step the student trains on a batch *sampled from the teacher*; record `KL(Q_t‖P_t)` via the unbiased Monte-Carlo estimator `log Q_t(X_t) − log P_t(X_t)` on the batch the student trains on (Qiu et al. Appendix C.1).
3. **Report** `L_req = Σ_t KL(Q_t‖P_t)` as the certified compressed size, and plug into the Finzi et al. bound (Qiu et al. Theorem D.1) for a generalization guarantee.
4. Use **teacher smoothing (EMA)** and **iso-loss projection** as in Qiu et al. Algorithm 1 to tighten the code.

### 6.3 The three comparisons that matter

| Comparison | What it tests |
|---|---|
| **C1. Requential code length: EP-trained vs backprop-trained, at equal validation loss** | *Does local learning discover equally-compressible (equally-generalizing) structure?* The central question. |
| **C2. Code-length-vs-loss Pareto frontier, both learners, across model sizes** | *Does the EP-trained family lie on the same frontier?* Qiu et al. show larger backprop models compress to fewer bits at fixed loss; does EP inherit this scaling? |
| **C3. Generalization bound (Finzi et al.) for the EP-trained model** | *Can we certify a non-vacuous generalization bound for a backprop-free LM?* Nobody has. This would be a first regardless of the perplexity gap. |

### 6.4 Predictions on the compression axis

> **[PREDICTION P5 — compressibility parity]** At matched validation loss, the EP-trained RWR model's requential code length is within a small constant factor (target: ≤1.5×) of the backprop-trained twin's. If it is *much* larger, EP is memorizing rather than learning structure — a decisive negative result, and one perplexity alone would have hidden.

> **[PREDICTION P6 — scaling inheritance]** The EP-trained model family exhibits the Qiu et al. signature — code length per parameter *decreasing* with model size at compute-optimal scaling — i.e. local learning does not break the "larger models are more compressible" phenomenon.

> **[PREDICTION P7 — certified generalization]** The Finzi et al. bound evaluated on the EP-trained model is non-vacuous at ~100M parameters. This is a milestone independent of the EP-vs-backprop comparison: **the first certified generalization bound for a backprop-free language model.**

### 6.5 A conceptual bridge (analogy, not theorem)

Worth one paragraph because it is genuinely suggestive and connects the two halves of the program.

Requential coding's primitive is **REC**: communicate a sample from `Q` using ~`KL(Q‖P)` bits by *selecting an index into a shared-randomness proposal stream drawn from `P`* — you pay only for `Q`'s departure from the already-shared `P`. VSA's primitive is **cleanup**: recover a symbol by *selecting the nearest codeword from a shared random codebook* — you pay (in interference) only for the query's departure from the stored content. Both are "shared randomness + select-by-agreement, charge only for the difference." The **blessing of dimensionality** that makes VSA cleanup work (near-orthogonal random codewords) is the same concentration phenomenon that makes REC's proposal streams efficient. This is an *analogy* — REC is an information-theoretic coding scheme and VSA cleanup is a geometric retrieval; they are not formally the same map. But it hints that a VSA model and a requential code are natural companions: both are theories of *paying only for the novel bits*. The program does not rest on this; it is offered as motivation and as a possible seed for OP5 (§7).

---
## 7. Theory: five open problems

Each is self-contained. OP1 and the T1 converse (§4.4) are the first to attack.

### OP1. Existence and uniqueness of the per-step read fixed point

**Statement.** The Warm Read (4.1) relaxes `r` on a modern-Hopfield energy with a clamped content bias `c_t`. Modern Hopfield energies can have multiple minima (that is the point — they store multiple patterns). For a *language model read* we need the relaxation to land in the basin corresponding to the query-retrieved content, reliably, at the chosen β.

**Sub-problems.** (a) For what β is the per-step read fixed point unique given `c_t`? (b) Does the content clamp `−Re⟨c_t,r⟩` guarantee basin selection (i.e. does a strong enough clamp make the energy effectively convex around the retrieved value)? (c) Characterize the trade-off: low β → unique but under-denoised read; high β → sharp but multi-basin. This is the sequence-read analogue of v1.1's OP1 (Soft Resonator fixed points).

**Why it matters.** If the read is not reliably single-valued, EP's two-phase difference is ill-defined. This gates P1.

### OP2. The binding-interference wall and effective context length

**Statement (carried and adapted from v1.1 §2.1.3).** The running memory `S_t` superposes bound key–value pairs. Retrieval SNR for a superposition of *M* items in *D* dims scales ~√(D/M). With gating `γ<1` the *effective* number of superposed items is ~1/(1−γ). So there is an **effective context length** set by (D, γ, d_v) beyond which binding interference destroys retrieval.

**Sub-problems.** (a) Derive the effective context length as a function of (D, γ, β). (b) Does the finite-temperature cleanup (the Hopfield read) *extend* it relative to the raw linear-attention read, by denoising interference? [CONJECTURE: yes — the DAM cleanup is a denoiser, so RWR should tolerate longer effective contexts than vanilla linear attention at equal D.] (c) How does this compare to the known limitations of linear attention (the "recall-throughput trade-off," [Arora et al. 2024](https://arxiv.org/abs/2402.18668))?

**Why it matters.** This is the quantitative form of "can a fixed-width VSA state do long-range language," and (b) is a potential *advantage* of RWR over standard linear attention worth testing directly.

### OP3. Full space-time EP (strategy S-B)

**Statement.** §4 develops strategy S-A (EP-within-step + eligibility trace). The alternative S-B relaxes the *entire unrolled sequence* to a space-time equilibrium and nudges the whole output sequence at once. This is closer to the original EP spirit and to Lagrangian/Hamiltonian EP ([Pourcel et al. 2025](https://arxiv.org/abs/2506.06248); [Massar 2025](https://doi.org/10.1103/PhysRevE.112.035304)).

**Sub-problems.** (a) Formulate a space-time energy whose stationary state is the causal forward pass. (b) Does the causal (triangular) dependence structure admit a valid EP nudge, or does causality break the symmetric free/nudged relaxation EP assumes? (c) Compare S-A and S-B on cost, bias, and locality.

**Why it matters.** S-B, if it works, is more parallelizable and closer to a genuine physical implementation. It is higher-risk (causality + EP is unexplored) and should be attempted only after S-A validates.

### OP4. Does compressibility discriminate optimizers? (the meta-question)

**Statement.** The program uses requential coding to compare EP-trained and backprop-trained models. But is the requential code length *sensitive to the optimizer*, or only to the final function? Qiu et al. code the *training trajectory*; two optimizers reaching the same function by different paths could yield different code lengths.

**Sub-problems.** (a) Do EP and backprop trajectories to the same loss have systematically different cumulative teacher–student KL? (b) If EP takes a "rougher" path (noisier updates), does that inflate its requential code even at equal final loss — and is that a real generalization difference or an artifact of trajectory coding? (c) Can one define a *path-independent* variant (e.g. code only the final EMA model's self-distillation) to separate "optimizer path" from "learned function"?

**Why it matters.** This is essential for interpreting C1/P5 correctly. If requential code length conflates path roughness with poor generalization, the comparison must be adjusted. This is a genuine open question about the *measurement instrument itself*, and resolving it would contribute back to the requential-coding literature.

### OP5. Is there a formal REC↔cleanup correspondence?

**Statement.** §6.5 gives an analogy between relative entropy coding and VSA cleanup: both "pay only for the departure from shared randomness." Is there a *formal* connection?

**Sub-problems.** (a) Can VSA cleanup against a random codebook be cast as an REC decode with a specific proposal distribution? (b) Does the retrieval SNR √(D/M) map to a KL / rate in the REC sense? (c) If so, does a VSA language model have a *native* requential-style code length, computable without the teacher–student apparatus?

**Why it matters.** Speculative and high-risk, but if real it would unify the program's two halves — the VSA model and its compression measure would be the same object. Pursue last.

---

## 8. Methods and infrastructure

### 8.1 Software

| Tool | Use | Link |
|---|---|---|
| **flash-linear-attention** (`fla`) | Reference implementations of GLA, DeltaNet, RWKV, gated linear attention. **Start here for the backprop baseline.** | [github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) |
| **torchhd** | FHRR/phase primitives, feature maps. | [github.com/hyperdimensional-computing/torchhd](https://github.com/hyperdimensional-computing/torchhd) — [Heddes et al., *JMLR* 24(255), 2023](https://jmlr.org/papers/v24/23-0300.html) |
| **JAX** | The EP implementation: custom fixed-point solvers for the read, eligibility traces, and `custom_root`/implicit-diff for BPTT baselines. | [jax.readthedocs.io](https://jax.readthedocs.io) |
| **Requential-coding reference** | Qiu et al. release code (teacher smoothing, iso-loss projection, KL estimator, bound evaluation). **Use verbatim for §6.** | [github.com/shikaiqiu/requential-coding](https://github.com/shikaiqiu/requential-coding) |
| **Existing EP codebases** | Laborieux (scaling, symmetric nudge), Scellier, holomorphic-EP. Do not reimplement EP from scratch. | [github.com/Laborieux-Axel/Equilibrium-Propagation](https://github.com/Laborieux-Axel/Equilibrium-Propagation) |
| **nanoGPT / TinyStories harness** | Small-LM training loop, tokenizers, eval. | [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT); [TinyStories: Eldan & Li 2023](https://arxiv.org/abs/2305.07759) |

### 8.2 Architecture and hyperparameters (adopted from Qiu et al. for commensurability)

To make §6 comparisons apples-to-apples with a known-good compression baseline, adopt Qiu et al.'s setup verbatim where possible:

- **Backbone:** GPT-2-style, 8 blocks, context length 512, model size varied by width. For the RWR/linear-attention variants, replace softmax attention with the linear-attention/RWR recurrence; keep depth, width schedule, MLP, norm placement identical to the baseline.
- **Optimizer (baseline):** Adam (β₁=0.9, β₂=0.95, no weight decay), constant LR with linear warmup, µP for LR transfer across widths (base LR 2 under µP).
- **Datasets:** **TinyStories** (primary small-scale sanity, [Eldan & Li 2023](https://arxiv.org/abs/2305.07759)); **FineWeb** (GPT-2 BPE, V=50257) and **OpenWebText** (char-level, V=96) for the compression experiments to match Qiu et al. exactly; **CIFAR-5M-as-tokens** only if an image control is wanted.
- **Scales:** start ~5–20M (TinyStories coherence regime), then the Qiu et al. sweep (widths 128→2752, 14M→1B) for the compression-scaling curves (C2/P6).
- **Compute-optimal:** D = 20N (Chinchilla) for the generalization-bound experiments.

### 8.3 Datasets and what each is for

| Dataset | Role |
|---|---|
| **TinyStories** | Primary trainability sanity (P2, P3): small enough to iterate, rich enough that a 5–20M model produces coherent English; GPT-4-graded coherence/grammar/creativity rubric from the original paper. |
| **FineWeb (BPE)** | Compression experiments (C1–C3, P5–P7) at Qiu et al.'s exact settings. |
| **OpenWebText (char)** | Second compression dataset (Qiu et al. show largest requential gains here). |
| **A synthetic order-sensitive task** | [NEW] To isolate P4 (S-A0 vs S-A1): a task where order carries all the signal (e.g. induction-head / copy / associative-recall à la [Arora et al. 2024](https://arxiv.org/abs/2402.18668)), so the value of sequence credit assignment is measurable. |
| **Associative recall / MQAR** | Standard linear-attention stress test for effective context (OP2). |

### 8.4 Baselines — do not skip any

| Baseline | Why |
|---|---|
| **Backprop-trained linear attention (GLA/DeltaNet)** | The twin. Everything is measured against it, in both perplexity and requential code length. |
| **Backprop-trained softmax transformer** | Upper-reference for perplexity; the requential-coding paper's own baseline. |
| **RWR trained by *truncated BPTT*** | Ablation: isolates "does the RWR *architecture* help" from "does the *EP rule* work." T1 says RWR-BPTT at β→∞ = linear-attention-BPTT. |
| **S-A0 (local-in-time) vs S-A1 (eligibility trace)** | Isolates the value of sequence credit assignment (P4). |
| **e-prop on the same recurrence** | The closest existing local sequence-learning rule ([Bellec et al. 2020](https://www.nature.com/articles/s41467-020-17236-y)); RWR/EP should match or beat it, and the linear recurrence makes RWR's trace *exact* where e-prop's is approximate. |
| **Requential vs prequential vs PTQ** | The three compression references from Qiu et al., for C1–C3. |

### 8.5 Metrics

**Primary (capability):** validation perplexity / bits-per-token; TinyStories coherence rubric.
**Primary (compression):** requential code length `Σ_t KL(Q_t‖P_t)`; code-length-vs-loss Pareto; Finzi et al. certified bound.
**Diagnostic:** EP-vs-BPTT gradient cosine (P1); per-step read fixed-point uniqueness / iterations (OP1); effective context length (OP2); posterior entropy of the read; eligibility-trace magnitude vs. dependency length (P4).
**Secondary (report, do not lead with):** energy/latency on neuromorphic hardware. This program is about *whether local learning works for language and whether the result generalizes*, not about Joules.

### 8.6 Hardware targets (later milestones)

| Platform | Why | Reference |
|---|---|---|
| **Intel Loihi 2 (Lava)** | Spiking phasor neurons already implement FHRR reads; three-factor plasticity supports (4.4)–(4.6). | [Renner et al. 2024](https://www.nature.com/articles/s42256-024-00848-0) |
| **Coupled oscillator arrays** | The substrate EP already trains; the phase-domain read maps directly. | [Rageau & Grollier 2025](https://iopscience.iop.org/article/10.1088/2634-4386/adebaa); [Wang, Wanjura & Marquardt 2024](https://arxiv.org/abs/2402.08579) |
| **BrainScaleS-2** | Analog multi-compartment, on-chip plasticity. | Heidelberg |

---

## 9. Reference implementations (pseudocode)

Written for clarity, not speed. State in ℂ^D (amplitude + phase); codewords unit-magnitude.

### 9.1 FHRR feature map and the RWR forward pass

```python
import numpy as np

def phi(u):
    """FHRR feature map: real projection -> unit-magnitude phases (complex)."""
    return np.exp(1j * u)                      # (D,) complex, |phi_j| = 1

def rwr_forward(X, Wk, Wv, Wq, V, beta, gamma, read_steps=10, dt=0.3):
    """
    X    : (T, d_in) input token embeddings
    Wk,Wq: (D, d_in) key/query projections   Wv: (d_v, d_in) value projection
    V    : (d_v, K)  value/vocab codebook (output embeddings)
    beta : read inverse temperature   gamma: gate/forget in [0,1)
    returns per-step next-token distributions p (T, K) and read states r (T, d_v)
    """
    T = X.shape[0]; D = Wk.shape[0]; d_v = Wv.shape[0]; K = V.shape[1]
    S = np.zeros((D, d_v), dtype=complex)      # associative memory (bound superposition)
    z = np.zeros(D, dtype=complex)             # normalizer
    P = np.zeros((T, K)); R = np.zeros((T, d_v))
    for t in range(T):
        k = phi(Wk @ X[t]); q = phi(Wq @ X[t]); v = Wv @ X[t]     # key, query, value
        S = gamma * S + np.outer(k, v)         # bind (outer) + bundle (accumulate)
        z = gamma * z + k
        # query-retrieved content c_t = Re(q^H S) / Re(q^H z)
        num = np.real(np.conj(q) @ S); den = np.real(np.conj(q) @ z) + 1e-8
        c = num / den                          # (d_v,)  -- the linear-attention read
        # WARM READ: relax r on the modern-Hopfield energy (4.1)
        r = c.copy()
        for _ in range(read_steps):
            s = V.T @ r; s -= s.max()
            p = np.exp(beta * s); p /= p.sum()
            r = r + dt * (-r + c + V @ p)       # dr/dt = -dE/dr
        s = V.T @ r; s -= s.max()
        p = np.exp(beta * s); p /= p.sum()
        P[t] = p; R[t] = r
    return P, R
```

### 9.2 Per-step EP: free and nudged read

```python
def warm_read_relax(c, V, beta, target=None, beta_n=0.0, read_steps=30, dt=0.3):
    """Relax the read energy (4.1); if target given, add EP nudge -beta_n * dC/dr."""
    d_v = c.shape[0]; r = c.copy()
    for _ in range(read_steps):
        s = V.T @ r; s -= s.max()
        p = np.exp(beta * s); p /= p.sum()
        dr = -r + c + V @ p
        if target is not None:
            # C = -log p[target];  -dC/dr = beta * (V[:,target] - V @ p)
            dr = dr + beta_n * beta * (V[:, target] - V @ p)
        r = r + dt * dr
    s = V.T @ r; s -= s.max()
    p = np.exp(beta * s); p /= p.sum()
    return r, p

def ep_step_signals(c, target, V, beta, beta_n):
    """One step's EP teaching signal delta_t (4.5) and codebook update (4.4)."""
    r_free, p_free = warm_read_relax(c, V, beta, target=None,   beta_n=0.0)
    r_pos,  p_pos  = warm_read_relax(c, V, beta, target=target, beta_n=+beta_n)
    r_neg,  p_neg  = warm_read_relax(c, V, beta, target=target, beta_n=-beta_n)  # symmetric
    delta = (r_pos - r_neg) / (2 * beta_n)                      # (4.5) local content error
    dV = (np.outer(r_pos, p_pos) - np.outer(r_neg, p_neg)) / (2 * beta_n)  # (4.4) CHL on codebook
    return delta, dV, r_free
```

### 9.3 Sequence credit assignment via exact eligibility trace (S-A1)

```python
def rwr_ep_update(X, Wk, Wv, Wq, V, beta, gamma, beta_n, lr):
    """
    One full-sequence EP update (strategy S-A1).
    Forward the exact recurrence; at each step compute the local content error delta_t;
    propagate through the LINEAR recurrence with an EXACT eligibility trace.
    NO backprop-through-time, NO weight transport, NO stored activations beyond (S,z,trace).
    """
    T = X.shape[0]; D = Wk.shape[0]; d_v = Wv.shape[0]
    S = np.zeros((D, d_v), complex); z = np.zeros(D, complex)
    # eligibility traces: d S / d W, carried forward (exact because recurrence is linear in S)
    # For W_v: trace_v[a,b, :] accumulates gamma^{t-i} * outer(k_i, e_a) * x_i[b]
    # Implemented compactly: we accumulate the pieces needed to contract with delta_t.
    trace_kv = []   # list of (k_i, x_i, i) with decay applied on the fly (schematic)
    gWk = np.zeros_like(Wk); gWv = np.zeros_like(Wv); gWq = np.zeros_like(Wq); gV = np.zeros_like(V)

    for t in range(T):
        k = phi(Wk @ X[t]); q = phi(Wq @ X[t]); v = Wv @ X[t]
        S = gamma * S + np.outer(k, v); z = gamma * z + k
        num = np.real(np.conj(q) @ S); den = np.real(np.conj(q) @ z) + 1e-8
        c = num / den
        target = int(X_next_index[t])                          # next-token id
        delta, dV, _ = ep_step_signals(c, target, V, beta, beta_n)
        gV += dV

        # --- exact linear-recurrence credit assignment of delta onto W_k, W_v, W_q ---
        # c depends on S (hence on all past k_i v_i^T) and on q_t.
        # d c / d v_i  = gamma^{t-i} * Re(conj(q) . k_i) / den   (value pathway)
        # => gWv gets  outer( delta * dc/dv_i , x_i )  summed over i<=t
        # d c / d k_i  enters through both S (value) and z (normalizer): phase-domain term
        # d c / d q_t  is the current-step query pathway
        # (Full expressions in Appendix B.3; the point is they are EXACT and cheap because
        #  the recurrence is linear, so the trace is a decayed sum, not a BPTT unroll.)
        gWk, gWv, gWq = accumulate_trace_updates(gWk, gWv, gWq, delta, q, k, v,
                                                 X, t, gamma, den)  # Appendix B.3
    # descend
    Wk -= lr * gWk; Wv -= lr * gWv; Wq -= lr * gWq; V -= lr * gV
    return Wk, Wv, Wq, V
```

> **Note on locality and cost.** The eligibility trace is a *decayed sum over past steps*, O(1) state per parameter per step (because the recurrence is linear in `S`), not a stored unrolled graph. `delta_t` is a two-phase contrastive difference computed from two read relaxations. There is no gradient flowing backward through time; credit reaches the past only through the forward-carried trace. This is what makes the rule online and hardware-mappable.

### 9.4 Requential code length of a trained model (evaluation, per Qiu et al.)

```python
def requential_code_length(teacher_init, student_init, train_real, update, n_steps,
                           ema_alpha=0.01, smoothing=True):
    """
    Evaluate the requential code length WITHOUT running REC (Qiu et al. runtime trick):
    replace REC encode/decode by sampling X_t directly from the (EMA) teacher, and
    accumulate the unbiased KL estimate on the student's training batch.
    Returns L_req = sum_t KL(Q_t || P_t)  (bits), the certified compressed model size.
    """
    Q = teacher_init.copy(); P = student_init.copy()
    Q_ema = Q.copy(); P_ema = P.copy()
    L = 0.0
    for t in range(n_steps):
        Z = next(train_real)                       # real batch for the teacher
        Xt = sample(Q_ema)                          # synthetic batch ~ EMA teacher  (== REC decode)
        # unbiased KL estimate on the student's training batch: log Q(Xt) - log P(Xt)
        L += (logprob(Q_ema, Xt) - logprob(P_ema, Xt)).sum()   # nats; /ln2 for bits
        P = update(P, Xt)                           # student trains on synthetic
        Q = update(Q, Z)                            # teacher trains on real
        P_ema = ema(P_ema, P, ema_alpha)
        if smoothing: Q_ema = ema(Q_ema, Q, ema_alpha)
        else: Q_ema = Q
    return L / np.log(2)                             # bits
```

---
## 10. Milestones

18-month program, front-loaded with kill criteria. Every stage has an exit condition; a failed kill criterion means publish the negative result, not push on.

| M | Weeks | Task | Exit criterion | Kill? |
|---|---|---|---|---|
| **M0** | 1–3 | **Backprop baseline + its requential code.** Train a small linear-attention LM (GLA/DeltaNet) on TinyStories; reproduce coherent generation; compute its requential code length per Qiu et al. **Get the whole measurement pipeline working before touching EP.** | Baseline perplexity in known range; requential code computed and sane | ✋ Nothing is interpretable without this. |
| **M1** | 3–5 | **RWR forward pass.** Implement the Recurrent Warm Read; verify it does associative recall and next-token prediction by (truncated) BPTT, matching the linear-attention baseline. **Already partially validated — see Appendix D: RWR gets 100% associative recall.** | RWR-BPTT ≈ linear-attention-BPTT perplexity | ☠️ Kills the architecture if RWR can't match linear attention even with BPTT |
| **M2** | 5–8 | **EP-gradient validation (P1).** S-A1 eligibility-trace EP update vs truncated-BPTT gradient, cosine as β_n→0. **Per-step version already validated in Appendix D (cos→1.000).** Extend to the full sequence trace. | **P1:** cos > 0.9 as β_n→0 | ☠️ **Kills the program** |
| **M3** | 8–14 | **End-to-end EP training on TinyStories (P2).** Train RWR by S-A1; compare perplexity and generation to the backprop twin. | **P2:** within 20% perplexity | ☠️ Kills the core trainability claim |
| **M4** | 12–18 | **The compression comparison (C1/P5).** Requential code length: EP-trained vs backprop-trained at matched loss. | **P5:** within ~1.5× | ✅ Cannot fail informatively — a large gap is a decisive (publishable) memorization finding |
| **M5** | 14–20 | **Finite-β dividend + S-A0 vs S-A1 (P3, P4).** Ablate temperature and sequence-credit-assignment. | P3, P4 | — |
| **M6** | 18–28 | **Scaling (C2/P6) + certified bound (C3/P7).** Qiu et al. width sweep; Finzi et al. bound on the EP-trained model. | **P7:** non-vacuous bound at ~100M | 🏆 First certified generalization bound for a backprop-free LM |
| **M7** | 24–40 | **Effective context (OP2)** on associative recall / MQAR; test whether the Warm Read extends effective context vs vanilla linear attention. | OP2(b) resolved | — |
| **M8** | 32–52 | **Space-time EP (S-B, OP3)** — the higher-risk parallel formulation. | Formulated + tested | — |
| **M9** | 44–72 | **Hardware** — Loihi 2 / oscillator array, on-chip RWR read + local update. | On-chip learning demo | — |

**If you have one person and three months:** M0 → M1 → M2 → M3. That is a complete, publishable unit: a backprop-free rule that trains a linear-attention/VSA language model to within-tolerance perplexity on TinyStories, with the EP-gradient validation and the T1 theorem as the theoretical spine. M4 (the compression comparison) is the natural fourth beat and is what elevates it from "it works" to "it generalizes."

**Parallelism.** M0–M4 (does it work + does it compress) is the critical path. M7–M8 (effective context, space-time EP) are independent theory tracks. M9 (hardware) waits on M3.

---

## 11. Risks, failure modes, and the value of negative results

### 11.1 Enumerated risks

| # | Risk | Likelihood | Mitigation | If it happens |
|---|---|---|---|---|
| **R1** | **The honest ceiling:** even if RWR/EP works, it only matches linear attention, which already exists — so "why bother." | **Certain** as stated; it's the framing, not a failure. | Lead with the *two* things that are new: a **local backprop-free rule** for language (⇒ on-chip training) and a **compression-certified** generalization result. Never claim "beats transformers." | Reframe, don't retreat. The contribution is the *learning rule* and the *certificate*, not a new SOTA perplexity. |
| **R2** | EP gradient too biased/noisy at usable β_n to train a full LM. | Medium | Symmetric nudge (Laborieux 2021); **holomorphic EP** (Laborieux & Zenke 2022) for exact complex-domain gradients — natural fit for FHRR. Per-step validation already clean (App. D). | Fall back to implicit differentiation for the science; treat locality as a later hardware goal. |
| **R3** | **Sequence credit assignment (S-A1) doesn't capture long dependencies** — the eligibility trace, though exact for the linear recurrence, decays too fast (γ) to reach far back. | Medium-high | This *is* the linear-attention context-length limitation (OP2). Gated/delta variants; test whether the Warm Read's denoising extends effective context. | Quantify it honestly (P4, OP2). A short effective context is a known property of linear attention, not a bug unique to EP. |
| **R4** | **Compression comparison confounded** — requential code length reflects the *training path*, and EP's noisier path inflates its code even at equal final loss (OP4). | Medium | Add the path-independent variant (OP4c): code only the final EMA model's self-distillation. Report both. | The confound is itself a finding about the measurement instrument; contribute it back. |
| **R5** | **The finite-β dividend (P3) is zero** — the Hopfield cleanup buys nothing over the raw linear-attention read. | Medium | Then RWR collapses to linear attention (T1) and the contribution is purely the *learning rule*, not the architecture. Still publishable. | Drop the Warm Read; apply the EP rule directly to vanilla linear attention. Simpler result. |
| **R6** | Read fixed point non-unique (OP1) ⇒ EP two-phase difference ill-defined. | Medium | Strong content clamp; low-enough β; characterize the safe regime (OP1). | Constrain β to the single-basin regime; report the constraint. |
| **R7** | Prior art exists (someone already did EP-on-linear-attention or VSA-LM). | Low-medium | §12 log. If found, build on it. | Cite, adjust, contribute the compression angle regardless. |

### 11.2 The value of the negative results

The program is designed so every outcome informs:

- **R1 (certain):** forces the correct, defensible framing — the contribution is the *rule* and the *certificate*, and that framing is itself worth publishing because it clarifies what "neuromorphic for LLMs" can and cannot mean.
- **R3/OP2:** a sharp measurement of how much *sequence* (vs. per-step) credit assignment a local rule can achieve through a linear recurrence — directly relevant to the whole linear-attention field.
- **R4/OP4:** a contribution back to requential coding — does trajectory-coding conflate optimizer path with generalization?
- **P5 failure:** a decisive demonstration that a backprop-free learner *memorizes* rather than learns structure — exactly the finding perplexity alone would hide, and exactly what the compression instrument was adopted to catch.

**Publish the negative results.** The reason this seam looked empty (§12) is the file-drawer problem.

---

## 12. Prior-art search log

**So the negative claims can be checked and overturned.** Negative literature findings are weak evidence; treat as "we could not find it."

### 12.1 Queries run (July 2026; arXiv, Semantic Scholar, OpenReview, Google Scholar, lab pages)

```
equilibrium propagation language model / transformer / next-token / autoregressive
equilibrium propagation linear attention / state space model / RWKV / recurrence
vector symbolic architecture language model / next-token / sequence
hyperdimensional computing language modeling / autoregressive / GPT
contrastive Hebbian learning sequence / language / transformer
predictive coding language model / transformer / autoregressive (Bogacz/Salvatori)
forward-forward language model / sequence / next-token
e-prop language model / transformer / linear attention
local learning rule transformer / attention backprop-free
linear attention as associative memory / Hopfield / vector symbolic
requential coding / prequential coding language model compression generalization
holographic reduced representation language model / HRR sequence
```

### 12.2 Results

**FOUND — adjacent:**

| Work | What it is | Why not the thing |
|---|---|---|
| **Linear attention = fast weights / associative memory** ([Schlag, Irie & Schmidhuber, ICML 2021](https://arxiv.org/abs/2102.11174)) | Establishes linear transformers as fast-weight associative memories — the representational half of §3.3 | Representation, not *local learning*. Trained by backprop. |
| **DeltaNet, GLA, Mamba, RWKV** | Linear-attention/SSM language models | All backprop-trained. The architecture this program proposes to train *locally*. |
| **Predictive coding for sequences / language** (various, Bogacz/Salvatori/Millidge) | PC applied to some sequence tasks | Not VSA; not linear-attention; PC's own depth-scaling limits (v1.1 §B) apply; no compression evaluation. |
| **e-prop** ([Bellec et al., *Nat. Commun.* 11:3625, 2020](https://www.nature.com/articles/s41467-020-17236-y)) | Local online learning for recurrent SNNs via eligibility traces | The closest *mechanism* (traces), and a baseline (§8.4). But e-prop targets nonlinear RNNs (approximate trace) and small tasks, not linear-attention LMs; no VSA framing, no compression certificate. |
| **EP scaling / holomorphic EP / oscillator EP** (Laborieux, Scellier, Grollier, Marquardt) | EP on ConvNets, complex domain, physical oscillators | None target language or autoregressive loss. The enabling substrate, not the application. |
| **Requential coding** ([Qiu et al. 2026](https://arxiv.org/abs/2607.11883)) | Compression-based generalization measurement for LMs | A *measurement* method (adopted in §6), not a learning method, and not VSA. |

**NOT FOUND — the actual seam:**
- Equilibrium propagation applied to an autoregressive language-modeling loss: **nothing**
- EP or CHL training a linear-attention / SSM / RWKV model: **nothing**
- A VSA / hyperdimensional *language model* trained by a local backprop-free rule: **nothing**
- Requential/compression evaluation of any backprop-free learner: **nothing**
- The T1 correspondence (linear-attention training as the zero-temperature limit of EP on a Hopfield read): **nothing**

### 12.3 Limitations of this search

1. English, web-indexed only.
2. Workshop papers (NICE, NeurIPS/ICLR workshops on associative memory, neuromorphic) are under-indexed. **Sweep NICE 2022–2026 and the NeurIPS Associative Memory & Hopfield workshops before claiming novelty.**
3. File-drawer risk: EP-on-language may have been tried and failed silently.
4. Fast-moving area: linear-attention and local-learning literatures both move monthly.

### 12.4 Due diligence before publication

- [ ] Full-text sweep of **NICE 2022–2026** and the **NeurIPS 2023/2024 Associative Memory workshops**.
- [ ] OpenReview full-text: `"equilibrium propagation" AND ("language" OR "autoregressive")`; `"linear attention" AND ("local learning" OR "Hebbian")`.
- [ ] Citation-graph intersection of Ernoult et al. 2019 (EP-matches-BPTT) × Katharopoulos et al. 2020 (linear attention). Should be near-empty; read anything in it.
- [ ] **Email the people who would know:** Benjamin Scellier, Maxence Ernoult, Axel Laborieux, Friedemann Zenke; Julie Grollier, Damien Querlioz, Florian Marquardt (oscillator EP); Guillaume Bellec, Wolfgang Maass (e-prop); Songlin Yang, Yoon Kim (linear attention); Shikai Qiu, Marc Finzi, Andrew Gordon Wilson (requential coding). A five-line email could save six months.

---

## 13. Bibliography

All links verified July 2026. Organized by theme.

### 13.1 VSA / HDC foundations
- Kanerva, P. (2009). **Hyperdimensional Computing.** *Cognitive Computation* 1(2):139–159. https://doi.org/10.1007/s12559-009-9009-8
- Plate, T. A. (1995). **Holographic Reduced Representations.** *IEEE Trans. Neural Networks* 6(3):623–641. https://doi.org/10.1109/72.377968
- Frady, E. P., Kleyko, D. & Sommer, F. T. (2018). **A Theory of Sequence Indexing and Working Memory in Recurrent Neural Networks.** *Neural Computation* 30(6):1449–1513. https://doi.org/10.1162/neco_a_01084
- Kleyko, D., Rachkovskij, D. A., Osipov, E. & Rahimi, A. (2022/2023). **A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Parts I & II.** *ACM Computing Surveys*. https://dl.acm.org/doi/10.1145/3538531 · https://dl.acm.org/doi/10.1145/3558000
- Frady, E. P. & Sommer, F. T. (2019). **Robust Computation with Rhythmic Spike Patterns.** *PNAS* 116(36):18050–18059. https://doi.org/10.1073/pnas.1902653116
- Heddes, M. et al. (2023). **Torchhd.** *JMLR* 24(255). https://jmlr.org/papers/v24/23-0300.html · https://github.com/hyperdimensional-computing/torchhd

### 13.2 Linear attention, state-space models, and the associative-memory view
- Katharopoulos, A., Vyas, A., Pappas, N. & Fleuret, F. (2020). **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.** ICML 2020. https://arxiv.org/abs/2006.16236
- Schlag, I., Irie, K. & Schmidhuber, J. (2021). **Linear Transformers Are Secretly Fast Weight Programmers.** ICML 2021. https://arxiv.org/abs/2102.11174 — *the representational half of §3.3*
- Yang, S., Wang, B., Shen, Y., Panda, R. & Kim, Y. (2024). **Gated Linear Attention Transformers with Hardware-Efficient Training** (GLA). https://arxiv.org/abs/2312.06635
- Yang, S., Wang, B., Zhang, Y., Shen, Y. & Kim, Y. (2024). **Parallelizing Linear Transformers with the Delta Rule over Sequence Length** (DeltaNet). https://arxiv.org/abs/2406.06484
- Gu, A. & Dao, T. (2023). **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** https://arxiv.org/abs/2312.00752
- Peng, B. et al. (2023). **RWKV: Reinventing RNNs for the Transformer Era.** EMNLP Findings 2023. https://arxiv.org/abs/2305.13048
- Arora, S. et al. (2024). **Zoology: Measuring and Improving Recall in Efficient Language Models** (recall–throughput trade-off, MQAR). https://arxiv.org/abs/2402.18668

### 13.3 Modern Hopfield / Dense Associative Memory / attention
- Ramsauer, H. et al. (2021). **Hopfield Networks Is All You Need.** ICLR 2021. https://arxiv.org/abs/2008.02217
- Krotov, D. & Hopfield, J. J. (2016). **Dense Associative Memory for Pattern Recognition.** NeurIPS 2016. https://arxiv.org/abs/1606.01164
- Demircigil, M. et al. (2017). **On a Model of Associative Memory with Huge Storage Capacity.** *J. Stat. Phys.* 168:288–299. https://arxiv.org/abs/1702.01929
- Lucibello, C. & Mézard, M. (2024). **Exponential Capacity of Dense Associative Memories.** *Phys. Rev. Lett.* 132:077301. https://doi.org/10.1103/PhysRevLett.132.077301
- Bricken, T. & Pehlevan, C. (2021). **Attention Approximates Sparse Distributed Memory.** NeurIPS 2021. https://arxiv.org/abs/2111.05498
- Hoover, B. et al. (2023). **Energy Transformer.** NeurIPS 2023. https://arxiv.org/abs/2302.07253

### 13.4 Equilibrium propagation and backprop-free learning
- Scellier, B. & Bengio, Y. (2017). **Equilibrium Propagation.** *Front. Comput. Neurosci.* 11:24. https://www.frontiersin.org/articles/10.3389/fncom.2017.00024/full
- Ernoult, M., Grollier, J., Querlioz, D., Bengio, Y. & Scellier, B. (2019). **Updates of Equilibrium Prop Match Gradients of Backprop Through Time in an RNN with Static Input.** NeurIPS 2019. https://arxiv.org/abs/1905.13633 — ***the theorem that lets EP touch sequence models***
- Laborieux, A. et al. (2021). **Scaling Equilibrium Propagation to Deep ConvNets by Drastically Reducing Its Gradient Estimator Bias.** *Front. Neurosci.* 15:633674. https://www.frontiersin.org/articles/10.3389/fnins.2021.633674/full — *symmetric nudge*
- Laborieux, A. & Zenke, F. (2022). **Holomorphic Equilibrium Propagation Computes Exact Gradients Through Finite Size Oscillations.** NeurIPS 2022. https://arxiv.org/abs/2209.00530 — *exact complex-domain gradients; natural for FHRR*
- Scellier, B., Goyal, A., Binas, J., Mesnard, T. & Bengio, Y. (2018). **Generalization of Equilibrium Propagation to Vector Field Dynamics.** https://arxiv.org/abs/1808.04873
- Pourcel, G., Basu, D., Ernoult, M. & Gilra, A. (2025). **Lagrangian-Based Equilibrium Propagation ... Equivalence with Hamiltonian Echo Learning.** https://arxiv.org/abs/2506.06248
- Massar, S. (2025). **Equilibrium Propagation for Learning in Lagrangian Dynamical Systems.** *Phys. Rev. E* 112:035304. https://doi.org/10.1103/PhysRevE.112.035304
- Millidge, B., Song, Y., Salvatori, T., Lukasiewicz, T. & Bogacz, R. (2022). **Backpropagation at the Infinitesimal Inference Limit of Energy-Based Models: Unifying Predictive Coding, Equilibrium Propagation, and Contrastive Hebbian Learning.** https://arxiv.org/abs/2206.02629
- Bartunov, S. et al. (2018). **Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures.** NeurIPS 2018. https://arxiv.org/abs/1807.04587 — *the canonical negative result*
- Hinton, G. (2022). **The Forward-Forward Algorithm.** https://arxiv.org/abs/2212.13345

### 13.5 EP on physical / oscillator substrates
- Laydevant, J., Marković, D. & Grollier, J. (2024). **Training an Ising Machine with Equilibrium Propagation.** *Nat. Commun.* 15:3671. https://www.nature.com/articles/s41467-024-46879-4
- Wang, Q., Wanjura, C. C. & Marquardt, F. (2024). **Training Coupled Phase Oscillators as a Neuromorphic Platform Using Equilibrium Propagation.** https://arxiv.org/abs/2402.08579
- Rageau, T. & Grollier, J. (2025). **Training and Synchronizing Oscillator Networks with Equilibrium Propagation.** *Neuromorphic Computing and Engineering* 5(3):034008. https://iopscience.iop.org/article/10.1088/2634-4386/adebaa
- Momeni, A. et al. (2025). **Training of Physical Neural Networks.** *Nature*. https://www.nature.com/articles/s41586-025-09384-2

### 13.6 Eligibility traces and local sequence learning
- Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R. & Maass, W. (2020). **A Solution to the Learning Dilemma for Recurrent Networks of Spiking Neurons** (e-prop). *Nat. Commun.* 11:3625. https://www.nature.com/articles/s41467-020-17236-y

### 13.7 Compression, generalization, and the evaluation backbone
- **Qiu, S., Finzi, M., Zheng, Y., Zhang, K. & Wilson, A. G. (2026). Requential Coding: Pushing the Limits of Model Compression with Self-Generated Training Data.** arXiv:2607.11883. https://arxiv.org/abs/2607.11883 · code: https://github.com/shikaiqiu/requential-coding — ***the evaluation backbone (§6)***
- Finzi, M. A., Kapoor, S., Granziol, D., Gu, A., De Sa, C., Kolter, J. Z. & Wilson, A. G. (2025). **Compute-Optimal LLMs Provably Generalize Better with Scale.** ICLR 2025. — *the PAC-Bayes bound plugged into §6*
- Blier, L. & Ollivier, Y. (2018). **The Description Length of Deep Learning Models.** NeurIPS 2018. — *prequential coding*
- Lotfi, S., Finzi, M., Kuang, Y., Rudner, T. G. J., Goldblum, M. & Wilson, A. G. (2023). **Non-Vacuous Generalization Bounds for Large Language Models.** https://arxiv.org/abs/2312.17173
- Lotfi, S., Kuang, Y., Amos, B., Goldblum, M., Finzi, M. & Wilson, A. G. (2024). **Unlocking Tokens as Data Points for Generalization Bounds on Larger Language Models.** NeurIPS 2024. https://arxiv.org/abs/2407.18158

### 13.8 Datasets and tooling
- Eldan, R. & Li, Y. (2023). **TinyStories: How Small Can Language Models Be and Still Speak Coherent English?** https://arxiv.org/abs/2305.07759
- Penedo, G. et al. (2024). **The FineWeb Datasets.** NeurIPS 2024. https://arxiv.org/abs/2406.17557
- Karpathy, A. **nanoGPT.** https://github.com/karpathy/nanoGPT
- **flash-linear-attention.** https://github.com/fla-org/flash-linear-attention

### 13.9 The prior version
- v1.1, **"Learning the Codebook: Equilibrium Propagation for Resonator Networks, and Contrastive Hebbian Learning for Hyperdimensional Computing"** (this program, classification/factorization variant; superseded by the present document). Internal reference: Theorem B1, the coherence analysis, and the torus-state self-falsification (Appendix D of that document).

---

## Appendix A — Notation

| Symbol | Meaning |
|---|---|
| `D` | hypervector / key dimension |
| `d_v` | value dimension |
| `K` | vocabulary / codebook size |
| `T` | sequence length |
| `⊗`, `+`, `ρ` | binding, bundling, permutation |
| `φ(·)` | FHRR feature map, `φ(u)_j = e^{i u_j}` |
| `k_t, q_t, v_t` | key, query, value at step t |
| `S_t` | running associative memory (bound superposition), ℂ^{D×d_v} |
| `z_t` | normalizer, ℂ^D |
| `γ` | gate / forget factor |
| `c_t` | query-retrieved content (the linear-attention read) |
| `r` | Warm-Read state (relaxes on E_read) |
| `V` | value/vocabulary codebook, ℝ^{d_v×K} |
| **`β`** | **read inverse temperature** (model hyperparameter) |
| **`β_n`** | **EP nudge strength** (learning hyperparameter) |
| `δ_t` | EP content teaching signal, `(r_t^+ − r_t^-)/β_n` |
| `p_t` | next-token distribution |
| `L_req` | requential code length, `Σ_t KL(Q_t‖P_t)` |

> ⚠️ **`β` (read temperature) and `β_n` (EP nudge) are different quantities.** Conflating them is the most likely implementation bug.

---

## Appendix B — Derivations

### B.1 The Warm Read energy and dynamics

Energy (4.1): `E_read(r) = ½‖r‖² − Re⟨c,r⟩ − (1/β) lse(β Vᵀr)`. With `p = softmax(βVᵀr)`:

```
∂E/∂r = r − c − V p
```

Gradient flow `dr/dt = −∂E/∂r = −r + c + V·softmax(βVᵀr)`, converging to `r* = c + V·softmax(βVᵀr*)`. This is the modern-Hopfield retrieval with an external content bias `c`. ∎

### B.2 The EP content teaching signal (4.5)

`c` enters `E_read` only through the linear bias `−Re⟨c,r⟩`, so `∂E/∂c = −r`. By the EP theorem, the loss gradient w.r.t. `c` is
```
dC/dc = lim_{β_n→0} (1/β_n)[ ∂E/∂c|^{β_n} − ∂E/∂c|^0 ] = lim_{β_n→0} (1/β_n)( −r^+ + r^- ) = −δ_t,
```
where `δ_t = (r^+ − r^-)/β_n`. **`δ_t` is (minus) the loss gradient on the content**, computed by two relaxations. 

> **Numerically confirmed (Appendix D):** with the symmetric nudge, `cos(δ_t, −dC/dr*) → −0.997…` and magnitude ratio → 1.0 as β_n→0. (The sign: `δ_t` points along descent of the cost, `dC/dr*` along ascent; they are anti-parallel, as required.) ∎

### B.3 The codebook update needs TWO paths — a correction found by running the code

This is the most important derivation subtlety in the document, and it was **found empirically** (Appendix D), not anticipated.

The codebook `V` enters the per-step loss `C = −log softmax(βVᵀr*)[y]` through **two** paths:

1. **The read-energy path:** `V` is the stored-pattern matrix in `E_read`, so it shapes the fixed point `r*`. EP captures this path: `∂E/∂V_k = −p_k r`, giving the contrastive term
```
ΔV_k^{(read)} ∝ (1/β_n)[ p_k^+ r^+ − p_k^- r^- ].                                     (4.4, read path)
```

2. **The readout path:** `V` *also* appears explicitly in the final softmax `p_t = softmax(βVᵀr*)`. This is a direct dependence not mediated by the relaxation, with the standard cross-entropy-through-softmax gradient
```
ΔV_k^{(readout)} ∝ β · r* · (p_k^* − 1{k=y}).                                          (readout path)
```

**The full codebook update is the sum:**
```
ΔV_k = ΔV_k^{(read)} + ΔV_k^{(readout)}.                                               (4.4-corrected)
```

> **Numerically confirmed (Appendix D):** the read path alone has `cos ≈ 0.74` with the true finite-difference gradient; **read + readout gives `cos = 1.0000`.** The naive (read-only) update trains in the *wrong direction* and increases loss; the corrected two-path update drives loss from 5.05 → 0.004. **Any implementation must include the readout term.** This is the sequence-model analogue of the fact that in a standard energy-based classifier the output weights get both a relaxation-mediated and a direct readout gradient. ∎

**Projection updates.** `W_k, W_q, W_v` enter only through `c_t` (via the exact forward recurrence `S_t, z_t`). Given the content error `δ_t = −dC/dc_t`, the chain to each projection is through `c_t = Re(q_t^⊤ S_t)/Re(q_t^⊤ z_t)`:
- **Value path:** `∂c_t/∂v_i = γ^{t-i} Re(\overline{q_t}·k_i)/den_t`, so `ΔW_v ∝ Σ_{i≤t} δ_t · [γ^{t-i} Re(\overline{q_t}·k_i)/den_t] · x_i^⊤` — an eligibility-trace-weighted Hebbian term.
- **Key path:** `k_i` enters both `S_t` (numerator) and `z_t` (denominator); the derivative is a phase-domain term `∝ Im(...)` carried in the same trace.
- **Query path:** current-step only, `∂c_t/∂q_t` through numerator and denominator.

Because the recurrence is **linear** in `S`, all traces are decayed sums (O(1) state per parameter per step), not BPTT unrolls. ∎

### B.4 Theorem T1 (linear attention as the β→∞ EP limit)

**Claim.** As β→∞ with a single read step and the symmetric nudge in its β_n→0 limit, the RWR per-step computation reduces to standard linear attention, and the S-A1 update reduces to truncated-BPTT training of a linear-attention layer.

**Proof.**
*Read collapse.* As β→∞, `softmax(βVᵀr) → e_{argmax}`. One step of `dr/dt` from `r=c` gives `r ← c + V e_{argmax(Vᵀc)}`; with the single-step/hard-cleanup convention the retrieved content is the nearest codeword and the *logits* are `βVᵀc` up to the argmax — i.e. `p_t → softmax(βVᵀc_t)`, the ordinary linear-attention readout over the value codebook. The content `c_t = Re(q_t^⊤S_t)/Re(q_t^⊤z_t)` is exactly the linear-attention read (§2.2.1).

*Gradient collapse.* By B.2, `δ_t = (r^+−r^-)/β_n → −dC/dc_t` as β_n→0. At β→∞ the read is the identity map `c ↦ c` (one step), so `dC/dc_t` is the plain cross-entropy gradient of the linear-attention logits w.r.t. the read. Feeding this through the exact linear-recurrence eligibility trace (B.3) reconstructs the truncated-BPTT gradient of a linear-attention layer (the trace *is* the linearized backward recurrence for a linear state update). ∎

**Consequence.** Standard linear-attention training is the zero-temperature, single-Newton-step corner of the RWR/EP family. The finite-β, multi-step, two-phase rule is the local generalization. This mirrors v1.1's Theorem B1 (OnlineHD as the zero-temperature EP limit of a Hopfield readout) at the sequence-model level.

---

## Appendix C — The one-paragraph pitch

> Linear attention — the architecture behind Mamba, RWKV, DeltaNet, and GLA — is, structurally, a Vector-Symbolic Architecture: it binds keys to values, bundles them into a running associative memory, and reads by unbinding with a query. It is VSA with a *learned* codebook, and it is competitive with transformers for language. But it is trained by backpropagation. We propose to train it by a **local, backpropagation-free rule**: reformulate each step's read as a finite-temperature modern-Hopfield retrieval (a gradient system), apply equilibrium propagation to get a two-phase contrastive teaching signal on the retrieved content, and assign credit through the *linear* state recurrence with an *exact* eligibility trace — no backprop through time, no weight transport, no stored activations. We prove that standard linear-attention training is the zero-temperature limit of this rule (Theorem T1), and we evaluate not by perplexity alone but by **requential coding** (Qiu et al. 2026): at matched loss, does the locally-trained model compress — hence provably generalize — as well as its backprop-trained twin? If yes, neuromorphic hardware gets a route to language-model *training*, and we get the first certified generalization bound for a backprop-free language model. If no, the compression instrument tells us exactly where local learning falls short — in optimization, representation geometry, or memorization. **Either way the field learns something it does not currently know, and every claim in this document is falsifiable, staged, and equipped with a kill criterion.**

---

## Appendix D — Numerical exploration log (July 2026)

**Status: preliminary, and it changed the document.** Two of the central claims were tested at small scale before writing the derivations. One passed cleanly; one exposed a sign/structure error in the codebook update that is now corrected in the text (B.3). Reported in full so the corrections are auditable.

Environment: NumPy, float64, single-threaded. All scripts reconstructible from §9.

### D.1 The RWR forward pass works (Test 1)

Associative-recall task: write *K* key–value pairs into the running memory `S`, then query the keys in random order and read via the Warm Read (4.1)–(4.2). Result: **100% recall** across gates γ ∈ {0.9, 1.0} and read temperatures β ∈ {2, 5, 10}, for K=6 pairs in d_v=16. The core forward mechanism — bind, bundle, query-retrieve, Hopfield-clean-up — reads associative memory correctly. **The architecture is sound; M1 is de-risked.**

### D.2 The EP content signal recovers the true gradient (Test 2 — this is P1 in miniature)

For a single read, compared the EP teaching signal `δ_t = (r^+ − r^-)/(2β_n)` (symmetric nudge) against the finite-difference gradient `dC/dc` through the fixed point:

| β | β_n | cos(δ, dC/dc) | ‖δ‖/‖grad‖ |
|---|---|---|---|
| 2 | 0.5 | −0.977 | 1.17 |
| 2 | 0.1 | −0.999 | 1.09 |
| 2 | 0.02 | **−1.0000** | **1.003** |
| 5 | 0.02 | **−1.0000** | 1.000 |

As β_n→0, `δ_t` recovers the gradient **exactly** (cos → −1, ratio → 1; the sign is the descent/ascent convention, §B.2). **P1 holds at the per-step level.** Large-β_n rows are badly biased (e.g. β=5, β_n=0.1 gives cos −0.34) — the expected finite-nudge EP bias, cancelled only in the limit; use small β_n and the symmetric nudge (Laborieux 2021).

### D.3 The codebook update needed a second term (Test 3 — found here, fixed in B.3)

**The naive read-only update (4.4) trained in the wrong direction**: loss *rose* 5.05 → 27.6 and stuck. Diagnosis by comparison to the finite-difference gradient of the full loss:

- `δ_t` recovers the **state** gradient `dC/dr*` cleanly (cos −0.997). Good.
- But `V` enters the loss through a **second, explicit readout path** (the final softmax `softmax(βVᵀr*)`) that the read-EP does not see.
- **Read path alone: cos 0.74** with the true gradient. **Read + readout path: cos 1.0000.**
- With the corrected two-path update (B.3, eq. 4.4-corrected), loss drops **5.05 → 0.004** in 8 steps.

**This is now baked into the derivation (B.3) and the pseudocode should include the readout term.** It is the sequence-model analogue of the standard fact that output weights in an energy-based classifier receive both a relaxation-mediated and a direct readout gradient. It is exactly the kind of error the "run the code" discipline is meant to catch before it reaches a reader.

### D.4 Honest assessment

- ✅ **Architecture (forward read): validated.** 100% associative recall.
- ✅ **EP content signal: validated.** Recovers the true gradient exactly as β_n→0 (P1 in miniature).
- ⚠️ **Codebook update: corrected.** The first-written form was wrong (missing readout path); the corrected two-path form is exact (cos 1.0) and trains (loss → 0.004).
- ❓ **Not yet tested at any scale:** the full sequence eligibility trace (S-A1), end-to-end language training (P2), and — the whole point of the reorientation — whether the local rule matches backprop *and compresses as well* (P5). Those are M2–M4 and are not de-riskable from an afternoon of NumPy.

**Reproduction.** All three tests rebuild from §9 in minutes. Reproduce D.2 and D.3 before trusting Appendix B; reproduce D.1 before trusting §4.2.

---

## Document changelog

| Version | Date | Change |
|---|---|---|
| 1.0 / 1.1 | July 2026 | Classification/factorization program (superseded). v1.1 added the torus-state self-falsification. |
| **2.0** | **July 2026** | **Full reorientation to autoregressive next-token prediction.** New central obstacle (bundling is order-invariant); linear-attention-as-VSA bridge; Recurrent Warm Read architecture; causal-EP derivation (strategies S-A/S-B) with exact linear-recurrence eligibility trace; Theorem T1 (linear attention as the β→∞ EP limit); **requential coding (Qiu et al. 2026) adopted as the evaluation backbone** with three compression predictions; five reoriented open problems; milestones with kill criteria. **Appendix D:** RWR forward pass validated (100% recall) and per-step EP gradient validated (cos→1.0); **codebook update corrected** (readout-path term added) after numerics showed the first form trained backwards. |

**License:** CC BY 4.0. Reuse, adapt, and — especially — *falsify* freely.

**If you run any of these experiments, report the result, including the negative ones. The file-drawer problem is why this seam looked empty.**
