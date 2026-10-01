# 11 — Weight transport & backward-pass-free credit assignment: literature status (July 2026)

**Role:** External literature check on the L1-VF obstruction (`09-spacetime-ep.md`, `experiments/S8-L1VF-result.md`)
and on the neuromorphic-training aspiration (`00 §5`, whitepaper `§1.4(1)`, `§8.6`).

**Method:** Adversarial multi-source research pass, 2026-07-21. Six search angles → 29 sources → 145 extracted
claims → top 25 put to a 3-vote adversarial verification panel → **11 survived, 14 refuted for overreach**.
Findings below are labelled with their vote. Claims that failed verification are listed in §7 so they are not
accidentally reused.

**Headline:** As of mid-2026 **no published method is simultaneously transport-free, backward-pass-free, and
demonstrated on a causal sequence model.** The gap L1-FA sits in is the field's open gap, not a local failure.
But the L1-VF negative result **is not novel** and must be cited, not claimed — and its current phrasing is
**too strong** and would be read as refuted by a reviewer who knows ICML 2026.

---

## 1. The L1-VF obstruction has direct published prior art

**Scurria, Vanden Abeele, Mognetti & Massar, "Equilibrium Propagation for Non-Conservative Systems."**
ICML 2026, PMLR 306 (arXiv:2602.03670, v1 2026-02-03, v2 2026-06-01). *Peer-reviewed.* [vote 3-0]

- Verbatim (Sec. 2): *"as noted in (Scellier et al., 2018), Eq. (12) does not align with the true gradient
  dC/dθ(x⁰) and is exact only if the force is conservative."*
- Quantitative characterisation: *"the gradient error increases with the spectral radius of
  (S_J(x⁰,θ))⁻¹ A_J(x⁰,θ)"* — symmetric-part inverse times antisymmetric part.
- Pathological limit: where the Jacobian is purely antisymmetric (S_J = 0), *"the update of VF gives the
  negative of the true gradient, maximizing the cost rather than minimizing it."*
- Empirical (Table 2, CIFAR-10, epoch 40, 5 seeds, identical deep feedforward CNN, estimator is the only
  variable): **BP 90.66 ± 0.25 · Dyadic EP 90.69 ± 0.14 · AsymEP 89.74 ± 0.14 · VF 10.00 ± 0.00.**
  Narrative: *"VF makes slight initial progress (peaking near 30%) before collapsing to chance level (10%)"*,
  attributed to *"the missing transpose in the post-synaptic term."*

**Read across to us.** A causal block stack is exactly the non-reciprocal/asymmetric regime this covers
(the paper "extends EP to arbitrary non-conservative systems, including feedforward networks"). Their VF
collapse to chance (10% of 10 classes) is the published analogue of our collapse to 1/64 on MQAR (0.012).
The analogy is **qualitative-mechanistic, not quantitative** — different chance levels, different task.

**Citation chain to use:**
- Scellier & Bengio, vector-field / generalized EP (arXiv:1808.04873) — exactness only under conservative force.
- Scurria et al. ICML 2026 (arXiv:2602.03670) — spectral-radius error bound + empirical chance collapse.
- Ernoult et al. 2020, **appendix** — prior empirical observation that antisymmetry degrades EP.

**Required reframing of `S8-L1VF-result.md`:** present L1-VF as **independent confirmation in a new
(causal-sequence, time-varying-input) regime**, not as a discovery.

---

## 2. …and the same paper narrows how strongly we may state the negative

Scurria et al. present two fixes that reach BP parity on asymmetric/feedforward graphs:

- **AsymEP** — corrective force `−2·A_J(x̄⁰,θ)(x − x̄⁰)`, proportional to the antisymmetric part of the
  Jacobian at the free equilibrium. 89.74 ± 0.14% vs BP 90.66 ± 0.25%.
- **Dyadic EP** — doubles the state to `(z, z′)` with `H(z,z′,θ) = −(z−z′)ᵀF((z+z′)/2, θ)`, evolving both
  **forward simultaneously**; Sec. 6 verbatim: *"by solving the forward and adjoint equations simultaneously
  via relaxation, it circumvents a separate backward-in-time pass."* 90.69 ± 0.14%. [vote 3-0]

**Therefore our current sentence is unsafe.**

> ❌ Current: *"a spurious-free cyclic EP relaxation that computes the correct gradient does not exist for this stack"*
>
> ✅ Defensible: *"no spurious-free cyclic relaxation exists for this stack that is **also transport-free and
> local** — the known corrections either reconstruct the transpose or double the state, and **neither has been
> demonstrated for time-varying inputs**."*

Why the narrowed version still holds:
- The authors state **"AsymEP recovers the exact transpose required by BPTT"** — i.e. it buys exactness by
  reconstructing transport, and they concede *"a physical mechanism to obtain the local corrective force at the
  synapse level remains a subject for future work."*
- Dyadic EP *"would require a doubling of the degrees of freedom on the physical hardware."*
- **Both are explicitly scoped to stationary states with FIXED inputs.** Neither touches a causal recurrence.

---

## 3. The open gap — and why we are standing in it

The literature splits into two axes that **no verified source spans simultaneously**:

| Axis | Coverage | Limitation for us |
|---|---|---|
| **Feedforward depth credit** | Bartunov 2018; Xiao 2019 (sign-symmetry); Akrout 2019 (weight mirrors/Kolen-Pollack); Launay 2020 (DFA); Boeshertz 2026 (rank collapse); Scurria 2026 (VF/AsymEP/Dyadic) | All feedforward; Scurria explicitly fixed-input. **Silent on temporal credit.** |
| **Temporal credit, no backward sweep** | Essentially only RHEL / Hamiltonian Echo Learning (Pourcel & Ernoult, arXiv:2506.05259) | **Does not transfer** — see below |

**RHEL does not apply to the WNN.** [vote 3-0]
Sec. 2.1 verbatim: *"The most fundamental requirement of the whole proposed approach for the Hamiltonian models
at use is time-reversal symmetry."* Theorem 3.2 is stated on Hamiltonian Recurrent Units — *"discrete-time,
symplectic integrators of separable Hamiltonian dynamics"* — requiring `H[Φ,θ,u] = H[Σ_z·Φ,θ,u]` with
`Σ_z = diag(I,−I)`, a position/momentum conjugate-pair structure our state lacks. The paper explicitly excludes
the dissipative class: *"RHEL also extends the design of scalable self-learning machines **beyond dissipative
systems** emulating implicit differentiation."* Our `S_t = γS_{t−1} + k_t⊗v_t` with γ<1 is a strict contraction
and the Hopfield warm read is dissipative relaxation. The "γ<1 is invertible hence reversible" escape fails:
the requirement is time-reversal symmetry under momentum flip *plus* symplecticity, not mere invertibility.

**The gift.** RHEL's own Limitations section, verbatim:
> *"there is ultimately a need for a temporal credit assignment algorithm tailored for truly dissipative systems."*

That is a published, citable statement of exactly the gap this project occupies — a stronger position to write
from than a novelty claim. Their endorsed (expensive) workaround — *"embedding a dissipative system within a
larger conservative one"* — would mean re-architecting the phasor state as a conjugate (position, momentum)
pair with γ absorbed into a larger conservative system.

⚠️ RHEL's *positive* capability claims (exact gradients from three forward passes; extension to stacks /
"Hamiltonian SSMs") were **voted down 0-3 and 1-2** in verification. Treat them as unverified pending a direct
read; only the applicability boundary above is solid.

---

## 4. Where L1-FA stands against 2026 state of the art

**Bartunov et al. 2018 still stands — with mandatory qualifications.** [vote 2-1: verification unanimous,
contest was about scope] Abstract verbatim: *"for CIFAR and ImageNet we find that TP and FA variants perform
significantly worse than BP."* Nobody has rebutted those experiments. **But** three peer-reviewed papers
qualify the generalisation: Xiao et al. ICLR 2019 (sign-symmetry near BP on ImageNet — though sign-symmetry
still transports the *sign* of each forward weight); Akrout et al. NeurIPS 2019 (weight mirrors / Kolen-Pollack
*"nearly match backprop"* on ImageNet and *"outperform both feedback alignment and the newer sign-symmetry
method"* — genuinely transport-free, co-authored by a Bartunov co-author, i.e. the same group qualified itself
within ~12 months); Launay et al. NeurIPS 2020 (DFA on transformers/NeRF/graph nets).
**Citing Bartunov alone as "the 2026 verdict" is an outdated framing.** All four are feedforward-depth results,
silent on temporal credit.

**The most on-target negative for us.** [surfaced during verification]
Filipovich et al., *"Scaling Laws Beyond Backpropagation"* (arXiv:2210.14593, NeurIPS 2022 workshop) studied DFA
on **causal decoder-only transformers**: *"DFA fails to offer more efficient scaling than backpropagation: there
is never a regime for which the degradation in loss incurred by using DFA is worth the potential reduction in
compute budget… at variance with previous beliefs in the alternative training methods community."* From LightOn
— the same lab as Launay 2020's "DFA scales." **Cite this directly when positioning L1-FA** rather than letting
a reviewer raise it.

**Strongest 2026 positive — and our best defensive argument.** [vote 2-1 + 3-0]
GrAPE, *"Scaling Direct Feedback Learning with Jacobian Alignment Guarantees"*, **accepted ICLR 2026 poster**
(openreview kasbbmwk3s; confirmed at iclr.cc/virtual/2026/poster/10007774). Forward-mode JVPs estimate rank-1
Jacobians; each layer's feedback matrix is aligned by a local cosine-alignment loss. **But:** *"To curb drift in
very deep models, GrAPE performs infrequent BP anchor steps on a single mini-batch, preserving mostly parallel
updates… closing a large fraction of the gap to BP."*
→ **The published state of the art still reintroduces full weight transport AND a full backward sweep on its
anchor steps.** A scheme keeping only a transposeless backward sweep with exact forward-carried temporal traces
is *strictly more hardware-local than published SOTA.* This is the strongest honest argument for accepting
L1-FA as "neuromorphic enough."
⚠️ Do **not** instead argue from "cosine 0.04 still learns" — the theoretical justification for that
(positive-expected-cosine stochastic approximation) was **voted down 0-3** and is unverified.

**FA depth failure is still the standing premise.** [vote 3-0]
Boeshertz, Pascanu & Clopath, *"Overcoming Rank Collapse in Feedback Alignment"* (arXiv:2606.11123, 2026-06-09;
Imperial / Mila-DeepMind): *"Feedback Alignment (FA) occurs in MLPs and very shallow CNNs but does not scale
well to deeper architectures."* Notable because the paper is *selling a fix* and still states the negative
premise. Fix = Muon optimizer + activation normalisation: CIFAR-100/ResNet-18 — BP 71.9, FA 1.4, FA+BN 37.1,
FA+Muon 25.3, FA+Muon+BN 46.1 (25.8pp gap remains; 15.7pp on Tiny-ImageNet). No transformer/sequence/LM
experiments.
**Hardware-relevant caveat stated by the authors:** Muon *"depends on coordinated information about the spectrum
of an entire weight-update matrix rather than on purely synapse-local quantities"*, and BN needs batch
statistics. **Transport-free ≠ synapse-local.** They also report a negative worth citing: a synapse-local
decorrelation alternative *"seemed to suggest that it would not scale to the larger networks."*
⚠️ The specific *rank-collapse mechanism* framing was voted down 0-3 as an over-reading — treat it as the
paper's hypothesis, not consensus. The depth-failure premise itself is solid.

**Forward gradients are not an escape route.** [vote 2-1]
Qin & Huang, Split-FG (arXiv:2607.16612, 2026-07-18 — three days old). WikiText-103, GPT-2-style causal
transformer, 44 layers, d_model 256, 16.1M params, matched update count: **BP 149.61 ppl · best hybrid
(ρ=0.03) 386.70 (2.58×) · pure forward gradient 2,884.97 (19.3×)** · frozen trunk + exact head 667.64.
Corroborated in direction by arXiv:2306.06968 and arXiv:2511.03110. Split-FG is a *hybrid* (head gradient in
closed form); single-seed; weak BP baseline. **The useful lesson:** forward-mode is worth using to *estimate*
Jacobians (as GrAPE does), not to replace the gradient wholesale.

---

## 5. Our own metrics are exposed to a 2026 methodological critique

Hao, Wan & Zhai (UIUC), *"What Accuracy and Gradient Cosine Miss: Evaluating Feedback Alignment via Scale
Stability, Reference Validity, and Depth Utility"* (arXiv:2606.21126, 2026-06-19). [vote 3-0 on the
insufficiency assertion]

The paper argues accuracy + aggregate-cosine reporting is insufficient and prescribes operational checks:
- **Scale stability:** flag if `ρ = max_ℓ ‖h_{ℓ+1}‖/‖h_ℓ‖ > 50`.
- **Reference validity:** flag if `‖g_L‖ < 10·ε` (ε = cosine denominator floor; 1e-8 fp32 → 1e-7) — a BP
  reference gradient can collapse to the numerical floor in terminal-normalisation residual architectures.
  **The WNN is RMSNorm + additive residual bus, i.e. that architecture class.**
- **Depth utility:** gap < 2pp vs an architecture-matched **frozen-blocks** baseline.
- **Aggregation collapse:** *"the aggregate cosine masks layerwise heterogeneity that concentrates credit at
  one end of the network"* → report per-layer, not aggregate, cosines.
- Verbatim: *"the standard reporting pair gives no signal of failure in any audited case, while our protocol
  identifies all failures with wide calibration margins."*

**What this does and does not imply for us.** It lands on the *scalar* cosines (RIPPLE 0.99999; L1-FA 0.04),
not on everything — our 0.763 figure is already per-block, and our real depth evidence is behavioural (MQAR).
⚠️ Two stronger versions of this critique were **voted down 0-3**: that FA actually failed the depth-utility
control in that audit, and the specific scale-explosion mechanism claim. Do not assert our cosine *is* invalid;
assert only that the control has not been run.
Caveats: single unreviewed preprint, one month old, audits only ResMLP/ViT-Mini/SmallCNN/StudentNet on
CIFAR-10/100 — no LM, no sequence tasks, no EP discussion.

---

## 6. Three experiments this literature suggests, ranked

1. **Frozen-blocks control + per-block cosines.** Re-run L1-FA's MQAR (0.999 @ L=2, 0.996 @ L=3) against an
   architecture-matched baseline whose residual blocks are frozen, and report per-block cosines and
   `ρ = max‖h_{ℓ+1}‖/‖h_ℓ‖`, plus the `‖g_L‖ > 10ε` reference-validity check. Cheap; pre-empts the most
   obvious reviewer attack (arXiv:2606.21126).
2. **AsymEP correction on the two-block causal stack.** Our spurious term `J_s1·K_10·J_s0·f_0` is a
   one-directional — i.e. maximally antisymmetric — coupling, precisely what the corrective force
   `−2·A_J(x̄⁰,θ)(x − x̄⁰)` targets. Test whether it cancels exactly when specialised to the causal read
   coupling `K_10`, and whether it can be computed **without** reconstructing the transpose (the authors say
   AsymEP "recovers the exact transpose," so this may be a negative — which is itself publishable).
3. **Kolen-Pollack learned feedback** in place of fixed random `B_k/B_q/B_v` (Akrout et al. 2019); measure
   whether MQAR holds at depth 3–6.

**Unclaimed contribution on our critical path:** extending **Dyadic EP** (joint forward relaxation of the
adjoint, no backward-in-time sweep) from stationary states with fixed inputs to **time-varying inputs on a
causal recurrence**. The authors explicitly scope themselves to fixed inputs. Neighbouring prior art to check
first: Høier, Staudt & Zach, *Dual propagation* (ICML 2023); Høier, Kalinin, Ernoult & Zach, *Dyadic learning in
recurrent and feedforward models* (NeurIPS 2024 workshop); Pourcel et al., Lagrangian EP ≡ Hamiltonian Echo
Learning (arXiv:2506.06248).

---

## 7. Claims that FAILED verification — do not reuse

Listed so they are not recycled from memory. These were rejected as **over-readings of their sources**, which
is not the same as the underlying papers being wrong.

| Rejected claim | Vote |
|---|---|
| AsymEP restores the *exact* gradient for arbitrary non-conservative systems, directly contradicting our conclusion | 0-3 |
| RHEL computes exact gradients from three forward passes with no backward sweep (and extends to stacks / "Hamiltonian SSMs") | 0-3 / 1-2 |
| Vector-field EP "provably does not compute the exact gradient" stated as that paper's central premise | 0-3 |
| A published theoretical justification exists for accepting cosine ≈ 0.04 (positive-expected-cosine stochastic approximation) | 0-3 |
| FA/DFA failed the depth-utility frozen-blocks control in the arXiv:2606.21126 audit | 0-3 |
| Terminal-LayerNorm scale explosion drives the BP reference to the numerical floor (specific mechanism) | 0-3 |
| FA failure is *rank collapse* rather than cosine misalignment (mechanism framing) | 0-3 |
| Naive forward-gradient training is worse than a frozen random trunk | 0-3 |
| Forward-gradient's BP gap widens monotonically with the forward-mode-trained fraction | 0-3 |
| ImageNet transport-free variants land "essentially at chance" (Bartunov Table 2 framing) | 0-3 |
| Extending EP to time-varying inputs is "an explicitly open and hard problem" (as that paper's statement) | 0-3 |
| Plain DFA asserted by an ICLR 2026 accepted paper to still fail on deep CNNs / transformers | 1-2 |

---

## 8. Standing caveats on this whole review

- **Scope mismatch is pervasive.** Nearly every empirical result above is a vision CNN at CIFAR/Tiny-ImageNet
  scale. The only LM-scale evidence is a 16M-parameter forward-gradient study and Filipovich 2022.
  **Nothing in the verified corpus tests transport-free or backward-pass-free learning on a 162M causal
  linear-attention model.** All transfer to the WNN is extrapolation.
- **Source age.** Only two sources are peer-reviewed at a venue: Scurria et al. (ICML 2026) and GrAPE
  (ICLR 2026). arXiv:2607.16612 is three days old; arXiv:2606.11123 and arXiv:2606.21126 are ~5–6 weeks old and
  unreplicated.
- **Budget exhaustion.** Multiple verifier sessions hit their 200-search limit. "No contradicting evidence
  found" means "none surfaced within budget," not "none exists."
- **UNRESEARCHED — a second pass is required.** Zero verified claims cover: neuromorphic **on-chip training**
  (Loihi 2, BrainScaleS-2, memristive crossbars, Ising/oscillator hardware — including *what the largest model
  ever trained on-chip is*, and whether any attention-like or sequence model has ever been trained on-chip);
  **RTRL** and its modern scalable approximations; **e-prop**; **Forward-Forward / NoProp / predictive coding /
  local blockwise losses**. These sub-questions must be reported as *unresearched*, **not** as answered
  negatively.
- **The single most architecture-specific open question, unaddressed by any source:** since our recurrence
  `S_t = γS_{t−1} + k_t⊗v_t` is **linear in the state**, is exact **RTRL** tractable at 162M scale (is the
  influence Jacobian diagonal/structured rather than dense?) — and would that eliminate the reverse-time
  adjoint `Q_t` entirely, with no approximation? This is the most promising escape route available and nobody
  has looked at it for this architecture class.
