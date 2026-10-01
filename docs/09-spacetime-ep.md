# 09 — Space-time Vector-Field EP (L1-VF): design note & derivation

**Status:** design (Phase A of the L1-VF plan). Implements the transport-free, **backward-pass-free**
depth-credit rule promised in `docs/03` §7 (L1-VF) and resolves the open question `docs/06` §5 /
whitepaper OP3(b): *does the causal (triangular) recurrence admit a valid EP nudge, or does causality
force an anticausal component that reduces to a backward pass?*

Predecessors on the depth-credit ladder (`docs/03` §7): **L1-exact** (`ripple_grads_l1`, transpose,
exact) and **L1-FA** (`ripple_grads_l1fa`, transport-free *spatial* routing; Stage 1,
`experiments/S8-L1FA-result.md`, recovers MQAR to ceiling at L=2/3). Both keep an explicit inter-block
**backward sweep** and the reverse-time **temporal adjoint** `Q_t`. L1-VF removes the backward sweep by
making the credit emerge from a **joint relaxation**.

---

## 1. The EP identity we instantiate at the stack level

For a settle with content `c`, free energy `F(r)=E(r)+β_n C(r)`, free eq `r⁻`, nudged eq `r^{±β_n}`
(`docs/01` §7):

    dC/dθ|_{r⁻} = lim_{β_n→0} (1/β_n)[ ∂E/∂θ|_{r^{β_n}} − ∂E/∂θ|_{r⁻} ].            (EP identity)

Since `c` enters `E` only via `−⟨c,r⟩` (so `∂E/∂c=−r`), with the symmetric nudge:

    δ := (r⁺ − r⁻)/(2β_n),        ∂C/∂c = −δ,     bias O(β_n²).                    (content signal)

The δ→parameter-gradient pipeline (`projection_grads` exact traces + two-path codebook +
`_rmsnorm_vjp`/`_unit_cols_vjp` + gate/scale tail) is reused **verbatim** from L1. L1-VF changes only
how each block's **downstream cost / downward force** is produced: from a relaxation, not a sweep.

## 2. The downstream cotangent and two corrections that are load-bearing

Block ℓ affects the loss only through `r_ℓ`, which feeds its own readout and `bus_{ℓ+1}=bus_ℓ+r_ℓ`.
The settle receives the augmented cost `C_ℓ^{tot}=CE_ℓ+⟨g_top^ℓ,r⟩` with the **downstream cotangent**
`g_top^ℓ=∂C/∂bus_{ℓ+1}` injected as the force `−g_top^ℓ`. Down the additive bus:

    ∂C/∂bus_L = 0;   ∂C/∂bus_ℓ = ∂C/∂bus_{ℓ+1} + rmsnorm_vjp( CR_adj(−δ_ℓ) ⊙ g^ℓ ).

- **F1 — normalize by 1/β_n (else the rule degenerates to L0).** `cleanup.settle` injects
  `dr += β_n·(force)`, so a nudged neighbour's raw deviation `r_{ℓ+1}^±−r_{ℓ+1}^free` is **O(β_n)**. A
  force built from that raw deviation makes block ℓ's inter-block credit O(β_n) → 0 as β_n→0 (i.e. L0,
  the opposite of the EP identity). The downward force must be built from the **normalized content
  signal** `δ_{ℓ+1}=(r_{ℓ+1}^+−r_{ℓ+1}^-)/2β_n` (already `/2β_n`), rebuilt between settle passes.
- **F3 — the correct operator is the content cotangent through `CR_adj`, not a static r→r matrix.**
  `CR_adj` (the ContextRead input pullback, `contextual_input_grad{,_fa}`) couples `c_{ℓ,t}` to
  block-(ℓ+1) deviations at **all t′≥t** (weights `γ^{t′−t}Re(q̄k)`) — a temporal structure a
  position-wise matrix cannot carry. Transport-free spatial part: `B` in place of `Wᵀ` at the final
  projection (Stage-1 `Bk/Bq/Bv`, audited by `tests/test_transport_free.py`).

## 3. Space-time relaxation and the causal obstruction (OP3(b))

State = the unrolled block reads `R={r_{ℓ,t}}`. The forward pass is the fixed point of the causal
forward map `Φ` (the **free equilibrium**; deviation feedback is 0 there). The **nudged** phase clamps
the top-final readout by `±β_n` and relaxes the augmented system by an outer iteration that (a) re-runs
`Φ` (depth up, time forward) and (b) applies the downward feedback (F1/F3). `δ_ℓ=(r_ℓ^+−r_ℓ^-)/2β_n`.

**Where the feedback enters (the β_n trap).** `dr=(−r+c+Mσ_β(M^⊤r)) + β_n(readout−g_top)`. The
inter-block feedback is part of the *dynamics*, not the output nudge — it must be β_n-free at leading
order. Two formulations satisfy this (both must pass G1/G2, §5):
- **Formulation A (recommended):** keep the `g_top` slot but feed the *normalized* `δ_{ℓ+1}` (the
  `/2β_n` cancels the slot's `β_n`); an **outer loop re-runs `CR`** so `c_{ℓ+1}` responds to `r_ℓ`
  (the space-time cycle) and iterates the top→bottom δ→g_top pass to a joint fixed point. = "L1-FA's
  per-block EP + F3 routing, wrapped in a genuine relaxation." G2 (iterate≠sweep) distinguishes it.
- **Formulation B (purity study):** nudge only the top; add the feedback to the **base** `dr` (not the
  β_n slot) via `rmsnorm_vjp∘CR_adj∘φ(J_s^⊤)`; deviations are O(β_n) natively. Extra `J_s^⊤` bias.

**The obstruction (theorem to test, not assume).** Anticausal (query→source) credit — what a 2-layer
induction head needs — requires the relaxation to propagate `t′→t` backward in time. `Φ` is strictly
causal; a **symmetric energy** coupling `t↔t′` is impossible (it would make `c_t` depend on the future
— this IS OP3(b)); the minimal backward-in-time feedback is exactly the reverse-time adjoint `Q_t`.
Hence the predictions: **(T1)** with the symmetric nudge, VF-EP's gradient equals **L1-FA's to
O(β_n²)**; **(T2)** a forward-only relaxation cannot carry anticausal credit, so the working variant
uses `Q_t` and thereby **reduces to L1-FA**. Stage 2's contribution is the **mechanism** (no explicit
backward pass) + this **measured resolution**, not a capability gain.

**Two variants = the experiment.** *VF-ST-adjoint*: temporal part of `CR_adj` uses `Q_t` (removes the
inter-block sweep; keeps the temporal adjoint) — the tractable, guaranteed-correct rule. *VF-ST-forward*:
`Q_t` disabled (current-step S-A0 pullback only), temporal credit attempted forward-only via the `CR`
re-runs — the OP3(b) probe (= Stage-1's deferred drop-the-adjoint control). Their MQAR comparison
(final-only) resolves OP3(b): forward-only recovers ⇒ obstruction false; only the adjoint recovers ⇒
obstruction confirmed.

## 4. Convergence

Linearizing one depth cycle (β_n cancels after F1), the loop gain is
`G_ℓ=−J_s^ℓ·λ_fb·[rmsnorm_vjp∘CR_adj]·J_s^{ℓ+1}`. Contraction needs
`‖J_s^ℓ‖·λ_fb‖CR_adj‖·‖J_norm‖·‖J_s^{ℓ+1}‖<1`. Divergence drivers: `‖B‖≈2` (Bai–Yin, `1/√d` Gaussian),
`‖CR_adj‖∼1/(1−γ)` (long-γ heads), `‖J_s‖` near the safe-β boundary. Safeguards: bounded feedback gain
`λ_fb<1`, under-relaxation `η_o`, outer cap `n_outer`; telemetry gate `ρ(G)<1` by power iteration. The
uncorrected scheme has `ρ(G)=O(β_n)` — "converges" trivially but to L0; a healthy `ρ(G)∈(0,1)` needing
several sweeps is the signature of a genuine relaxation.

## 5. Validation gates (see the plan / `tests/test_spacetime_ep.py`)

- **G1** exact coupling (`B=W`, `λ_fb=1`, VF-ST-adjoint) ⇒ reproduces L1-exact/BPTT at **cos=1.0** on
  every parameter. (2A-control fails G1 by construction.)
- **G2** iterate-to-convergence ≠ one-sweep (certifies a relaxation, not a covert sweep).
- **G3** `ρ(G)<1`; stable training. **G4** `cos(Δ_VF,Δ_L1FA)→1` as β_n→0 (T1). **G5** final-only MQAR:
  VF-ST-forward vs -adjoint = the OP3(b) datum (T2). **G6** cost vs L1-FA (OP3(c)).

Every gradient/capability claim is measured against the exact **L1-exact** reference. Implementation is
gated by **independent verification** (code-vs-spec, the C2/transport-free/genuine-relaxation
invariants, an independent math re-derivation vs autograd BPTT, and results reproduction) before any
commit. Scope: L=2, T≈8–16 (real-T is intractable — the honest OP3(c) statement).

## 6. Empirical resolution — OP3(b) obstruction CONFIRMED (independently verified)

`ripple_grads_l1vf` implements the joint relaxation (`vf_rerun` toggles the space-time cycle). Result
(gradient level, 2-layer float64, `tests/test_spacetime_ep.py`), independently reproduced and
adjudicated by three independent verifiers (C2 / transport-free / genuine-relaxation invariants: PASS;
independent math re-derivation of δ_ℓ + F1 + exact-coupling correctness: PASS):

- **Plumbing correct (G1).** No cycle + exact coupling reproduces L1-exact to fp and BPTT to the
  shared O(β_n²) EP bias. F1 (÷β_n) is load-bearing: without it the rule collapses to L0.
- **The cyclic relaxation is genuinely obstructed.** The clean, *provably convergent* demonstration
  uses `feedback_gain=0` (triangular loop ⇒ no positive-feedback divergence): the space-time cycle
  still corrupts the **top block** (`M.1` cos≈0.76, converged and O(1) in β_n) while the acyclic
  version is exactly correct (cos=1.0). Mechanism, proven by a falsifiable control: re-running the
  **causal** contextual_read in the nudged phase feeds block-0's local nudge *up* into `c_1` — a
  spurious O(1) forward-coupling absent from `∂C/∂c`; removing the lower local nudge (**final-only
  supervision**) makes the top block recover to cos=1.0 exactly. No spurious-free stable cyclic
  relaxation exists across swept `γ, λ_fb, η_o, n_outer`.
- **Caveat (verification-caught).** At exact coupling `feedback_gain=1` the cyclic loop additionally
  **diverges** (positive feedback), so magnitudes there are transient, not a clean measurement — the
  decisive result is the stable `feedback_gain=0` demonstration above.

**Conclusion.** For a *causal feed-forward* stack, a spurious-free cyclic EP relaxation that computes
the correct gradient **does not exist**; the correct transport-free credit is exactly the **acyclic
reverse-time structure (= L1-FA)**. This resolves OP3(b) / `docs/06` §5: causality forces an
anticausal component (the reverse-time adjoint `Q_t`), i.e. the genuine "no-backward-pass" relaxation
reduces to L1-FA. The transport-free depth-credit *capability* is delivered by Stage 1 (L1-FA); the
space-time relaxation is obstructed, exactly as predicted (T1/T2).
