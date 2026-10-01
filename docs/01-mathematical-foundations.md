# 01 — Mathematical Foundations

**Role:** The formal core. Defines the phasor algebra, proves the properties the
architecture relies on, derives capacity/interference bounds, the cleanup energy and
its fixed-point guarantees, the equilibrium-propagation teaching signal, the two-path
codebook gradient, the **exact eligibility trace** (the result that makes local
sequence credit assignment cheap), the RoPE≡phase-binding identity, and Theorem T1.

Every derivation is step-complete and self-contained. Where a result is from the
literature it is cited with equation/theorem specificity; where it is original to this
program it is marked `[WNN]`. Symbols follow Appendix A of the whitepaper; the notation
table is repeated in §0 for self-containment.

> ⚠️ **`β` (read inverse temperature, a model hyperparameter) and `β_n` (EP nudge
> strength, a learning hyperparameter) are different quantities.** Conflating them is
> the most likely implementation bug. See `03` for their roles.

---

## 0. Spaces and notation

| Symbol | Space / type | Meaning |
|---|---|---|
| `D` | ℕ | phasor (key/query) dimension, `10²–10⁴` |
| `d_v` | ℕ | value dimension |
| `d_in` | ℕ | model width feeding the projections |
| `K` | ℕ | codebook / vocabulary size |
| `T` | ℕ | sequence length |
| `x_t` | ℝ^{d_in} | block input at step `t` |
| `W_k, W_q` | ℝ^{D×d_in} | key/query projections (real; phase is `φ` of the projection) |
| `W_v` | ℝ^{d_v×d_in} | value projection |
| `φ(u)` | ℝ^D→ℂ^D | FHRR feature map, `φ(u)_j = e^{i u_j}`, `|φ(u)_j| = 1` |
| `k_t, q_t` | ℂ^D, unit-magnitude | key, query phasors: `k_t = φ(W_k x_t)`, `q_t = φ(W_q x_t)` |
| `v_t` | ℝ^{d_v} | value, `v_t = W_v x_t` |
| `S_t` | ℂ^{D×d_v} | contextual memory (bound superposition) |
| `z_t` | ℂ^D | normalizer |
| `γ` | [0,1) | gate/forget factor (per head) |
| `c_t` | ℝ^{d_v} | query-retrieved content (the linear-attention read) |
| `r` | ℝ^{d_v} | cleanup (Warm-Read) state |
| `V` | ℝ^{d_v×K} | value/vocabulary codebook (columns `V_k`) |
| `M` | ℝ^{d_v×P} | knowledge codebook (`P` learned patterns), per block |
| `p_t` | Δ^{K-1} | next-token distribution |
| `β` | ℝ₊ | read inverse temperature |
| `β_n` | ℝ₊ | EP nudge strength |
| `δ_t` | ℝ^{d_v} | EP content teaching signal |

**Complex convention.** ℂ^D is realized in code as ℝ^{2D} (interleaved real/imag or,
equivalently, `torch.complex64`). Hermitian inner product `⟨a,b⟩ = Σ_j \bar{a}_j b_j`;
`a^†` denotes conjugate-transpose. `Re`, `Im` act componentwise. `lse(x) = log Σ_k e^{x_k}`.
`softmax(x)_k = e^{x_k}/Σ_l e^{x_l}`.

---

## 1. The FHRR phasor algebra

Atomic symbols receive **i.i.d. random unit phasors**: `c_j = e^{iθ_j}`, `θ_j ~ U[−π,π)`.
This is the Fourier Holographic Reduced Representation of Plate (*IEEE TNN* 6(3):623,
1995). Three operations, each defined componentwise.

### 1.1 Binding `⊗` (role–filler / key–value)

**Definition.** `(a ⊗ b)_j := a_j b_j` (componentwise complex product). For unit
phasors this is **phase addition**: `(a⊗b)_j = e^{i(α_j+β_j)}`.

**Properties.**
- *Commutative, associative:* immediate from complex multiplication.
- *Invertible.* The inverse of a unit phasor `b` is `b^{-1}_j = \bar{b}_j = e^{-iβ_j}`,
  since `b_j \bar{b}_j = |b_j|^2 = 1`. **Unbinding:** `(a⊗b)⊗b^{-1} = a`. *Proof:*
  `(a_j b_j)\bar{b}_j = a_j |b_j|^2 = a_j`. ∎
- *Distributes over bundling* (see §1.2): `(a+b)⊗c = a⊗c + b⊗c`, from componentwise
  distributivity of `·` over `+`. **This is the property that lets a superposition of
  bound pairs be queried by one key** (§4).
- *Dissimilarity:* for random `a,b`, `E[sim(a⊗b, a)] = 0` (a bound pair is unlike its
  factors), because `a⊗b` has phases `α_j+β_j` with `β_j` uniform, decorrelating from `α_j`.

### 1.2 Bundling `+` (superposition / sets)

**Definition.** `(a+b)_j := a_j + b_j` (componentwise complex sum). The result is **not
unit-magnitude**: the magnitudes encode superposition weights. This is the reason the
*state* leaves the torus (§2).

**Properties.**
- *Commutative, associative.* Immediate. **This is exactly why bundling alone cannot
  model order** (whitepaper §3.2): `bundle(t_0,…,t_{n-1})` is invariant to permutation
  of its arguments. Order must come from binding/permutation (§3), not bundling.
- *Similarity-preserving:* `sim(a+b, a) > 0` — the bundle resembles each summand,
  with expected similarity `1/√(number of summands)` after normalization (see §4.1).

### 1.3 Permutation `ρ` (order / position)

**Definition (FHRR phase form).** Fix a base frequency vector `ω ∈ ℝ^D`. The `k`-fold
permutation is componentwise phase advance:
```
ρ^k(z)_j := z_j · e^{i k ω_j}.
```
Equivalently `ρ^k(z) = z ⊗ p_k` where `p_k := φ(kω)` is the **positional phasor**.

**Properties.**
- *Unitary, invertible:* `ρ^{-k}(ρ^k(z)) = z`, since `e^{ikω_j}e^{-ikω_j}=1`. Norm-preserving.
- *Order-sensitive:* `ρ^i(t) ≠ ρ^j(t)` for `i≠j` (generic `ω`), so
  `Σ_k ρ^k(t_k)` distinguishes orderings — this is the mechanism that repairs C1's
  order problem, and it is identical to RoPE (§9).
- *Relative position under the read:* `⟨ρ^m(q), ρ^n(k)⟩ = Σ_j \bar{q}_j k_j e^{i(n−m)ω_j}`
  depends only on `n−m`. Position information is **relative**, for free.

### 1.4 Similarity and cleanup

**Similarity.** `sim(a,b) := (1/D) Re⟨a,b⟩ = (1/D) Σ_j Re(\bar a_j b_j)`. For unit
phasors `= (1/D) Σ_j cos(α_j − β_j) ∈ [−1,1]`. A **bank of FHRR hypervectors is a bank
of phase oscillators** and similarity is their mean phase coherence (Frady & Sommer,
*PNAS* 116(36):18050, 2019).

**Cleanup.** Given a noisy `y ∈ ℂ^D` (or a content `c ∈ ℝ^{d_v}`) and a codebook,
return the codeword most similar to `y`. Hard cleanup: `argmax_k sim(y, c_k)`. **Soft,
differentiable, iterative cleanup is the modern-Hopfield read** (§6) — the per-step
primitive of WNN.

---

## 2. The state must carry amplitude (the torus is impossible)

**Claim [carried from whitepaper v1.1 Appendix D].** A running memory constrained to
the unit torus (`|state_j| = 1 ∀j`) *cannot* represent a weighted superposition of two
or more items.

**Argument.** A superposition `s = Σ_i w_i c_i` of unit phasors `c_i` has, at component
`j`, magnitude `|s_j| = |Σ_i w_i e^{iθ_{ij}}|`, which is a random variable in `[0, Σ|w_i|]`
almost never equal to 1. The information "how much of item `i` is present" lives in the
**magnitude** `|s_j|` and the relative phase; forcing `|s_j|=1` erases the weights.
Therefore the codewords stay on the torus (unit phasors) but **the state lives in
ℂ^{D×d_v} with free amplitude**. ∎

**Design consequence (commitment 1 in `02`).** `k_t, q_t` are unit-magnitude; `S_t, z_t`
are not. This is a *solved* constraint, not an open question — it is why WNN is
complex-amplitude, not pure-phase.

---

## 3. Sequences: encoding order in a superposition

An ordered context `(t_0,…,t_{n-1})` is encoded so that order survives bundling:
```
context = Σ_{k} ρ^k(t_k)          (permutation form)
        = Σ_{k} t_k ⊗ p_k          (positional-binding form, p_k = φ(kω))
```
Retrieval of the token at position `k`: `sim( ρ^{-k}(context), · )` cleaned against the
vocabulary. Both forms break commutativity (§1.2, §1.3). In WNN the recurrence itself
supplies order (accumulation is causal), and positional binding is available as an
explicit reinforcement (`02` §positional). This is the classic VSA sequence treatment
(Plate 1995; Frady, Kleyko & Sommer, *Neural Computation* 30(6):1449, 2018; Kanerva,
*Cogn. Comput.* 1:139, 2009).

---

## 4. Capacity and interference (the √(D/M) wall)

The single most important quantitative fact governing a fixed-width holographic memory.

### 4.1 Bundling capacity

**Setup.** Bundle `M` i.i.d. random unit phasors `s = Σ_{i=1}^M c_i ∈ ℂ^D`. Query with
`c_1`:
```
⟨c_1, s⟩ = ⟨c_1,c_1⟩ + Σ_{i=2}^M ⟨c_1,c_i⟩ = D + Σ_{i=2}^M ⟨c_1,c_i⟩.
```
**Signal** `= D` (real, deterministic). **Noise** `N = Σ_{i=2}^M ⟨c_1,c_i⟩`. Each cross
term `⟨c_1,c_i⟩ = Σ_j e^{i(θ_{ij}−θ_{1j})}` is a sum of `D` i.i.d. unit phasors with
uniform phase, hence (CLT) approximately complex-Gaussian with `E=0`,
`Var(⟨c_1,c_i⟩) = D`. The `M−1` cross terms are independent, so `Var(N) = (M−1)D`.

**Amplitude SNR.**
```
SNR = |signal| / √Var(N) = D / √((M−1)D) = √( D / (M−1) )  ≈  √(D/M).      (4.1)
```
This is the theorem of Frady, Kleyko & Sommer (2018). It is verified empirically in
`tests/test_capacity.py` (measured SNR must match (4.1) to within Monte-Carlo error).

### 4.2 Key–value memory and the read

WNN stores bound pairs, not raw items: `S = Σ_{i} k_i v_i^⊤` (`k_i∈ℂ^D` unit, `v_i∈ℝ^{d_v}`).
Reading with query `q = k_1`:
```
q^† S = Σ_i (q^† k_i) v_i^⊤ = (q^† k_1) v_1^⊤ + Σ_{i≠1}(q^† k_i) v_i^⊤ = D·v_1^⊤ + (noise).
```
`q^† k_1 = Σ_j|k_{1,j}|^2 = D`; cross terms `q^† k_i` are complex-Gaussian with variance
`D` as in §4.1. Taking `Re(q^† S)/Re(q^† z)` recovers `v_1` up to interference of
relative amplitude `√(M/D)`. The read is exact in the limit `M ≪ D` and degrades
gracefully — the **blessing of dimensionality**.

### 4.3 Effective context length under gating `[WNN]`

With decay `γ<1`, `S_t = Σ_{i≤t} γ^{t-i} k_i v_i^⊤`. The effective number of items in
superposition (the "participation number") is
```
M_eff = ( Σ_{i≤t} γ^{t-i} )^2 / Σ_{i≤t} γ^{2(t-i)}  →  (1/(1−γ))^2 / (1/(1−γ^2)) = (1+γ)/(1−γ).   (4.2)
```
Substituting into (4.1), retrieval is reliable while
```
D / M_eff ≳ SNR_min^2   ⇔   D ≳ SNR_min^2 · (1+γ)/(1−γ).                        (4.3)
```
**Interpretation.** Effective context grows like `1/(1−γ)`; to double it, halve `1−γ`
or (for equal SNR) double `D`. This is the quantitative form of "can a fixed-width VSA
state do long-range language" (whitepaper OP2). Multi-timescale heads (`02`) place
`γ_h` on a log-spaced grid so different heads cover different `M_eff`. **Conjecture
[WNN, tested in `05` P6]:** the finite-temperature cleanup (§6) denoises the read and
raises the usable `M_eff` beyond the raw-read bound (4.3).

---

## 5. The contextual memory as linear attention (representation)

Reproducing the whitepaper §2.2 bridge, in the exact form the code uses.

**Update (bind + bundle):**
```
S_t = γ S_{t-1} + k_t v_t^⊤.                                                     (5.1)
```
**Read (unbind + normalize):**
```
c_t = Re(q_t^† S_t) / den_t,      den_t = γ den_{t-1} + d_k   ( > 0 always ).      (5.2)
```
`den_t = Σ_{i≤t} γ^{t-i} d_k` is a **positive, deterministic count normalizer** — a
weighted-average denominator. This is the form the code uses (`wavenn/ripple.py::contextual_read`)
and it is **not** cosmetic: FHRR similarities `Re(q^†k)` are **signed** (unlike positive-
feature-map linear attention), so the naïve query-dependent normalizer `Re(q_t^†z_t)`
(with `z_t = γz_{t-1}+k_t`) can cross zero and go negative as the projections grow during
training, exploding the read. The count normalizer is bounded away from 0, keeps `c_t` a
stable weighted average, and — because it does **not** depend on `q` or `k` — keeps the
recurrence linear in `S` while *removing* the denominator terms from the key/query
eligibility traces (§9). Empirically this is what makes training converge (`experiments/
S4a-denominator-stability-fix.md`). `ε`-guarded in code.
This is O(1)-memory, O(T)-time — a *recurrent* form (Katharopoulos et al., ICML 2020).
The VSA reading: outer product `k_t v_t^⊤` = **binding**; running sum = **bundling**;
`q_t^† S_t` = **unbinding + implicit cleanup**; `φ` = **encoder**; `S_t` = **item
memory** (Schlag, Irie & Schmidhuber, ICML 2021). WNN adds, on top of (5.2), the
explicit finite-temperature cleanup of §6 (the "knowledge memory"), and — where enabled
— a **delta-rule** write that first removes the current key's stored value:
```
S_t = γ S_{t-1} + k_t (v_t − \hat v_t)^⊤,   \hat v_t = Re(k_t^† S_{t-1})/den_{t-1},   (5.3)
```
(the same count normalizer `den_{t-1}`) — the Widrow–Hoff correction (DeltaNet, Yang et
al. 2024); it is itself a local error-correcting rule and improves recall at fixed `D`
(`05` P6). In WNN it is a forward/recall option (P6), not part of the fidelity-validated
trained rule, because its exact sequence-credit trace is a research refinement (`03` §4.6).

---

## 6. The Warm Read: modern-Hopfield cleanup energy and its fixed point

### 6.1 Energy and dynamics

Let `V ∈ ℝ^{d_v×K}` (or the block's `M`) be the cleanup codebook, `c ∈ ℝ^{d_v}` the read
content (5.2), `β>0`. Define
```
E_read(r) = ½‖r‖² − ⟨c, r⟩ − (1/β) lse(β Vᵀ r),   r ∈ ℝ^{d_v}.                    (6.1)
```
With `p(r) := softmax(β Vᵀ r) ∈ Δ^{K−1}`,
```
∂E_read/∂r = r − c − V p(r).                                                      (6.2)
```
Gradient flow `dr/dτ = −∂E_read/∂r = −r + c + V p(r)` converges (§6.2) to a fixed point
```
r* = c + V p(r*),                                                                 (6.3)
```
the modern-Hopfield retrieval (Ramsauer et al., ICLR 2021) with **external content bias
`c`**. The next-token / pattern distribution is `p_t = softmax(β Vᵀ r*)` (eq. 4.2 of
the whitepaper). **`c` is the linear-attention read; `E_read` cleans it up against the
codebook at temperature `β`.** In the `β→∞` corner (§10, Theorem T1), the read reduces to a
single hard cleanup whose argmax is preserved, so `p_t→softmax(βVᵀc)` — the ordinary
linear-attention logits (stated there as a corner-case correspondence, not a one-step identity
`r*=c`).

### 6.2 Existence and uniqueness of the fixed point `[WNN]` — resolves whitepaper OP1

**Hessian.**
```
H(r) = ∂²E_read/∂r² = I − β V ( diag(p) − p pᵀ ) Vᵀ.                              (6.4)
```
`Σ(p) := diag(p) − p pᵀ` is the covariance of the categorical `p`; it is PSD with
`λ_max(Σ(p)) ≤ 1/2` (the maximum, `1/2`, is attained by a two-point distribution at
`½,½`; general bound: for any `p`, `λ_max ≤ ½`). Hence
```
λ_max( β V Σ(p) Vᵀ ) ≤ β σ_max(V)² · ½,
H(r) ⪰ ( 1 − ½ β σ_max(V)² ) I.
```
**Theorem (safe regime).** *If*
```
β < 2 / σ_max(V)²,                                                               (6.5)
```
*then `E_read` is strictly convex on all of ℝ^{d_v}, so the fixed point (6.3) exists and
is unique, and the gradient flow (6.2) converges to it from any initialization.* ∎

The quantity `1 − ½βσ_max(V)²` is the **strong-convexity margin of the continuous flow**
(the lower bound on `λ_min(H)`), not the per-iteration contraction factor. A discrete Euler
step of size `η` (the settle update `r ← r + η(−∂E_read/∂r)`, `02` §5.1) contracts the error
by `1 − η·(1 − ½βσ_max(V)²)` per step; the continuous margin is recovered as `η→0`.

**Consequences.**
- The two-phase EP difference (§7) is well-defined **iff the read is single-valued**;
  (6.5) is the checkable condition guaranteeing it. Training keeps `β` under (6.5) by
  spectral-normalizing `V` (or clamping `β`); `tests/test_fixed_point.py` verifies
  convergence and uniqueness across the `β`-grid and flags violations of (6.5).
- (6.5) is *sufficient*, not necessary: above it the energy may still be locally convex
  around the retrieved basin when the content clamp `c` is strong. The safe-regime
  guarantee is what the design relies on; the stronger-clamp regime is characterized
  empirically (whitepaper OP1(b,c)).
- The `β→∞` corner deliberately violates (6.5); there the read collapses to a single
  hard-cleanup step (Theorem T1) and uniqueness is replaced by the argmax convention.

### 6.3 Adaptive settling `[WNN]`

The number of fixed-point iterations `n(r)` is not fixed: iterate until
`‖r^{(s+1)}−r^{(s)}‖ < τ_tol`, with a hard cap `n_max`. Easy tokens (peaked `p`) converge
in 1–2 steps; ambiguous tokens take more. This is **native adaptive compute** (mutation M4)
and is measured (`05`, iterations vs token surprise). The settle's stop rule is exactly the
`‖Δr‖<τ_tol` (+ `n_max`) criterion (`wavenn/cleanup.py::settle`); the read-entropy plateau
`H(p) = −Σ p_k log p_k` is used only as an **offline diagnostic** of how many steps hard
tokens take (`wavenn/rlm.py::adaptive_settle_stats`), not as the settle's termination test.

---

## 7. Equilibrium propagation on the read: the content teaching signal

### 7.1 The EP identity

Energy-based learning (Scellier & Bengio, *Front. Comput. Neurosci.* 11:24, 2017). Total
energy with cost `C(r) = −log softmax(βVᵀr)[y]` (`y` the target token):
```
F(r) = E_read(r) + β_n C(r).
```
Relax to the **free** equilibrium `r⁻` (`β_n=0`) and a **nudged** equilibrium `r^{β_n}`.
The EP theorem gives, for any parameter/quantity `θ` entering `E_read`,
```
dC/dθ (at r⁻) = lim_{β_n→0} (1/β_n)[ ∂E/∂θ|_{r^{β_n}} − ∂E/∂θ|_{r⁻} ].            (7.1)
```
No backward pass, no weight transport, no stored activations — two relaxations and a
local difference.

### 7.2 The content signal `δ_t` `[WNN]` — derives whitepaper (4.5)

The content `c` enters `E_read` *only* through the linear term `−⟨c,r⟩`, so `∂E/∂c = −r`.
Applying (7.1) with `θ = c`:
```
dC/dc = lim_{β_n→0} (1/β_n)( −r^{β_n} + r⁻ ) = −δ,    δ := (r^{β_n} − r⁻)/β_n.     (7.2)
```
**`δ` is (minus) the loss gradient on the content, computed by two relaxations.** With
the **symmetric nudge** (Laborieux et al. 2021) — relax at `+β_n` and `−β_n` — the O(β_n)
bias cancels:
```
δ_t = ( r_t^{+} − r_t^{−} ) / (2 β_n).                                            (7.3)
```
*Numerically confirmed at the per-step level (whitepaper App. D.2):* `cos(δ, −dC/dc)→+1.000`
and `‖δ‖/‖grad‖→1.000` as `β_n→0`. δ estimates `−∂C/∂c` (eq 7.2), so it is `+1`-aligned with
`−∂C/∂c` and `−1`-aligned with `+∂C/∂c` — the sign is fixed by the derivation, not a free
convention. This is `05` P1 in miniature, and `tests/test_gradient_fidelity.py` re-verifies it
as a regression (it asserts `cos(∂C/∂c, autograd ∂C/∂c) → +1`, i.e. `cos(δ, −∂C/∂c) → +1`).

---

## 8. The codebook gradient needs TWO paths `[WNN]` — derives whitepaper B.3/D.3

The single most important implementation subtlety, found by running code, not
anticipated. `V` enters the per-step loss `C = −log softmax(βVᵀr*)[y]` by **two** paths.

**(a) Read path.** `V` is the stored-pattern matrix in `E_read`, so it shapes `r*`.
`∂E_read/∂V_k = −p_k r`, so by EP (7.1):
```
ΔV_k^{read} ∝ (1/β_n)[ p_k^{+} r^{+} − p_k^{−} r^{−} ]   (symmetric: /(2β_n)).      (8.1)
```
**(b) Readout path.** `V` *also* appears explicitly in the final softmax
`p = softmax(βVᵀr*)`, a direct dependence not mediated by relaxation:
```
∂C/∂V_k = β r* ( p_k − 1{k=y} )  ⇒  ΔV_k^{readout} ∝ β r* ( 1{k=y} − p_k ).        (8.2)
```
**Full update:**
```
ΔV_k = ΔV_k^{read} + ΔV_k^{readout}.                                              (8.3)
```
*Numerically confirmed (whitepaper D.3):* read path alone has `cos ≈ 0.74` with the true
finite-difference gradient and **trains in the wrong direction** (loss rises 5.05→27.6);
read + readout gives `cos = 1.0000` and drives loss `5.05→0.004` in 8 steps. **Any
implementation must include the readout term.** This is locked by
`tests/test_two_path_codebook.py` as a permanent regression. It is the sequence-model
analogue of the standard fact that an energy-based classifier's output weights receive
both a relaxation-mediated and a direct readout gradient.

---

## 9. Exact eligibility traces through the linear recurrence `[WNN]` — the enabling theorem

This is the result that turns the per-step content error `δ_t` (§7) into weight updates
for `W_k, W_v, W_q` **online, exactly, and cheaply** — the mechanism that makes local
*sequence* credit assignment feasible. It is exact *because the recurrence is linear in
`S`* (contrast e-prop's approximate traces on nonlinear RNNs; Bellec et al. 2020).

> **Normalizer note (matches the code).** The derivations below are written with the
> generic normalizer `den_t`. In the implemented model `den_t = γ den_{t-1} + d_k` is the
> **deterministic count normalizer** (§5), so it does **not** depend on `q_t` or `k_i`.
> This *simplifies* the query and key pathways versus a query-dependent `den`: the
> `∂den/∂q` and `∂den/∂k` terms vanish, leaving only the numerator terms
> (`∂c_{t,m}/∂u_j^q = +Im(q̄_j S_{j,m})/den_t`, `∂c_{t,m}/∂u_j^k = −γ^{t-i}Im(q̄_j k_{i,j})v_{i,m}/den_t`),
> and it adds a `dotden_t = den_{t-1} + γ dotden_{t-1}` term to the decay (`γ`) trace.
> All pathways are validated equal to autograd in `tests/test_traces.py`
> (`wavenn/ripple.py::projection_grads` is the source of truth).

### 9.1 The value pathway (full derivation)

From (5.1), `S_t = Σ_{i≤t} γ^{t-i} k_i v_i^⊤`, so the read (5.2) is
```
c_{t,m} = Σ_{i≤t} a_{t,i} v_{i,m},     a_{t,i} := γ^{t-i} Re(q_t^† k_i) / den_t,    den_t := Re(q_t^† z_t)+ε.   (9.1)
```
The block's contribution to the loss at step `t` is `δ_t^⊤ c_t` (with `δ_t` the content
error from §7). With `v_i = W_v x_i`, the total gradient w.r.t. `W_v` is
```
∂L/∂W_v = Σ_t δ_t ⊗ ( Σ_{i≤t} a_{t,i} x_i ).                                       (9.2)
```
The inner sum couples the *current* query `q_t` with *past* keys `k_i`, so it is not a
naïve running sum. Factor through the phasor dimension `j`:
```
Σ_{i≤t} γ^{t-i} Re(q_t^† k_i) x_i
  = Re( Σ_j \bar q_{t,j} Σ_{i≤t} γ^{t-i} k_{i,j} x_i )
  = Re( q_t^† G_t ),   where   G_t := Σ_{i≤t} γ^{t-i} k_i x_i^⊤ ∈ ℂ^{D×d_in}.       (9.3)
```
`G_t` is a **holographic eligibility trace** obeying the same linear recurrence as `S_t`:
```
G_t = γ G_{t-1} + k_t x_t^⊤.                                                       (9.4)
```
Therefore the online, exact value update is
```
ΔW_v ⟵ ΔW_v + δ_t ⊗ ( Re(q_t^† G_t) / den_t ),                                     (9.5)
```
where `(q_t^† G_t)_n = Σ_j \bar q_{t,j} G_{t,j,n} ∈ ℂ^{d_in}`.

**Theorem [WNN — exactness & cost].** *Equations (9.4)–(9.5) compute `∂L/∂W_v` exactly
(equal to truncated-BPTT through (5.1)–(5.2)) using `O(D·d_in)` trace memory,
independent of `T`, and `O(D·d_in + d_v·d_in)` compute per step — the same order as the
forward pass.* *Proof:* (9.5) summed over `t` equals (9.2) by construction of `G_t`
(9.3); `G_t` is maintained by the constant-size recurrence (9.4); no activation history
is stored. ∎

`tests/test_traces.py` verifies (9.5) against autograd through (5.1)–(5.2) to
floating-point tolerance — the exactness claim is a hard test, not an assertion.

### 9.2 The query and key pathways (phase-domain terms)

The query enters only at the current step: `∂c_t/∂q_t` through numerator and
denominator of (5.2). Here `q_t = φ(W_q x_t)` with `∂φ(u)/∂u_j = i e^{iu_j}`, but the read
numerator holds the **conjugate** `\bar q_{t,j}` (an unbind), so the chain factor is
`∂\bar q_{t,j}/∂(W_q x_t)_j = −i \bar q_{t,j}` — the phase derivative carries `−i`, not `+i`:
```
∂c_{t,m}/∂(W_q x_t)_j = Re( −i · \bar q_{t,j} (S_t)_{j,m} )/den_t − c_{t,m}·Re( −i·\bar q_{t,j}(z_t)_j )/den_t
                     = +Im( \bar q_{t,j}(S_t)_{j,m} )/den_t − c_{t,m} Im( \bar q_{t,j}(z_t)_j )/den_t.   (9.6)
```
(Using `Re(−i w) = Im(w)`.) This is a **current-step** term (no trace needed):
`ΔW_q ⟵ ΔW_q + [Σ_m δ_{t,m}·(9.6)] x_t^⊤`. The sign of (9.6) — `+Im` on the numerator term,
`−c·Im` on the denominator term — was verified against autograd
(`tests/test_traces.py::test_query_trace_matches_autograd`); it matches the code
`wavenn/ripple.py::projection_grads` (query path).

The key enters *both* `S_t` (numerator) and `z_t` (denominator) and persists across time,
so it uses **two traces** analogous to `G_t`: one carrying `k_i x_i^⊤` weighted into the
numerator gradient, one into the denominator. With `k_t = φ(W_k x_t)` the derivative
again carries the `i e^{iu}` phase factor. Full expressions and the trace recurrences are
in `03-learning-rule-spec.md` §key-path; they are all `O(D·d_in)` decayed sums (exact,
linear recurrence). The denominator dependence may optionally be treated as a
stop-gradient (a documented approximation, ablated in `05`) to halve the key-path cost.

### 9.3 Why this is the crux

`δ_t` (§7) is a *local, two-phase* error on each step's content; (9.4)–(9.5) turns it
into weight updates by an *exact eligibility trace through a linear recurrence*. Nowhere
is there backpropagation through time, weight transport, or stored activations beyond
`(S_t, z_t, G_t)` and their key-path siblings. The rule is online and maps to
oscillator/phasor hardware. **Exactness is bought by the linearity of (5.1)** — the
architectural constraint that every design change must preserve (`00` §1, C1↔C2 synergy).

---

## 10. Theorem T1 — linear attention is the `β→∞` limit `[WNN, from whitepaper B.4]`

**Claim.** *As `β→∞` with a single read step and the symmetric nudge in its `β_n→0`
limit, the Warm Read's per-step computation reduces to standard linear attention, and
the §9 eligibility-trace update reduces to truncated-BPTT training of a linear-attention
layer.*

**Proof (limiting/corner-case identification, not an exact single-step identity).**
*Read collapse.* As `β→∞`, `softmax(βVᵀr) → e_{argmax(Vᵀr)}`. One Euler step of the flow
(6.2) from `r⁰=c` gives `r¹ = c + η V·softmax(βVᵀc) → c + η V_{argmax(Vᵀc)}`, which is **not**
`c`: the read is the exact identity `c↦c` only in the zero-settle-step (or `η→0`) convention.
What survives is the **argmax**: for near-orthonormal unit codewords the added term
`η V_{argmax(Vᵀc)}` lies along the winning column and preserves `argmax(Vᵀc)`, so the block's
logits satisfy `argmax(βVᵀr*) = argmax(βVᵀc)` and, as `β→∞`, `p_t → softmax(βVᵀc_t)` — the
ordinary linear-attention readout, with `c_t = Re(q_t^†S_t)/Re(q_t^†z_t)` exactly the
linear-attention read (5.2). This is a corner-case identification of the two distributions,
not a one-step equality of the states. ∎(read, corner case)
*Gradient collapse.* By (7.2), `δ_t = (r^{+}−r^{−})/(2β_n) → −dC/dc_t` as `β_n→0`. In the
same corner (`β→∞` with the `η→0`/argmax-preserving read reducing to `c↦c`), `dC/dc_t` is the
plain cross-entropy gradient of the linear-attention logits; feeding it through the exact
linear-recurrence trace (9.4)–(9.5) reconstructs the truncated-BPTT gradient of a
linear-attention layer, because the trace *is* the linearized backward recurrence of a linear
state update. ∎(gradient, corner case)

**Honest caveat on rigor.** At `β→∞` the cleanup energy (6.1) is **non-convex** (the
safe-regime bound (6.5) is deliberately violated, §6.2), so EP's differentiability premise
(§7) is strained: the fixed point is no longer guaranteed unique and the two-phase difference
is no longer guaranteed single-valued — uniqueness is replaced by the argmax convention.
T1 is therefore a **corner-case correspondence** identifying the family's zero-temperature
edge with linear attention, *not* a theorem carrying the same rigor as the finite-`β` results
(§6.2, §7, §9). It is validated only as the corner is **approached** (`05` P1: cosine→1 at
large-but-finite `β`), not exactly at `β=∞`.

**Consequence.** Standard linear-attention training is the zero-temperature,
single-Newton-step corner of the RIPPLE family; the finite-`β`, multi-step, two-phase
rule is the local generalization. This (a) sanity-checks the construction, (b) predicts
finite-`β` should *differ* from standard training via soft negatives over the whole
vocabulary, a denoised read, and updates on correct predictions too, and (c) gives a
continuous knob (`β`, read-steps, trace vs current-step-only) interpolating "cheap and
local" ↔ "exact and global." Verified as `05` P1 (cosine→1 as the corner is approached).

---

## 11. RoPE is FHRR positional binding `[WNN]`

**Claim.** Rotary position embedding (Su et al., RoFormer 2021) is exactly the
permutation operator `ρ` of §1.3 applied in the real projection.

**Derivation.** RoPE pairs real dimensions `(x_{2j}, x_{2j+1})` and rotates pair `j` at
position `t` by angle `θ_j t`:
```
[x_{2j}; x_{2j+1}] ↦ R(θ_j t) [x_{2j}; x_{2j+1}],   R(α)=[[cosα,−sinα],[sinα,cosα]].
```
Identify the pair with a complex number `z_j := x_{2j} + i x_{2j+1}`. Rotation by `R(α)`
is complex multiplication by `e^{iα}`:
```
z_j ↦ z_j e^{iθ_j t} = (ρ^t z)_j,      with ω_j := θ_j  ⇒  ρ^t z = z ⊗ φ(tθ).       (11.1)
```
So RoPE **is** binding the content with the positional phasor `p_t = φ(tθ)` — the FHRR
permutation `ρ^t`. The attention score's dependence on relative position `m−n`
(§1.3) is precisely RoPE's celebrated relative-position property. **WNN treats phase
binding as the primitive and RoPE as its real-domain shadow**; the model is complex, so
position, order, and content live in one algebra rather than being stapled together.

**Implementation note (the frequency schedule).** `wavenn/algebra.py::default_frequencies`
assigns `ω_j = base^{-2⌊j/2⌋/D}` over the `D` phasor components — i.e. it **duplicates each
base frequency across the two adjacent components** (`⌊j/2⌋` pairs them), reproducing exactly
the RoPE pair-rotation schedule (Su et al. 2021) rather than assigning a distinct `ω` per
complex component. This is the RoPE-schedule *choice*, and it is mathematically harmless: the
order-sensitivity (§1.3) and the relative-position identity `⟨ρ^m q, ρ^n k⟩` depending only on
`n−m` hold for **any** `ω` (generic or duplicated). The duplication is a compatibility
convention with real-domain RoPE, not a correctness requirement.

---

## 12. Summary of guarantees

| Object | Guarantee | Where | Test |
|---|---|---|---|
| Binding | invertible, distributes over bundling | §1.1 | `test_algebra` |
| Bundling | order-invariant (⇒ needs §3 for order) | §1.2 | `test_algebra` |
| Permutation / RoPE | unitary, order-sensitive, relative position | §1.3, §11 | `test_algebra` |
| State | must carry amplitude (torus impossible) | §2 | design invariant |
| Capacity | read SNR `= √(D/(M−1))`; `M_eff=(1+γ)/(1−γ)` | §4 | `test_capacity` |
| Read fixed point | unique & convergent if `β < 2/σ_max(V)²` | §6.2 | `test_fixed_point` |
| EP content signal | `cos(δ,−∇C)→+1` (⇔ `cos(δ,+∇C)→−1`), ratio→1 as `β_n→0` | §7 | `test_gradient_fidelity` |
| Codebook update | exact iff read+readout paths included | §8 | `test_two_path_codebook` |
| Eligibility trace | exact = BPTT, `O(D·d_in)` memory, `O(T)` time | §9 | `test_traces` |
| `β→∞` limit | = linear attention + truncated BPTT (corner-case correspondence) | §10 | analysis only (§10; approached in `05` P1, not exercised at `β→∞`) |
