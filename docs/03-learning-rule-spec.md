# 03 — RIPPLE: the Backpropagation-Free Learning Rule

**RIPPLE** = **R**esonance-**I**nduced **P**lasticity with **P**hase-**L**ocal
**E**ligibility. This document specifies the complete learning rule: the equilibrium-
propagation content signal, the two-path codebook update, the exact eligibility traces
for **every** parameter tensor, three-factor neuromodulation, local Adam-style
normalization, the L0/L1/L2 depth-credit ladder, and the full hyperparameter table.

It satisfies constitutional requirement **C2** (`00` §1): no global reverse-mode autodiff
over model parameters. Autograd appears only in the BPTT-twin baselines and the
finite-difference tests (`05`). All math and symbols follow `01`.

> ⚠️ **`β` (read temperature, `02`/`01` §6) ≠ `β_n` (nudge strength, here).** The rule
> uses both; keep them distinct in code (`WNNConfig.beta` vs `RippleConfig.beta_n`).

---

## 1. Three plasticity timescales (mutation M1)

RIPPLE writes memory at three coupled rates; "learning" and "inference" are the same
operation at different decay constants (`00` §2).

| Timescale | Medium | Written by | Decays | Role |
|---|---|---|---|---|
| **Fast** | state `S_t, z_t` | forward recurrence (`02` §4) | `γ_h` per step | working memory / in-context |
| **Medium** | episodic buffer (`06`) | high-surprise bindings | per episode | test-time / continual learning |
| **Slow** | weights `W_*, M_*, V, E, γ` | RIPPLE updates (this doc) | never (consolidated) | knowledge |

v0 (S3–S5) implements **fast + slow**; the medium episodic path is `06`/S7. The *same*
RIPPLE update equations apply to slow weights during training and to consolidation
replay — this is the unification, not two separate mechanisms.

---

## 2. The per-step objective and the EP two-phase read

At each step `t`, block `ℓ` produces content `c_t` (`02` §4), settles the cleanup read
to `r*_t` (`02` §5), and predicts `p_t = softmax(β Vᵀ r*_t)`. Local cost:
```
C_t = − log p_t[ y_t ],     y_t = next token.                                     (2.1)
```

**Two-phase relaxation (symmetric nudge, `01` §7).** With `c_t` held fixed:
```
free:      r_t^0  = settle E_read(r; c_t)                       (β_n = 0)
nudged±:   r_t^± = settle E_read(r; c_t) ± β_n · [ −∂C_t/∂r ]    (± symmetric)
```
The nudge force is `−∂C_t/∂r = β ( V_{:,y_t} − V p_t )` (cross-entropy through the
readout softmax). The two nudged settles reuse the free settle as warm start.

**Content teaching signal (`01` eq 7.3):**
```
δ_t = ( r_t^+ − r_t^- ) / (2 β_n)  ∈ ℝ^{d_v}.                                      (2.2)
```
`δ_t` is the local, backprop-free stand-in for `−∂C_t/∂c_t`. `05` P1 requires
`cos(δ_t, −∂C_t/∂c_t) → 1` as `β_n→0` (per-step already validated, whitepaper D.2).

---

## 3. Codebook update — both paths (mandatory, `01` §8)

> **Sign convention (§3–§4, matches `wavenn/ripple.py`).** Every quantity assembled here and
> every `grad_*`/`ΔV`/`ΔW_*` the implementation returns is the **loss gradient** `∂C/∂θ`; the
> local optimizer (§6) applies the descent step `θ ← θ − lr·ĝ`. The EP content signal is
> `δ_t = −∂C/∂c_t` (`01` eq 7.2), so the content **gradient** the code contracts is
> `g^c_t := ∂C/∂c_t = −δ_t` (returned as `WarmReadEP.grad_c`). Consequently the `ΔV` and
> `ΔW_*` written below are the loss **gradient**, *not* a pre-negated descent step; do not
> negate them again before the optimizer.

The cleanup codebook `V` (and each block's `M_key, M_val`) receives a loss gradient `∂C/∂V`
that is the **sum** of the read path and the readout path. Read-path-only trains *backwards*
(wrong direction; whitepaper D.3).

```
# read path (EP contrastive, from ∂E_read/∂V_k = −p_k r):  gives ∂C/∂V_k via the two-phase diff
ΔV_k^read   = −( p_{t,k}^+ r_t^+  −  p_{t,k}^- r_t^- ) / (2 β_n)

# readout path (direct cross-entropy through the final softmax; β here = readout temp β_readout, §9):
ΔV_k^readout = β · r*_t · ( p_{t,k} − 1{k = y_t} )

ΔV_k = ΔV_k^read + ΔV_k^readout  =  ∂C/∂V_k                                         (3.1)
```
For the knowledge codebooks `M_key, M_val` inside the settling, the read-path term
applies (they shape `r*`); the readout path applies to whichever codebook feeds the
local softmax (`V`). `test_two_path_codebook.py` locks `cos(ΔV, ∇_V C)=1.0000` — i.e. `ΔV`
*is* `∇_V C`, up to the `O(β_n)` finite-nudge error.

---

## 4. Exact eligibility traces for the projections (`01` §9)

The content gradient `g^c_t := ∂C/∂c_t = −δ_t` (§3 convention) is assigned to
`W_v, W_q, W_k, E, γ` through the **exact** trace (exact because the recurrence is linear in
`S`). Every trace is a decayed sum updated by a constant-size recurrence — `O(D·d_in)` memory,
`O(T)` time, no BPTT unroll. The code contracts each trace with `g^c_t` (the `g_content`
argument of `projection_grads`), so every `ΔW_*` below is the loss **gradient** `∂C/∂W_*`
(§3 sign convention), fed straight to the descent step of §6. All derivations below are for
one head; heads are independent and batched.

### 4.1 Value path — `W_v`

Trace (holographic, same recurrence as `S`):
```
G_t^v = γ G_{t-1}^v + k_t x_t^⊤        ∈ ℂ^{d_k×d_in}                              (4.1a)
```
Update:
```
ΔW_v ← ΔW_v + g^c_t ⊗ ( Re(q_t^† G_t^v) / den_t )  =  ∂C/∂W_v   ∈ ℝ^{d_vh×d_in}    (4.1b)
```
where `(q_t^† G_t^v)_n = Σ_j \bar q_{t,j} G^v_{t,j,n}`, `den_t = Re(q_t^† z_t)+ε`, and
`g^c_t = ∂C/∂c_t = −δ_t` (§3). Derived in full at `01` §9.1; equals truncated-BPTT
`∂C/∂W_v` exactly (`test_traces.py`).

### 4.2 Query path — `W_q` (current step, phase-domain)

The query enters only at step `t`. With `q_t = φ(W_q x_t)`, `∂φ(u)_j/∂u_j = i e^{iu_j} = i q_{t,j}`;
the read holds the **conjugate** `\bar q_{t,j}`, so its phase derivative carries `−i`
(`∂\bar q_{t,j}/∂u_j^q = −i \bar q_{t,j}`, and `Re(−i w)=Im(w)`), matching `01` eq 9.6:
```
∂c_{t,m}/∂u_j^q = [ +Im( \bar q_{t,j} (S_t)_{j,m} ) − c_{t,m} Im( \bar q_{t,j} (z_t)_j ) ] / den_t     (4.2a)
```
Update (no trace; `S_t, z_t` are the live states, `u^q = W_q x_t`; contract with the content
gradient `g^c_t = −δ_t`):
```
g_j^q = Σ_m g^c_{t,m} · ∂c_{t,m}/∂u_j^q
ΔW_q ← ΔW_q + g^q ⊗ x_t  =  ∂C/∂W_q                                                (4.2b)
```
The sign of (4.2a) — `+Im` on the numerator term, `−c·Im` on the denominator term — is
verified against autograd (`tests/test_traces.py::test_query_trace_matches_autograd`).

### 4.3 Key path — `W_k` (two traces, phase-domain)

`k_t = φ(W_k x_t)` enters `S_t` (numerator) *and* `z_t` (denominator) and persists.
`∂k_{i,j}/∂u_{i,j}^k = i k_{i,j}`. Two traces carry the numerator and denominator
sensitivities:
```
G_t^{k,S} = γ G_{t-1}^{k,S} + (i k_t) ⊙_row (v_t^⊤)  ... x_t^⊤   # numerator, ℂ^{d_k×d_vh×d_in} (rank-structured)
G_t^{k,z} = γ G_{t-1}^{k,z} + (i k_t) x_t^⊤                       # denominator, ℂ^{d_k×d_in}
```
In practice the numerator trace is kept rank-factored (store `i k_t x_t^⊤ ∈ ℂ^{d_k×d_in}`
and `v_t`), so both traces are `O(d_k·d_in)`. The update contracts `δ_t` with the query
`q_t` (the current unbinding direction):
```
num-part:  Σ_m g^c_{t,m} · Re( q_t^† · ∂S_t/∂u_j^k )_m / den_t
den-part: −Σ_m (g^c_{t,m} c_{t,m}) · Re( q_t^† · ∂z_t/∂u_j^k ) / den_t
ΔW_k ← ΔW_k + [num-part + den-part] ⊗ x_t  =  ∂C/∂W_k   (accumulated via the traces above)  (4.3)
```
(with `g^c_t = −δ_t`, §3).
Full component expressions are in Appendix 03-A (below). **Denominator stop-gradient
option:** dropping `G^{k,z}` (treat `den_t` as constant) halves the key-path cost; it is
a documented approximation, ablated in `05` (its effect on `05` P1 cosine is the test).

### 4.4 Embedding path — `E`

The block input `x_t` at block 1 is the token embedding `E[token_t]`. `x_t` enters
`k_t, q_t, v_t` (current step) and, via `k_t`, all future reads. The embedding gradient
is the sum of the current-step projection Jacobians times `δ_t`, plus the future
contributions already captured by the traces `G^v, G^{k,·}` (they contain `x_t`). In
practice `E` accumulates the pulled-back error `W_v^⊤ (·) + phase terms` at the token's
row — a **sparse, per-token Hebbian write** (only rows for tokens seen in the batch
update). This is the natural "write the memory of this token" update.

### 4.5 Decay path — `log_gamma`

`γ_h` is plastic. `∂S_t/∂γ = Σ_{i<t}(t-i)γ^{t-i-1}k_i v_i^⊤` — itself a linear
recurrence, carried by an auxiliary trace `Ġ_t = γ Ġ_{t-1} + S_{t-1}` (since
`∂S_t/∂γ = S_{t-1} + γ ∂S_{t-1}/∂γ`). The decay update is
`Δlog_gamma_h ∝ Σ_m δ_{t,m} Re(q_t^† Ġ_t)_m/den_t · γ_h(1−γ_h)` (chain through `σ`). This
lets the model **learn its own memory horizons** rather than fixing the log-spaced grid.
Implemented in `projection_grads(..., return_gamma=True)` via the `dotS/dotz` traces and
validated equal to autograd in `tests/test_traces.py::test_decay_trace_matches_autograd`.

### 4.6 Implemented exactness scope and the delta-rule (auditable)

The exact-trace theorem (`01` §9) covers the **pure-linear** recurrence. The
implementation (`wavenn/ripple.py::projection_grads`) realizes it as:

| Pathway | Mode | Status |
|---|---|---|
| value `W_v` | exact eligibility trace `G^v` (S-A1) | validated = autograd (`test_traces`) |
| query `W_q` | exact, current-step (query enters only at `t`) | validated = autograd |
| key `W_k` (`key_full=True`, **default**) | exact trace `T^{kv}` (S-A1) | validated = autograd |
| key `W_k` (`key_full=False`) | current-step attribution (S-A0) | cheap P4 ablation; not fidelity-exact |
| decay `log_gamma` | exact trace `Ġ` | validated = autograd |
| norm gain `norm_gain` | RMSNorm-gain gradient from the current-step input cotangent (S-A0) | complete, local |
| embedding `E` | tied-readout gradient + block-1 current-step input pullback (S-A0) | complete, local |

**`key_full`** is the concrete S-A0↔S-A1 knob of Prediction **P4** (`05`): the exact key
trace `T^{kv}` is `O(d_k·d_vh·d_in)` (large at scale). The **faithful default is
`key_full=True`** (the exact trace, `wavenn/rlm.py::RLMConfig`); `key_full=False` selects the
cheap current-step S-A0 key path, and P4 measures how much the exact key trace adds.

**Delta-rule heads** (`α_h>0`, M6) — a forward/recall option, **not** the trained rule. The
forward uses the DeltaNet write `S_t = γS_{t-1} + k_t(v_t − α_h v̂_t)^⊤` (`01` §5.3); its
recall benefit is P6 (`05`). For *learning*, the direct value/key/query pathways are computed
on the actual delta-rule state (correct), but the **delta-feedback** term — the dependence of
`v̂_t` on past `W_v` — is **dropped**, not "attributed at the current step". Because
`v̂_t = Re(k_t^†S_{t-1})/den'` depends only on **past** writes (`S_{t-1}`), its current-step
component is exactly **zero**, so there is nothing to attribute at step `t`; capturing the
past-write dependence would require an extra DeltaNet eligibility trace (the documented S-A1
refinement), which is not maintained — so that gradient term is simply dropped. The decay-path
(`log_gamma`) trace inherits the same gap under the delta-rule, so its delta-feedback term is
dropped as well (accordingly the code requests the `log_gamma` gradient **only** when the
delta-rule is off, `wavenn/rlm.py::ripple_grads`). Consequently the delta-rule's local gradient
does **not** track BPTT; it is therefore offered as a faithful, complete, local *forward/recall*
rule (the **P6** ablation), **not** as part of the fidelity-validated trained rule.
Pure-accumulation heads (the trained default) are always exact.

---

## 5. Three-factor neuromodulation

Every local update is gated by a **global scalar modulator** `μ_t` (the third factor,
alongside pre- and post-synaptic activity):
```
Δθ  ⟵  μ_t · ( local RIPPLE update for θ ).                                        (5.1)
```
`μ_t` is a bounded function of **prediction surprise**:
```
μ_t = clip( g0 + g1 · ( C_t − \bar C )  , μ_min, μ_max ),   \bar C = EMA of C_t.   (5.2)
```
Rationale (Frémaux & Gerstner 2016): surprising tokens (high `C_t` above the running
mean) drive larger plastic changes; well-predicted tokens drive small ones. This
(a) stabilizes learning (down-weights already-learned content), (b) is the substrate for
**continual learning without catastrophic interference** (novel = surprising = large
update; familiar = protected), and (c) is a single cheap scalar, hardware-trivial.
Defaults: `g0=1, g1=0.5, μ∈[0.1, 3]`. `μ_t≡1` recovers ungated RIPPLE (ablation).

---

## 6. Local optimizer (Adam-style, no graph)

The accumulated `Δθ` are the raw local **loss gradients** `∂C/∂θ` assembled in §3–§4 (the
sign convention stated there), so the descent step below subtracts them. They are conditioned
by **per-parameter moment statistics kept locally** (no autograd, just running moments):
```
m ← β1 m + (1−β1) Δθ ;   v ← β2 v + (1−β2) Δθ²
θ ← θ − lr · m̂ / (√v̂ + ε)     (bias-corrected)                                    (6.1)
```
This is ordinary Adam arithmetic applied to *locally computed* gradients — permitted
under C2 because the moments are element-wise functions of the local `Δθ`, never a
backward pass. Defaults `β1=0.9, β2=0.99, ε=1e-8`, `lr` per §9. µP-style width scaling
(`04`) transfers `lr` across model sizes.

**Gradient scale (matches the BPTT twin).** Before the update, the assembled RIPPLE gradient
is scaled by `1/(n_layers·B·T)` (`wavenn/rlm.py::ripple_grads`): the local terms *sum* over the
`B·T` tokens and the `L=n_layers` blocks, whereas the BPTT twin's loss is a *mean* over tokens
and a mean over the per-block deep-supervision losses (`rlm.py::bptt_loss`). Dividing by
`n_layers·B·T` puts the two gradients on the same mean-of-means scale, so a learning rate
transfers between the RIPPLE rule and its BPTT twin.

---

## 7. Depth-credit ladder: L0 → L1 → L2 (the research core)

How error reaches lower blocks is the genuinely open question. RIPPLE defines a ladder
of three modes, each strictly more coupled, and **the gap between them is measured, never
assumed** (`05` P2/P3).

### L0 — Pure local (per-block readouts) — **IMPLEMENTED (`wavenn/rlm.py::ripple_grads`)**
Every block has its own vocabulary readout `p_ℓ = softmax(β Vᵀ r*_ℓ)` and its own local
cost `C_ℓ = −log p_ℓ[y_t]`. Each block runs its own EP two-phase read and updates its own
parameters from its own `δ_ℓ`. **No error flows between blocks.** This is maximally local
and maximally parallel (all blocks learn simultaneously). It is the deep-supervision /
predictive-coding-per-layer regime (mutation M3). Cheapest, most hardware-friendly, and
the first thing trained. L0 is a *complete* backprop-free training rule (not a reduced L1):
every block has a real local objective and every parameter a real local gradient
(`tests/test_full_model.py::test_full_ripple_grads_all_params_learn`). The block bundles
its cleaned read `r*_ℓ` directly onto the bus (no separate `W_o`, which would have no local
gradient in L0), so all parameters are trainable — a defensible design decision, not a gap.

### L1 — Stack credit assignment (inter-block, `07` R1) — **IMPLEMENTED (`ripple_grads_l1`)**

*The obstruction, stated honestly.* The residual stack is **feedforward between blocks**: the
bus only flows up (`bus_{ℓ+1}=bus_ℓ+r_ℓ`) and the inter-block maps (`W_k,W_q,W_v` + the
linear-attention recurrence) are asymmetric with no downward path. So the naive "nudge only
the top readout and let it propagate down through the forward weights" produces **zero**
perturbation at a lower block's content `c_ℓ` — genuine top-down EP needs a bidirectional
coupling this architecture does not have. The literature is unambiguous (survey in `07` §6):
for an asymmetric feedforward stack, *no scheme is both transpose-free and provably
BPTT-matching* — one must (a) add a symmetric feedback coupling [Ernoult et al. 2019,
arXiv:1905.13633; Laborieux & Zenke 2022, arXiv:2209.00530 — exact], (b) keep the asymmetry
and accept a bias bounded by the Jacobian's degree of symmetry [Scellier et al. 2018,
Vector-Field EP, arXiv:1808.04873], or (c) use the transpose. L1 is therefore a **ladder of
two concrete modes**, and the gap between them is measured, never assumed.

**L1-exact (implemented).** The additive bus makes the exact inter-block cotangent trivial to
route *down*: `∂L/∂r_ℓ = ∂L/∂bus_{ℓ+1}`, accumulated top→bottom. Each block runs its EP settle
with that top-down cotangent injected as the linear-cost force `−g_top` (`stack_read_ep`,
`cleanup.settle`), so `δ_ℓ=(r_ℓ^+−r_ℓ^−)/2β_n` estimates `−∂C_total/∂c_ℓ` — the content
signal that already carries the credit of every block above (EP identity 7.1 with the
augmented cost `C_total = CE_local + ⟨g_top, r⟩`). The input pullback is then pushed onto the
bus via the **exact reverse-time input trace** `contextual_input_grad`
(`Q_t=conj(q_t)⊗g_t/den_t+γQ_{t+1}`; value path `Re(Σ_j k_t[j]Q_t[j,·])`, key path
`−Im(k_t⊙(Q_t·v_t))`, query path current-step) — exact because the recurrence is linear in S;
the cheap current-step `g_a` of `projection_grads` would drop the temporal cross-terms that
carry the recall credit. The result is the **exact gradient** of the deep-supervision (or
final-only) loss, assembled with *no autograd over parameters* (C2): it reproduces the
2-layer BPTT gradient to `cos = 1.0000` on **every** parameter, including the block-1 codebook
`M.0` that L0 gets at `cos ≈ 0.24` (`tests/test_stack_ep_fidelity.py`). Because it uses
inter-block weight transposes for the `g_a` routing it is the C2-compliant **exact-credit
reference**, not the transport-free rule; its job is to answer `07`'s causal question
decisively — *does closing depth-credit recover the capability L0 lacked?*

> **Recall result (`07` R1, `experiments/S8-L1-result.md`):** RIPPLE-L1 reaches 2-layer MQAR
> recall **0.998** (N=4) and **0.999** (N=8), matching its BPTT twin (0.996 / 0.998) and a
> transformer (≈0.995), up from the L0 floor of ~0.26 (N=4) / 0.14 (N=8). A 2×2 (rule × key-path) factorial shows recovery needs **both** the
> stack-EP depth-credit **and** the exact key gradient (`key_full=True`): L0 stays at ~0.26
> even with the exact key path, and L1 stays at ~0.26 with the cheap key path — only L1 +
> exact key reaches the ceiling. This confirms `07` R1 and adds the co-requisite key-gradient
> finding.

*(The exactness check also surfaced and fixed a latent L0 bug: the tied-readout→embedding
Jacobian used the wrong normalization axis — a ~55% error on the embedding's readout gradient,
invisible to the free-`V` fidelity test. `tests/test_tied_readout_grad.py` locks it.)*

**L1-FA (transport-free routing, spatial only) — ✅ DONE (`experiments/S8-L1FA-result.md`).**
The cheapest transport-free step: keep the entire L1-exact scaffold and replace *only* the
inter-block input projection's transpose with fixed random feedback `Bk/Bq/Bv` (feedback
alignment, Lillicrap et al. 2016), reading cached forward activations — no forward weight in the
error projection (`contextual_input_grad_fa`; audited, `tests/test_transport_free.py`). Result:
**it recovers MQAR to the ceiling at both L=2 (0.999) and L=3 (0.996)** in the clean final-only
isolation (block-0 has no readout of its own; 3 seeds), i.e. *removing spatial weight transport
costs nothing here* — and it holds at depth 3, where FA was expected to break. Striking: the
routed update is only ~0.04–0.33-cosine aligned with the true gradient yet still trains the
induction head — plausibly because the **exact temporal adjoint is preserved** (only the input-
dimension mixing is randomized), so credit still lands on the right timesteps. Scope: this keeps
the reverse-time adjoint and an explicit backward sweep — it is the *spatial-transport* study,
not the full plausibility claim.

**L1-VF (genuine Vector-Field EP, no backward pass) — ✅ DONE: obstructed by causality
(`experiments/S8-L1VF-result.md`, `docs/09`).** Built the space-time relaxation (`ripple_grads_l1vf`,
`depth_mode="L1VF"`): propagate the top nudge down through a joint stack relaxation using only forward
operators. Result (independently verified by three independent reviews): the plumbing is correct (acyclic +
exact coupling reproduces L1-exact at cos=1.0; F1 ÷β_n and F3 operator verified), but the genuine
**cyclic** relaxation is *obstructed* — re-running the causal read to propagate the nudge forward-in-
time injects an O(1) spurious forward-coupling (block-0's nudge fed up into c_1), corrupting the
gradient even at a converged fixed point (top-block cos 0.76; final-only recovery to 1.0 proves the
mechanism). Carrying anticausal (query→source) credit requires the reverse-time adjoint `Q_t` — the
"no-backward-pass" relaxation reduces to L1-FA. **So the transport-free depth-credit capability is
delivered by L1-FA; a genuine forward-only EP relaxation on this causal stack does not exist.** This
resolves OP3(b) / `06` §5 (negative).

### L2 — Space-time EP (research track R1, `06`)
Relax the entire unrolled trajectory to a space-time equilibrium and nudge the whole
output sequence (whitepaper S-B/OP3). Higher-risk, more parallel, attempted only after
L1 validates.

**Measurement (`05`).** For each depth `L`, report: (i) `cos` of the L0/L1 update vs the
BPTT-twin gradient per block; (ii) final val loss of L0, L1, and BPTT. The claim is *not*
"local = backprop"; it is "here is exactly how much local depth-credit recovers, per
depth" — a quantitative contribution regardless of outcome.

---

## 8. The RIPPLE step (pseudocode, L0 and L1)

```python
def ripple_step(model, x_tokens, y_tokens, cfg):        # one sequence (batched)
    reset_states(model)        # S,z,G_v,G_kS,G_kz,G_dotgamma = 0 for all blocks/heads
    grads = zeros_like(model.params)
    Cbar  = model.Cbar         # EMA surprise
    for t in range(T):
        x = embed(x_tokens[:, t])
        for ℓ, blk in enumerate(model.blocks):
            k, q, v = blk.project(x)                     # φ on k,q; positional ρ^t
            blk.write(k, v)                              # S,z update (delta-rule optional)
            blk.update_traces(k, q, v, x)                # G_v, G_kS, G_kz, G_dotgamma  (§4)
            c = blk.read(q)                              # content c_t  (02 §4)
            # --- EP two-phase on this block's cleanup read (03 §2) ---
            r0      = settle(c, blk.M_key, blk.M_val, cfg.beta, nudge=0)
            rp, pp  = settle(c, ..., nudge=+cfg.beta_n, target=y_tokens[:,t], warm=r0)
            rn, pn  = settle(c, ..., nudge=-cfg.beta_n, target=y_tokens[:,t], warm=r0)
            delta   = (rp - rn) / (2*cfg.beta_n)         # content signal δ_t
            gc      = -delta                             # content GRADIENT ∂C/∂c = −δ_t (§3)
            rstar, p = readout(r0, model.V, cfg.beta_readout)  # final read + distribution (F3, §9)
            C_t = nll(p, y_tokens[:, t])
            mu  = clip(cfg.g0 + cfg.g1*(C_t - Cbar), cfg.mu_min, cfg.mu_max)   # §5
            Cbar = ema(Cbar, C_t, cfg.ema)
            # --- codebook (both paths, §3); each grad is ∂C/∂θ ---
            grads.V      += mu * two_path_codebook(rp,pp,rn,pn, rstar,p, y, cfg)
            grads.M_key  += mu * read_path_M(...);  grads.M_val += mu * read_path_M(...)
            # --- projections via exact traces (§4); contract the trace with gc ⇒ ∂C/∂θ ---
            grads.W_v    += mu * gc ⊗ (Re(q^† blk.G_v)/den)
            grads.W_q    += mu * qpath(gc, q, blk.S, blk.z, den, x)
            grads.W_k    += mu * kpath(gc, q, blk.G_kS, blk.G_kz, den, x)
            grads.log_gamma += mu * gammapath(gc, q, blk.G_dotgamma, den)
            grads.E[x_tokens[:,t]] += mu * embed_path(...)
            x = blk.bundle(rstar)           # onto the bus (02 §6)
    scale_grads(grads, 1.0/(cfg.n_layers*B*T))                   # mean-of-means scale (F8, §6)
    local_adam_update(model.params, grads, model.moments, cfg)   # §6
    model.Cbar = Cbar
```

In **L0** (above), each block's `settle`/`delta` uses its *own* target readout independently
and the bus is detached between blocks. In **L1-exact** (`ripple_grads_l1`), the forward pass
builds the bus and stores each block's free read `r_ℓ`; then a **top→bottom sweep** carries the
bus cotangent `∂L/∂bus_{ℓ+1}` into each block as the top-down force `g_top` of its ± nudged
settles (`stack_read_ep`), so `δ_ℓ` now includes inter-block credit, and the exact input trace
(`contextual_input_grad`) pushes the pullback back onto the bus for the next lower block. The
trace and codebook updates are identical to L0; only the source of `δ_ℓ` (and the exact input
routing) differ. See §7 for the derivation and the transpose-free L1-VF follow-on.

---

## 9. Hyperparameters (defaults, ranges, justification)

| Symbol | Default | Range | Justification |
|---|---|---|---|
| `β` (settle temp) | 1.0 | 0.5–4 | knowledge-memory **settle** temperature; kept modest so the read is converged / single-valued (the EP precondition, `01` §6.2). The sufficient bound `2/σ_max(M)²` is *conservative* — with unit-norm columns `σ_max(M)≥1` ⇒ the bound is `≤2`, so it cannot license a large β; the operating point is validated **empirically** (`tests/test_training_fidelity.py`), and the safe limit is exposed via `safe_beta_limit()` / logged in telemetry (`codebook_health`). |
| `β_readout` (readout temp) | 6.0 | 2–20 | **readout** softmax temperature (F3), decoupled from the settle so the next-token distribution can be sharp without threatening read uniqueness; `wavenn/rlm.py::RLMConfig.beta_readout`. |
| `β_n` (nudge) | 0.05 | 0.01–0.2 | small ⇒ EP bias `O(β_n)`; symmetric nudge cancels leading bias (Laborieux 2021); too small ⇒ FP noise in `δ` |
| read steps `n_max` | 8 | 1–20 | adaptive stop usually 1–3; cap bounds worst case (`02` §5.1) |
| `η` settle step | 0.3 | 0.1–0.5 | gradient-flow step; convergence rate `1−½βσ²` (`01` §6.2) |
| `γ_h` grid | logspace(1−γ∈[1,1e-3]) | learned | per-head timescales (`02` §4.2); plastic via §4.5 |
| delta-rule `α_h` | **off** (`delta_rule=False`) | {0,1} or learned | trained-rule default is delta **OFF** (pure accumulation ⇒ exact traces, §4.6); the delta-rule write is a forward/recall **P6 ablation** (`01` §5, `05` P6) — when enabled it acts on the middle-third heads |
| `key_full` | **True** | {True, False} | trained default is the **exact** key eligibility trace (§4.6); `key_full=False` is the cheap current-step **S-A0** key path, the **P4** ablation |
| `μ` gate `(g0,g1)` | (1, 0.5) | g1∈[0,1] | surprise modulation (§5); g1=0 disables |
| `μ` clip | [0.1, 3] | — | stability bound on plastic step |
| local Adam `(β1,β2)` | (0.9, 0.99) | — | standard; moments are local |
| `lr` | 3e-3 (µP base) | 1e-4–1e-2 | µP transfer across width (`04`); higher than BPTT since updates are unit-scaled |
| depth mode | L0 → L1 | {L0,L1,L2} | ladder (§7); start L0 |
| chunk `L_c` | 64 | 16–128 | throughput vs memory (`02` §4.3) |
| `D`, `d_v` | `= d_model` | — | capacity `√(D/M)` (`01` §4) |
| `P` knowledge | `2·d_model` | 1–8× | FFN-expansion analogue (`02` §5) |

---

## 10. C2 compliance checklist (audited in code review)

- [ ] No `loss.backward()` / `torch.autograd` in the WNN training loop (`ripple.py`, `train.py`).
- [ ] All `δ_t` from two settles; all projection grads from traces (§4); codebook from §3.
- [ ] Traces reset per sequence; `O(D·d_in)` per trace, no `T`-indexed storage.
- [ ] Autograd used **only** in the BPTT twin (`wavenn/rlm.py::bptt_loss` + `wavenn/train.py::train_bptt`) and `tests/*` (finite diff).
- [ ] `β` and `β_n` are separate config fields, never aliased.
- [ ] Two-path codebook present (both terms of eq 3.1); regression test green.

---

## Appendix 03-A — Key-path component expressions (full)

For head `h`, with `u^k = W_k x_t`, `k_{t,j}=e^{iu^k_j}`, and the persistent traces from
§4.3, the exact contribution of `δ_t` to `W_k` decomposes into numerator and denominator
parts (both `O(d_k·d_in)`):

**Numerator** (`k` in `S_t`): for each past write `i≤t`, `S_t` contains `γ^{t-i}k_i v_i^⊤`.
`∂(k_{i,j})/∂u^k_{i,j}=i k_{i,j}`. The read numerator `Re(q_t^†S_t)_m = Σ_i γ^{t-i}Re(q_t^†k_i)v_{i,m}`
has sensitivity to `u^k_{i,j}`:
```
∂/∂u^k_{i,j} [ Re(q_t^† S_t)_m ] = γ^{t-i} Re( \bar q_{t,j} · i k_{i,j} ) v_{i,m}
                                 = −γ^{t-i} Im( \bar q_{t,j} k_{i,j} ) v_{i,m}.
```
Summed over `m` against `δ_t` and pulled onto `x_i`, this is exactly the trace
`G_t^{k,S}` contracted with `δ_t`.

**Denominator** (`k` in `z_t`): `den_t = Re(q_t^†z_t) = Σ_i γ^{t-i}Re(q_t^†k_i)`, with
`∂den_t/∂u^k_{i,j} = −γ^{t-i}Im(\bar q_{t,j}k_{i,j})`. Since `c_{t,m}=num_m/den_t`,
```
∂c_{t,m}/∂u^k_{i,j} = [ −γ^{t-i}Im(\bar q_{t,j}k_{i,j})v_{i,m}  −  c_{t,m}·(−γ^{t-i}Im(\bar q_{t,j}k_{i,j})) ] / den_t
                    = −γ^{t-i} Im(\bar q_{t,j}k_{i,j}) ( v_{i,m} − c_{t,m} ) / den_t.
```
Both terms share the factor `γ^{t-i}Im(\bar q_{t,j}k_{i,j})`, so a **single** complex
trace `G_t^{k} = γG_{t-1}^{k} + (i k_t) x_t^⊤` plus the running `S_t` (for `v_i`) and the
scalar `c_{t,m}` suffice; the update is
```
ΔW_k ← ΔW_k − Σ_m δ_{t,m} · Re( q_t^† [ (∂S/∂u^k as G^k with v) − c_{t,m}(∂z/∂u^k as G^k) ] )_j / den_t ⊗ x_t
```
computed from the traces, `O(d_k·d_in)` per step, exact. `test_traces.py` checks this
against autograd through the recurrence to `1e-5` relative error.
