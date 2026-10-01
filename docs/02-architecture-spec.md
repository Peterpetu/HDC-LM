# 02 — Architecture Specification

**Role:** The complete forward pass of the Wave Neural Network, with explicit shapes,
dtypes, the multi-timescale memory schedule, delta-rule writes, adaptive settling, the
bundling bus, normalization, tokenizers, and parameter/memory budgets for a 16 GB GPU.
Every design commitment traces to a section of `00` (axioms) or `01` (math).

Notation and all algebra follow `01`. Reference implementation targets are named per
module (`wavenn/*.py`); this document is the contract those modules satisfy.

---

## 1. Design commitments (from `00`, `01`)

1. **State in ℂ (amplitude+phase), codewords unit-magnitude** — the torus is
   impossible for superposition (`01` §2). `k_t, q_t` unit; `S_t, z_t` free amplitude.
2. **Order via the causal recurrence** (accumulation order = position), optionally
   reinforced by positional phase binding `ρ^t` (`01` §3, §11).
3. **Finite-temperature associative read** — each block cleans its read against a
   learned codebook via modern-Hopfield settling (`01` §6); this is an energy-based
   model to which EP applies (`03`).
4. **Learned codebook** — all projections and codebooks are plastic parameters trained
   by the local rule.
5. **Linearity of the state recurrence is sacred** — it is what makes the eligibility
   trace exact (`01` §9). No nonlinearity may be inserted between `S_{t-1}` and `S_t`.

---

## 2. Tensor conventions

- Batch `B`, sequence `T`, model width `d_model` (= `d_in` for block projections).
- Complex tensors: `torch.complex64` for phasors; states may be `bfloat16`-real-pair on
  the forward path with `float32` accumulation for `S_t, z_t` (see §9 precision policy).
- Shape assertions at every module boundary (project rules).
- **Phasor dimension `D`**, **value dimension `d_v`**, **heads `H`**; per-head
  `d_k = D/H`, `d_vh = d_v/H`. Default `d_v = d_model`, `D = d_model` (so `d_k = d_vh = d_model/H`).

---

## 3. The block (attention + FFN, both as memories)

A WNN block has two memories on a shared **bundling bus** (the residual stream, `00` M7).
Two realizations exist and both implement this contract:

- **General reference (`wavenn/model.py::WNNBlock`, forward-only, tested):** two *parallel*
  sublayers each bundling onto the bus, with a contextual output projection `W_o`:
  ```
  x ─► [Contextual Memory] ─(W_o, bundle)─► x' ─► [Knowledge Memory] ─(bundle)─► x''
  ```
- **RIPPLE-trainable realization (`wavenn/rlm.py::RippleLM`, the trained model):** the two
  memories are *composed* — the contextual read `c` is the content bias the knowledge
  memory cleans, and the single cleaned result `r*` is bundled onto the bus:
  ```
  a = RMSNorm(bus);  c = Contextual(a);  r* = KnowledgeSettle(c);  bus = bus + r*
  ```
  This composition (rather than two parallel bundles with a separate `W_o`) is chosen for
  the local rule: in L0, `W_o` would have no local gradient path, so folding the
  contextual output through the knowledge read keeps **every** parameter trainable (a
  defensible design decision, `03` §7). `d_v = d_model` so `r*` bundles directly.

Both are pre-normalized (`01`-consistent RMS, §6) and bundle (add) onto the bus. The bus
carries the running superposition; magnitude normalization keeps it inside the SNR budget
(`01` §4). The RippleLM realization is the one trained by RIPPLE (`03`) and the BPTT twin
(`05`); the WNNBlock reference validated the forward primitives (recall/causality, `05` S2).

### 3.1 Per-block parameter inventory

| Tensor | Shape | Role |
|---|---|---|
| `W_q, W_k` | `ℝ^{D×d_model}` | query/key projections → phase via `φ` |
| `W_v` | `ℝ^{d_v×d_model}` | value projection |
| `W_o` | `ℝ^{d_model×d_v}` | contextual output projection (bundle onto bus) |
| `log_gamma` | `ℝ^{H}` | per-head decay `γ_h = σ(log_gamma_h)` (§4.2) |
| `M_key` | `ℝ^{d_v×P}` | knowledge codebook keys (Hopfield stored patterns) |
| `M_val` | `ℝ^{d_v×P}` | knowledge codebook values (Hopfield readout) |
| `g_block` | `ℝ^{K}` (readout head) | per-block vocabulary readout for local loss (`03` M3); tied to `V` where possible |
| RMS scales | `ℝ^{d_model}` ×2 | pre-norm gains |

`P` (knowledge patterns) defaults to `2·d_model` (FFN-expansion analogue, `01` §5 note).

---

## 4. Contextual Memory (the attention replacement)

### 4.1 Forward (per head `h`, causal recurrence)

```
k_t^h = φ(W_k^h x_t) ⊙ ρ^t        # unit phasor, positionally bound (01 §3,§11)   ∈ ℂ^{d_k}
q_t^h = φ(W_q^h x_t) ⊙ ρ^t        # same positional phasor                         ∈ ℂ^{d_k}
v_t^h = W_v^h x_t                                                                    ∈ ℝ^{d_vh}

# write (bind + bundle); delta-rule optional on medium-timescale heads (01 eq 5.3)
S_t^h = γ_h S_{t-1}^h + k_t^h (v_t^h − α_h \hat v_t^h)^⊤        \hat v_t^h = Re(k_t^{h†}S_{t-1}^h)/den'   ∈ ℂ^{d_k×d_vh}
z_t^h = γ_h z_{t-1}^h + k_t^h                                                        ∈ ℂ^{d_k}

# read (unbind + normalize by the POSITIVE count normalizer den_t=γden+d_k)  (01 eq 5.2)
c_t^h = Re(q_t^{h†} S_t^h) / den_t^h,     den_t^h = γ_h den_{t-1}^h + d_k          ∈ ℝ^{d_vh}
```
The normalizer is the **deterministic count** `den_t=Σγ^{t-i}d_k`, not the query-dependent
`Re(q^†z)`: FHRR similarities are signed, so `Re(q^†z)` can go negative during training and
explode the read (`01` §5, `experiments/S4a-denominator-stability-fix.md`).

Concatenate heads → `c_t ∈ ℝ^{d_v}`. The contextual sublayer output bundled onto the bus
is `W_o · CLEANUP(c_t)` where `CLEANUP` is the block's knowledge memory (§5) OR identity
(configurable: cleanup can live in the contextual path, the knowledge path, or both —
the P6 ablation, `05`).

- `ρ^t` = positional phasor `φ(tθ)` (RoPE-as-primitive, `01` §11); `θ ∈ ℝ^{d_k}` fixed
  log-spaced frequencies. Applied to **both** `q` and `k` so scores are relative.
- `α_h ∈ {0,1}` (or learned in `[0,1]`) enables the delta-rule write per head. When
  enabled, the head layout is: fast heads `α=0` (pure accumulation), medium (middle-third)
  heads `α=1` (error-correcting), slow heads `α=0`. The **RIPPLE-trained default keeps the
  delta-rule OFF** (`wavenn/rlm.py::RLMConfig.delta_rule=False`, pure accumulation ⇒ exact
  traces); the delta-rule write is a forward/recall **P6 ablation** (`03` §4.6, `05`).

### 4.2 Multi-timescale schedule (mutation M6, `01` §4.3)

Heads' decays are **log-spaced** to cover distinct effective contexts `M_eff = (1+γ)/(1−γ)`:
```
γ_h  chosen so that  1−γ_h  ∈  logspace(1−γ_max, 1−γ_min, H).
```
Defaults: `γ_min = 0.0` (one purely local head, `M_eff≈1`), `γ_max = 0.999`
(`M_eff≈2000`). `log_gamma` is a plastic parameter (learned by the trace rule, `03`),
initialized to the log-spaced grid. This gives the block a spectrum of memory horizons
from one token to thousands — the WNN analogue of multi-head attention's implicit
timescale mixture, made explicit and writeable.

### 4.3 Chunked scan (throughput, `04`)

The recurrence (4.1) is computed in chunks of length `L_c` (default 64–128): within a
chunk, the intra-chunk contribution is a masked matmul (parallel), and the inter-chunk
carry is the recurrent state `S`. This is the standard linear-attention chunked form
(GLA, Yang et al. 2024) and gives near-transformer training throughput while preserving
the O(1) recurrent generation form. The eligibility trace `G_t` (`01` §9) is scanned in
the same chunks.

---

## 5. Knowledge Memory (the FFN replacement)

A finite-temperature modern-Hopfield cleanup — the block's "what do I know" memory,
addressed by the contextual read `c_t`.

### 5.1 Forward (adaptive settling, `01` §6.3)

```
r^{(0)} = c_t
repeat  s = 0,1,…  until convergence (‖Δr‖ < τ_tol; hard cap n_max):
    a^{(s)} = softmax( β · M_keyᵀ r^{(s)} )          ∈ Δ^{P−1}
    r^{(s+1)} = r^{(s)} + η ( −r^{(s)} + c_t + M_val a^{(s)} )
r* = r^{(last)}                                       ∈ ℝ^{d_v}
```

The settle's stop rule is exactly `‖Δr‖ < τ_tol` (with the `n_max` cap), as implemented in
`wavenn/cleanup.py::settle`. The read-entropy plateau is **not** part of the stop test; it is
an offline adaptive-compute **diagnostic** (`wavenn/rlm.py::adaptive_settle_stats`, the M4
measurement of iterations vs token surprise).

This is the gradient flow of `E(r)=½‖r‖²−⟨c_t,r⟩−(1/β)lse(β M_keyᵀ r)`. **The
implementation uses the symmetric form `M_key = M_val = M`** (`wavenn/rlm.py`,
`wavenn/cleanup.py`), which is the exact `01` §6 modern-Hopfield **energy** and is
therefore the EP-trainable choice — the separate-key/value ("attention over stored
patterns", Ramsauer et al. 2021) form is *not* an energy in general, so EP would not
apply. Convergence & uniqueness hold in the safe regime `β < 2/σ_max(M)²` (`01` §6.2).
**`M`'s columns are kept unit-norm** (`F.normalize`, VSA codewords are unit magnitude,
`01` §2), which bounds `σ_max(M)` and keeps the read `r* = c + M·softmax(βMᵀr*)` a bounded
`c` + convex-combination-of-unit-vectors — the read cannot blow up as `M` trains. The
readout codebook `V` (tied to the embedding) is unit-normalized for the same reason. This
is a correctness requirement, not a convenience (project rules).

- **Adaptive compute (M4):** `n` iterations vary per token by difficulty; logged vs
  token surprise (`05`). `n_max` caps worst case (default 8).
- **Block-local readout:** `p_t^{(block)} = softmax(β V^ᵀ r*)` feeds the block's local EP
  loss (`03` M3). The final block's readout is the model's next-token distribution.

### 5.2 Why cleanup is not decorative (C1 test)

If `β→∞`, one step, `M`-cleanup collapses to the raw read `r*=c_t` and the block is
vanilla linear attention (Theorem T1, `01` §10). The finite-`β` multi-step read is the
*only* place the knowledge codebook `M` shapes the representation. The P6 ablation
(`05`) tests whether it buys measurable quality; if it does not, cleanup is dropped from
the contextual path and kept only as the FFN-role knowledge memory — a clean,
documented fallback (`00` §4, R5).

---

## 6. The bundling bus and normalization

- **Residual = bundling** (`00` M7): `x' = x + W_o·(contextual out)`,
  `x'' = x' + (knowledge out)`. Additions are VSA superposition on the shared bus.
- **Magnitude normalization** (phasor-native RMS): before each sublayer,
  `x̃ = x / RMS(x) · g`, `RMS(x)=√(mean_j |x_j|²)+ε`. This keeps the bus magnitude inside
  the SNR budget (`01` §4) — the interference wall is a function of superposed
  magnitude, so normalization is a *correctness* requirement, not a convenience
  (project rules). Real-valued bus uses standard RMSNorm; complex
  bus uses magnitude RMS.
- **Embeddings.** Token → `d_model` real embedding `E ∈ ℝ^{K×d_model}`; the input is
  lifted to the phasor domain by `φ` only inside the projections (keys/queries), so the
  bus itself is real (values + content), keeping the FFN-role arithmetic real and cheap.
  The vocabulary codebook `V` is **tied** to `E` where dimensions match.

---

## 7. Full model

```
tokens ─► Embed E ─► x_0
for ℓ in 1..L:
    x_ℓ = Block_ℓ(x_{ℓ-1})            # §3: contextual then knowledge, on the bus
    (optional) block-local readout p_ℓ = softmax(β Vᵀ r*_ℓ)   for local EP loss (03)
logits = β Vᵀ r*_L                      # final cleanup read → next-token distribution
p_T    = softmax(logits)
```

- **Depth = timescale/abstraction ladder**, not just composition: lower blocks use
  shorter `γ` grids and smaller `P`; higher blocks longer memory and larger knowledge
  codebooks. (Configurable per-block; default uniform for the first experiments.)
- **Tied vocabulary codebook** `V = Eᵀ` (or a learned `d_v×K`), so the readout cleanup
  retrieves against the same space tokens are embedded in.

---

## 8. Tokenizers

| Dataset | Tokenizer | `K` | Rationale |
|---|---|---|---|
| TinyStories | SentencePiece BPE | 8192 | Small vocab keeps `V` (and its two-path update, `01` §8) cheap while covering the simple-English distribution; matches the coherence-regime scales. |
| The Pile / FineWeb | GPT-2 BPE | 50257 | Commensurability with GLA/GPT-2 baselines and the requential-coding literature (`05`). |
| (research) char-level | bytes | 256 | For the OpenWebText-char compression control and the "reset tokenization" track (`06`). |

Tokenization is **not load-bearing** for the constitutional claims; the "reset
tokenization" exploration (learned holographic sub-word codes) is a `06` research track,
deferred until after the core rule validates.

---

## 9. Precision policy (`04` detail)

| Quantity | dtype | Why |
|---|---|---|
| Plastic weights `W_*, M_*, V, E` | `float32` | local updates accumulate; master copy fp32 |
| Optimizer-moment statistics (local Adam, `03`) | `float32` | stability of per-parameter scaling |
| Phasors `k,q`, states `S,z` | `complex64` (fp32 real pairs) | phase precision matters near the read denominator; accumulation must not drift |
| Eligibility traces `G_t` and siblings | `float32` | exactness claim (`01` §9) requires no accumulation error |
| Bus activations (forward, within chunk) | `bfloat16` | throughput; recomputed, not stored across chunks |
| Read settling `r` | `float32` | fixed-point convergence tolerance is tight |

Rule of thumb: **anything that accumulates over time is fp32; per-step transient
activations may be bf16.** No autograd graph is stored, so the usual activation-memory
pressure of BPTT is absent — precision is spent where accumulation demands it.

---

## 10. Parameter and memory budgets (16 GB)

**Per-block params** ≈ `4·d_model²` (contextual: `W_q,W_k,W_v,W_o`) `+ 2·d_model·P`
(knowledge). With `P=2d_model` → ≈ `8·d_model²` per block. Plus tied embed `d_model·K`.

| Config | `d_model` | `L` | `H` | `K` | Params | Target |
|---|---|---|---|---|---|---|
| WNN-XS | 384 | 6 | 6 | 8192 | ≈ 10 M | TinyStories iterate |
| WNN-S | 512 | 8 | 8 | 8192 | ≈ 21 M | TinyStories P2/P4 |
| WNN-M | 768 | 12 | 12 | 50257 | ≈ 95 M | Pile P5 |
| WNN-L | 1024 | 16 | 16 | 50257 | ≈ 185 M | Pile scaling (stretch) |

**Training-memory advantage.** Because there is **no unrolled autograd graph**, training
memory is `params + optimizer moments + traces + one chunk of activations`:
- params (fp32): `4·N` bytes; optimizer moments (2×): `8·N` bytes → `12·N`.
- traces `G_t` and siblings: `O(L·D·d_model)` fp32, ≈ the size of a few weight matrices
  (e.g. WNN-M: `12·768·768·4·3 traces ≈ 85 MB`).
- states `S,z`: `O(L·D·d_v)` current-only (no `T` factor), ≈ 85 MB at WNN-M.
- one chunk of activations: `B·L_c·d_model` bf16.

For WNN-M (95 M): `12·95M ≈ 1.1 GB` for params+moments, `<0.3 GB` traces+states, leaving
the rest of 16 GB for large batch. **This is the concrete 16 GB payoff of C2:** a
locally-trained model of a given size fits a far larger batch (or a larger model) than
its BPTT twin, because activation memory does not scale with `T`.

---

## 11. Shape-assertion contract (module boundaries)

Every module asserts on entry/exit (enforced in code, the project rules):

| Module | Input | Output | Invariant |
|---|---|---|---|
| `algebra.phi` | `ℝ^{...×D}` | `ℂ^{...×D}`, `|·|=1` | unit magnitude to `1e-5` |
| `memory.write` | `S:(B,H,d_k,d_vh)`, `k,v` | `S'` same shape | linear in `S` (no nonlinearity) |
| `memory.read` | `S,z,q` | `c:(B,d_v)` | real output |
| `cleanup.settle` | `c,M_key,M_val,β` | `r*:(B,d_v)`, `p:(B,P)` | safe-β limit `2/σmax(M_key)²` exposed via `safe_beta_limit()` and logged in telemetry (`codebook_health`); **not** enforced at runtime — operating β validated by the fidelity test |
| `block.forward` | `x:(B,T,d_model)` | `x'':(B,T,d_model)` | bus norm bounded |
| `model.forward` | tokens `(B,T)` | logits `(B,T,K)` | causal (no future leak) |

---

## 12. What "the architecture is done" means (per the project rules)

The forward pass is complete when: (1) `algebra`/`memory`/`cleanup`/`block`/`model`
implement §3–§7 exactly with the §11 assertions passing; (2) associative recall
reproduces whitepaper D.1 (100% recall, `05`/S2); (3) the read fixed point converges in
the safe regime (`01` §6.2, `test_fixed_point`); (4) a randomly-initialized model runs
end-to-end on a TinyStories batch producing a valid causal next-token distribution with
no NaN/overflow at `T=512`. Only then does `03`'s learning rule attach.
