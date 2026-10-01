# S8 / R1 — L1 stack-EP closes the depth-credit gap: 2-layer MQAR recall recovered

**Date:** 2026-07-15. Follow-on to `docs/07-architectural-analysis.md` R1. The question:
*can a backprop-free rule route inter-block credit well enough to recover the 2-layer
in-context-recall capability that the pure-local L0 rule lacked?* Answer: **yes** — RIPPLE-L1
reaches MQAR recall **0.998**, matching its BPTT twin (0.996) and a transformer (≈0.995), up
from the L0 floor of ~0.26. Along the way the exactness check corrected docs/07's single-cause
diagnosis and fixed a latent L0 gradient bug.

---

## 1. What was built (`wavenn/ripple.py`, `wavenn/rlm.py`, docs/03 §7)

**L1-exact** — inter-block credit via the residual-bus adjoint + EP-within-block:

- The additive bus makes the exact inter-block cotangent trivial to route *down*:
  `∂L/∂r_ℓ = ∂L/∂bus_{ℓ+1}`, accumulated top→bottom (`ripple_grads_l1`).
- Each block runs its EP settle with that top-down cotangent injected as the linear-cost
  force `−g_top` (`stack_read_ep`, `cleanup.settle`, derivation in docs/03 §7), so
  `δ_ℓ=(r_ℓ⁺−r_ℓ⁻)/2β_n` estimates `−∂C_total/∂c_ℓ` — carrying the credit of every block above.
- The input pullback is pushed onto the bus by an **exact reverse-time input trace**
  (`contextual_input_grad`: `Q_t=conj(q_t)⊗g_t/den_t+γQ_{t+1}`; value `Re(Σ_j k_t[j]Q_t[j,·])`,
  key `−Im(k_t⊙(Q_t·v_t))`), needed because the cheap current-step `g_a` drops the temporal
  cross-terms that carry the recall credit. Validated vs autograd to **rel-err 1.8e-16**.

No autograd over parameters (C2). It uses inter-block transposes for routing, so it is the
**exact-credit reference** (not the transport-free rule — that is L1-VF, docs/03 §7, next).

**Gradient fidelity (`tests/test_stack_ep_fidelity.py`).** Against the exact 2-layer BPTT
gradient (key_full=True), L1 reaches **cos = 1.0000 on every parameter**, including the lower
block's codebook `M.0` (L0: 0.24) and, in the pure-top-down / final-only case, params where L0
is *negatively* correlated. The L0 depth-credit gap is closed exactly.

**Latent L0 bug found + fixed (`tests/test_tied_readout_grad.py`).** The exactness check
exposed the tied-readout→embedding Jacobian using the wrong normalization axis
(`_unit_cols_vjp(..., dim=1)` vs the forward's `dim=0`): a ~55% error on the embedding's
readout gradient (cos 0.89 vs autograd), invisible to the free-`V` fidelity test. Fixed in both
`ripple_grads` (L0) and `ripple_grads_l1`; L0's embedding cosine improved as a result.

---

## 2. Decisive experiment — 2-layer MQAR (`mqar_l1.py`, lr=1e-3, 3000 steps, 2 seeds)

Same `RippleLM` (d_model=128, 4 heads, 2 layers, β=1.0, β_ro=6.0, n_settle=15), masked
query-only objective, keys/values in disjoint 64-token bands. Recall = argmax accuracy at query
positions (chance 1/64 ≈ 0.016).

### The 2×2 (rule × key-path) factorial, N=4

| | key_full=False (cheap key) | key_full=True (exact key) |
|---|---|---|
| **RIPPLE-L0** (pure local) | 0.242 | 0.261 |
| **RIPPLE-L1** (stack-EP) | 0.262 | **0.998** |
| **BPTT** twin | ≈0.996 | 0.996 |

*(BPTT is autograd — key_full does not apply; ≈1.0 either way.)*

**N=8, key_full=True** (harder — more binding interference): **L0 = 0.142, L1 = 0.999, BPTT =
0.998.** The gap widens (L0 drops from 0.26 to 0.14 as N grows) and L1 still tracks BPTT exactly
— the recovery is robust, not an N=4 artifact.

### Reading

Recovery requires **both** fixes; neither alone suffices — a clean double dissociation:

- **Depth-credit is necessary.** L0 stays at ~0.26 *even with the exact key gradient*
  (key_full=True → 0.261). Deep-supervision credit alone cannot build the 2-layer induction
  head: block 0 is trained to predict the next token *directly*, not to *support* block 1.
- **The exact key gradient is necessary.** L1 stays at 0.26 with the cheap key path
  (key_full=False → 0.262). The current-step (S-A0) key attribution caps `∂L/∂W_k` at cos ≈
  0.84–0.88 (measured), and the key projection *is* the induction match — an approximate key
  gradient cannot learn it, no matter how credit is routed.
- **Both together → the ceiling.** RIPPLE-L1 + key_full=True = **0.998**, statistically equal
  to its BPTT twin (0.996) and a transformer (≈0.995, docs/07). A fully backprop-free rule
  reaches the in-context-recall ceiling at 2 layers.

---

## 3. Correction to `docs/07`

docs/07 attributed the weak 2-layer recall to "the L0 depth-credit gap." That is **correct but
incomplete**: the RIPPLE-L0 runs there used `key_full=False`, so the documented 0.24 conflated
*two* deficiencies — the depth-credit gap **and** the cheap-key gradient approximation. The
factorial isolates them: both are real, both bind, and **L1 (stack-EP) + the exact key path
closes both to the transformer ceiling**. The causal claim of docs/07 R1 — that closing
depth-credit recovers the capability — is confirmed, with the added finding that the exact key
gradient is a co-requisite.

---

## 4. Scope, limitations, next

- **Verified:** the update *is* the (near-)exact gradient (cos→1.0, key_full=True), and it
  *trains* to the BPTT recall ceiling. C2 holds (no autograd over parameters).
- **Cost.** L1 is noticeably slower per step than L0 (observed): the exact reverse-time input
  trace and the per-block backward sweep add work, and the trace's sequence-length Python loops
  are unvectorized. Exactness over speed for the reference; a chunked/vectorized trace is an
  engineering follow-up (S6). A precise throughput ratio is not yet measured.
- **Transport.** L1-exact uses inter-block transposes, so it forgoes EP's "no weight transport"
  property between blocks (within blocks it is preserved). The transport-free **L1-VF**
  (Vector-Field EP, Scellier et al. 2018, + symmetric ±β nudging, Laborieux et al. 2021) is the
  next experiment; its asymmetry bias will be *measured* against this L1-exact reference
  (per-block `cos(Δ_VF, Δ_BPTT)`), per docs/03 §7.
- **Re-audit at scale.** With key_full's importance now clear, the S4/S5 language-modeling
  comparisons (P2/P4) should be re-checked at key_full=True (they used the cheap key path).

---

## 5. Provenance & reproduction

- Fidelity + bug fix: `tests/test_stack_ep_fidelity.py`, `tests/test_tied_readout_grad.py`
  (`pytest -q` → 71 green). Exact input trace vs autograd: rel-err 1.8e-16.
- Recall: `python experiments/mqar_l1.py --N 4 --seeds 0 1 --steps 3000 --lr 1e-3 [--key_full]`;
  the `--key_full` toggle reproduces both rows of the factorial (writes
  `S8-L1-mqar-kf{0,1}.json`). Gradient-cosine-under-training-config diagnostic in the same run
  family.
- L1 as a training mode: `RLMConfig(depth_mode="L1")` → `wavenn.train.train_ripple`.
