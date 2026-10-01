# 04 — Implementation Plan

**Role:** How `docs/00–03` become running code on one RTX 4080 SUPER (16 GB, CUDA 11.8,
PyTorch 2.7.1, Python 3.11, Windows-native). Covers module layout, the no-autograd core,
chunked-scan kernels, memory budget, precision, and the throughput engineering plan.

---

## 1. Environment (verified)

- GPU: NVIDIA RTX 4080 SUPER, 16 GB (16376 MiB). `torch.cuda.is_available() == True`.
- PyTorch `2.7.1+cu118`, Python `3.11.2`, Windows 11.
- Complex support: `torch.complex64` ops (mul, matmul, real/imag, conj) are used
  throughout; verified available on this build. Where a complex op lacks a fast CUDA
  path, the interleaved-real-pair fallback (`algebra.py`) is used.
- Determinism: fixed seeds, `torch.use_deterministic_algorithms(True)` in tests;
  training logs the seed, `torch`/CUDA versions, and full config (project rules).

---

## 2. Module layout (`wavenn/`)

| File | Contents | Depends on |
|---|---|---|
| `algebra.py` | phasor ops: `phi`, `bind`, `bundle`, `permute`/`rho`, `similarity`, `mag_norm`; complex↔real-pair. | — |
| `memory.py` | `ContextualMemory`: multi-timescale state, delta-rule write, chunked scan, read; trace maintenance hooks. | algebra |
| `cleanup.py` | `settle` (free/nudged, adaptive), `readout`, safe-`β` spectral norm. | algebra |
| `block.py` | `WNNBlock`: contextual + knowledge on the bundling bus, pre-norm. | memory, cleanup |
| `model.py` | `WNN`: embed, stack, per-block readouts, tied `V`, final logits. | block |
| `ripple.py` | RIPPLE: EP signals, two-path codebook, exact traces (`§4` of `03`), 3-factor gate, local Adam, L0/L1. | model |
| `rlm.py` | `RippleLM` trained model **and** its **BPTT twin** `bptt_loss` (autograd, identical forward); `ripple_grads` (L0). GLA / small-softmax-transformer references are separate baseline configs. The BPTT twin is *not* a separate module. | model, ripple, torch.autograd (twin only) |
| `data.py` | TinyStories + Pile pipelines, tokenizers, chunked loaders. | — |
| `train.py` | config system, seed/log, train loop (RIPPLE **or** BPTT twin), checkpointing. | all |
| `generate.py` | O(1)-state recurrent sampler. | model |
| `configs/` | YAML configs per experiment (WNN-XS/S/M/L, ablations). | — |
| `experiments/` | run scripts + negative-result writeups. | — |

**Separation of concerns for C2:** `ripple.py` never imports `torch.autograd`; the BPTT twin
— `wavenn/rlm.py::bptt_loss` (built differentiably) trained by `wavenn/train.py::train_bptt`,
where the sole `.backward()` on model params lives — is the *only* place autograd touches
model parameters. The code reviewer greps for this (`03` §10).

---

## 3. The no-autograd core

- The WNN forward pass (`model.py`) runs under `torch.no_grad()` during RIPPLE training.
- Gradients are **assembled explicitly** in `ripple.py` from `δ_t`, the traces, and the
  codebook rule. Parameters are plain `torch.nn.Parameter` with `requires_grad=False`
  during RIPPLE; updates are applied in-place via the local Adam (`03` §6).
- The **trace state** (`G_v, G_kS, G_kz, Ġ_γ`) lives on-device as fp32 tensors, reset per
  sequence, updated by the chunked scan alongside `S, z`.
- This is why 16 GB stretches far (`02` §10): no computation graph is retained. The peak
  memory is `params + moments + traces + one chunk of activations`, none scaling with `T`.

---

## 4. Chunked-scan kernels (throughput)

The correctness-first reference is a pure-Python token loop (`memory.py::scan_reference`),
used by tests. The performance path is the **chunked** form:

1. Split `T` into chunks of `L_c` (default 64).
2. **Intra-chunk:** compute contributions with a masked matmul (parallel over the chunk)
   — the standard linear-attention chunk trick (GLA, Yang et al. 2024).
3. **Inter-chunk:** carry `S, z` (and traces `G`) recurrently across chunks.
4. The **trace scan** reuses the same chunk boundaries; `G_v = γG + kxᵀ` is a linear
   recurrence identical in shape-flow to `S`, so the same kernel handles both.

Implementation order: (a) reference loop → tests green; (b) vectorized chunk in pure
PyTorch (`einsum`, `torch.cumsum`-style decayed prefix) → throughput; (c) only if
profiling demands, a Triton kernel for the decayed matmul (deferred; `06`/S6). Kernel
correctness is gated by equality with `scan_reference` to `1e-5` (`tests/`).

---

## 5. Memory budget (WNN-M, 95 M, `02` §10)

| Item | Size | Note |
|---|---|---|
| params fp32 | ≈ 380 MB | master weights |
| local-Adam moments (2×) | ≈ 760 MB | fp32 |
| traces `G_v,G_kS,G_kz,Ġ_γ` | ≈ 85 MB | `O(L·D·d_model)`, no `T` |
| states `S,z` | ≈ 85 MB | current-only |
| activations (1 chunk, bf16) | ≈ 0.2–1 GB | `B·L_c·d_model·L`, tunable via `B,L_c` |
| **total** | **≈ 1.5–2.3 GB** | leaves ≳13 GB for batch scaling |

The freed memory is spent on **large batch** (better local-gradient SNR) or **larger
model**. BPTT twins, by contrast, store `O(B·T·d_model·L)` activations and are the
memory-limited comparison point.

---

## 6. Precision policy (implements `02` §9)

- fp32: weights, moments, traces, states `S,z`, settle `r` (anything accumulating).
- bf16: transient bus activations within a chunk.
- Complex: `complex64`. Guard the read denominator `den_t = Re(q^†z)+ε` with `ε=1e-6`
  and clamp `|den_t|` away from 0 (numerical-stability requirement, the project rules).
- Log-space where relevant: the readout uses `log_softmax`; the settle uses the
  stabilized `lse` (subtract max).

---

## 7. Throughput engineering plan (`05` P7)

Targets: training tokens/sec within 3× of a matched-size GLA-BPTT baseline; recurrent
generation ≥ transformer tokens/sec at 2k+ context.

1. **Baseline profile.** Measure GLA-BPTT and WNN-RIPPLE tokens/sec at WNN-S; identify
   the bottleneck (expected: the per-step two-phase settle and the trace updates).
2. **Batch the settles.** Free/nudged± settles are three relaxations sharing a warm
   start; batch them as one `(3B)`-tensor pass. Cap iterations by the adaptive stop.
3. **Fuse the trace scan** into the chunked forward (§4) so `S`, `z`, `G` update in one
   pass.
4. **Amortize EP:** the settle count dominates; use `n_max` adaptive stop (`02` §5.1) and
   a low default `β` to converge fast (`01` §6.2 rate).
5. **Mixed precision + chunk tuning:** sweep `L_c`, `B` for the 16 GB sweet spot.
6. **Deferred:** Triton decayed-matmul kernel; only if PyTorch chunk path misses the 3×
   target.

Generation is inherently O(1)-state (no KV cache growth), so long-context sampling
throughput is a structural advantage measured directly (`05` P7).

---

## 8. Configuration & reproducibility

- YAML configs under `configs/`; a config fully determines a run (seed, arch, rule mode,
  hyperparameters, dataset, tokenizer, `β/β_n`, chunk, precision).
- `train.py` logs: config hash, seed, git-less version stamp of `docs/` + `wavenn/`,
  torch/CUDA versions, GPU name, wall-clock, tokens/sec, val curves.
- Checkpoints store weights + local-Adam moments + `Cbar` + RNG state, so training
  resumes bit-reproducibly.
- Every experiment writes a short `experiments/<name>.md` with the config, the result
  (including negative results), and the plot paths.

---

## 9. Build order (matches `05` stages)

1. `algebra.py` + `tests/test_algebra.py`, `test_capacity.py` (S1).
2. `memory.py`, `cleanup.py`, `block.py`, `model.py` + recall/fixed-point tests (S2).
3. `ripple.py` + `tests/test_gradient_fidelity.py`, `test_two_path_codebook.py`,
   `test_traces.py` — the **kill gate** (S3).
4. the BPTT twin (`rlm.py::bptt_loss` + `train.py::train_bptt`), `data.py`, `train.py` → TinyStories L0/L1 vs BPTT twin (S4).
5. throughput + Pile (S6), long-context + continual (S7), research tracks (S8).

No module is "done" until its boundary assertions (`02` §11) and its tests pass
(project rules).
