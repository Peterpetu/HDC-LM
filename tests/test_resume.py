"""S6 checkpoint/resume certificate (plan A.4): 20-step split-run reproducibility.

A straight 20-step run must equal a 10-step run → full checkpoint (model incl. Cbar,
LocalAdam moments, RNG states, data cursor) → FRESH objects → restore → 10 more
steps, with BIT-IDENTICAL final parameters.

Determinism scope (documented, docs/10): this test runs on CPU, where every op in
the pipeline (incl. embed.index_add_) is deterministic — certifying that the
checkpoint captures the COMPLETE training state. On CUDA, index_add_ uses
non-deterministic atomics unless torch.use_deterministic_algorithms(True); the
production A.4 re-check at the tuned config compares under that flag.

The data cursor is exercised through the real ``PileTrainBatcher`` over a synthetic
reference-format int32 memmap cache built in tmp_path.
"""

import json

import pytest
import torch

from wavenn.pile_data import PileMemmap, PileTrainBatcher
from wavenn.ripple import LocalAdam
from wavenn.rlm import RLMConfig, RippleLM, ripple_grads_l1_fast
from wavenn.train import accumulate_step, load_s6_checkpoint, save_s6_checkpoint


def _make_cache(tmp_path, n_tokens=6000, vocab=20, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, vocab, size=n_tokens, dtype=np.int32)
    p = tmp_path / "toy_tokens.bin"
    arr.tofile(p)
    with open(str(p) + ".meta.json", "w") as f:
        json.dump({"length": n_tokens, "dtype": "int32"}, f)
    return str(p)


def _cfg():
    return RLMConfig(
        vocab_size=20, d_model=12, n_layers=2, n_heads=3, P=24, beta=0.4,
        beta_readout=3.0, beta_n=0.02, n_settle=8, eta=0.3, key_full=True,
        deep_supervision=True, depth_mode="L1", scan_Lc=4, row_chunk=7,
        cbar_init=3.0,
    )


def _fresh(bin_path, seed=42):
    torch.manual_seed(0)
    model = RippleLM(_cfg())
    opt = LocalAdam(dict(model.named_parameters()), lr=1e-3)
    store = PileMemmap(bin_path)
    batcher = PileTrainBatcher(store, B=2, seq_len=16, device="cpu", seed=seed)
    return model, opt, batcher


def _step(model, opt, batcher):
    micros = [batcher.batch() for _ in range(2)]  # accum 2 exercises the wrapper
    accumulate_step(model, ripple_grads_l1_fast, opt, micros, clip_norm=1.0)


def test_split_run_bit_identical(tmp_path):
    bin_path = _make_cache(tmp_path)

    # straight 20 steps
    m_a, o_a, b_a = _fresh(bin_path)
    for _ in range(20):
        _step(m_a, o_a, b_a)

    # 10 steps → checkpoint → fresh objects → restore → 10 more
    m_b, o_b, b_b = _fresh(bin_path)
    for _ in range(10):
        _step(m_b, o_b, b_b)
    ck = str(tmp_path / "ck.pt")
    save_s6_checkpoint(ck, step=10, model=m_b, opt=o_b, batcher=b_b, min_free_gb=0.0)

    m_c, o_c, b_c = _fresh(bin_path, seed=7)  # deliberately different init state
    for _ in range(3):
        _step(m_c, o_c, b_c)  # drift the fresh objects; restore must overwrite it all
    load_s6_checkpoint(ck, model=m_c, opt=o_c, batcher=b_c)
    for _ in range(10):
        _step(m_c, o_c, b_c)

    for (n1, p1), (n2, p2) in zip(m_a.named_parameters(), m_c.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"{n1}: split-run differs from straight run"
    assert torch.equal(m_a.Cbar, m_c.Cbar)
    assert b_a.batches_served == b_c.batches_served
    assert torch.equal(b_a.gen.get_state(), b_c.gen.get_state())
    for k in o_a.m:
        assert torch.equal(o_a.m[k], o_c.m[k]) and torch.equal(o_a.v[k], o_c.v[k])


def test_checkpoint_preflight_disk_floor(tmp_path):
    bin_path = _make_cache(tmp_path)
    m, o, b = _fresh(bin_path)
    with pytest.raises(RuntimeError, match="pre-flight"):
        save_s6_checkpoint(str(tmp_path / "x.pt"), step=0, model=m, opt=o,
                           batcher=b, min_free_gb=10_000_000.0)
