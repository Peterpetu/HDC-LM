"""Recurrent generation tests (docs/04 §7, docs/06 §3).

The O(1)-state recurrent sampler must produce the SAME next-token logits as the batched
causal forward pass (otherwise generation is inconsistent with training). Also checks the
generator runs, is deterministic given a seed, and extends the prompt.
"""

import torch

from wavenn.generate import generate, _BlockState, _step
from wavenn.rlm import RLMConfig, RippleLM


def _tiny():
    cfg = RLMConfig(vocab_size=64, d_model=48, n_layers=2, n_heads=4, P=96, beta=1.0,
                    beta_readout=6.0, n_settle=20, key_full=False, delta_rule=False)
    return RippleLM(cfg), cfg


def test_recurrent_matches_batched_forward():
    torch.manual_seed(0)
    model, cfg = _tiny()
    model.eval()
    tokens = torch.randint(0, 64, (1, 12))
    # batched causal forward: logits at each position
    with torch.no_grad():
        batched = model.forward_blocks(tokens)[-1]  # (1, T, K)
    # recurrent: feed tokens one at a time, collect next-token logits per position
    H, d_k = cfg.n_heads, model.d_k
    d_vh = model.embed.weight.shape[1] // H
    states = [_BlockState(1, H, d_k, d_vh, tokens.device) for _ in range(cfg.n_layers)]
    for t in range(tokens.shape[1]):
        lg = _step(model, tokens[:, t], t, states)  # (1, K) — logits to predict token t+1
        assert torch.allclose(lg, batched[:, t], atol=1e-4), (
            f"pos {t}: recurrent vs batched max diff {(lg - batched[:, t]).abs().max():.2e}"
        )


def test_generate_extends_and_is_deterministic():
    torch.manual_seed(0)
    model, _ = _tiny()
    prompt = torch.randint(0, 64, (1, 4))
    a = generate(model, prompt, 10, temperature=0.8, top_k=20, seed=123)
    b = generate(model, prompt, 10, temperature=0.8, top_k=20, seed=123)
    assert a.shape == (1, 14)
    assert torch.equal(a, b)  # deterministic given seed
    assert torch.equal(a[:, :4], prompt)  # prompt preserved


def test_generate_greedy_runs():
    torch.manual_seed(0)
    model, _ = _tiny()
    out = generate(model, torch.randint(0, 64, (2, 3)), 8, greedy=True)
    assert out.shape == (2, 11)
    assert torch.isfinite(out.float()).all()
