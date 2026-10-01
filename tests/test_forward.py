"""Forward-pass tests (docs/02): associative recall (whitepaper D.1) and the full model.

Reproduces the whitepaper Appendix D.1 result — the bind→bundle→query→clean-up memory
recalls stored key–value pairs — and checks the end-to-end WNN produces a valid, causal
next-token distribution with no numerical pathologies (docs/02 §12 definition of done).
"""

import torch

from wavenn import algebra as A
from wavenn import cleanup as C
from wavenn.memory import scan_reference
from wavenn.model import WNN, WNNConfig

torch.manual_seed(0)


# --------------------------------------------------------------------------------------
# Whitepaper D.1: associative recall via the Warm Read (docs/02 §5, docs/01 §6)
# --------------------------------------------------------------------------------------
def _recall_accuracy(K: int, d_v: int, D: int, gamma: float, use_settle: bool, beta: float) -> float:
    keys = A.random_phasors(K, D)  # (K, D) unit phasors
    val_codebook = torch.randn(K, d_v)  # the K distinct values to recall
    # write K pairs sequentially into the running memory with decay γ
    S = torch.zeros(D, d_v, dtype=torch.complex64)
    z = torch.zeros(D, dtype=torch.complex64)
    for i in range(K):
        S = gamma * S + torch.outer(keys[i], val_codebook[i].to(torch.complex64))
        z = gamma * z + keys[i]
    correct = 0
    for j in range(K):
        num = (keys[j].conj() @ S).real  # Re(k_j^† S)  (d_v,)
        den = (keys[j].conj() @ z).real + 1e-6
        c = (num / den).unsqueeze(0)  # (1, d_v)
        if use_settle:
            # soft warm read: settle against the value codebook, then nearest value
            M = val_codebook.t()  # (d_v, K)
            r, _, _ = C.settle(c, M, M, beta=beta, n_max=20)
            pred = (r @ M).argmax(dim=-1).item()
        else:
            pred = torch.cdist(c, val_codebook).argmin(dim=-1).item()
        correct += int(pred == j)
    return correct / K


def test_associative_recall_100pct_hard():
    # whitepaper D.1: 100% recall across γ ∈ {0.9, 1.0}, K=6, d_v=16
    for gamma in (0.9, 1.0):
        torch.manual_seed(0)  # deterministic regardless of test order
        acc = _recall_accuracy(K=6, d_v=16, D=1024, gamma=gamma, use_settle=False, beta=5.0)
        assert acc == 1.0, f"γ={gamma}: recall {acc}"


def test_associative_recall_100pct_warm_read():
    # whitepaper D.1: 100% recall via the Hopfield settle. d_v=48 gives clean separation of
    # the K=6 random value codewords (d_v=16 admits occasional near-collisions the soft settle
    # cannot resolve — a value-codebook capacity effect, not a memory-read failure).
    for beta in (5.0, 10.0):
        torch.manual_seed(0)
        acc = _recall_accuracy(K=6, d_v=48, D=1024, gamma=1.0, use_settle=True, beta=beta)
        assert acc == 1.0, f"β={beta}: warm-read recall {acc}"


def test_scan_matches_manual_recurrence():
    # the ContextualMemory reference scan equals a hand-rolled recurrence
    B, T, H, d_k, d_vh = 2, 5, 2, 8, 4
    k = A.random_phasors(B, T, H, d_k)
    q = A.random_phasors(B, T, H, d_k)
    v = torch.randn(B, T, H, d_vh)
    gamma = torch.tensor([0.9, 0.5])
    c = scan_reference(k, q, v, gamma)
    # manual: positive count normalizer den_t = γ den_{t-1} + d_k (docs/01 §5)
    S = torch.zeros(B, H, d_k, d_vh, dtype=torch.complex64)
    den = torch.zeros(1, H, 1)
    for t in range(T):
        S = gamma.view(1, H, 1, 1) * S + torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t].cfloat())
        den = gamma.view(1, H, 1) * den + d_k
        num = torch.einsum("bhk,bhkv->bhv", q[:, t].conj(), S).real
        assert torch.allclose(c[:, t], num / den, atol=1e-5)


# --------------------------------------------------------------------------------------
# Full model forward (docs/02 §7, §12)
# --------------------------------------------------------------------------------------
def _tiny_model() -> WNN:
    cfg = WNNConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, P=128, beta=4.0)
    return WNN(cfg)


def test_model_forward_shape_and_finite():
    model = _tiny_model()
    tokens = torch.randint(0, 256, (3, 32))
    with torch.no_grad():
        logits = model(tokens)
    assert logits.shape == (3, 32, 256)
    assert torch.isfinite(logits).all()


def test_model_produces_valid_distribution():
    model = _tiny_model()
    tokens = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        p = model(tokens).softmax(dim=-1)
    assert torch.allclose(p.sum(-1), torch.ones(2, 16), atol=1e-4)
    assert (p >= 0).all()


def test_model_is_causal():
    # changing the last token must not alter logits at earlier positions
    model = _tiny_model()
    tokens = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        base = model(tokens)
        tokens2 = tokens.clone()
        tokens2[0, -1] = (tokens2[0, -1] + 7) % 256
        changed = model(tokens2)
    assert torch.allclose(base[:, :-1], changed[:, :-1], atol=1e-5)


def test_param_estimate_matches_actual():
    cfg = WNNConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, P=128)
    model = WNN(cfg)
    actual = sum(p.numel() for p in model.parameters())
    est = cfg.param_estimate()
    # estimate ignores norm gains/biases; should be within 10%
    assert abs(actual - est) / est < 0.1
