"""Online training diagnostics for the RIPPLE learner (docs/05 §5, §8).

The headline diagnostic is the **online P1 check**: periodically, on a real batch, compute
both the RIPPLE local gradient and the autograd BPTT gradient for the SAME model and report
their cosine similarity per parameter group. This tracks whether the local rule keeps
tracking the true gradient as the weights move away from initialization — a property the
static kill-gate test (tests/test_gradient_fidelity.py) only checks at init.

Autograd here is a *measurement instrument* (like a test), not part of the RIPPLE training
path, so it does not violate C2 (docs/00 §1).
"""

from __future__ import annotations

import torch

from .rlm import RippleLM, ripple_grads


@torch.no_grad()
def _flatten(grads: dict, names) -> torch.Tensor:
    return torch.cat([grads[n].reshape(-1) for n in names if grads.get(n) is not None])


def ep_bptt_cosine(model: RippleLM, x: torch.Tensor, y: torch.Tensor) -> dict:
    """Cosine(RIPPLE update, BPTT gradient) per parameter group and overall, on batch (x,y).

    Returns a dict of cosines. The RIPPLE gradient is ∂C/∂θ estimated by the local rule;
    the BPTT gradient is the autograd ∂/∂θ of the same deep-supervised loss. They should be
    strongly aligned (docs/05 P1); a decaying cosine during training would be an early
    warning that the local rule is drifting.
    """
    # RIPPLE (no autograd)
    _, rip = ripple_grads(model, x, y)
    # BPTT on an identical copy of the parameters (autograd — measurement only)
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    loss = model.bptt_loss(x, y)
    loss.backward()
    bptt = {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(False)

    def cos(a, b):
        if a is None or b is None:
            return float("nan")
        return float(a.reshape(-1) @ b.reshape(-1) / (a.norm() * b.norm() + 1e-30))

    groups: dict[str, list] = {}
    for n in rip:
        base = n.split(".")[0]
        groups.setdefault(base, []).append(n)
    out = {}
    for base, names in groups.items():
        rr = _flatten(rip, names)
        bb = _flatten({n: bptt.get(n) for n in names}, names)
        out[f"cos_{base}"] = cos(rr, bb)
    all_names = list(rip.keys())
    out["cos_overall"] = cos(_flatten(rip, all_names), _flatten({n: bptt.get(n) for n in all_names}, all_names))
    return out


@torch.no_grad()
def adaptive_compute_report(model: RippleLM, x: torch.Tensor) -> dict:
    """M4 native-adaptive-compute evidence (docs/02 §5.1, docs/06 §2): correlation between
    per-token settle iterations and read entropy (harder tokens 'think longer')."""
    from .rlm import adaptive_settle_stats

    iters, ents = adaptive_settle_stats(model, x)
    if iters.numel() < 2 or iters.std() == 0 or ents.std() == 0:
        corr = float("nan")
    else:
        corr = float(torch.corrcoef(torch.stack([iters, ents]))[0, 1])
    return {
        "settle_iters_mean": float(iters.mean()),
        "settle_iters_max": int(iters.max()),
        "read_entropy_mean": float(ents.mean()),
        "iters_entropy_corr": corr,
    }
