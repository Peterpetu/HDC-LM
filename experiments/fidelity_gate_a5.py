"""S6 Phase A.5 — the fidelity mini-gate (docs/10 §6; plan §5.A / [M1]).

Measures, against the autograd BPTT twin (the independent ground truth), the
per-parameter-group gradient cosines of:
  * LEGACY   — ``ripple_grads_l1`` (fixed n_settle, live readout force): the
               committed, S8-validated rule (its cosines are the baseline);
  * FAST     — ``ripple_grads_l1_fast`` (frozen readout force §M.4 + the free-phase
               per-row iteration schedule §M.4a + chunked kernels §M.1/M.2).

GATE (pre-registered): every core-parameter-group cosine of FAST must be within 1%
of LEGACY's. This isolates the ONLY two approximations the fast path adds — the
kernels themselves are exact (tests/test_chunked_scan.py ≤1e-9) and the fast
assembly in legacy-faithful mode ≡ legacy (tests/test_settle_rowchunk.py ≤1e-9).

Config: an S6-like scaled twin — P = 14.5·d codebook, γ_max = 0.999, β_n = 0.05,
n_settle 15 — small enough for the autograd twin (float64, T spanning many chunks).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavenn.rlm import (  # noqa: E402
    RLMConfig, RippleLM, ripple_grads_l1, ripple_grads_l1_fast,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def masked_deepsup_loss(model, tok, tgt):
    K = model.cfg.vocab_size
    bl = model.forward_blocks(tok)
    return sum(
        F.cross_entropy(lg.reshape(-1, K), tgt.reshape(-1), ignore_index=-100)
        for lg in bl
    ) / len(bl)


def bptt_grads(model, tok, tgt):
    ps = [p for _, p in model.named_parameters()]
    ns = [n for n, _ in model.named_parameters()]
    for p in ps:
        p.requires_grad_(True)
    g = torch.autograd.grad(masked_deepsup_loss(model, tok, tgt), ps)
    for p in ps:
        p.requires_grad_(False)
    return dict(zip(ns, g))


def group_cos(g_rule: dict, g_ref: dict) -> dict:
    groups: dict[str, list] = {}
    for n in g_rule:
        groups.setdefault(n.split(".")[0], []).append(n)
    out = {}
    for base, names in sorted(groups.items()):
        a = torch.cat([g_rule[n].flatten() for n in names])
        b = torch.cat([g_ref[n].flatten() for n in names])
        out[base] = float(a @ b / (a.norm() * b.norm() + 1e-30))
    a = torch.cat([g_rule[n].flatten() for n in g_rule])
    b = torch.cat([g_ref[n].flatten() for n in g_rule])
    out["overall"] = float(a @ b / (a.norm() * b.norm() + 1e-30))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    P = int(round(14.5 * args.d))  # the S6 codebook ratio
    common = dict(
        vocab_size=args.vocab, d_model=args.d, n_layers=args.layers, n_heads=4,
        P=P, beta=1.0, beta_readout=6.0, beta_n=0.05, n_settle=15, eta=0.3,
        key_full=True, deep_supervision=True, depth_mode="L1", gamma_max=0.999,
        scan_Lc=64, row_chunk=1024, settle_tol=1e-4,
    )

    torch.manual_seed(args.seed)
    tok = torch.randint(0, args.vocab, (args.B, args.T), device=DEV)
    tgt = torch.cat([tok[:, 1:],
                     torch.full((args.B, 1), -100, device=DEV, dtype=torch.long)], 1)

    results = {}
    g_ref = None
    for name, over in (
        ("legacy", dict(fixed_schedule=True, frozen_force=False)),
        ("fast", dict(fixed_schedule=False, frozen_force=True)),
    ):
        torch.manual_seed(args.seed)
        m = RippleLM(RLMConfig(**{**common, **over})).double().to(DEV)
        if g_ref is None:  # same init both times (same seed) — compute BPTT once
            g_ref = bptt_grads(m, tok, tgt)
        if name == "legacy":
            _, g = ripple_grads_l1(m, tok, tgt, ignore_index=-100, clip_norm=None)
        else:
            _, g = ripple_grads_l1_fast(m, tok, tgt, ignore_index=-100, clip_norm=None)
        results[name] = group_cos(g, g_ref)
        print(f"[{name:6s}] " + "  ".join(f"{k}={v:+.5f}" for k, v in results[name].items()),
              flush=True)

    gate_ok = True
    deltas = {}
    for k in results["legacy"]:
        base, fast = results["legacy"][k], results["fast"][k]
        deltas[k] = fast - base
        if fast < base - 0.01:  # the pre-registered 1% band
            gate_ok = False
    print(f"\nA.5 fidelity mini-gate: {'PASS' if gate_ok else 'FAIL'}")
    print("  deltas (fast − legacy): " +
          "  ".join(f"{k}={v:+.5f}" for k, v in deltas.items()))
    out = {"config": {**common, "B": args.B, "T": args.T, "seed": args.seed},
           "cos_vs_bptt": results, "deltas": deltas, "gate_pass": gate_ok,
           "env": {"torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}}
    path = os.path.join(_HERE, "S6-a5-fidelity.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
