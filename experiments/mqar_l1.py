"""S8 / R1 — does L1 stack-EP recover 2-layer MQAR recall? (docs/07 R1, S8-L1-result.md)

Compares, at 2 layers, matched seeds/steps/lr, the backprop-free depth-credit modes against the
exact-gradient target:
  - RIPPLE-L0   (ripple_grads)      — pure-local, the depth-credit floor
  - RIPPLE-L1   (ripple_grads_l1)   — stack-EP, inter-block credit via the residual-bus adjoint
  - BPTT        (masked deep-sup)   — autograd, the ceiling

The ``--key_full`` toggle exposes the 2×2 (rule × key-path) factorial that decomposes the weak
recall into its two co-causes (docs/07 was single-cause): recovery needs BOTH the stack-EP
depth-credit AND the exact key gradient. Result: L1 + key_full=True → 0.998 = BPTT.

Usage:
    python experiments/mqar_l1.py --N 4 --seeds 0 1 --steps 3000 --lr 1e-3 --key_full
    python experiments/mqar_l1.py --N 4 --seeds 0 1 --steps 3000 --lr 1e-3   # cheap-key row
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.mqar import eval_recall, make_mqar_batch  # noqa: E402
from wavenn.ripple import LocalAdam  # noqa: E402
from wavenn.rlm import RLMConfig, RippleLM, ripple_grads, ripple_grads_l1  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def train(rule, mcfg, *, n_pairs, n_keys, n_vals, steps, lr, seed, B=48):
    """Train one (rule, config, seed) and return final MQAR recall.

    rule ∈ {"L0","L1","BPTT"}. L0/L1 use the backprop-free assembler + LocalAdam; BPTT uses the
    masked deep-supervision loss + torch.optim.Adam over the SAME model (the exact-gradient twin).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    model = RippleLM(mcfg).to(DEV)
    nq = max(4, n_pairs // 2)
    K = mcfg.vocab_size

    if rule == "BPTT":
        opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    else:
        opt = LocalAdam(dict(model.named_parameters()), lr=lr)
    gfun = ripple_grads if rule == "L0" else ripple_grads_l1

    for step in range(1, steps + 1):
        cur = lr * min(1.0, step / 40)  # linear warmup
        x, tgt = make_mqar_batch(B, n_pairs, nq, n_keys, n_vals, gen)
        if rule == "BPTT":
            for gp in opt.param_groups:
                gp["lr"] = cur
            opt.zero_grad(set_to_none=True)
            block_logits = model.forward_blocks(x)
            loss = sum(
                F.cross_entropy(lg.reshape(-1, K), tgt.reshape(-1), ignore_index=-100)
                for lg in block_logits
            ) / len(block_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            opt.lr = cur
            _, g = gfun(model, x, tgt, ignore_index=-100, clip_norm=1.0)
            opt.step(g, clip=None)
    return eval_recall(model, n_pairs, nq, n_keys, n_vals, gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--key_full", action="store_true", help="exact key trace (else cheap S-A0)")
    ap.add_argument("--rules", type=str, nargs="+", default=["L0", "L1", "BPTT"])
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()

    n_keys, n_vals = 64, 64
    vocab = n_keys + n_vals
    res = []
    out = os.path.join(_HERE, f"S8-L1-mqar-kf{int(args.key_full)}.json")
    for N in args.N:
        for rule in args.rules:
            accs = []
            for sd in args.seeds:
                mcfg = RLMConfig(
                    vocab_size=vocab, d_model=128, n_heads=4, n_layers=2, beta=1.0,
                    beta_readout=6.0, key_full=args.key_full, n_settle=15, delta_rule=False,
                    deep_supervision=True,
                )
                acc = train(rule, mcfg, n_pairs=N, n_keys=n_keys, n_vals=n_vals,
                            steps=args.steps, lr=args.lr, seed=sd, B=args.batch)
                accs.append(acc)
                print(f"[N={N} {rule:4s} kf={args.key_full} seed={sd}] recall={acc:.3f}", flush=True)
            res.append({"N": N, "rule": rule, "key_full": args.key_full,
                        "mean": sum(accs) / len(accs), "accs": accs})
            json.dump(res, open(out, "w"), indent=2)
    print(f"\n=== 2-layer MQAR recall (lr={args.lr}, key_full={args.key_full}) ===")
    for N in args.N:
        row = {r["rule"]: r["mean"] for r in res if r["N"] == N}
        print(f"N={N}: " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()))


if __name__ == "__main__":
    main()
