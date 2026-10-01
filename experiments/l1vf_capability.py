"""Capability-level demonstration of the L1-VF obstruction (Stage 2, S8-L1VF-result §4).

Trains, on final-only MQAR (block-0 gets NO local readout — all its learning is routed credit):
  - L1FA          : transport-free acyclic routing (Stage 1)          -> expected ceiling
  - L1VF-acyclic  : ripple_grads_l1vf, vf_rerun=False (== L1-FA)       -> expected == L1FA
  - L1VF-cyclic   : ripple_grads_l1vf, vf_rerun=True (genuine relax.)  -> expected FAIL (obstruction)
  - BPTT          : autograd twin                                       -> ceiling

Purpose: convert the gradient-level obstruction (docs/09 §6) into a recall-level statement — the
genuine cyclic relaxation cannot train the induction head, while the acyclic rule = L1-FA recovers.
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
from wavenn.rlm import (  # noqa: E402
    RLMConfig, RippleLM, ripple_grads_l1fa, ripple_grads_l1vf,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_HERE = os.path.dirname(os.path.abspath(__file__))


def mk_cfg(rule, n_outer):
    return RLMConfig(
        vocab_size=128, d_model=128, n_heads=4, n_layers=2, beta=1.0, beta_readout=6.0,
        key_full=True, n_settle=15, delta_rule=False, deep_supervision=False,  # final-only
        feedback_lambda=1.0, feedback_gain=1.0, vf_variant="VF-ST-adjoint",
        vf_rerun=(rule == "L1VFc"), vf_n_outer=n_outer, vf_relax=0.5, vf_tol=1e-3,
    )


def train(rule, *, n_pairs, n_keys, n_vals, steps, lr, seed, B, n_outer):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    model = RippleLM(mk_cfg(rule, n_outer)).to(DEV)
    nq = max(4, n_pairs // 2)
    K = 128
    if rule == "BPTT":
        opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    else:
        opt = LocalAdam(dict(model.named_parameters()), lr=lr)
    gfun = {"L1FA": ripple_grads_l1fa, "L1VFa": ripple_grads_l1vf, "L1VFc": ripple_grads_l1vf}.get(rule)
    diverged = False
    for step in range(1, steps + 1):
        cur = lr * min(1.0, step / 40)
        x, tgt = make_mqar_batch(B, n_pairs, nq, n_keys, n_vals, gen)
        if rule == "BPTT":
            for gp in opt.param_groups:
                gp["lr"] = cur
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model.forward_blocks(x)[-1].reshape(-1, K), tgt.reshape(-1),
                            ignore_index=-100).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            opt.lr = cur
            nll, g = gfun(model, x, tgt, ignore_index=-100, clip_norm=1.0)
            if not (nll == nll):  # NaN guard (cyclic divergence)
                diverged = True
                break
            opt.step(g, clip=None)
    acc = eval_recall(model, n_pairs, nq, n_keys, n_vals, gen)
    return acc, diverged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", nargs="+", default=["L1FA", "L1VFa", "L1VFc", "BPTT"])
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--n_outer", type=int, default=6)  # kept small: cyclic is expensive + diverges
    args = ap.parse_args()
    res = []
    out = os.path.join(_HERE, "S8-l1vf-capability.json")
    for rule in args.rules:
        accs, divs = [], 0
        for sd in args.seeds:
            acc, div = train(rule, n_pairs=args.N, n_keys=64, n_vals=64, steps=args.steps,
                             lr=args.lr, seed=sd, B=args.batch, n_outer=args.n_outer)
            accs.append(acc); divs += int(div)
            print(f"[{rule:6s} N={args.N} s{sd}] recall={acc:.3f}{' (diverged)' if div else ''}", flush=True)
        res.append({"rule": rule, "N": args.N, "mean": sum(accs) / len(accs), "accs": accs,
                    "diverged": divs})
        json.dump(res, open(out, "w"), indent=2)
    print("\n=== L1-VF capability (final-only MQAR, N={}) ===".format(args.N))
    for r in res:
        print(f"  {r['rule']:6s} recall={r['mean']:.3f}  {r['accs']}" +
              (f"  [{r['diverged']} diverged]" if r['diverged'] else ""))


if __name__ == "__main__":
    main()
