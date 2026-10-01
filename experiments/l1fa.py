"""Stage 1 — L1-FA transport-free depth-credit: dose-response + controls (plan §1f-1h).

Tests whether depth credit survives removing weight transport, and at what cost, measured
against the exact L1 reference. L1-FA reuses the entire L1-exact scaffold and replaces only the
inter-block projection transpose with fixed random feedback (feedback alignment). This script:

- runs L0 / L1(exact) / L1FA(λ) / BPTT at 2 layers, key_full=True, on MQAR;
- **final-only** (deep_supervision=False) isolates ROUTED inter-block credit (deep-sup lets the
  local readout do block-0's work — review F3); both modes supported;
- the **λ dose-response** (B_eff=(1-λ)W+λB): λ=0≡L1-exact, λ=1≡transport-free (review C2);
- an **N-sweep** (partial FA recovery is most visible at higher interference — review P4);
- **learning curves + alignment**: FA converges only after it aligns, so a null at a short budget
  is undertraining not a kill (review F4). Logs (i) update-cos cos(Δ_rule, Δ_BPTT) over block-0
  projections and (ii) weight-alignment cos(Wk.0, Bk) — if update-cos rises while weight-alignment
  is flat, recovery is NOT feedback alignment (top-block compensation — review C4);
- controls: **--freeze_block0** (C3 floor), and λ<1 is the dose-response knob.

Usage:
  python experiments/l1fa.py --rules L0 L1 L1FA BPTT --lam 1.0 --N 4 8 --final_only \
      --seeds 0 1 2 --steps 20000 --lr 1e-3 --log_every 500
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
    RLMConfig, RippleLM, ripple_grads, ripple_grads_l1, ripple_grads_l1fa,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_HERE = os.path.dirname(os.path.abspath(__file__))
GFUN = {"L0": ripple_grads, "L1": ripple_grads_l1, "L1FA": ripple_grads_l1fa}
BLOCK0 = ["Wk.0", "Wq.0", "Wv.0"]  # the routed projections whose credit L1-FA must recover


def _masked_loss(model, x, tgt, deep_sup):
    K = model.cfg.vocab_size
    bl = model.forward_blocks(x)
    if deep_sup:
        return sum(F.cross_entropy(l.reshape(-1, K), tgt.reshape(-1), ignore_index=-100)
                   for l in bl) / len(bl)
    return F.cross_entropy(bl[-1].reshape(-1, K), tgt.reshape(-1), ignore_index=-100)


def _bptt_grads(model, x, tgt, deep_sup):
    ps = [p for _, p in model.named_parameters()]
    ns = [n for n, _ in model.named_parameters()]
    prev = [p.requires_grad for p in ps]  # restore afterwards (BPTT training needs grad)
    for p in ps:
        p.requires_grad_(True)
    g = torch.autograd.grad(_masked_loss(model, x, tgt, deep_sup), ps)
    for p, r in zip(ps, prev):
        p.requires_grad_(r)
    return dict(zip(ns, g))


def _cos(a, b):
    return (a.flatten() @ b.flatten() / (a.norm() * b.norm() + 1e-30)).item()


def _alignment(model, x, tgt, deep_sup, rule):
    """(update-cos over block-0 projections vs BPTT, weight-alignment cos(Wk.0,Bk)).

    Not under no_grad: the BPTT reference needs autograd (_bptt_grads restores requires_grad).
    """
    bptt = _bptt_grads(model, x, tgt, deep_sup)
    with torch.no_grad():
        if rule == "BPTT":
            upd = 1.0
        else:
            _, g = GFUN[rule](model, x, tgt, ignore_index=-100, clip_norm=None)
            ca = torch.cat([g[n].flatten() for n in BLOCK0])
            cb = torch.cat([bptt[n].flatten() for n in BLOCK0])
            upd = _cos(ca, cb)
        walign = _cos(model.Wk[0], model.Bk[0])  # FA mechanism: does W rotate toward B?
    return upd, walign


def train(rule, mcfg, *, n_pairs, n_keys, n_vals, steps, lr, seed, B, log_every, freeze_block0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    model = RippleLM(mcfg).to(DEV)
    nq = max(4, n_pairs // 2)
    deep_sup = mcfg.deep_supervision
    frozen = {f"{p}.0" for p in ("Wk", "Wq", "Wv", "M", "log_gamma", "norm_gain")} if freeze_block0 else set()

    if rule == "BPTT":
        params = [p for n, p in model.named_parameters() if n not in frozen]
        opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99))
    else:
        pdict = {n: p for n, p in model.named_parameters() if n not in frozen}
        opt = LocalAdam(pdict, lr=lr)

    probe = make_mqar_batch(B, n_pairs, nq, n_keys, n_vals, gen)  # fixed probe for alignment
    curve = []
    for step in range(1, steps + 1):
        cur = lr * min(1.0, step / 40)
        x, tgt = make_mqar_batch(B, n_pairs, nq, n_keys, n_vals, gen)
        if rule == "BPTT":
            for gp in opt.param_groups:
                gp["lr"] = cur
            opt.zero_grad(set_to_none=True)
            _masked_loss(model, x, tgt, deep_sup).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        else:
            opt.lr = cur
            _, g = GFUN[rule](model, x, tgt, ignore_index=-100, clip_norm=1.0)
            opt.step({k: v for k, v in g.items() if k not in frozen}, clip=None)
        if log_every and (step % log_every == 0 or step == steps):
            rec = eval_recall(model, n_pairs, nq, n_keys, n_vals, gen, iters=8, B=64)
            upd, wal = _alignment(model, probe[0], probe[1], deep_sup, rule)
            curve.append({"step": step, "recall": rec, "update_cos": upd, "weight_align": wal})
            print(f"    [{rule:5s} λ={mcfg.feedback_lambda} N={n_pairs} s{seed} step {step}] "
                  f"recall={rec:.3f} updcos={upd:+.3f} walign={wal:+.3f}", flush=True)
    final = eval_recall(model, n_pairs, nq, n_keys, n_vals, gen)
    return final, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", type=str, nargs="+", default=["L0", "L1", "L1FA", "BPTT"])
    ap.add_argument("--lam", type=float, nargs="+", default=[1.0])  # dose-response (L1FA only)
    ap.add_argument("--N", type=int, nargs="+", default=[4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--feedback_seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--final_only", action="store_true", help="deep_supervision=False (isolate routing)")
    ap.add_argument("--freeze_block0", action="store_true")
    ap.add_argument("--n_layers", type=int, default=2, help="depth (review Q1: L=3-4 finds the FA break)")
    ap.add_argument("--tag", type=str, default="l1fa")
    args = ap.parse_args()

    n_keys, n_vals = 64, 64
    vocab = n_keys + n_vals
    out = os.path.join(_HERE, f"S8-{args.tag}-fo{int(args.final_only)}.json")
    res = []

    def mk(rule, lam, fbseed):
        return RLMConfig(
            vocab_size=vocab, d_model=128, n_heads=4, n_layers=args.n_layers, beta=1.0, beta_readout=6.0,
            key_full=True, n_settle=15, delta_rule=False, deep_supervision=not args.final_only,
            depth_mode=rule if rule in ("L1", "L1FA") else "L0",
            feedback_lambda=lam, feedback_seed=fbseed,
        )

    for N in args.N:
        for rule in args.rules:
            # λ and feedback-seed only vary L1FA; other rules run once per (N, seed)
            lams = args.lam if rule == "L1FA" else [0.0]
            fbs = args.feedback_seeds if rule == "L1FA" else [1234]
            for lam in lams:
                for fbseed in fbs:
                    accs, curves = [], []
                    for sd in args.seeds:
                        acc, curve = train(rule, mk(rule, lam, fbseed), n_pairs=N, n_keys=n_keys,
                                           n_vals=n_vals, steps=args.steps, lr=args.lr, seed=sd,
                                           B=args.batch, log_every=args.log_every,
                                           freeze_block0=args.freeze_block0)
                        accs.append(acc)
                        curves.append(curve)
                        print(f"[N={N} {rule} λ={lam} fb={fbseed} s{sd}] final recall={acc:.3f}", flush=True)
                    res.append({"N": N, "rule": rule, "lam": lam, "feedback_seed": fbseed,
                                "final_only": args.final_only, "freeze_block0": args.freeze_block0,
                                "mean": sum(accs) / len(accs), "accs": accs, "curves": curves})
                    json.dump(res, open(out, "w"), indent=2)
    print(f"\n=== L1-FA MQAR (final_only={args.final_only}, freeze_block0={args.freeze_block0}) ===")
    for r in res:
        lab = f"{r['rule']}" + (f"(λ={r['lam']})" if r["rule"] == "L1FA" else "")
        print(f"  N={r['N']:2d} {lab:14s} recall={r['mean']:.3f}  {r['accs']}")


if __name__ == "__main__":
    main()
