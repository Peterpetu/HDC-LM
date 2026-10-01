"""S6 Phase A.6 — throughput/memory profile of the fast L1 path at the target config.

Measures the per-micro-batch step time of ``ripple_grads_l1_fast`` at the S6 headline
configuration (docs/10 §4: d=768, L=12, H=12, P=11,116, vocab 50,304, T=1024,
micro-batch 8) and projects the optimizer-step throughput and the 3,815-step run
wall-clock. Gate A (docs/10 / plan §5.A): proceed bar >= 600 tok/s.

Usage: python experiments/profile_s6.py [--iters 3] [--batch 8] [--Lc 128]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavenn.rlm import RLMConfig, RippleLM, ripple_grads_l1_fast  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--Lc", type=int, default=128)
    ap.add_argument("--row_chunk", type=int, default=2048)
    ap.add_argument("--n_settle", type=int, default=15)
    ap.add_argument("--settle_tol", type=float, default=1e-4)
    ap.add_argument("--P", type=int, default=11116)
    ap.add_argument("--tag", type=str, default="a6")
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = RLMConfig(
        vocab_size=50304, d_model=768, n_layers=12, n_heads=12, P=args.P,
        beta=1.0, beta_readout=6.0, beta_n=0.05, n_settle=args.n_settle, eta=0.3,
        key_full=True, deep_supervision=True, depth_mode="L1", gamma_max=0.999,
        cbar_init=10.826, scan_Lc=args.Lc, row_chunk=args.row_chunk,
        frozen_force=True, settle_tol=args.settle_tol,
    )
    m = RippleLM(cfg).cuda()
    n_par = sum(p.numel() for p in m.parameters())
    print(f"params: {n_par:,}  (target 162,321,552)")

    x = torch.randint(0, 50304, (args.batch, args.T), device="cuda")
    y = torch.cat(
        [x[:, 1:], torch.full((args.batch, 1), -100, device="cuda", dtype=torch.long)], 1
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    nll, _g = ripple_grads_l1_fast(m, x, y, clip_norm=1.0)
    torch.cuda.synchronize()
    print(f"warmup: {time.time() - t0:.2f}s  nll={nll:.3f}")

    ts = []
    for _ in range(args.iters):
        t0 = time.time()
        nll, _g = ripple_grads_l1_fast(m, x, y, clip_norm=1.0)
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    tok = args.batch * args.T
    best, mean = min(ts), sum(ts) / len(ts)
    peak = torch.cuda.max_memory_allocated() / 2**30
    tps = tok / mean
    run_h = 3815 * 8 * mean / 3600
    print(f"micro-step: best {best:.3f}s mean {mean:.3f}s -> {tps:,.0f} tok/s")
    print(f"peak VRAM: {peak:.2f} GB")
    print(f"projected 3,815-step (x8 accum) run: {run_h:.1f} h")
    print(f"Gate-A bar (>=600 tok/s): {'PASS' if tps >= 600 else 'BELOW'}")
    out = {
        "config": {"batch": args.batch, "T": args.T, "Lc": args.Lc,
                   "row_chunk": args.row_chunk, "n_settle": args.n_settle,
                   "settle_tol": args.settle_tol, "P": args.P},
        "params": n_par, "step_times_s": ts, "tok_per_s": tps,
        "peak_vram_gb": peak, "projected_run_h": run_h,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }
    path = os.path.join(_HERE, f"S6-profile-{args.tag}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
