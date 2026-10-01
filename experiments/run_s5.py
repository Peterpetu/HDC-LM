"""S5 driver — longer, higher-capacity RIPPLE run with periodic generation samples.

Trains the real WNN model (RIPPLE, backprop-free) FIRST with concise generation samples
logged every `gen_every` steps, then the BPTT twin for the P2-at-scale comparison.

Usage:  python experiments/run_s5.py [steps]
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavenn.rlm import RLMConfig  # noqa: E402
from wavenn.train import TrainConfig, train_bptt, train_ripple  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    mcfg = RLMConfig(
        vocab_size=8192, d_model=192, n_layers=1, n_heads=6,
        beta=1.0, beta_readout=6.0, beta_n=0.05, n_settle=20,
        key_full=False, delta_rule=False, deep_supervision=True,
    )
    tcfg = TrainConfig(
        steps=steps, batch=32, seq_len=96, lr=2e-3, warmup=40,
        eval_every=50, eval_iters=20, diag_every=500,
        gen_every=350, gen_new=55, seed=0,
    )
    print("=== S5 quality run ===")
    print("model:", mcfg)
    print("train:", tcfg, flush=True)

    print("\n=== RIPPLE (backprop-free — THE REAL MODEL), with generation samples ===", flush=True)
    _, log_r = train_ripple(mcfg, tcfg, run_name="ripple_S5")
    print("\n=== BPTT twin (baseline) ===", flush=True)
    _, log_b = train_bptt(mcfg, tcfg, run_name="bptt_S5")

    vb = json.load(open(os.path.join(log_b.dir, "summary.json")))["final"]["val_nll"]
    vr = json.load(open(os.path.join(log_r.dir, "summary.json")))["final"]["val_nll"]
    gap = (vr - vb) / vb
    result = {"steps": steps, "ripple_val_nll": vr, "bptt_val_nll": vb,
              "gap_fraction": gap, "P2_within_20pct": bool(gap <= 0.20)}
    json.dump(result, open(os.path.join(_HERE, "S5-results.json"), "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(f"RIPPLE val_nll {vr:.4f} ({vr/0.6931:.3f} bpt)   BPTT {vb:.4f}   gap {gap*100:+.1f}%")
    try:
        from analyze_run import main as analyze
        analyze([log_r.dir, log_b.dir])
    except Exception as e:
        print(f"[analyze] skipped: {e}")


if __name__ == "__main__":
    main()
