"""S4 driver — RIPPLE (L0) vs BPTT twin on TinyStories (docs/05 P2/P3), fully logged.

Runs the faithful configuration validated by the audit + tests/test_training_fidelity.py:
pure-accumulation heads (exact traces), exact key trace, decoupled settle/readout
temperatures, converged settle. Trains both learners, writes telemetry to runs/, and
produces a comparison + HTML reports via analyze_run.py.

Usage:  python experiments/run_s4.py [n_layers] [steps]
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavenn.rlm import RLMConfig  # noqa: E402
from wavenn.train import TrainConfig, train_bptt, train_ripple  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 600

    mcfg = RLMConfig(
        vocab_size=8192, d_model=128, n_layers=n_layers, n_heads=4,
        beta=1.0, beta_readout=6.0, beta_n=0.05, n_settle=20,
        key_full=False, delta_rule=False, deep_supervision=True,  # fast S-A0 key (cos>0.9)
    )
    tcfg = TrainConfig(
        steps=steps, batch=24, seq_len=64, lr=2e-3, warmup=30, eval_every=25,
        eval_iters=20, diag_every=100, seed=0,
    )
    tag = f"L{n_layers}"
    print("=== S4 faithful config ===")
    print("model:", mcfg)
    print("train:", tcfg, flush=True)

    # Train the REAL model (RIPPLE, backprop-free) FIRST so its telemetry leads the log,
    # then the BPTT twin baseline.
    print(f"\n=== RIPPLE L0 ({tag}, backprop-free — THE REAL MODEL) ===", flush=True)
    _, log_r = train_ripple(mcfg, tcfg, run_name=f"ripple_{tag}")
    print(f"\n=== BPTT twin ({tag}, autograd baseline) ===", flush=True)
    _, log_b = train_bptt(mcfg, tcfg, run_name=f"bptt_{tag}")

    # summaries
    def final_val(logdir):
        with open(os.path.join(logdir, "summary.json"), encoding="utf-8") as f:
            return json.load(f)["final"]["val_nll"]

    vb = final_val(log_b.dir)
    vr = final_val(log_r.dir)
    gap = (vr - vb) / vb
    result = {
        "n_layers": n_layers, "steps": steps,
        "bptt_final_val_nll": vb, "ripple_final_val_nll": vr,
        "gap_fraction": gap, "P2_within_20pct": bool(gap <= 0.20),
        "runs": {"bptt": log_b.dir, "ripple": log_r.dir},
    }
    with open(os.path.join(_HERE, f"S4-results-{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\n=== SUMMARY ===")
    print(f"BPTT   final val NLL: {vb:.4f} (bpt {vb/0.6931:.4f})")
    print(f"RIPPLE final val NLL: {vr:.4f} (bpt {vr/0.6931:.4f})")
    print(f"gap: {gap*100:+.1f}%   P2 (<=20%): {result['P2_within_20pct']}")

    # analysis artifacts
    try:
        from analyze_run import main as analyze
        analyze([log_b.dir, log_r.dir])
    except Exception as e:  # analysis is best-effort; training result already saved
        print(f"[analyze] skipped: {e}")


if __name__ == "__main__":
    main()
