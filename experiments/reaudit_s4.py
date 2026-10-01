"""Re-audit of the S4/P2 TinyStories gap under the exact-key path and lr sweep.

S4 measured RIPPLE-L0 within ~9% of its BPTT twin, but at `key_full=False` (cheap S-A0 key)
and `lr=2e-3`. Two findings since call those into question:
  - S8 (`S8-L1-result.md`): the cheap key path caps ∂L/∂W_k at cos≈0.85 — it materially hurt
    MQAR. Does it also inflate the LM gap?
  - docs/07 §5: the lr=3e-3 MQAR pathology; re-check the LM gap at lr≤1e-3.
  - the L1 work also fixed a tied-readout→embedding Jacobian bug (was in the S4 code), which
    improves the embedding gradient the S4 doc flagged as a ~0.76-cos limitation.

`key_full` affects ONLY the RIPPLE gradient assembly; the BPTT twin (autograd) is identical
either way, so this is a clean isolation. Reports the gap (RIPPLE−BPTT)/BPTT for each cell.

Usage:  python experiments/reaudit_s4.py [n_layers] [steps] [seed]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavenn.rlm import RLMConfig  # noqa: E402
from wavenn.train import TrainConfig, train_bptt, train_ripple  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def _final_val(log):
    with open(os.path.join(log.dir, "summary.json"), encoding="utf-8") as f:
        return json.load(f)["final"]["val_nll"]


def main():
    n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    def mk_cfg(key_full):
        return RLMConfig(
            vocab_size=8192, d_model=128, n_layers=n_layers, n_heads=4,
            beta=1.0, beta_readout=6.0, beta_n=0.05, n_settle=20,
            key_full=key_full, delta_rule=False, deep_supervision=True, depth_mode="L0",
        )

    def mk_train(lr):
        return TrainConfig(
            steps=steps, batch=24, seq_len=64, lr=lr, warmup=30, eval_every=50,
            eval_iters=20, diag_every=100000, gen_every=0, seed=seed,  # diag/gen off; val only
        )

    results = []
    for lr in (2e-3, 1e-3):
        tcfg = mk_train(lr)
        lrtag = f"{lr:.0e}".replace("-0", "-")
        # BPTT once per lr (key_full-independent)
        print(f"\n=== BPTT twin (lr={lr}) ===", flush=True)
        _, logb = train_bptt(mk_cfg(False), tcfg, run_name=f"reaudit_bptt_{lrtag}_s{seed}")
        vb = _final_val(logb)
        for kf in (False, True):
            print(f"\n=== RIPPLE L0 key_full={kf} (lr={lr}) ===", flush=True)
            _, logr = train_ripple(mk_cfg(kf), tcfg, run_name=f"reaudit_ripple_kf{int(kf)}_{lrtag}_s{seed}")
            vr = _final_val(logr)
            gap = (vr - vb) / vb
            results.append({"lr": lr, "key_full": kf, "bptt_val": vb, "ripple_val": vr,
                            "gap": gap})
            print(f"  -> RIPPLE {vr:.4f} vs BPTT {vb:.4f}  gap {gap*100:+.2f}%", flush=True)
            json.dump(results, open(os.path.join(_HERE, f"S8-reaudit-L{n_layers}-s{seed}.json"), "w"),
                      indent=2)

    print("\n=== S4 RE-AUDIT: RIPPLE-vs-BPTT gap (TinyStories, "
          f"L{n_layers}, {steps} steps, seed {seed}) ===")
    print(f"{'lr':>7s}  {'key_full':>8s}  {'BPTT':>7s}  {'RIPPLE':>7s}  {'gap':>8s}")
    for r in results:
        print(f"{r['lr']:>7.0e}  {str(r['key_full']):>8s}  {r['bptt_val']:>7.4f}  "
              f"{r['ripple_val']:>7.4f}  {r['gap']*100:>+7.2f}%")


if __name__ == "__main__":
    main()
