"""S6 headline/pilot training driver (docs/10 §3-§5; plan Phases B.1, B.3, C).

Trains the WNN-162M full-config RIPPLE-L1 model on the reference Pile cache under
the frozen protocol:
  * data: reference-matched sampling over the 99.5% train region (seed 42);
  * budget: micro-batch 8 × T=1024 × accum 8 = 65,536 tokens/step (the reference's);
  * schedule: cosine → 0.1× with 188-step linear warmup over --schedule_steps
    (default 3,815 — pilots run a PREFIX of the real schedule);
  * eval: FROZEN-EVAL (the full 153-batch val set) every --eval_every steps via the
    deterministic fast forward (rlm.fast_logits);
  * telemetry: γ/M_eff, safe-β, gate μ, grad norms + the §M.7 codebook-utilization
    signals (H̄/ln P, PR per block) at every eval;
  * halt rule (docs/10 §5): on full-scale runs, dead-codebook verdict at the first
    eval ≥ step 200 (H̄>0.98 on ≥10/12 blocks, flat since the first eval) ⇒ halt;
  * checkpoints: resume-complete latest.pt every --ckpt_every steps (rolling) +
    best.pt (weights only); disk pre-flight ≥8 GB;
  * variant B (--variant on): episodic replay — surprise-prioritized A-Res buffer
    (3,000 sequences), replay_frac 0.4 ⇒ per step 5 fresh + 3 replayed micro-batches
    at the same fixed update budget (fresh ≈ 150M tokens; docs/10 §4).

PAUSE / CONTINUE (user-controlled long runs):
  * START:    python experiments/run_s6.py --tag C1 --variant off --lr <lr> --steps 3815
  * PAUSE:    press Ctrl+C in the training window (graceful: saves latest.pt, exits) —
              or, from any other terminal, create the sentinel file
              runs/s6_<tag>/PAUSE   (the run saves latest.pt and exits at the next
              step boundary; the sentinel is consumed on the next start).
  * CONTINUE: re-run the EXACT same start command — --auto_resume is ON by default:
              if runs/s6_<tag>/latest.pt exists, training continues from it
              (bit-consistent state: model, optimizer moments, gate EMA, RNG, data
              cursor, replay buffer — certified in tests/test_resume.py).
  * Hard shutdown (power off, crash): at most --ckpt_every (default 100 ≈ 1.5 h)
    steps of progress are lost; CONTINUE works the same way.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_pile import frozen_eval_ce  # noqa: E402
from wavenn import telemetry as TM  # noqa: E402
from wavenn.episodic import ReservoirBuffer  # noqa: E402
from wavenn.pile_data import (  # noqa: E402
    IGNORE_INDEX, PILE_250M_BIN, FrozenEval, PileMemmap, PileTrainBatcher,
)
from wavenn.ripple import LocalAdam  # noqa: E402
from wavenn.rlm import (  # noqa: E402
    RLMConfig, RippleLM, codebook_utilization, fast_logits, ripple_grads_l1_fast,
)
from wavenn.train import accumulate_step, load_s6_checkpoint, save_s6_checkpoint  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

MICRO_B = 8
SEQ_LEN = 1024
ACCUM = 8
WARMUP = 188
MIN_LR_RATIO = 0.1
REPLAY_CAP = 3000
REPLAY_MICROS = 3  # of 8 → replay_frac ≈ 0.4 (docs/10 §4; S7-validated 0.4)


def s6_config(lr_unused=None) -> RLMConfig:
    """The frozen S6 headline configuration (docs/10 §4)."""
    return RLMConfig(
        vocab_size=50304, d_model=768, n_layers=12, n_heads=12, P=11116,
        beta=1.0, beta_readout=6.0, beta_n=0.05, n_settle=15, eta=0.3,
        key_full=True, delta_rule=False, deep_supervision=True, depth_mode="L1",
        gamma_max=0.999, cbar_init=10.826, scan_Lc=128, row_chunk=2048,
        frozen_force=True, settle_tol=1e-4,
    )


def cosine_lr(base_lr: float, step: int, schedule_steps: int) -> float:
    if step < WARMUP:
        return base_lr * step / max(1, WARMUP)
    t = (step - WARMUP) / max(1, schedule_steps - WARMUP)
    t = min(1.0, t)
    return base_lr * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--variant", choices=["off", "on"], default="off")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=3815)
    ap.add_argument("--schedule_steps", type=int, default=3815)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--ckpt_every", type=int, default=100)
    ap.add_argument("--resume", type=str, default=None,
                    help="explicit checkpoint path (overrides --auto_resume)")
    ap.add_argument("--no_auto_resume", action="store_true",
                    help="do NOT auto-continue from runs/s6_<tag>/latest.pt")
    ap.add_argument("--no_halt_check", action="store_true",
                    help="disable the dead-codebook halt rule (pilot runs)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    mcfg = s6_config()
    model = RippleLM(mcfg).to(args.device)
    n_par = sum(p.numel() for p in model.parameters())
    assert n_par == 162_321_552, f"param count drifted: {n_par:,}"
    opt = LocalAdam(dict(model.named_parameters()), lr=args.lr)

    store = PileMemmap(PILE_250M_BIN)
    batcher = PileTrainBatcher(store, B=MICRO_B, seq_len=SEQ_LEN,
                               device=args.device, seed=args.seed)
    frozen = FrozenEval(store, seq_len=SEQ_LEN, batch_size=8)
    buffer = ReservoirBuffer(REPLAY_CAP, seed=args.seed) if args.variant == "on" else None

    run_dir = os.path.join(_REPO, "runs", f"s6_{args.tag}")
    log = TM.RunLogger(os.path.join(_REPO, "runs"), f"s6_{args.tag}",
                       {"model": mcfg, "args": vars(args), "tokens_per_step": MICRO_B * SEQ_LEN * ACCUM},
                       repo_root=_REPO, seed=args.seed, total_steps=args.steps)
    pause_file = os.path.join(run_dir, "PAUSE")
    latest_path = os.path.join(run_dir, "latest.pt")
    if os.path.exists(pause_file):
        os.remove(pause_file)  # consume a leftover sentinel so we don't re-pause at once
    resume_path = args.resume
    if resume_path is None and not args.no_auto_resume and os.path.exists(latest_path):
        resume_path = latest_path
    start_step = 0
    resumed_extra: dict = {}
    if resume_path:
        payload = load_s6_checkpoint(resume_path, model=model, opt=opt,
                                     batcher=batcher, replay_buffer=buffer)
        start_step = int(payload["step"])
        resumed_extra = payload.get("extra") or {}
        log.print(f"[resume] {resume_path} -> continuing from step {start_step}")
        if start_step >= args.steps:
            log.print(f"[resume] run already complete ({start_step}/{args.steps}); nothing to do")
            return

    # full-run accounting — restored across pause/resume (verified in the pause test)
    fresh_tokens = int(resumed_extra.get("fresh_tokens", 0))
    replayed_tokens = int(resumed_extra.get("replayed_tokens", 0))
    best_ce = float(resumed_extra.get("best_ce", float("inf")))
    util0: dict | None = resumed_extra.get("util0")
    curve = list(resumed_extra.get("curve", []))
    rows_box: dict = {}

    def _extra() -> dict:
        return {"best_ce": best_ce, "fresh_tokens": fresh_tokens,
                "replayed_tokens": replayed_tokens, "util0": util0, "curve": curve}

    def gfn(m, x, y, *, clip_norm=None, raw=True):
        nll, g, n, rows = ripple_grads_l1_fast(
            m, x, y, clip_norm=clip_norm, raw=True, want_rows_nll=True)
        rows_box["rows"] = rows
        return nll, g, n

    # ---- graceful pause machinery (docs/10; user-controlled long runs) ----
    # Ctrl+C is DEFERRED to the next step boundary: an interrupt inside opt.step()
    # would leave a partially-applied update, so the handler only sets a flag; the
    # loop saves a boundary-consistent checkpoint and exits. Second Ctrl+C = force
    # quit (resume then falls back to the last periodic checkpoint).
    import signal

    stop_req = {"flag": False}

    def _sigint(_signum, _frame):
        if stop_req["flag"]:
            raise KeyboardInterrupt
        stop_req["flag"] = True
        print("[pause] finishing the current step, then saving state... "
              "(Ctrl+C again to force-quit WITHOUT saving)", flush=True)

    signal.signal(signal.SIGINT, _sigint)

    def _pause_exit(completed_step: int, reason: str):
        # pause/final saves use a HARD minimum (2.5 GB ≈ one checkpoint + margin):
        # losing the pause state is worse than a tight write. Periodic saves keep
        # the full 8 GB floor and skip-with-warning instead (drive is 99% full and
        # free space fluctuates; measured 5–18 GB during Phase A).
        save_s6_checkpoint(latest_path, step=completed_step, model=model, opt=opt,
                           batcher=batcher, replay_buffer=buffer,
                           extra=_extra(), min_free_gb=2.5)
        log.print(f"[PAUSED] ({reason}) state saved at step {completed_step}/{args.steps} "
                  f"-> {latest_path}")
        log.print(f"[PAUSED] CONTINUE later with the SAME start command "
                  f"(auto-resume picks up latest.pt).")

    t0 = time.time()
    tok_since = 0
    for step in range(start_step + 1, args.steps + 1):
        if stop_req["flag"] or os.path.exists(pause_file):
            if os.path.exists(pause_file):
                os.remove(pause_file)
            _pause_exit(step - 1, "Ctrl+C" if stop_req["flag"] else "PAUSE file")
            return
        opt.lr = cosine_lr(args.lr, step, args.schedule_steps)
        micros = []
        fresh_micros = []
        n_replay = REPLAY_MICROS if (buffer is not None and len(buffer) >= MICRO_B) else 0
        for _ in range(ACCUM - n_replay):
            xb, yb = batcher.batch()
            micros.append((xb, yb))
            fresh_micros.append((xb, yb))
            fresh_tokens += xb.numel()
        for _ in range(n_replay):
            xb, yb = buffer.sample(MICRO_B, args.device)
            micros.append((xb, yb))
            replayed_tokens += xb.numel()

        priorities = []
        if buffer is not None:
            orig = gfn

            def gfn_track(m, x, y, *, clip_norm=None, raw=True):
                out = orig(m, x, y, clip_norm=clip_norm, raw=raw)
                pr = rows_box["rows"].sum(1) / (y != IGNORE_INDEX).sum(1).clamp(min=1)
                priorities.append(pr.detach().cpu())
                return out

            mean_nll, _g = accumulate_step(model, gfn_track, opt, micros, clip_norm=1.0)
            for i, (xb, yb) in enumerate(fresh_micros):  # fresh micros come first
                buffer.add(xb, yb, priority=priorities[i])
        else:
            mean_nll, _g = accumulate_step(model, gfn, opt, micros, clip_norm=1.0)
        tok_since += MICRO_B * SEQ_LEN * ACCUM

        if not math.isfinite(mean_nll):
            log.checkpoint(step, model)
            raise SystemExit(f"[HALT] non-finite NLL at step {step} — diagnose (docs/10 §3)")

        if step % 10 == 0 or step == 1:
            dt = time.time() - t0
            tps = tok_since / max(dt, 1e-9)
            log.log(step, {"train_nll": mean_nll, "lr": opt.lr, "tokens_per_s": tps,
                           "fresh_tokens": fresh_tokens, "replayed_tokens": replayed_tokens,
                           "Cbar": model.Cbar.item()},
                    tag="train",
                    console_msg=f"[s6_{args.tag}] step {step:4d}/{args.steps} "
                                f"nll {mean_nll:.4f} lr {opt.lr:.2e} {tps:,.0f} tok/s")
            t0, tok_since = time.time(), 0

        if step % args.eval_every == 0 or step == args.steps:
            ev = frozen_eval_ce(lambda x: fast_logits(model, x), frozen, args.device)
            x0, _ = next(frozen.batches(device=args.device))
            util = codebook_utilization(model, x0)
            if util0 is None:
                util0 = util
            rec = {"val_ce": ev["ce"], "val_ppl": ev["ppl"], **util,
                   **TM.learned_timescales(model), **TM.codebook_health(model),
                   "fresh_tokens": fresh_tokens, "replayed_tokens": replayed_tokens}
            curve.append({"step": step, "ce": ev["ce"], "ppl": ev["ppl"],
                          "Hbar": [util[f"util_Hbar_L{i}"] for i in range(12)]})
            log.log(step, rec, tag="eval",
                    console_msg=f"[s6_{args.tag}] step {step:4d} FROZEN-EVAL "
                                f"ce {ev['ce']:.4f} ppl {ev['ppl']:.2f} "
                                f"Hbar[0]={util['util_Hbar_L0']:.3f} "
                                f"Hbar[11]={util['util_Hbar_L11']:.3f}")
            if ev["ce"] < best_ce:
                best_ce = ev["ce"]
                torch.save({"step": step, "model_state": model.state_dict(),
                            "val_ce": ev["ce"]}, os.path.join(run_dir, "best.pt"))
            # dead-codebook halt rule (docs/10 §5): first eval ≥ 200 on full runs
            if (not args.no_halt_check) and step >= 200 and args.steps >= 1000:
                dead = sum(1 for i in range(12)
                           if util[f"util_Hbar_L{i}"] > 0.98
                           and abs(util[f"util_Hbar_L{i}"] - util0[f"util_Hbar_L{i}"]) < 0.005)
                if dead >= 10:
                    save_s6_checkpoint(os.path.join(run_dir, "latest.pt"), step=step,
                                       model=model, opt=opt, batcher=batcher,
                                       replay_buffer=buffer, extra=_extra(),
                                       min_free_gb=2.5)
                    raise SystemExit(
                        f"[HALT] dead-codebook verdict at step {step}: {dead}/12 blocks "
                        f"H̄>0.98 and flat (docs/10 §5) — user decision required")

        if args.ckpt_every and step % args.ckpt_every == 0:
            try:
                save_s6_checkpoint(os.path.join(run_dir, "latest.pt"), step=step,
                                   model=model, opt=opt, batcher=batcher,
                                   replay_buffer=buffer, extra=_extra())
                log.print(f"[ckpt] latest.pt @ step {step}")
            except RuntimeError as e:  # disk floor breached: skip, warn, retry next cadence
                log.print(f"[ckpt SKIPPED @ step {step}] {e} — training continues; "
                          f"free disk space to restore checkpointing")

    # final: frozen eval + resume-complete final checkpoint + results JSON
    ev = frozen_eval_ce(lambda x: fast_logits(model, x), frozen, args.device)
    save_s6_checkpoint(os.path.join(run_dir, "final.pt"), step=args.steps, model=model,
                       opt=opt, batcher=batcher, replay_buffer=buffer, extra=_extra(),
                       min_free_gb=2.5)
    result = {"tag": args.tag, "variant": args.variant, "lr": args.lr,
              "steps": args.steps, "schedule_steps": args.schedule_steps,
              "seed": args.seed, "final_ce": ev["ce"], "final_ppl": ev["ppl"],
              "best_ce": min(best_ce, ev["ce"]),
              "fresh_tokens": fresh_tokens, "replayed_tokens": replayed_tokens,
              "curve": curve, "params": n_par,
              "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30}
    out = os.path.join(_HERE, f"S6-run-{args.tag}.json")
    json.dump(result, open(out, "w"), indent=2)
    log.close({"final": result})
    print(f"[s6_{args.tag}] FINAL frozen-eval ce {ev['ce']:.4f} ppl {ev['ppl']:.2f} -> {out}")


if __name__ == "__main__":
    main()
