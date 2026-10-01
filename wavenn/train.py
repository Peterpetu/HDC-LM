"""Training harness with academic-grade telemetry (docs/04 §8, docs/05 §7).

Trains the RIPPLE LM by the backprop-free rule OR its BPTT twin (autograd baseline). Both
share architecture, data order, and seed; only the learning rule differs, so P2/P3 are a
pure learning-rule comparison (docs/05 §3).

Every run writes a structured telemetry directory (wavenn/telemetry.py): a machine-readable
metrics time series, a live status file, per-eval WNN diagnostics (learned timescales,
codebook safe-β health, gradient norms, the three-factor gate μ), a periodic online
EP-vs-BPTT fidelity check (docs/05 P1 tracked during training), the M4 adaptive-compute
report, checkpoints, and a self-contained HTML report via experiments/analyze_run.py.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch

from . import data as D
from . import diagnostics as DG
from . import telemetry as TM
from .ripple import LocalAdam
from .rlm import RippleLM, RLMConfig, ripple_grads, ripple_grads_l1, ripple_grads_l1fa

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TrainConfig:
    steps: int = 500
    batch: int = 32
    seq_len: int = 128
    lr: float = 2e-3
    warmup: int = 30  # linear lr warmup steps (stabilizes early training)
    grad_clip: float = 1.0
    eval_every: int = 25
    eval_iters: int = 20
    diag_every: int = 100  # online P1 fidelity + adaptive-compute (expensive)
    gen_every: int = 200  # emit a short generation sample every N steps (0 = off)
    gen_prompt: str = "Once upon a time"
    gen_new: int = 40  # tokens to generate for the sample
    ckpt_every: int = 0  # 0 = only at end
    seed: int = 0
    device: str = "cuda"
    runs_dir: str = field(default_factory=lambda: os.path.join(_REPO, "runs"))


def _lr_at(tcfg: "TrainConfig", step: int) -> float:
    return tcfg.lr * min(1.0, step / max(1, tcfg.warmup))


# ======================================================================================
# S6: gradient accumulation (docs/10 §6 M.6) and checkpoint/resume (plan A.4)
# ======================================================================================
@torch.no_grad()
def accumulate_step(model, grad_fn, opt, micro_batches, *, clip_norm: float = 1.0):
    """One optimizer step from several micro-batches — the §M.6 exact semantics.

    Collects each micro-batch's RAW scaled gradient (clip/gate/Cbar deferred via
    ``grad_fn(..., raw=True)`` → (mean_nll, grads, n_valid)), forms the
    TOKEN-WEIGHTED average (≡ the union-batch gradient exactly, §M.6 identity), then
    applies ONE global-norm clip → ONE surprise-gate μ (from the token-weighted mean
    NLL vs Cbar) → ONE LocalAdam step → ONE Cbar EMA update. The gate timescale is
    therefore per optimizer step (documented; docs/10 §6).

    Returns (mean_nll, grads) — grads post clip+gate, as the legacy single-call
    path returns them (for telemetry).
    """
    from .ripple import surprise_gate

    cfg = model.cfg
    acc: dict | None = None
    n_tot = 0
    nll_wsum = 0.0
    for x, y in micro_batches:
        nll_mu, g_mu, n_mu = grad_fn(model, x, y, clip_norm=None, raw=True)
        nll_wsum += nll_mu * n_mu
        if acc is None:
            acc = {k: g_mu[k] * float(n_mu) for k in g_mu}
        else:
            for k in acc:
                acc[k].add_(g_mu[k], alpha=float(n_mu))
        n_tot += n_mu
    assert acc is not None and n_tot > 0, "accumulate_step requires ≥1 micro-batch"
    for k in acc:
        acc[k].div_(n_tot)  # token-weighted average == union-batch gradient (§M.6)
    mean_nll = nll_wsum / n_tot
    cs = 1.0
    if clip_norm is not None:
        total = (sum((g.detach() ** 2).sum() for g in acc.values()) + 1e-12) ** 0.5
        cs = min(1.0, clip_norm / (float(total) + 1e-12))
    mu = surprise_gate(
        torch.tensor(mean_nll), model.Cbar.item(), cfg.gate_g0, cfg.gate_g1,
        cfg.gate_lo, cfg.gate_hi,
    )
    for k in acc:
        acc[k].mul_(cs * mu)
    opt.step(acc, clip=None)
    model.Cbar.mul_(1 - cfg.gate_ema).add_(cfg.gate_ema * mean_nll)
    return mean_nll, acc


def save_s6_checkpoint(path: str, *, step: int, model, opt, batcher=None,
                       replay_buffer=None, extra: dict | None = None,
                       min_free_gb: float = 8.0):
    """Resume-complete checkpoint (plan A.4): model (incl. Cbar buffer), LocalAdam
    moments, RNG states, the data cursor, and the replay buffer. Atomic write; hard
    disk pre-flight (docs/10: abort below ``min_free_gb``)."""
    import shutil

    free_gb = shutil.disk_usage(os.path.dirname(path) or ".").free / 2**30
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"checkpoint pre-flight: {free_gb:.1f} GB free < {min_free_gb} GB floor"
        )
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "batcher_state": batcher.state_dict() if batcher is not None else None,
        "replay_state": (
            {"x": replay_buffer.x, "y": replay_buffer.y, "n_seen": replay_buffer.n_seen,
             "heap": replay_buffer._heap, "gen": replay_buffer.gen.get_state()}
            if replay_buffer is not None else None
        ),
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_s6_checkpoint(path: str, *, model, opt, batcher=None, replay_buffer=None):
    """Restore everything ``save_s6_checkpoint`` wrote. Returns the payload."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    opt.load_state_dict(payload["opt_state"])
    torch.set_rng_state(payload["rng"])
    if payload["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    if batcher is not None and payload["batcher_state"] is not None:
        batcher.load_state_dict(payload["batcher_state"])
    if replay_buffer is not None and payload["replay_state"] is not None:
        rs = payload["replay_state"]
        replay_buffer.x = rs["x"]
        replay_buffer.y = rs["y"]
        replay_buffer.n_seen = rs["n_seen"]
        replay_buffer._heap = rs["heap"]
        replay_buffer.gen.set_state(rs["gen"])
    return payload


@torch.no_grad()
def _sample_and_log(model, tok, log, step, tcfg):
    """Emit ONE concise generation sample (truncated, single line) — a qualitative check
    that does not clutter the log (docs/05: coherence is the TinyStories rubric axis)."""
    from .generate import generate

    ids = tok.encode(tcfg.gen_prompt).ids or [tok.token_to_id("<|endoftext|>")]
    x = torch.tensor([ids], device=tcfg.device)
    out = generate(model, x, tcfg.gen_new, temperature=0.8, top_k=40, seed=step)
    text = tok.decode(out[0].tolist()).replace("\n", " ").strip()
    text = " ".join(text.split())[:160]  # collapse whitespace, cap length
    log.log(step, {"sample": text}, tag="sample", console_msg=f'   ↳ gen: "{text}…"')


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: RippleLM, val: D.Batcher, iters: int) -> dict:
    """Held-out next-token NLL (nats) and bits/token, plus per-block NLL (deep supervision)."""
    tot, n = 0.0, 0
    per_block = None
    for _ in range(iters):
        x, y = val.batch()
        block_logits = model.forward_blocks(x)
        K = model.cfg.vocab_size
        losses = [
            torch.nn.functional.cross_entropy(lg.reshape(-1, K), y.reshape(-1)).item()
            for lg in block_logits
        ]
        per_block = losses if per_block is None else [a + b for a, b in zip(per_block, losses)]
        tot += losses[-1]  # last block is the model output
        n += 1
    per_block = [v / n for v in per_block]
    nll = tot / n
    return {"val_nll": nll, "val_bpt": nll / 0.69314718, "val_nll_per_block": per_block}


def _wnn_diag(model: RippleLM, grads: dict, mean_nll: float) -> dict:
    d = {"train_nll": mean_nll, "train_bpt": mean_nll / 0.69314718, "Cbar": model.Cbar.item()}
    d.update(TM.gradient_norms(grads))
    d.update(TM.learned_timescales(model))
    d.update(TM.codebook_health(model))
    return d


def train_ripple(mcfg: RLMConfig, tcfg: TrainConfig, run_name: str = "ripple"):
    """Train by the backprop-free RIPPLE rule, fully logged.

    ``mcfg.depth_mode`` selects the credit-assignment mode (docs/03 §7): "L0" (pure-local,
    the default) or "L1" (stack-EP, inter-block credit via the residual-bus adjoint). Both
    are backprop-free (C2); L1 additionally routes the top-loss cotangent down the bus.
    """
    set_seed(tcfg.seed)
    train, val, tok = D.split_batchers(
        mcfg.vocab_size, tcfg.batch, tcfg.seq_len, tcfg.device, seed=tcfg.seed
    )
    model = RippleLM(mcfg).to(tcfg.device)
    opt = LocalAdam(dict(model.named_parameters()), lr=tcfg.lr)
    grad_fn = {"L1": ripple_grads_l1, "L1FA": ripple_grads_l1fa}.get(mcfg.depth_mode, ripple_grads)
    log = TM.RunLogger(tcfg.runs_dir, run_name, {"model": mcfg, "train": tcfg},
                       repo_root=_REPO, seed=tcfg.seed, total_steps=tcfg.steps)
    log.print(f"[RIPPLE {mcfg.depth_mode}] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
              f"safe_beta_M~{TM.codebook_health(model)['safe_beta_M_min']:.3f} beta={mcfg.beta}")
    t0 = time.time()
    for step in range(1, tcfg.steps + 1):
        x, y = train.batch()
        opt.lr = _lr_at(tcfg, step)  # linear warmup
        mean_nll, grads = grad_fn(model, x, y, clip_norm=tcfg.grad_clip)
        opt.step(grads, clip=None)  # already clipped inside the grad fn (before the gate)

        if step % tcfg.eval_every == 0 or step == 1:
            ev = evaluate(model, val, tcfg.eval_iters)
            toks = step * tcfg.batch * tcfg.seq_len
            tps = toks / (time.time() - t0)
            rec = {**_wnn_diag(model, grads, mean_nll), **ev, "lr": _lr_at(tcfg, step),
                   "tokens": toks, "tokens_per_s": tps}
            log.log(step, rec, tag="eval",
                    console_msg=f"[RIPPLE] step {step:4d} train_nll {mean_nll:.4f} "
                                f"val_nll {ev['val_nll']:.4f} val_bpt {ev['val_bpt']:.4f} "
                                f"mu*Cbar={model.Cbar.item():.2f} {tps/1e3:.1f}k tok/s")
        if step % tcfg.diag_every == 0:
            xf, yf = val.batch()
            fid = DG.ep_bptt_cosine(model, xf, yf)
            ac = DG.adaptive_compute_report(model, xf[:1])
            log.log(step, {**fid, **ac}, tag="diag",
                    console_msg=f"[RIPPLE] step {step:4d} P1 cos_overall={fid['cos_overall']:.3f} "
                                f"settle_iters~{ac['settle_iters_mean']:.1f} "
                                f"iters~entropy r={ac['iters_entropy_corr']:.2f}")
        if tcfg.gen_every and (step % tcfg.gen_every == 0 or step == tcfg.steps):
            _sample_and_log(model, tok, log, step, tcfg)
        if tcfg.ckpt_every and step % tcfg.ckpt_every == 0:
            log.checkpoint(step, model, extra={"step": step})

    ev = evaluate(model, val, tcfg.eval_iters * 2)
    log.checkpoint(tcfg.steps, model)
    log.close({"final": ev, "params_M": sum(p.numel() for p in model.parameters()) / 1e6})
    return model, log


def train_bptt(mcfg: RLMConfig, tcfg: TrainConfig, run_name: str = "bptt"):
    """Train the identical architecture by BPTT (autograd) — the twin baseline, logged."""
    set_seed(tcfg.seed)
    train, val, tok = D.split_batchers(
        mcfg.vocab_size, tcfg.batch, tcfg.seq_len, tcfg.device, seed=tcfg.seed
    )
    model = RippleLM(mcfg).to(tcfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, betas=(0.9, 0.99))
    log = TM.RunLogger(tcfg.runs_dir, run_name, {"model": mcfg, "train": tcfg},
                       repo_root=_REPO, seed=tcfg.seed, total_steps=tcfg.steps)
    log.print(f"[BPTT] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    t0 = time.time()
    for step in range(1, tcfg.steps + 1):
        x, y = train.batch()
        for g in opt.param_groups:
            g["lr"] = _lr_at(tcfg, step)  # linear warmup
        loss = model.bptt_loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        if step % tcfg.eval_every == 0 or step == 1:
            ev = evaluate(model, val, tcfg.eval_iters)
            toks = step * tcfg.batch * tcfg.seq_len
            tps = toks / (time.time() - t0)
            rec = {"train_nll": loss.item(), "train_bpt": loss.item() / 0.69314718,
                   "gnorm_total": float(gnorm), "lr": _lr_at(tcfg, step), "tokens": toks,
                   "tokens_per_s": tps, **ev, **TM.learned_timescales(model),
                   **TM.codebook_health(model)}
            log.log(step, rec, tag="eval",
                    console_msg=f"[BPTT]   step {step:4d} train_loss {loss.item():.4f} "
                                f"val_nll {ev['val_nll']:.4f} val_bpt {ev['val_bpt']:.4f} "
                                f"{tps/1e3:.1f}k tok/s")
        if tcfg.gen_every and (step % tcfg.gen_every == 0 or step == tcfg.steps):
            _sample_and_log(model, tok, log, step, tcfg)
        if tcfg.ckpt_every and step % tcfg.ckpt_every == 0:
            log.checkpoint(step, model)
    ev = evaluate(model, val, tcfg.eval_iters * 2)
    log.checkpoint(tcfg.steps, model)
    log.close({"final": ev, "params_M": sum(p.numel() for p in model.parameters()) / 1e6})
    return model, log
