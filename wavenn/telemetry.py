"""Training telemetry for academic post-training analysis and live inspection.

Design goals (docs/04 §8, docs/05 §7 reproducibility):
  * **Machine-readable time series** — one JSON object per logged step in ``metrics.jsonl``,
    so a later analysis (pandas / the bundled ``experiments/analyze_run.py``) can plot and
    compare runs without re-parsing console text.
  * **Full reproducibility metadata** — config, seed, library/hardware versions, and a
    content hash of ``wavenn/`` + ``docs/`` captured in ``meta.json`` at run start.
  * **Live inspection** — ``status.json`` is rewritten on every log with the latest metrics,
    elapsed time, throughput, and an ETA, so an ongoing run can be inspected at a glance
    (``python -m wavenn.telemetry <run_dir>`` prints it).
  * **WNN-specific diagnostics** — per-block loss, per-layer learned timescales γ, the
    three-factor gate μ, the safe-β margin, gradient norms per parameter group, and the
    periodic EP-vs-BPTT gradient cosine (an online P1 check).
  * **Same schema for RIPPLE and BPTT** so the two learners are directly comparable.

Dependency-light: only the standard library + torch (for checkpoint saving).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, is_dataclass
from typing import Any

import torch


def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return {k: _jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist() if x.numel() < 64 else f"<tensor {tuple(x.shape)}>"
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return str(x)


def _source_hash(root: str) -> str:
    """Content hash of the source (wavenn/*.py + docs/*.md) for exact reproducibility."""
    h = hashlib.sha256()
    for sub in ("wavenn", "docs"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith((".py", ".md")):
                with open(os.path.join(d, name), "rb") as f:
                    h.update(name.encode())
                    h.update(f.read())
    return h.hexdigest()[:16]


def environment_meta(repo_root: str, seed: int) -> dict:
    """Reproducibility metadata captured once at run start (docs/05 §7)."""
    meta = {
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "source_hash": _source_hash(repo_root),
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["cuda"] = torch.version.cuda
        props = torch.cuda.get_device_properties(0)
        meta["gpu_total_mem_MB"] = round(props.total_memory / 1e6, 1)
    return meta


class RunLogger:
    """Structured logger for one training run.

    Files under ``<runs_dir>/<run_name>/``:
        meta.json      — reproducibility metadata + config (written once).
        metrics.jsonl  — one record per logged step (the analysis time series).
        status.json    — latest snapshot, rewritten every log (live inspection).
        summary.json   — final summary (written by :meth:`close`).
        ckpt_*.pt      — checkpoints (weights + optimizer state + RNG).
        console.log    — a copy of the human-readable lines.
    """

    def __init__(
        self,
        runs_dir: str,
        run_name: str,
        config: Any,
        *,
        repo_root: str,
        seed: int = 0,
        total_steps: int | None = None,
        console: bool = True,
    ):
        self.dir = os.path.join(runs_dir, run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.run_name = run_name
        self.total_steps = total_steps
        self.console = console
        self.t0 = time.time()
        self._metrics_fp = open(os.path.join(self.dir, "metrics.jsonl"), "a", encoding="utf-8")
        self._console_fp = open(os.path.join(self.dir, "console.log"), "a", encoding="utf-8")
        self._n_records = 0
        meta = {
            "run_name": run_name,
            "config": _jsonable(config),
            "env": environment_meta(repo_root, seed),
            "total_steps": total_steps,
            "start_unix": self.t0,
        }
        with open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        self.print(f"[run:{run_name}] dir={self.dir} src={meta['env']['source_hash']}")

    # ------------------------------------------------------------------ logging
    def print(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        if self.console:
            print(line, flush=True)
        self._console_fp.write(line + "\n")
        self._console_fp.flush()

    def log(self, step: int, metrics: dict, *, tag: str = "train", console_msg: str | None = None):
        """Append one metrics record and refresh the live status file."""
        elapsed = time.time() - self.t0
        rec = {"step": step, "tag": tag, "elapsed_s": round(elapsed, 2), **_jsonable(metrics)}
        self._metrics_fp.write(json.dumps(rec) + "\n")
        self._metrics_fp.flush()
        self._n_records += 1
        # live status snapshot
        eta = None
        if self.total_steps and step > 0:
            eta = round(elapsed / step * (self.total_steps - step), 1)
        status = {
            "run_name": self.run_name,
            "last_step": step,
            "total_steps": self.total_steps,
            "elapsed_s": round(elapsed, 1),
            "eta_s": eta,
            "records": self._n_records,
            "latest": rec,
        }
        with open(os.path.join(self.dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        if console_msg:
            self.print(console_msg)

    # -------------------------------------------------------------- checkpoints
    def checkpoint(self, step: int, model, optimizer_state: dict | None = None, extra: dict | None = None):
        path = os.path.join(self.dir, f"ckpt_{step:06d}.pt")
        payload = {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer_state,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "extra": extra or {},
        }
        torch.save(payload, path)
        self.print(f"[ckpt] step {step} -> {os.path.basename(path)}")
        return path

    def close(self, summary: dict | None = None):
        if summary is not None:
            with open(os.path.join(self.dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(_jsonable(summary), f, indent=2)
        self._metrics_fp.close()
        self._console_fp.close()


# --------------------------------------------------------------------- WNN diagnostics
@torch.no_grad()
def gradient_norms(grads: dict) -> dict:
    """Global and per-group L2 norms of the assembled gradient (for stability tracking)."""
    groups: dict[str, float] = {}
    total_sq = 0.0
    for name, g in grads.items():
        if g is None:
            continue
        base = name.split(".")[0]  # embed / Wk / Wq / Wv / M / log_gamma / norm_gain
        sq = float((g.detach() ** 2).sum())
        groups[base] = groups.get(base, 0.0) + sq
        total_sq += sq
    out = {f"gnorm_{k}": v**0.5 for k, v in groups.items()}
    out["gnorm_total"] = total_sq**0.5
    return out


@torch.no_grad()
def codebook_health(model) -> dict:
    """Safe-β margin and codebook conditioning (docs/01 §6.2): read stays unique while
    β < 2/σ_max(M)². Tracks whether training keeps the reads in the stable regime."""
    from .cleanup import safe_beta_limit

    V = model.V()
    m = {"sigma_max_V": float(torch.linalg.matrix_norm(V, ord=2)), "safe_beta_V": safe_beta_limit(V)}
    sig_M, safe_M = [], []
    for ell in range(model.cfg.n_layers):
        M = model.M_used(ell)
        sig_M.append(float(torch.linalg.matrix_norm(M, ord=2)))
        safe_M.append(safe_beta_limit(M))
    m["sigma_max_M_mean"] = sum(sig_M) / len(sig_M)
    m["safe_beta_M_min"] = min(safe_M)
    m["beta"] = model.cfg.beta
    m["beta_over_safe_M"] = model.cfg.beta / max(min(safe_M), 1e-9)  # <1 ⇒ in the safe regime
    return m


@torch.no_grad()
def learned_timescales(model) -> dict:
    """Per-layer mean effective context M_eff=(1+γ)/(1-γ) — how the model sets its horizons."""
    out = {}
    for ell in range(model.cfg.n_layers):
        g = model.gammas(ell)
        meff = ((1 + g) / (1 - g).clamp_min(1e-4)).mean().item()
        out[f"gamma_mean_L{ell}"] = g.mean().item()
        out[f"Meff_L{ell}"] = meff
    return out


if __name__ == "__main__":  # `python -m wavenn.telemetry <run_dir>` prints live status
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else "."
    sp = os.path.join(d, "status.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            st = json.load(f)
        print(json.dumps(st, indent=2))
    else:
        print(f"no status.json under {d}")
