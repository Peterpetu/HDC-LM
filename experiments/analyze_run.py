"""Post-training analysis for academic review (docs/05 §7).

Reads one or more run directories' ``metrics.jsonl`` and produces:
  * a printed summary table (final/best metrics, throughput, source hash);
  * PNG plots (loss curves, gradient norms, learned timescales, gate μ, safe-β margin,
    online EP-vs-BPTT cosine) under ``<run_dir>/plots/``;
  * a self-contained ``report.html`` embedding the plots and metadata for easy inspection;
  * a comparison view when multiple runs are given (e.g. RIPPLE vs its BPTT twin).

Usage:
    python experiments/analyze_run.py runs/ripple runs/bptt
    python experiments/analyze_run.py runs/ripple            # single run
"""

from __future__ import annotations

import base64
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def load_run(run_dir: str):
    meta_path = os.path.join(run_dir, "meta.json")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    rows = []
    if os.path.exists(metrics_path):
        for line in open(metrics_path, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    return meta, df


def _series(df: pd.DataFrame, col: str, tag: str = "eval"):
    if col not in df.columns or "tag" not in df.columns:
        return None, None
    sub = df[df["tag"] == tag].dropna(subset=[col])
    if sub.empty:
        sub = df.dropna(subset=[col])
    if sub.empty:
        return None, None
    return sub["step"].to_numpy(), sub[col].to_numpy()


def make_plots(run_dir: str, meta: dict, df: pd.DataFrame) -> list[str]:
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    paths = []
    name = meta.get("run_name", os.path.basename(run_dir))

    panels = [
        ("val_nll", "eval", "validation NLL (nats)"),
        ("val_bpt", "eval", "validation bits/token"),
        ("gnorm_total", "train", "gradient global norm"),
        ("mu", "train", "three-factor gate μ"),
        ("beta_over_safe_M", "train", "β / safe-β(M)  (<1 = stable)"),
        ("cos_overall", "diag", "EP-vs-BPTT gradient cosine"),
    ]
    for col, tag, ylabel in panels:
        xs, ys = _series(df, col, tag)
        if xs is None:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(xs, ys, marker="o", ms=3, lw=1.3)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{name}: {ylabel}")
        ax.grid(alpha=0.3)
        p = os.path.join(plot_dir, f"{col}.png")
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)

    # learned timescales: all layers on one axis
    gamma_cols = [c for c in df.columns if c.startswith("Meff_L")]
    if gamma_cols:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for c in sorted(gamma_cols):
            xs, ys = _series(df, c, "train")
            if xs is not None:
                ax.plot(xs, ys, marker=".", ms=3, lw=1.0, label=c.replace("Meff_", ""))
        ax.set_xlabel("step")
        ax.set_ylabel("effective context M_eff")
        ax.set_title(f"{name}: learned timescales")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        p = os.path.join(plot_dir, "timescales.png")
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths


def summarize(meta: dict, df: pd.DataFrame) -> dict:
    out = {"run_name": meta.get("run_name"), "source_hash": meta.get("env", {}).get("source_hash")}
    for col in ("val_nll", "val_bpt"):
        xs, ys = _series(df, col, "eval")
        if xs is not None and len(ys):
            out[f"final_{col}"] = float(ys[-1])
            out[f"best_{col}"] = float(min(ys))
    if "elapsed_s" in df.columns and len(df):
        out["wall_s"] = float(df["elapsed_s"].max())
    return out


def _embed(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'


def _samples_html(df: pd.DataFrame) -> str:
    """Render the generation-sample progression (if any) as a compact table."""
    if "sample" not in df.columns:
        return ""
    sub = df.dropna(subset=["sample"])
    if sub.empty:
        return ""
    rows = "\n".join(
        f"<tr><td style='white-space:nowrap;vertical-align:top'>{int(r.step)}</td>"
        f"<td>{str(r.sample)}</td></tr>"
        for r in sub.itertuples()
    )
    return (
        "<h2>Generation samples</h2>"
        "<table style='border-collapse:collapse;width:100%'>"
        "<tr><th style='text-align:left'>step</th><th style='text-align:left'>sample</th></tr>"
        f"{rows}</table>"
    )


def write_html(run_dir: str, meta: dict, summary: dict, plot_paths: list[str], df: pd.DataFrame):
    imgs = "\n".join(f"<div>{_embed(p)}</div>" for p in plot_paths)
    samples = _samples_html(df)
    html = f"""<!doctype html><meta charset=utf-8>
<title>{meta.get('run_name','run')} — WNN training report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}}
pre{{background:#f4f4f4;padding:1rem;overflow:auto}}div{{margin:1rem 0}}
table td,table th{{border-bottom:1px solid #ddd;padding:4px 8px;font-size:0.9em}}</style>
<h1>WNN training report — {meta.get('run_name','run')}</h1>
<h2>Summary</h2><pre>{json.dumps(summary, indent=2)}</pre>
{samples}
<h2>Environment</h2><pre>{json.dumps(meta.get('env', {}), indent=2)}</pre>
<h2>Config</h2><pre>{json.dumps(meta.get('config', {}), indent=2)}</pre>
<h2>Curves</h2>{imgs}
"""
    p = os.path.join(run_dir, "report.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    return p


def main(run_dirs: list[str]):
    summaries = []
    for run_dir in run_dirs:
        meta, df = load_run(run_dir)
        if df.empty:
            print(f"[skip] {run_dir}: no metrics")
            continue
        plots = make_plots(run_dir, meta, df)
        summ = summarize(meta, df)
        summaries.append(summ)
        html = write_html(run_dir, meta, summ, plots, df)
        print(f"[{summ['run_name']}] {json.dumps(summ)}")
        print(f"  report: {html}  ({len(plots)} plots)")
    if len(summaries) > 1:
        print("\n=== comparison ===")
        print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python experiments/analyze_run.py <run_dir> [<run_dir> ...]")
        sys.exit(1)
    main(sys.argv[1:])
