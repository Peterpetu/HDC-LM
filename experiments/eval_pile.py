"""S6 FROZEN-EVAL evaluator + Gate V0 (protocol-fidelity) runner.

Implements plan §3 / §M.5 and docs/10-s6-protocol.md:

  * ``frozen_eval_ce``  — the model-agnostic deterministic evaluator: entire val
    region of the reference Pile cache as fixed non-overlapping 1024-token windows,
    fp32, CE (nats) over the T-1 within-window positions (token-weighted),
    PPL = exp(min(CE, 20)). Bit-identical across repeated invocations.
  * ``load_ncalm_gpt2`` — loads the reference GPT-2/NeoX-162M checkpoint with the
    reference's OWN model class (loaded standalone via importlib from the read-only
    C:/koodi/NCALM tree — no NCALM package side effects, nothing written there).
  * Gate V0 (``python experiments/eval_pile.py --v0``):
      (i)   checkpoint-load + 1-step forward/backward/AdamW training smoke on the
            reference model class (Phase E trains this class);
      (ii)  side-by-side agreement <= 1e-4 between OUR evaluator and the
            REFERENCE-side evaluator (the model's own forward_train CE accumulation,
            Trainer.evaluate semantics) on IDENTICAL fixed batches;
      (iii) re-anchor final_model.pt and best_model.pt under FROZEN-EVAL ->
            CE*_ref-final / CE*_ref-best (the headline anchors; the archived
            3.3459/27.57 are historical, sigma~0.05 — plan §1).
    Writes experiments/S6-v0-anchor.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavenn.pile_data import (  # noqa: E402
    IGNORE_INDEX, NCALM_ROOT, PILE_250M_BIN, FrozenEval, PileMemmap,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
REF_RUN_DIR = os.path.join(
    NCALM_ROOT, "experiments", "GPT-2", "pile_250m_gpt2_20260319_141247"
)


# ======================================================================================
# The frozen evaluator (model-agnostic; plan §M.5)
# ======================================================================================
@torch.no_grad()
def frozen_eval_ce(get_logits, frozen: FrozenEval, device: str = "cuda") -> dict:
    """Deterministic CE/PPL of any causal LM under FROZEN-EVAL.

    Args:
        get_logits: callable (B, T) long -> (B, T, V) logits. Must be causal.
        frozen:     the FrozenEval window set (fixed order, no RNG).

    Returns dict(ce, ppl, supervised_tokens, n_windows). CE accumulated in fp64 from
    per-batch fp32 sums (token-weighted; all windows have T-1 targets).
    """
    total_nll = 0.0
    total_tok = 0
    for x, y in frozen.batches(device=device):
        logits = get_logits(x)
        assert logits.shape[:2] == x.shape and logits.dim() == 3
        nll = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="sum",
        )
        total_nll += float(nll.item())
        total_tok += int((y != IGNORE_INDEX).sum().item())
    assert total_tok == frozen.supervised_tokens, (total_tok, frozen.supervised_tokens)
    ce = total_nll / total_tok
    return {
        "ce": ce,
        "ppl": math.exp(min(ce, 20)),
        "supervised_tokens": total_tok,
        "n_windows": frozen.K,
    }


# ======================================================================================
# Reference model loading (standalone importlib — no NCALM package import, no writes)
# ======================================================================================
def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_ncalm_gpt2_class():
    """Load the reference GPT2Model class from the read-only NCALM tree.

    Loads exactly three files (ncalm/model.py, models/gpt2/config.py,
    models/gpt2/model.py) as standalone modules with package stubs — avoiding
    ncalm/__init__.py (which imports every other model family). Read-only.
    """
    if "ncalm.models.gpt2.model" in sys.modules:
        return sys.modules["ncalm.models.gpt2.model"].GPT2Model
    for pkg in ("ncalm", "ncalm.models", "ncalm.models.gpt2", "ncalm.data"):
        if pkg not in sys.modules:
            stub = types.ModuleType(pkg)
            stub.__path__ = []  # mark as package
            sys.modules[pkg] = stub
    base = os.path.join(NCALM_ROOT, "ncalm")
    _load_module("ncalm.model", os.path.join(base, "model.py"))
    _load_module("ncalm.models.gpt2.config", os.path.join(base, "models", "gpt2", "config.py"))
    mod = _load_module("ncalm.models.gpt2.model", os.path.join(base, "models", "gpt2", "model.py"))
    return mod.GPT2Model


def load_ncalm_checkpoint(path: str, device: str = "cuda"):
    """Load a reference checkpoint into the reference's own model class (eval mode)."""
    GPT2Model = load_ncalm_gpt2_class()
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # The checkpoint stores the plain-dict config alongside tensors; it is the
        # user's own artifact, loaded read-only.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT2Model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def reference_side_ce(model, frozen: FrozenEval, device: str = "cuda") -> float:
    """The REFERENCE's evaluator semantics on our fixed batches.

    Trainer.evaluate (ncalm/trainer.py:398-435) accumulates forward_train's
    info['ce'] (mean over B*(T-1) shifted positions) weighted by B*(T-1). Run here
    WITHOUT autocast (fp32 — the frozen protocol; docs/10 §fp32). The batches are
    the identical FROZEN-EVAL windows our evaluator sees.
    """
    total_ce = 0.0
    total_tok = 0
    model.eval()
    for x, _y in frozen.batches(device=device):
        B, T = x.shape
        _, info = model.forward_train(x)  # eval mode: no checkpointing, no dropout
        ce_val = info["ce"]
        ce_val = ce_val.item() if isinstance(ce_val, torch.Tensor) else float(ce_val)
        total_ce += ce_val * B * (T - 1)
        total_tok += B * (T - 1)
    return total_ce / total_tok


def training_smoke(device: str = "cuda") -> dict:
    """Gate V0(i): 1-step forward/backward/AdamW on a FRESH reference-class model.

    Phase E trains this class sequentially; load-only compatibility is insufficient
    (plan §3.2). Uses a small synthetic batch; touches nothing in NCALM.
    """
    GPT2Model = load_ncalm_gpt2_class()
    cfg = {"d_model": 768, "n_layer": 12, "n_head": 12, "d_ff": 3072,
           "vocab_size": 50304, "seq_len": 1024}
    model = GPT2Model(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, 50304, (2, 256), device=device)
    loss, info = model.forward_train(x)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    loss2, _ = model.forward_train(x)
    out = {"loss_before": float(loss.item()), "loss_after": float(loss2.item()),
           "grad_norm": float(gnorm)}
    del model, opt
    torch.cuda.empty_cache()
    return out


# ======================================================================================
# Gate V0
# ======================================================================================
def run_v0(device: str = "cuda") -> dict:
    store = PileMemmap(PILE_250M_BIN)
    frozen = FrozenEval(store, seq_len=1024, batch_size=8)
    print(f"[V0] cache: {store.bin_path} tokens={len(store):,}")
    print(f"[V0] FROZEN-EVAL: K={frozen.K} windows, {frozen.n_batches} batches, "
          f"{frozen.supervised_tokens:,} supervised tokens")
    assert frozen.K == 1220 and frozen.n_batches == 153, "plan §M.5 window count"

    print("[V0.i] training smoke on the reference model class ...", flush=True)
    smoke = training_smoke(device)
    smoke_ok = smoke["loss_after"] < smoke["loss_before"]
    print(f"       loss {smoke['loss_before']:.4f} -> {smoke['loss_after']:.4f} "
          f"(gnorm {smoke['grad_norm']:.2f})  {'OK' if smoke_ok else 'FAIL'}")

    results: dict = {"smoke": {**smoke, "ok": bool(smoke_ok)},
                     "frozen": {"K": frozen.K, "batches": frozen.n_batches,
                                "supervised_tokens": frozen.supervised_tokens}}

    for tag in ("final_model", "best_model"):
        path = os.path.join(REF_RUN_DIR, f"{tag}.pt")
        print(f"[V0] loading {tag}.pt ...", flush=True)
        model, ckpt = load_ncalm_checkpoint(path, device)
        ours = frozen_eval_ce(model.get_logits, frozen, device)
        entry = {"step": int(ckpt.get("step", -1)), "ours_ce": ours["ce"],
                 "ours_ppl": ours["ppl"]}
        if tag == "final_model":
            # V0.ii: side-by-side agreement on IDENTICAL batches (reference-side
            # evaluator = the model's own forward_train CE accumulation, fp32).
            print("[V0.ii] reference-side evaluator on identical batches ...", flush=True)
            ref_ce = reference_side_ce(model, frozen, device)
            agree = abs(ref_ce - ours["ce"])
            entry.update({"ref_side_ce": ref_ce, "agreement_abs": agree,
                          "agreement_ok": bool(agree <= 1e-4)})
            print(f"        ours {ours['ce']:.6f} vs ref-side {ref_ce:.6f} "
                  f"|diff| {agree:.2e}  {'OK' if agree <= 1e-4 else 'FAIL'}")
        print(f"[V0.iii] {tag}: CE*_ref = {ours['ce']:.4f}  PPL = {ours['ppl']:.2f} "
              f"(step {entry['step']})")
        results[tag] = entry
        del model, ckpt
        torch.cuda.empty_cache()

    results["env"] = {
        "torch": torch.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "protocol": "FROZEN-EVAL v1 (docs/10-s6-protocol.md, plan §M.5)",
        "archived_reference": {"final_val_ce": 3.3459, "final_val_ppl": 28.39,
                               "best_val_ppl": 27.57,
                               "note": "historical; stochastic-eval draws, sigma~0.05"},
    }
    results["gate_v0_pass"] = bool(
        results["smoke"]["ok"] and results["final_model"]["agreement_ok"]
    )
    out = os.path.join(_HERE, "S6-v0-anchor.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"[V0] gate {'PASS' if results['gate_v0_pass'] else 'FAIL'} -> {out}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", action="store_true", help="run Gate V0 (smoke + agreement + re-anchor)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.v0:
        run_v0(args.device)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
