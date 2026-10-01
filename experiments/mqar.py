"""S7 — in-context / test-time learning: multi-query associative recall (MQAR, P6).

The complement to continual training: the FAST plasticity timescale. The running state
`S_t` IS a learned associative memory (a fast weight), so the model does test-time learning
by construction — it binds novel key→value pairs presented in-context and recalls them
later WITHOUT any weight update. MQAR (Arora et al. 2024, "Zoology") is the standard probe.

Sequence: k1 v1 k2 v2 … kN vN  q1 a1 q2 a2 …  — after N key–value pairs, each query qj
repeats a key and the model must predict its value aj (loss/accuracy measured only at the
query positions). Keys and values are disjoint token bands; the assignment is random per
example, so success requires in-context recall, not memorization.

This also tests **P6**: does the finite-temperature Hopfield cleanup (and the delta-rule
write) EXTEND the recallable number of pairs N at fixed D, by denoising binding interference
(docs/01 §4.3 conjecture)? We sweep N for cleanup-on vs cleanup-off.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavenn.ripple import LocalAdam  # noqa: E402
from wavenn.rlm import RLMConfig, RippleLM, ripple_grads  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_mqar_batch(B, n_pairs, n_queries, n_keys, n_vals, gen):
    """Return (tokens, targets) with targets=-100 except at query positions (docs/06 §2).

    Layout: [k1 v1 … kN vN  q1 a1 q2 a2 …]. Keys ∈ [0,n_keys); values ∈ [n_keys,n_keys+n_vals).
    """
    key_off, val_off = 0, n_keys
    L = 2 * n_pairs + 2 * n_queries
    x = torch.zeros(B, L, dtype=torch.long, device=DEV)
    tgt = torch.full((B, L), -100, dtype=torch.long, device=DEV)
    for b in range(B):
        keys = torch.randperm(n_keys, generator=gen)[:n_pairs]
        vals = torch.randint(0, n_vals, (n_pairs,), generator=gen) + val_off
        seq = torch.empty(2 * n_pairs, dtype=torch.long)
        seq[0::2] = keys + key_off
        seq[1::2] = vals
        # queries: a random subset of the pairs, repeated
        qidx = torch.randperm(n_pairs, generator=gen)[:n_queries]
        qseq = torch.empty(2 * n_queries, dtype=torch.long)
        qseq[0::2] = keys[qidx] + key_off  # query key
        qseq[1::2] = vals[qidx]  # its value (teacher-forced answer)
        full = torch.cat([seq, qseq]).to(DEV)
        x[b] = full
        # target at each query-key position = its value (predict the NEXT token)
        qpos = 2 * n_pairs + torch.arange(n_queries, device=DEV) * 2
        tgt[b, qpos] = vals[qidx].to(DEV)
    return x, tgt


@torch.no_grad()
def eval_recall(model, n_pairs, n_queries, n_keys, n_vals, gen, iters=20, B=64):
    correct = total = 0
    for _ in range(iters):
        x, tgt = make_mqar_batch(B, n_pairs, n_queries, n_keys, n_vals, gen)
        lg = model.logits(x)
        mask = tgt != -100
        pred = lg.argmax(-1)
        correct += (pred[mask] == tgt[mask]).sum().item()
        total += mask.sum().item()
    return correct / total


def train_mqar(mcfg, *, n_pairs, n_keys, n_vals, steps=1500, B=48, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    model = RippleLM(mcfg).to(DEV)
    opt = LocalAdam(dict(model.named_parameters()), lr=lr)
    nq = max(4, n_pairs // 2)
    for step in range(1, steps + 1):
        opt.lr = lr * min(1.0, step / 40)
        x, tgt = make_mqar_batch(B, n_pairs, nq, n_keys, n_vals, gen)
        # MASKED objective (audit M3): tgt supervises ONLY the query-key positions (the
        # value to recall as the next token); all other positions are ignore_index=-100, so
        # the KV-block and query answers (random) don't dilute the recall signal.
        _, g = ripple_grads(model, x, tgt, ignore_index=-100, clip_norm=1.0)
        opt.step(g, clip=None)
    acc = eval_recall(model, n_pairs, nq, n_keys, n_vals, gen)
    return model, acc


def main():
    n_keys, n_vals = 64, 64
    vocab = n_keys + n_vals
    base = dict(vocab_size=vocab, d_model=128, n_heads=4, n_layers=1, beta=1.0,
                beta_readout=6.0, key_full=False)
    configs = {
        "cleanup_on (n_settle=15)": dict(n_settle=15, delta_rule=False),
        "cleanup_off (n_settle=1)": dict(n_settle=1, delta_rule=False),
        "cleanup+delta": dict(n_settle=15, delta_rule=True),
    }
    pair_counts = [4, 8, 16, 32]
    results = {}
    for name, extra in configs.items():
        results[name] = {}
        for N in pair_counts:
            mcfg = RLMConfig(**base, **extra)
            _, acc = train_mqar(mcfg, n_pairs=N, n_keys=n_keys, n_vals=n_vals, steps=2000, lr=3e-3, seed=0)
            results[name][N] = acc
            print(f"[{name}] N={N:2d} pairs -> recall {acc:.3f}", flush=True)
    json.dump(results, open(os.path.join(_HERE, "S7-mqar-results.json"), "w"), indent=2)
    print("\n=== MQAR in-context recall accuracy vs #pairs (P6: does cleanup extend recall?) ===")
    print(f"{'config':26s} " + " ".join(f"N={N:<5d}" for N in pair_counts))
    for name in configs:
        print(f"{name:26s} " + " ".join(f"{results[name][N]:.3f}" for N in pair_counts))


if __name__ == "__main__":
    main()
