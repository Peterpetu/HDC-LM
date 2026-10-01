"""S7 — continual / test-time learning benchmark (docs/06 §1).

Can the model acquire a stream of domains WITHOUT catastrophically forgetting earlier ones
(the M1 attribute)? Sequential "Markov-language" domains share a vocabulary but have distinct
next-token statistics, so a later domain *conflicts* with earlier ones in the shared weights
— the classic catastrophic-forgetting paradigm (McCloskey & Cohen 1989) on sequence prediction.

Learners (same architecture / seed / byte-identical domain batches per seed):
  * RIPPLE naive           — backprop-free, no gate, no replay
  * RIPPLE + gate          — three-factor neuromodulation (docs/03 §5): global surprise-modulated
                             plasticity. NB a *global* scalar cannot selectively protect prior-domain
                             weights; its role is to make replay efficient (low-surprise, already-
                             consolidated replayed batches drive small updates) — not to protect on
                             its own. We test it to *measure* that, honestly.
  * RIPPLE + replay        — episodic replay consolidation (wavenn/episodic.py), the CLS mechanism
  * RIPPLE + replay + gate — the full complementary-learning-systems story
  * BPTT + replay          — the autograd twin with the same replay (baseline)

FAIRNESS: every learner does the SAME number of optimizer updates per domain (audit M1).
Replay learners spend a fraction `replay_frac` of that fixed budget rehearsing buffered
(past) data instead of fresh data — a method choice within the same compute, not extra steps.

Metrics (standard CL, Chaudhry et al. 2018): average final loss, average FORGETTING over the
first n−1 domains, backward transfer, and excess over the (correctly computed) Bayes floor.
Reported as mean ± std over several seeds (audit H1/M1/M4/M5).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace as dc_replace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavenn.episodic import ReservoirBuffer  # noqa: E402
from wavenn.ripple import LocalAdam  # noqa: E402
from wavenn.rlm import RLMConfig, RippleLM, ripple_grads  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class MarkovDomain:
    """A distinct next-token 'language': each token → k random successors with a random peaked
    distribution. Learnable; domains conflict in the shared readout/weights."""

    def __init__(self, vocab: int, k: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        T = torch.zeros(vocab, vocab)
        for i in range(vocab):
            succ = torch.randperm(vocab, generator=g)[:k]
            T[i, succ] = torch.softmax(torch.randn(k, generator=g) * 1.5, dim=0)
        self.T = T.to(DEV)
        self.vocab = vocab
        self.seed = seed
        self.reset_rng(0)
        # one pool of parallel chains (walk vectorized over chains ⇒ few sequential steps)
        n_chains, chain_len = 3000, 128
        pool = torch.zeros(n_chains, chain_len, dtype=torch.long, device=DEV)
        pool[:, 0] = torch.randint(0, vocab, (n_chains,), generator=self.gen, device=DEV)
        for t in range(chain_len - 1):
            pool[:, t + 1] = torch.multinomial(self.T[pool[:, t]], 1, generator=self.gen).squeeze(-1)
        self.pool = pool
        self.n_chains, self.chain_len = n_chains, chain_len
        # exact Bayes floor: NLL of the true model on the pool's transitions (audit H1) —
        # = marginal-weighted conditional entropy H(next|current), the achievable optimum.
        curr, nxt = pool[:, :-1].reshape(-1), pool[:, 1:].reshape(-1)
        self._floor = float(-(self.T[curr, nxt] + 1e-12).log().mean())

    def reset_rng(self, run_seed: int):
        """Re-seed the batch/eval generator so every learner sees byte-identical data (M2)."""
        self.gen = torch.Generator(device=DEV).manual_seed(self.seed * 100003 + run_seed)

    def bayes_nll(self) -> float:
        return self._floor

    def batch(self, B: int, L: int):
        rows = torch.randint(0, self.n_chains, (B,), generator=self.gen, device=DEV)
        offs = torch.randint(0, self.chain_len - L - 1, (B,), generator=self.gen, device=DEV)
        cols = offs[:, None] + torch.arange(L + 1, device=DEV)[None, :]
        win = self.pool[rows[:, None], cols]
        return win[:, :L].contiguous(), win[:, 1:].contiguous()


@torch.no_grad()
def eval_domain(model: RippleLM, dom: MarkovDomain, B: int, L: int, gen: torch.Generator, iters=16):
    tot = 0.0
    for _ in range(iters):
        rows = torch.randint(0, dom.n_chains, (B,), generator=gen, device=DEV)
        offs = torch.randint(0, dom.chain_len - L - 1, (B,), generator=gen, device=DEV)
        cols = offs[:, None] + torch.arange(L + 1, device=DEV)[None, :]
        win = dom.pool[rows[:, None], cols]
        x, y = win[:, :L], win[:, 1:]
        lg = model.logits(x)
        tot += F.cross_entropy(lg.reshape(-1, model.cfg.vocab_size), y.reshape(-1)).item()
    return tot / iters


def continual_run(mode, domains, mcfg, *, steps=400, B=32, L=48, lr=2e-3,
                  replay_frac=0.4, buffer_cap=3000, seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    for d in domains:
        d.reset_rng(seed)  # byte-identical batches across modes (M2)
    eval_gen = torch.Generator(device=DEV).manual_seed(12345)  # fixed eval stream
    mcfg = dc_replace(mcfg, gate_g1=(0.5 if "gate" in mode else 0.0))  # gate is the ONLY diff (N3)
    model = RippleLM(mcfg).to(DEV)
    use_replay = "replay" in mode
    is_bptt = mode.startswith("bptt")
    if is_bptt:
        opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))  # match LocalAdam (N4)
    else:
        opt = LocalAdam(dict(model.named_parameters()), lr=lr)
    buf = ReservoirBuffer(buffer_cap, seed=seed)
    rep_gen = torch.Generator(device=DEV).manual_seed(seed + 7)

    def step_update(x, y, lr_t):
        if is_bptt:
            for gp in opt.param_groups:
                gp["lr"] = lr_t
            loss = model.bptt_loss(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            opt.lr = lr_t
            _, g = ripple_grads(model, x, y, clip_norm=1.0)
            opt.step(g, clip=None)  # ripple_grads already clipped (before the gate)

    timeline = []
    for dom in domains:
        for step in range(1, steps + 1):
            lr_t = lr * min(1.0, step / 30)
            # FIXED budget: each step is EITHER a fresh step OR (for replay modes, with prob
            # replay_frac once the buffer is warm) a rehearsal step on buffered past data.
            do_replay = (use_replay and len(buf) > 0
                         and float(torch.rand(1, device=DEV, generator=rep_gen)) < replay_frac)
            if do_replay:
                xr, yr = buf.sample(B, DEV)
                step_update(xr, yr, lr_t)
            else:
                x, y = dom.batch(B, L)
                step_update(x, y, lr_t)
                buf.add(x, y)
        timeline.append([eval_domain(model, dm, B, L, eval_gen) for dm in domains])
    return timeline


def metrics(timeline, floors):
    n = len(timeline)
    final = timeline[-1]
    # forgetting_d = final loss on d − loss right after training d; averaged over the first
    # n−1 domains (the last cannot be forgotten) — standard CL (audit M5).
    forgetting = [final[d] - timeline[d][d] for d in range(n - 1)]
    return {
        "avg_final_loss": sum(final) / n,
        "avg_forgetting": sum(forgetting) / max(1, len(forgetting)),
        "backward_transfer": -sum(forgetting) / max(1, len(forgetting)),  # +ve = retained/improved
        "avg_excess_over_bayes": sum(final[d] - floors[d] for d in range(n)) / n,
        "final_per_domain": [round(v, 4) for v in final],
        "forgetting_per_domain": [round(v, 4) for v in forgetting],
    }


def _agg(dicts, key):
    vals = torch.tensor([d[key] for d in dicts])
    return float(vals.mean()), float(vals.std())


def main():
    vocab, n_domains, seeds = 256, 3, (0, 1, 2)
    modes = ["naive", "gate", "replay", "replay_gate", "bptt_naive", "bptt_replay"]
    mcfg = RLMConfig(vocab_size=vocab, d_model=96, n_layers=1, n_heads=4, beta=1.0,
                     beta_readout=6.0, n_settle=15, key_full=False, delta_rule=False)

    per_seed = {m: [] for m in modes}
    for sd in seeds:
        domains = [MarkovDomain(vocab, k=6, seed=100 + i) for i in range(n_domains)]
        floors = [d.bayes_nll() for d in domains]
        print(f"\n##### seed {sd} — Bayes floors {[round(f,3) for f in floors]} #####", flush=True)
        for mode in modes:
            tl = continual_run(mode, domains, mcfg, steps=300, seed=sd)
            m = metrics(tl, floors)
            per_seed[mode].append(m)
            print(f"  [{mode:12s}] final {m['avg_final_loss']:.3f}  forget {m['avg_forgetting']:+.3f}  "
                  f"per-dom-final {m['final_per_domain']}", flush=True)

    summary = {}
    for mode in modes:
        summary[mode] = {
            "avg_final_loss": _agg(per_seed[mode], "avg_final_loss"),
            "avg_forgetting": _agg(per_seed[mode], "avg_forgetting"),
            "backward_transfer": _agg(per_seed[mode], "backward_transfer"),
            "avg_excess_over_bayes": _agg(per_seed[mode], "avg_excess_over_bayes"),
        }
    json.dump({"seeds": list(seeds), "summary": summary, "per_seed": per_seed},
              open(os.path.join(_HERE, "S7-continual-results.json"), "w"), indent=2)

    print("\n=== CONTINUAL-LEARNING SUMMARY (mean±std over seeds; lower forget = better) ===")
    print(f"{'mode':14s} {'final_loss':>16s} {'forgetting':>16s} {'excess/Bayes':>16s}")
    for m in modes:
        s = summary[m]
        f = lambda k: f"{s[k][0]:.3f}±{s[k][1]:.3f}"
        print(f"{m:14s} {f('avg_final_loss'):>16s} {f('avg_forgetting'):>16s} {f('avg_excess_over_bayes'):>16s}")


if __name__ == "__main__":
    main()
