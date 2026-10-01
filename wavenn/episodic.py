"""Episodic memory + replay consolidation — the MEDIUM plasticity timescale (docs/06 §1, M1).

The core model has three coupled plasticity timescales (docs/03 §1): the fast running
state `S_t` (in-context / test-time learning, native), the SLOW weights (RIPPLE updates),
and — implemented here — the MEDIUM **episodic** timescale: a bounded buffer of past
samples that is **replayed** to consolidate them into the slow weights, exactly the
complementary-learning-systems mechanism (McClelland et al. 1995; sharp-wave-ripple
replay). Replay is what lets the model acquire new domains **without catastrophically
forgetting** earlier ones.

The buffer supports two retention policies:
  * **uniform** (default): Algorithm-R reservoir sampling — a uniform-in-expectation sample
    of everything seen with O(capacity) memory, so replay covers the whole history evenly.
  * **surprise-prioritized** (`priority` given to :meth:`add`): Efraimidis–Spirakis A-Res
    weighted reservoir — items are kept in proportion to their priority (e.g. prediction
    surprise), realizing the "episodic buffer of high-surprise bindings" of docs/06 §1.

The replay step itself is just an ordinary RIPPLE update on a buffered batch — replay and
learning are the same operation (M1).
"""

from __future__ import annotations

import heapq

import torch


class ReservoirBuffer:
    """Fixed-capacity buffer of past training rows. Uniform (Algorithm R) by default;
    priority-weighted (A-Res) when :meth:`add` is given per-row priorities."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.x: list[torch.Tensor] = []
        self.y: list[torch.Tensor] = []
        self.n_seen = 0
        self.gen = torch.Generator().manual_seed(seed)
        # min-heap of (key, slot) for the weighted policy; key = u^(1/priority)
        self._heap: list[tuple[float, int]] = []

    def add(self, x: torch.Tensor, y: torch.Tensor, priority: torch.Tensor | None = None):
        """Add a batch (each row a sequence). ``priority`` (per row, >0) enables A-Res
        weighted retention (higher priority ⇒ more likely to be kept); else uniform."""
        x, y = x.detach().cpu(), y.detach().cpu()
        n = x.shape[0]
        pr = priority.detach().cpu() if priority is not None else None
        for i in range(n):
            self.n_seen += 1
            if pr is None:
                # Algorithm R (uniform reservoir): pre-incremented n_seen; keep with prob cap/n_seen
                if len(self.x) < self.capacity:
                    self.x.append(x[i])
                    self.y.append(y[i])
                else:
                    j = int(torch.randint(0, self.n_seen, (1,), generator=self.gen))
                    if j < self.capacity:
                        self.x[j], self.y[j] = x[i], y[i]
            else:
                # A-Res weighted reservoir: key = u^(1/w); keep the `capacity` largest keys
                u = float(torch.rand(1, generator=self.gen))
                w = max(float(pr[i]), 1e-6)
                key = u ** (1.0 / w)
                if len(self.x) < self.capacity:
                    self.x.append(x[i])
                    self.y.append(y[i])
                    heapq.heappush(self._heap, (key, len(self.x) - 1))
                elif key > self._heap[0][0]:
                    _, slot = heapq.heapreplace(self._heap, (key, self._heap[0][1]))
                    self.x[slot], self.y[slot] = x[i], y[i]

    def __len__(self):
        return len(self.x)

    def sample(self, batch: int, device: str):
        """Draw a random replay batch (with replacement) → (x, y) on ``device``."""
        idx = torch.randint(0, len(self.x), (batch,), generator=self.gen)
        x = torch.stack([self.x[i] for i in idx]).to(device)
        y = torch.stack([self.y[i] for i in idx]).to(device)
        return x, y
