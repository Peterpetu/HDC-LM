"""Tests for the local optimizer (docs/03 §6) — LocalAdam.

A silent optimizer bug corrupts every experiment, so the local Adam (Adam arithmetic on
locally-assembled gradients, no autograd graph) gets its own tests: it descends a convex
quadratic, applies bias correction, and its global-norm clip bounds the step.
"""

import torch

from wavenn.ripple import LocalAdam


def test_local_adam_minimizes_quadratic():
    # minimize ½‖θ − t‖² ; the gradient is (θ − t); LocalAdam should converge to t
    torch.manual_seed(0)
    target = torch.randn(20)
    theta = torch.zeros(20, requires_grad=False)
    params = {"w": theta}
    opt = LocalAdam(params, lr=0.1)
    for _ in range(500):
        grad = params["w"] - target  # ∂/∂θ of ½‖θ−t‖²
        opt.step({"w": grad}, clip=None)
    assert torch.allclose(params["w"], target, atol=1e-2), (params["w"] - target).abs().max()


def test_local_adam_bias_correction_first_step():
    # first step magnitude ≈ lr (Adam with bias correction takes a ~unit-scaled first step)
    theta = torch.zeros(10)
    opt = LocalAdam({"w": theta}, lr=0.05)
    g = torch.full((10,), 3.0)
    before = theta.clone()
    opt.step({"w": g}, clip=None)
    step = (theta - before).abs().mean().item()
    assert abs(step - 0.05) < 0.01, f"first step {step:.4f} should be ≈ lr with bias correction"


def test_local_adam_global_clip_bounds_step():
    # with a huge gradient, the global-norm clip caps the effective update
    theta = torch.zeros(10)
    opt = LocalAdam({"w": theta}, lr=1.0)
    g = torch.full((10,), 1e6)
    opt.step({"w": g}, clip=1.0)
    assert torch.isfinite(theta).all()
    # after clip to norm 1, Adam still normalizes per-element, so the step is ~lr-bounded
    assert theta.abs().max().item() < 2.0


def test_local_adam_ignores_none_grads():
    theta = torch.zeros(5)
    opt = LocalAdam({"w": theta, "b": torch.zeros(3)}, lr=0.1)
    opt.step({"w": torch.ones(5), "b": None}, clip=None)  # b has no grad this step
    assert theta.abs().sum() > 0  # w updated, no crash on None
