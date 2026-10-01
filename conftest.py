"""Pytest bootstrap: make the repo root importable so `import wavenn` works, and keep
the global default dtype isolated between tests (the gradient-fidelity tests switch to
float64 for tight tolerance and must not leak that into the forward-pass tests)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    saved = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(saved)
