"""Unit tests for Grassman loss normalization."""

import importlib.util
from pathlib import Path

import torch

_SGD_LOSS_PATH = Path(__file__).resolve().parents[1] / "src" / "criterions" / "sgd_loss.py"
_spec = importlib.util.spec_from_file_location("sgd_loss", _SGD_LOSS_PATH)
sgd_loss = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(sgd_loss)

compute_grassman_loss = sgd_loss.compute_grassman_loss


def test_grassman_loss_uses_mean_not_sum():
    n = 10
    teacher = torch.eye(n)
    student = torch.eye(n) * 2.0
    loss = compute_grassman_loss(teacher, student)
    expected = ((teacher - student) ** 2).mean()
    assert torch.allclose(loss, expected)
    assert loss.item() < 2.0  # mean scale, not sum (~100 for n=10)


def test_grassman_loss_zero_when_identical():
    espace = torch.randn(8, 8)
    loss = compute_grassman_loss(espace, espace.clone())
    assert loss.item() == 0.0
