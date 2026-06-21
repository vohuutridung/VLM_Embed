"""Unit tests for Teacher-Anchored Facility Location token selection."""

import importlib.util
import math
from pathlib import Path

import torch

_SGD_LOSS_PATH = Path(__file__).resolve().parents[1] / "src" / "criterions" / "sgd_loss.py"
_spec = importlib.util.spec_from_file_location("sgd_loss", _SGD_LOSS_PATH)
sgd_loss = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(sgd_loss)

select_tokens_facility_location = sgd_loss.select_tokens_facility_location


def _make_clustered_tokens(num_per_cluster, cluster_centers, noise=0.01):
    """Build token rows around cluster centers."""
    rows = []
    for center in cluster_centers:
        for _ in range(num_per_cluster):
            rows.append(center + noise * torch.randn_like(center))
    return torch.stack(rows)


def test_toy_clusters_pick_one_per_group():
    """t1~t2, t3~t4, t5 isolated — should pick one rep per cluster, not both t1 and t2."""
    c12 = torch.tensor([1.0, 0.0, 0.0])
    c34 = torch.tensor([0.0, 1.0, 0.0])
    c5 = torch.tensor([0.0, 0.0, 1.0])

    tokens = torch.stack([
        c12 + torch.tensor([0.01, 0.0, 0.0]),
        c12 + torch.tensor([-0.01, 0.0, 0.0]),
        c34 + torch.tensor([0.0, 0.01, 0.0]),
        c34 + torch.tensor([0.0, -0.01, 0.0]),
        c5,
    ])

    selected_idx, info = select_tokens_facility_location(
        tokens,
        coverage_ratio=0.85,
        min_k=2,
        max_ratio=0.6,
    )

    assert info["k"] == 3
    assert set(selected_idx.tolist()) == {0, 2, 4} or set(selected_idx.tolist()) == {1, 3, 4}
    assert len(set(selected_idx.tolist())) == 3
    assert info["coverage"] >= 0.85
    assert not (0 in selected_idx.tolist() and 1 in selected_idx.tolist())
    assert not (2 in selected_idx.tolist() and 3 in selected_idx.tolist())


def test_small_n_returns_all():
    """N <= min_k must return all token indices."""
    tokens = torch.randn(2, 8)
    selected_idx, info = select_tokens_facility_location(
        tokens, min_k=2, max_ratio=0.6,
    )
    assert selected_idx.tolist() == [0, 1]
    assert info["k"] == 2
    assert info["N"] == 2
    assert info["coverage"] == 1.0
    assert info["k_ratio"] == 1.0


def test_random_tokens_no_duplicates_and_respects_max_k():
    torch.manual_seed(0)
    N, D = 20, 16
    tokens = torch.randn(N, D)

    selected_idx, info = select_tokens_facility_location(
        tokens,
        coverage_ratio=0.85,
        min_k=2,
        max_ratio=0.6,
    )

    max_k = min(max(int(N * 0.6), 2), N)
    assert selected_idx.numel() <= max_k
    assert len(selected_idx.unique()) == selected_idx.numel()
    assert info["k"] == selected_idx.numel()
    assert info["N"] == N
    if selected_idx.numel() >= 2:
        assert info["coverage"] >= 0.0
        assert math.isfinite(info["coverage"])


def test_nan_input_fallback():
    tokens = torch.randn(6, 4)
    tokens[2, 1] = float("nan")

    selected_idx, info = select_tokens_facility_location(tokens, min_k=2, max_ratio=0.6)

    assert selected_idx.tolist() == list(range(6))
    assert info["k"] == 6
    assert info["coverage"] == 1.0


def test_coverage_history_grows_with_selection():
    tokens = _make_clustered_tokens(
        2,
        [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0])],
    )
    _, info = select_tokens_facility_location(tokens, coverage_ratio=0.99, min_k=2, max_ratio=1.0)
    history = info["coverage_history"]
    assert len(history) >= 2
    assert history[-1] >= history[0]


def test_detach_does_not_require_grad():
    tokens = torch.randn(8, 4, requires_grad=True)
    selected_idx, _ = select_tokens_facility_location(tokens, detach=True)
    assert not selected_idx.requires_grad
    assert selected_idx.dtype == torch.long
