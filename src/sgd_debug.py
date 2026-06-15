"""Debug helpers for SGD spectral loss (kept separate from loss computation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GraphConfig:
    knn_neighbors: int
    num_eigenvectors: int
    laplacian_type: str


@dataclass
class ModalSpectralOutcome:
    loss: torch.Tensor
    num_nodes: int = 0
    vision_nodes: int = 0
    text_nodes: int = 0
    total_nodes: int = 0
    valid: bool = False
    skip_reason: Optional[str] = None
    w_teacher: Optional[torch.Tensor] = None
    w_student: Optional[torch.Tensor] = None


def loss_to_float(loss: torch.Tensor) -> float:
    return float(loss.detach().item()) if torch.isfinite(loss) else float("nan")


def metric_tensor(device, value: float) -> torch.Tensor:
    return torch.tensor(value, device=device, dtype=torch.float32)


def new_sample_extraction_debug(batch_idx: int, side: str, has_image: bool, num_text: int) -> dict:
    return {
        "batch_idx": batch_idx,
        "side": side,
        "has_image": bool(has_image),
        "num_text": int(num_text),
        "vision": {},
        "text": {},
    }


def sample_extraction_needs_warning(sample_debug: dict) -> bool:
    vision_ok = sample_debug.get("vision", {}).get("vision_reps_valid", False)
    text_ok = sample_debug.get("text", {}).get("text_reps_valid", True)
    if sample_debug.get("has_image") and not vision_ok:
        return True
    if sample_debug.get("num_text", 0) > 0 and not text_ok:
        return True
    return False


def summarize_weight_graph(
    W: Optional[torch.Tensor],
    *,
    knn_neighbors: int,
    num_eigenvectors: int,
    laplacian_type: str,
) -> Dict[str, Any]:
    """Summarize kNN / bipartite graph used by Grassman loss."""
    if W is None:
        return {"skip": "graph_not_built"}
    n = int(W.size(0))
    if n == 0:
        return {"nodes": 0, "skip": "empty_graph"}

    k_eff = min(knn_neighbors, n - 1) if n >= 2 else 0
    k_eig = min(num_eigenvectors, n - 1) if n >= 3 else 0
    with torch.no_grad():
        nonzero = int((W > 0).sum().item())
        edge_count = nonzero // 2
        if nonzero > 0:
            positive = W[W > 0]
            w_min = float(positive.min().item())
            w_max = float(positive.max().item())
            w_mean = float(positive.mean().item())
        else:
            w_min = w_max = w_mean = 0.0

    return {
        "nodes": n,
        "edges": edge_count,
        "knn_k_config": knn_neighbors,
        "knn_k_effective": k_eff,
        "eigenvectors_requested": num_eigenvectors,
        "eigenvectors_used": k_eig,
        "laplacian_type": laplacian_type,
        "weight_min": w_min,
        "weight_max": w_max,
        "weight_mean": w_mean,
    }


def summarize_graph_pair(
    w_teacher: torch.Tensor,
    w_student: torch.Tensor,
    graph_cfg: GraphConfig,
) -> tuple:
    kwargs = dict(
        knn_neighbors=graph_cfg.knn_neighbors,
        num_eigenvectors=graph_cfg.num_eigenvectors,
        laplacian_type=graph_cfg.laplacian_type,
    )
    return (
        summarize_weight_graph(w_teacher, **kwargs),
        summarize_weight_graph(w_student, **kwargs),
    )


def build_vision_modal_debug(outcome: ModalSpectralOutcome, graph_cfg: GraphConfig) -> dict:
    debug: Dict[str, Any] = {"batch_vision_nodes": outcome.num_nodes}
    if outcome.skip_reason:
        debug["skip_reason"] = outcome.skip_reason
        return debug
    if outcome.valid and outcome.w_teacher is not None and outcome.w_student is not None:
        debug["graph_teacher"], debug["graph_student"] = summarize_graph_pair(
            outcome.w_teacher, outcome.w_student, graph_cfg,
        )
        debug["vision_loss_valid"] = True
    return debug


def build_text_modal_debug(outcome: ModalSpectralOutcome, graph_cfg: GraphConfig) -> dict:
    debug: Dict[str, Any] = {"batch_text_nodes": outcome.num_nodes}
    if outcome.skip_reason:
        debug["skip_reason"] = outcome.skip_reason
        return debug
    if outcome.valid and outcome.w_teacher is not None and outcome.w_student is not None:
        debug["graph_teacher"], debug["graph_student"] = summarize_graph_pair(
            outcome.w_teacher, outcome.w_student, graph_cfg,
        )
        debug["text_loss_valid"] = True
    return debug


def build_cross_modal_debug(outcome: ModalSpectralOutcome, graph_cfg: GraphConfig) -> dict:
    debug: Dict[str, Any] = {
        "vision_nodes": outcome.vision_nodes,
        "text_nodes": outcome.text_nodes,
        "total_nodes": outcome.total_nodes,
        "cross_loss_valid": outcome.valid,
    }
    if outcome.skip_reason:
        debug["skip_reason"] = outcome.skip_reason
        return debug
    if outcome.valid and outcome.w_teacher is not None and outcome.w_student is not None:
        debug["graph_teacher"], debug["graph_student"] = summarize_graph_pair(
            outcome.w_teacher, outcome.w_student, graph_cfg,
        )
    return debug


def build_batch_side_debug_entry(
    side: str,
    loss_v: torch.Tensor,
    loss_t: torch.Tensor,
    loss_cross: torch.Tensor,
    vision_debug: dict,
    text_debug: dict,
    cross_debug: dict,
) -> dict:
    return {
        "type": "batch_side",
        "side": side,
        "vision": vision_debug,
        "text": text_debug,
        "cross": cross_debug,
        "losses": {
            "v": loss_to_float(loss_v),
            "t": loss_to_float(loss_t),
            "cross": loss_to_float(loss_cross),
        },
    }


def batch_side_stats_from_debug(batch_debug: dict) -> dict:
    vision = batch_debug.get("vision", {})
    text = batch_debug.get("text", {})
    cross = batch_debug.get("cross", {})
    return {
        "vision_nodes": int(vision.get("batch_vision_nodes", 0)),
        "text_nodes": int(text.get("batch_text_nodes", 0)),
        "total_nodes": int(cross.get("total_nodes", 0)),
    }


class SGDSpectralDebugSession:
    def __init__(self):
        self.entries: List[dict] = []
        self.batch_stats = {
            "qry_vision_nodes": 0,
            "qry_text_nodes": 0,
            "pos_vision_nodes": 0,
            "pos_text_nodes": 0,
        }

    def maybe_record_sample_warning(self, sample_debug: dict) -> None:
        if sample_extraction_needs_warning(sample_debug):
            self.entries.append(sample_debug)

    def record_batch_side(self, side: str, batch_debug: dict) -> None:
        self.entries.append(batch_debug)
        stats = batch_side_stats_from_debug(batch_debug)
        self.batch_stats[f"{side}_vision_nodes"] = stats["vision_nodes"]
        self.batch_stats[f"{side}_text_nodes"] = stats["text_nodes"]


def build_sgd_loss_dict(
    device,
    total_loss: torch.Tensor,
    contrastive_loss: torch.Tensor,
    rkd_loss: torch.Tensor,
    spectral_loss: torch.Tensor,
    spectral_loss_v: torch.Tensor,
    spectral_loss_t: torch.Tensor,
    spectral_loss_cross: torch.Tensor,
    batch_stats: dict,
) -> dict:
    return {
        "loss": total_loss,
        "contrastive_loss": contrastive_loss,
        "rkd_loss": rkd_loss,
        "spectral_loss": spectral_loss,
        "spectral_loss_v": spectral_loss_v,
        "spectral_loss_t": spectral_loss_t,
        "spectral_loss_cross": spectral_loss_cross,
        "batch_vision_nodes_qry": metric_tensor(device, batch_stats["qry_vision_nodes"]),
        "batch_text_nodes_qry": metric_tensor(device, batch_stats["qry_text_nodes"]),
        "batch_vision_nodes_pos": metric_tensor(device, batch_stats["pos_vision_nodes"]),
        "batch_text_nodes_pos": metric_tensor(device, batch_stats["pos_text_nodes"]),
    }


def _fmt_graph(name: str, graph: Dict[str, Any]) -> str:
    if graph.get("skip"):
        return f"{name}: SKIP ({graph['skip']})"
    return (
        f"{name}: nodes={graph.get('nodes', 0)} edges={graph.get('edges', 0)} "
        f"k_eff={graph.get('knn_k_effective', 0)}/{graph.get('knn_k_config', 0)} "
        f"eig={graph.get('eigenvectors_used', 0)}/{graph.get('eigenvectors_requested', 0)} "
        f"w=[{graph.get('weight_min', 0):.3e},{graph.get('weight_max', 0):.3e}]"
    )


def _split_grassman_debug_entries(grassman_debug: list):
    batch_entries = [e for e in grassman_debug if e.get("type") == "batch_side"]
    sample_entries = [e for e in grassman_debug if e.get("type") != "batch_side"]
    return batch_entries, sample_entries


def _format_batch_side_debug_entry(entry: dict) -> list:
    side = entry.get("side", "?")
    vision = entry.get("vision") or {}
    text = entry.get("text") or {}
    cross = entry.get("cross") or {}
    losses = entry.get("losses") or {}

    lines = [
        f"  [batch/{side}] "
        f"vision_nodes={vision.get('batch_vision_nodes', 0)} "
        f"text_nodes={text.get('batch_text_nodes', 0)} "
        f"total_nodes={cross.get('total_nodes', 0)} "
        f"v_loss_ok={vision.get('vision_loss_valid', False)} "
        f"t_loss_ok={text.get('text_loss_valid', False)} "
        f"cross_loss_ok={cross.get('cross_loss_valid', False)}"
    ]
    for label, section in (("vision", vision), ("text", text), ("cross", cross)):
        if section.get("skip_reason"):
            lines.append(f"    {label}_skip={section['skip_reason']}")
    for graph_key, graph_label in (
        ("graph_teacher", "batch_vision_graph_t"),
        ("graph_student", "batch_vision_graph_s"),
    ):
        if vision.get(graph_key):
            lines.append(f"    {_fmt_graph(graph_label, vision[graph_key])}")
    for graph_key, graph_label in (
        ("graph_teacher", "batch_text_graph_t"),
        ("graph_student", "batch_text_graph_s"),
    ):
        if text.get(graph_key):
            lines.append(f"    {_fmt_graph(graph_label, text[graph_key])}")
    for graph_key, graph_label in (
        ("graph_teacher", "batch_cross_graph_t"),
        ("graph_student", "batch_cross_graph_s"),
    ):
        if cross.get(graph_key):
            lines.append(f"    {_fmt_graph(graph_label, cross[graph_key])}")
    lines.append(
        "    losses: "
        f"v={losses.get('v', 0):.4f} t={losses.get('t', 0):.4f} "
        f"cross={losses.get('cross', 0):.4f}"
    )
    return lines


def _format_sample_warning_entry(entry: dict) -> str:
    vision = entry.get("vision") or {}
    text = entry.get("text") or {}
    skip_parts = []
    if vision.get("skip_reason"):
        skip_parts.append(f"vision={vision['skip_reason']}")
    if text.get("skip_reason"):
        skip_parts.append(f"text={text['skip_reason']}")
    skip_str = ", ".join(skip_parts) if skip_parts else "reps_invalid"
    return (
        f"    [b{entry.get('batch_idx', '?')}/{entry.get('side', '?')}] "
        f"has_image={entry.get('has_image')} num_text={entry.get('num_text')} "
        f"{skip_str}"
    )


def _batch_side_debug_has_warning(entry: dict) -> bool:
    vision = entry.get("vision") or {}
    text = entry.get("text") or {}
    cross = entry.get("cross") or {}

    if vision.get("batch_vision_nodes", 0) >= 2 and not vision.get("vision_loss_valid", False):
        return True
    if text.get("batch_text_nodes", 0) >= 2 and not text.get("text_loss_valid", False):
        return True
    if cross.get("total_nodes", 0) >= 3 and not cross.get("cross_loss_valid", False):
        return True
    for section in (vision, text, cross):
        for key in ("graph_teacher", "graph_student"):
            graph = section.get(key) or {}
            nodes = graph.get("nodes", 0)
            if graph and nodes > 0 and nodes < 3:
                return True
    return False


def _sample_warning_debug_has_warning(entry: dict) -> bool:
    vision = entry.get("vision") or {}
    text = entry.get("text") or {}
    return bool(vision.get("skip_reason") or text.get("skip_reason"))


def format_grassman_debug_lines(grassman_debug: list) -> list:
    if not grassman_debug:
        return ["grassman: no entries"]

    batch_entries, sample_entries = _split_grassman_debug_entries(grassman_debug)
    lines = [
        f"grassman: batch_sides={len(batch_entries)} "
        f"sample_warnings={len(sample_entries)}"
    ]

    for entry in batch_entries:
        lines.extend(_format_batch_side_debug_entry(entry))

    if sample_entries:
        lines.append("  sample extraction warnings:")
        lines.extend(_format_sample_warning_entry(entry) for entry in sample_entries)

    return lines


def grassman_debug_has_warning(grassman_debug: list) -> bool:
    batch_entries, sample_entries = _split_grassman_debug_entries(grassman_debug)
    if any(_batch_side_debug_has_warning(entry) for entry in batch_entries):
        return True
    if any(_sample_warning_debug_has_warning(entry) for entry in sample_entries):
        return True
    return False
