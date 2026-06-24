"""Trajectory-based cross-modal relation distillation for VLM2Vec.

Overview
--------
This module implements two cooperating components:

* :class:`TrajectoryLoss` — a **component loss** that distills teacher cross-modal
  relation trajectories (direct + two-hop cycle) from pre-aligned hidden states.
* :class:`ContrastiveLoss` — InfoNCE contrastive loss on student query↔positive
  embeddings (always active in :class:`TotalLoss`).
* :class:`TotalLoss` — the **top-level criterion** registered as ``kd_loss_type=trajectory``.
  It encodes student/teacher models, builds token alignment via
  :mod:`src.align_token_count`, and composes contrastive + trajectory terms.

End-to-end pipeline (``TotalLoss.forward``)::

    input_data
      → encode teacher (no grad) + student on query/positive pairs
      → :class:`ContrastiveLoss` on student embeddings
      → per sample: ``align_sample_tokens``  (A_v, A_t, masks; once per sample)
      → per layer:  ``extract_layer_sample`` → :class:`TrajectoryBatch`
      → :class:`TrajectoryLoss` (mean over layers)
      → total = contrastive + w_trajectory * trajectory + metric dict

Hyperparameters are defined in :class:`src.arguments.TrainingArguments` under the
**Trajectory distillation** section and parsed by :class:`TrajectoryLossConfig` /
:class:`TrajectoryAlignConfig`.

See Also
--------
:mod:`src.align_token_count` — visual/text token alignment and hidden-state slicing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.align_token_count import (
    VisualPreprocessMeta,
    align_text_hiddens_from_layer,
    align_text_tokens_from_input,
    align_visual_tokens_from_grids,
    count_text_tokens,
    extract_semantic_text,
    extract_vision_hidden_single,
    infer_image_size_from_vision_tokens,
    infer_vision_grid,
)

if TYPE_CHECKING:
    from src.arguments import TrainingArguments

MASK_FILL_VALUE = -1e4
"""Logit mask value for invalid keys before softmax in relation distributions."""

TRAJECTORY_KD_METRIC_KEYS = (
    "trajectory_loss",
    "direct_loss",
    "cycle_loss",
    "vt_loss",
    "tv_loss",
    "vv_loss",
    "tt_loss",
    "teacher_entropy_v",
    "teacher_entropy_t",
    "rho_v",
    "rho_t",
)
"""Component metrics from :class:`TrajectoryLoss` (excluding contrastive)."""

TOTAL_LOSS_METRIC_KEYS = ("loss", "contrastive_loss", *TRAJECTORY_KD_METRIC_KEYS)
"""Keys returned by :class:`TotalLoss` (used for zero-fill and logging)."""


# ---------------------------------------------------------------------------
# Configuration dataclasses (sourced from TrainingArguments)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryLossConfig:
    """Hyperparameters for :class:`TrajectoryLoss`.

    Attributes:
        tau: Softmax temperature for visual↔text conditional distributions.
        beta_direct: Weight of one-hop (direct) relation KL per layer.
        beta_cycle: Weight of two-hop (cycle) relation KL per layer.
        eps: Clamping epsilon for probabilities and denominators.
        eps_weight: Minimum teacher reliability ``rho`` floor.
        remove_self_in_cycle: If True, remove self-loops from cycle matrices
            before cycle KL.
    """

    tau: float
    beta_direct: float
    beta_cycle: float
    eps: float
    eps_weight: float
    remove_self_in_cycle: bool

    @classmethod
    def from_args(cls, args: TrainingArguments) -> TrajectoryLossConfig:
        """Build config from :class:`~src.arguments.TrainingArguments`."""
        return cls(
            tau=args.trajectory_tau,
            beta_direct=args.trajectory_beta_direct,
            beta_cycle=args.trajectory_beta_cycle,
            eps=args.trajectory_eps,
            eps_weight=args.trajectory_eps_weight,
            remove_self_in_cycle=args.trajectory_remove_self_in_cycle,
        )


@dataclass(frozen=True)
class TrajectoryAlignConfig:
    """Token-alignment settings for building A_v / A_t before TrajectoryLoss.

    Attributes:
        teacher_patch_size: Teacher vision patch size in pixels.
        student_patch_size: Student vision patch size in pixels.
        student_resize: Student canvas resize used for visual alignment.
        teacher_layers: Teacher hidden-state layer indices (one per trajectory step).
        student_layers: Student hidden-state layer indices (same length as teacher).
    """

    teacher_patch_size: int
    student_patch_size: int
    student_resize: int
    teacher_layers: Tuple[int, ...]
    student_layers: Tuple[int, ...]

    @classmethod
    def from_args(cls, args: TrainingArguments) -> TrajectoryAlignConfig:
        """Build alignment config from :class:`~src.arguments.TrainingArguments`."""
        teacher_layers = tuple(args.teacher_layer_mapping or (-1,))
        student_layers = tuple(args.student_layer_mapping or (-1,))
        if len(teacher_layers) != len(student_layers):
            raise ValueError(
                "teacher_layer_mapping and student_layer_mapping must have the same length, "
                f"got {len(teacher_layers)} and {len(student_layers)}"
            )
        return cls(
            teacher_patch_size=args.teacher_patch_size,
            student_patch_size=args.student_patch_size,
            student_resize=args.student_resize,
            teacher_layers=teacher_layers,
            student_layers=student_layers,
        )


# ---------------------------------------------------------------------------
# Structured intermediates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleTokenAlignment:
    """Token-level alignment for one (batch item, query|positive) pair.

    Independent of transformer layer: A_v, A_t and masks depend only on token
    geometry and text spans, not on hidden-state values.

    Attributes:
        av: Visual alignment matrix ``[R_v^S, R_v^T]`` (row-stochastic).
        at: Text alignment matrix ``[R_t^S, R_t^T]`` (row-stochastic).
        visual_mask: Valid student visual token rows ``[R_v^S]``.
        text_mask: Valid student text token rows ``[R_t^S]``.
        num_vision_student: Number of student vision tokens.
        num_vision_teacher: Number of teacher vision tokens.
        num_text: Number of text tokens (teacher count used for slicing).
    """

    av: torch.Tensor
    at: torch.Tensor
    visual_mask: torch.Tensor
    text_mask: torch.Tensor
    num_vision_student: int
    num_vision_teacher: int
    num_text: int


@dataclass
class AlignedLayerSample:
    """Vision/text hidden states for one sample at one selected layer.

    Attributes:
        student_visual: ``[R_v^S, d_S]``
        teacher_visual: ``[R_v^T, d_T]``
        student_text: ``[R_t^S, d_S]``
        teacher_text: ``[R_t^T, d_T]``
        token_alignment: Precomputed alignment reused across layers.
    """

    student_visual: torch.Tensor
    teacher_visual: torch.Tensor
    student_text: torch.Tensor
    teacher_text: torch.Tensor
    token_alignment: SampleTokenAlignment


@dataclass
class TrajectoryBatch:
    """Padded multi-sample batch fed into :class:`TrajectoryLoss`.

    All tensors share batch dimension ``B`` (number of valid aligned micro-samples
    pooled from query and positive sides). Sequence lengths are padded to the
    maximum within the batch.

    Attributes:
        student_visual: ``[B, R_v^S, d_S]``
        teacher_visual: ``[B, R_v^T, d_T]``
        student_text: ``[B, R_t^S, d_S]``
        teacher_text: ``[B, R_t^T, d_T]``
        av: ``[B, R_v^S, R_v^T]``
        at: ``[B, R_t^S, R_t^T]``
        visual_mask: ``[B, R_v^S]``
        text_mask: ``[B, R_t^S]``
    """

    student_visual: torch.Tensor
    teacher_visual: torch.Tensor
    student_text: torch.Tensor
    teacher_text: torch.Tensor
    av: torch.Tensor
    at: torch.Tensor
    visual_mask: torch.Tensor
    text_mask: torch.Tensor

    @classmethod
    def from_samples(cls, samples: Sequence[AlignedLayerSample]) -> TrajectoryBatch:
        """Pad variable-length aligned samples into a fixed-shape batch.

        Args:
            samples: Non-empty sequence of per-sample layer alignments.

        Returns:
            A :class:`TrajectoryBatch` with zero-padded tail positions and
            invalid mask rows left as ``False``.

        Raises:
            ValueError: If ``samples`` is empty.
        """
        if not samples:
            raise ValueError("Cannot build TrajectoryBatch from an empty sample list")

        max_rv_s = max(s.student_visual.size(0) for s in samples)
        max_rv_t = max(s.teacher_visual.size(0) for s in samples)
        max_rt_s = max(s.student_text.size(0) for s in samples)
        max_rt_t = max(s.teacher_text.size(0) for s in samples)

        device = samples[0].student_visual.device
        dtype_v = samples[0].student_visual.dtype
        dtype_t = samples[0].student_text.dtype
        n = len(samples)

        student_visual = torch.zeros(n, max_rv_s, samples[0].student_visual.size(-1), device=device, dtype=dtype_v)
        teacher_visual = torch.zeros(n, max_rv_t, samples[0].teacher_visual.size(-1), device=device, dtype=dtype_v)
        student_text = torch.zeros(n, max_rt_s, samples[0].student_text.size(-1), device=device, dtype=dtype_t)
        teacher_text = torch.zeros(n, max_rt_t, samples[0].teacher_text.size(-1), device=device, dtype=dtype_t)
        av = torch.zeros(n, max_rv_s, max_rv_t, device=device, dtype=dtype_v)
        at = torch.zeros(n, max_rt_s, max_rt_t, device=device, dtype=dtype_t)
        visual_mask = torch.zeros(n, max_rv_s, dtype=torch.bool, device=device)
        text_mask = torch.zeros(n, max_rt_s, dtype=torch.bool, device=device)

        for i, sample in enumerate(samples):
            rv_s, rv_t = sample.student_visual.size(0), sample.teacher_visual.size(0)
            rt_s, rt_t = sample.student_text.size(0), sample.teacher_text.size(0)
            align = sample.token_alignment

            student_visual[i, :rv_s] = sample.student_visual
            teacher_visual[i, :rv_t] = sample.teacher_visual
            student_text[i, :rt_s] = sample.student_text
            teacher_text[i, :rt_t] = sample.teacher_text
            av[i, :rv_s, :rv_t] = align.av
            at[i, :rt_s, :rt_t] = align.at
            visual_mask[i, :rv_s] = align.visual_mask
            text_mask[i, :rt_s] = align.text_mask

        return cls(
            student_visual=student_visual,
            teacher_visual=teacher_visual,
            student_text=student_text,
            teacher_text=teacher_text,
            av=av,
            at=at,
            visual_mask=visual_mask,
            text_mask=text_mask,
        )


# ---------------------------------------------------------------------------
# Trajectory relation math (internal)
# ---------------------------------------------------------------------------


def _zero_scalar(ref: torch.Tensor) -> torch.Tensor:
    """Return a zero scalar tensor on the same device/dtype graph as ``ref``."""
    return ref.sum() * 0.0


def _zero_trajectory_kd_output(ref: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Build zero trajectory-KD metrics (e.g. when no valid aligned samples exist)."""
    zero = _zero_scalar(ref)
    out = {key: zero for key in TRAJECTORY_KD_METRIC_KEYS}
    out["loss"] = zero
    return out


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    """Mean of ``x`` over rows, optionally restricted to ``mask`` positions."""
    if mask is None:
        return x.mean()
    mask_f = mask.to(dtype=x.dtype)
    return (x * mask_f).sum() / (mask_f.sum() + eps)


def _rowwise_kl(p_teacher: torch.Tensor, p_student: torch.Tensor, eps: float) -> torch.Tensor:
    """KL(P_teacher || P_student) summed over the last dimension (per row)."""
    p_teacher = p_teacher.clamp_min(eps)
    p_student = p_student.clamp_min(eps)
    return (p_teacher * (p_teacher.log() - p_student.log())).sum(dim=-1)


def _weighted_mean(
    row_values: torch.Tensor,
    weights: torch.Tensor,
    row_mask: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Weighted average of per-row scalars with optional row mask."""
    w = weights if row_mask is None else weights * row_mask.to(dtype=weights.dtype)
    return (row_values * w).sum() / (w.sum() + eps)


def _masked_softmax(
    cosine: torch.Tensor,
    temperature: float,
    key_mask: Optional[torch.Tensor],
    dim: int,
) -> torch.Tensor:
    """Temperature-scaled softmax with invalid keys masked to ``MASK_FILL_VALUE``."""
    logits = cosine / temperature
    if key_mask is not None:
        invalid = ~key_mask.unsqueeze(-2 if dim == -1 else -1)
        logits = logits.masked_fill(invalid, MASK_FILL_VALUE)
    return F.softmax(logits, dim=dim)


def _valid_key_counts(
    mask: Optional[torch.Tensor],
    num_keys: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-batch-item count of valid key tokens for entropy normalization."""
    if mask is None:
        return torch.full((batch_size,), num_keys, device=device, dtype=torch.long)
    return mask.sum(dim=-1).long()


def _teacher_reliability(
    probs: torch.Tensor,
    num_valid_keys: torch.Tensor,
    row_mask: Optional[torch.Tensor],
    eps: float,
    eps_weight: float,
) -> torch.Tensor:
    """Teacher row reliability ``rho = eps_weight + (1 - H_norm) * margin``.

    Low-entropy, high-margin teacher rows receive larger weights in the
    distillation objective. Output is intended to be detached before use.
    """
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
    denom = torch.log(num_valid_keys.float().clamp(min=2.0))
    norm_entropy = torch.where(
        num_valid_keys.unsqueeze(-1) > 1,
        entropy / denom.unsqueeze(-1),
        torch.zeros_like(entropy),
    )
    top2 = probs.topk(2, dim=-1).values
    margin = torch.where(
        num_valid_keys.unsqueeze(-1) <= 1,
        torch.ones_like(top2[..., 0]),
        top2[..., 0] - top2[..., 1],
    )
    rho = eps_weight + (1.0 - norm_entropy) * margin
    if row_mask is not None:
        rho = rho * row_mask.to(dtype=rho.dtype)
    return rho


def _remove_self_loops(probs: torch.Tensor, eps: float) -> torch.Tensor:
    """Zero the diagonal of a row-stochastic matrix and renormalize rows."""
    n = probs.size(-1)
    eye = torch.eye(n, device=probs.device, dtype=torch.bool).unsqueeze(0)
    out = probs.masked_fill(eye, 0.0)
    row_sum = out.sum(dim=-1, keepdim=True)
    return torch.where(row_sum > eps, out / row_sum.clamp_min(eps), out)


def _cross_modal_similarities(
    teacher_visual: torch.Tensor,
    teacher_text: torch.Tensor,
    student_visual: torch.Tensor,
    student_text: torch.Tensor,
    av: torch.Tensor,
    at: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute aligned teacher and student cross-modal cosine similarity matrices.

    Teacher similarities are aligned to student token resolution via
    ``C_T = A_v @ C_T_raw @ A_t^T``. Teacher tensors and alignment matrices
    are detached; student similarities remain in the autograd graph.

    Returns:
        ``(C_T, C_S)`` each of shape ``[B, R_v^S, R_t^S]``.
    """
    v_t = F.normalize(teacher_visual, dim=-1)
    t_t = F.normalize(teacher_text, dim=-1)
    v_s = F.normalize(student_visual, dim=-1)
    t_s = F.normalize(student_text, dim=-1)

    c_t_raw = v_t @ t_t.transpose(-1, -2)
    av_det, at_det = av.detach(), at.detach()
    c_t = (av_det @ c_t_raw @ at_det.transpose(-1, -2)).detach()
    c_s = v_s @ t_s.transpose(-1, -2)
    return c_t, c_s


def _direct_and_cycle_losses(
    c_t: torch.Tensor,
    c_s: torch.Tensor,
    cfg: TrajectoryLossConfig,
    visual_mask: Optional[torch.Tensor],
    text_mask: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute direct (one-hop) and cycle (two-hop) weighted KL losses for one layer.

    Direct terms align ``P_vt`` and ``P_tv``. Cycle terms align ``P_vv = P_vt P_tv``
    and ``P_tt = P_tv P_vt``. Teacher distributions and reliability weights are
    detached.

    Returns:
        Dict with ``loss``, component losses, and diagnostic teacher statistics.
    """
    batch_size = c_s.size(0)
    device = c_s.device
    num_visual, num_text = c_s.size(1), c_s.size(2)

    p_t_vt = _masked_softmax(c_t, cfg.tau, text_mask, dim=-1).detach()
    p_s_vt = _masked_softmax(c_s, cfg.tau, text_mask, dim=-1)
    p_t_tv = _masked_softmax(c_t.transpose(-1, -2), cfg.tau, visual_mask, dim=-1).detach()
    p_s_tv = _masked_softmax(c_s.transpose(-1, -2), cfg.tau, visual_mask, dim=-1)

    n_valid_text = _valid_key_counts(text_mask, num_text, batch_size, device)
    n_valid_visual = _valid_key_counts(visual_mask, num_visual, batch_size, device)

    rho_v = _teacher_reliability(p_t_vt, n_valid_text, visual_mask, cfg.eps, cfg.eps_weight).detach()
    rho_t = _teacher_reliability(p_t_tv, n_valid_visual, text_mask, cfg.eps, cfg.eps_weight).detach()

    l_vt = _weighted_mean(_rowwise_kl(p_t_vt, p_s_vt, cfg.eps), rho_v, visual_mask, cfg.eps)
    l_tv = _weighted_mean(_rowwise_kl(p_t_tv, p_s_tv, cfg.eps), rho_t, text_mask, cfg.eps)
    l_direct = l_vt + l_tv

    p_t_vv, p_s_vv = p_t_vt @ p_t_tv, p_s_vt @ p_s_tv
    p_t_tt, p_s_tt = p_t_tv @ p_t_vt, p_s_tv @ p_s_vt
    if cfg.remove_self_in_cycle:
        p_t_vv = _remove_self_loops(p_t_vv, cfg.eps)
        p_s_vv = _remove_self_loops(p_s_vv, cfg.eps)
        p_t_tt = _remove_self_loops(p_t_tt, cfg.eps)
        p_s_tt = _remove_self_loops(p_s_tt, cfg.eps)

    p_t_vv, p_t_tt = p_t_vv.detach(), p_t_tt.detach()
    l_vv = _weighted_mean(_rowwise_kl(p_t_vv, p_s_vv, cfg.eps), rho_v, visual_mask, cfg.eps)
    l_tt = _weighted_mean(_rowwise_kl(p_t_tt, p_s_tt, cfg.eps), rho_t, text_mask, cfg.eps)
    l_cycle = l_vv + l_tt

    h_t_v = -(p_t_vt * p_t_vt.clamp_min(cfg.eps).log()).sum(dim=-1)
    h_t_t = -(p_t_tv * p_t_tv.clamp_min(cfg.eps).log()).sum(dim=-1)

    layer_loss = cfg.beta_direct * l_direct + cfg.beta_cycle * l_cycle
    return {
        "loss": layer_loss,
        "direct_loss": l_direct,
        "cycle_loss": l_cycle,
        "vt_loss": l_vt,
        "tv_loss": l_tv,
        "vv_loss": l_vv,
        "tt_loss": l_tt,
        "teacher_entropy_v": _masked_mean(h_t_v, visual_mask, cfg.eps),
        "teacher_entropy_t": _masked_mean(h_t_t, text_mask, cfg.eps),
        "rho_v": _masked_mean(rho_v, visual_mask, cfg.eps),
        "rho_t": _masked_mean(rho_t, text_mask, cfg.eps),
    }


# ---------------------------------------------------------------------------
# Component loss
# ---------------------------------------------------------------------------


class TrajectoryLoss(nn.Module):
    """Distill teacher cross-modal relation trajectories across selected layers.

    This module compares **relation distributions** derived from hidden states,
    not raw hidden vectors. At each layer it builds visual↔text conditional
    distributions and aligns the student to the teacher via:

    1. Direct one-hop cross-modal matching (``P_vt``, ``P_tv``).
    2. Two-hop cycle matching (``P_vv``, ``P_tt``).
    3. Entropy/margin-based teacher reliability weighting on each row.

    Args:
        config: Hyperparameters from :class:`TrajectoryLossConfig`.
    """

    def __init__(self, config: TrajectoryLossConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        teacher_visual_hiddens: List[torch.Tensor],
        teacher_text_hiddens: List[torch.Tensor],
        student_visual_hiddens: List[torch.Tensor],
        student_text_hiddens: List[torch.Tensor],
        av: torch.Tensor,
        at: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run trajectory distillation over ``K`` aligned layers.

        Args:
            teacher_visual_hiddens: ``K`` tensors ``[B, R_v^T, d_T]``.
            teacher_text_hiddens: ``K`` tensors ``[B, R_t^T, d_T]``.
            student_visual_hiddens: ``K`` tensors ``[B, R_v^S, d_S]``.
            student_text_hiddens: ``K`` tensors ``[B, R_t^S, d_S]``.
            av: Visual alignment ``[B, R_v^S, R_v^T]`` (detached in loss).
            at: Text alignment ``[B, R_t^S, R_t^T]`` (detached in loss).
            visual_mask: Optional ``[B, R_v^S]`` valid visual rows.
            text_mask: Optional ``[B, R_t^S]`` valid text rows.

        Returns:
            Dict containing ``loss``, ``trajectory_loss``, component losses, and
            detached teacher diagnostics (entropy, ``rho``).
        """
        num_layers = len(student_visual_hiddens)
        ref = student_text_hiddens[0] if student_text_hiddens else av

        if num_layers == 0:
            return _zero_trajectory_kd_output(ref)

        expected = (len(teacher_visual_hiddens), len(teacher_text_hiddens), len(student_text_hiddens))
        if expected != (num_layers, num_layers, num_layers):
            raise ValueError(
                "All hidden-state lists must have the same length, got "
                f"teacher_visual={len(teacher_visual_hiddens)}, "
                f"teacher_text={len(teacher_text_hiddens)}, "
                f"student_visual={num_layers}, "
                f"student_text={len(student_text_hiddens)}"
            )

        layer_outputs = [
            self._forward_layer(
                teacher_visual_hiddens[k],
                teacher_text_hiddens[k],
                student_visual_hiddens[k],
                student_text_hiddens[k],
                av,
                at,
                visual_mask,
                text_mask,
            )
            for k in range(num_layers)
        ]
        return self._aggregate_layers(layer_outputs)

    def _forward_layer(
        self,
        teacher_visual: torch.Tensor,
        teacher_text: torch.Tensor,
        student_visual: torch.Tensor,
        student_text: torch.Tensor,
        av: torch.Tensor,
        at: torch.Tensor,
        visual_mask: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute trajectory loss for a single layer."""
        c_t, c_s = _cross_modal_similarities(teacher_visual, teacher_text, student_visual, student_text, av, at)
        return _direct_and_cycle_losses(c_t, c_s, self.config, visual_mask, text_mask)

    @staticmethod
    def _aggregate_layers(layer_outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Mean-pool per-layer metrics into a single output dict."""
        def mean_metric(key: str) -> torch.Tensor:
            return torch.stack([out[key] for out in layer_outputs]).mean()

        loss = mean_metric("loss")
        metrics = {key: mean_metric(key) for key in layer_outputs[0] if key != "loss"}
        metrics["loss"] = loss
        metrics["trajectory_loss"] = loss

        for key in ("teacher_entropy_v", "teacher_entropy_t", "rho_v", "rho_t"):
            metrics[key] = metrics[key].detach()
        return metrics


# ---------------------------------------------------------------------------
# Alignment & batch construction
# ---------------------------------------------------------------------------


def _align_visual_tokens(
    *,
    num_vision_student: int,
    num_vision_teacher: int,
    align_cfg: TrajectoryAlignConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Build visual alignment ``A_v`` and valid student visual mask.

    Delegates to :func:`src.align_token_count.align_visual_tokens_from_grids`.

    Returns:
        ``(av, visual_mask)`` or ``None`` if grid inference / alignment fails.
    """
    teacher_grid = infer_vision_grid(num_vision_teacher)
    student_grid = infer_vision_grid(num_vision_student)
    if teacher_grid is None or student_grid is None:
        return None

    img_w, img_h = infer_image_size_from_vision_tokens(num_vision_teacher, align_cfg.teacher_patch_size)
    teacher_rows, teacher_cols = teacher_grid
    student_canvas = float(align_cfg.student_resize)
    teacher_canvas = float(teacher_cols * align_cfg.teacher_patch_size)

    visual_align = align_visual_tokens_from_grids(
        num_student_tokens=num_vision_student,
        num_teacher_tokens=num_vision_teacher,
        student_patch_size=float(align_cfg.student_patch_size),
        teacher_patch_size=float(align_cfg.teacher_patch_size),
        student_meta=VisualPreprocessMeta(
            canvas_width=student_canvas,
            canvas_height=student_canvas,
            original_width=img_w,
            original_height=img_h,
        ),
        teacher_meta=VisualPreprocessMeta(
            canvas_width=teacher_canvas,
            canvas_height=float(teacher_rows * align_cfg.teacher_patch_size),
            original_width=img_w,
            original_height=img_h,
        ),
        student_grid=student_grid,
        teacher_grid=teacher_grid,
        map_to_original=True,
        device=device,
        dtype=dtype,
    )
    if not visual_align.valid_student.any():
        return None
    return visual_align.overlap_matrix, visual_align.valid_student


def align_sample_tokens(
    *,
    student_hs: Tuple[torch.Tensor, ...],
    teacher_hs: Tuple[torch.Tensor, ...],
    student_input: Dict,
    teacher_input: Dict,
    sample_idx: int,
    student_image_features,
    teacher_image_features,
    semantic_text: str,
    teacher_tokenizer,
    student_tokenizer,
    teacher_layer: int,
    student_layer: int,
    align_cfg: TrajectoryAlignConfig,
) -> Optional[SampleTokenAlignment]:
    """Build ``A_v``, ``A_t``, and masks for one sample (layer-agnostic).

    Hidden-state layer indices are used only to obtain device/dtype for
    alignment tensors. Token geometry does not change across layers.

    Args:
        student_hs: Student hidden-state tuple from ``encode_input``.
        teacher_hs: Teacher hidden-state tuple from ``encode_input``.
        student_input: Collated student processor inputs for this side.
        teacher_input: Collated teacher processor inputs for this side.
        sample_idx: Index within the batch dimension.
        student_image_features: Per-sample student vision token features list.
        teacher_image_features: Per-sample teacher vision token features list.
        semantic_text: De-templated text used for span alignment.
        teacher_tokenizer: Teacher tokenizer.
        student_tokenizer: Student tokenizer.
        teacher_layer: Any valid layer index (for device/dtype).
        student_layer: Any valid layer index (for device/dtype).
        align_cfg: Patch sizes and resize from training arguments.

    Returns:
        :class:`SampleTokenAlignment` or ``None`` if the sample cannot be aligned.
    """
    if (
        student_image_features is None
        or sample_idx >= len(student_image_features)
        or student_image_features[sample_idx] is None
    ):
        return None

    num_vision_student = student_image_features[sample_idx].size(0)
    num_vision_teacher = teacher_image_features[sample_idx].size(0)
    num_text = count_text_tokens(
        teacher_input["input_ids"][sample_idx],
        teacher_input["attention_mask"][sample_idx],
        teacher_tokenizer,
    )
    if num_text <= 0:
        return None

    layer_ref = student_hs[student_layer][sample_idx]

    visual = _align_visual_tokens(
        num_vision_student=num_vision_student,
        num_vision_teacher=num_vision_teacher,
        align_cfg=align_cfg,
        device=layer_ref.device,
        dtype=layer_ref.dtype,
    )
    if visual is None:
        return None
    av, visual_mask = visual

    text_align = align_text_tokens_from_input(
        teacher_tokenizer,
        student_tokenizer,
        teacher_input["input_ids"][sample_idx],
        teacher_input["attention_mask"][sample_idx],
        student_input["input_ids"][sample_idx],
        student_input["attention_mask"][sample_idx],
        semantic_text,
        dtype=layer_ref.dtype,
    )
    if text_align is None or not text_align.valid_student.any():
        return None

    return SampleTokenAlignment(
        av=av,
        at=text_align.overlap_matrix,
        visual_mask=visual_mask,
        text_mask=text_align.valid_student,
        num_vision_student=num_vision_student,
        num_vision_teacher=num_vision_teacher,
        num_text=num_text,
    )


def extract_layer_sample(
    token_alignment: SampleTokenAlignment,
    *,
    student_hs: Tuple[torch.Tensor, ...],
    teacher_hs: Tuple[torch.Tensor, ...],
    student_input: Dict,
    teacher_input: Dict,
    sample_idx: int,
    semantic_text: str,
    teacher_tokenizer,
    student_tokenizer,
    teacher_layer: int,
    student_layer: int,
) -> Optional[AlignedLayerSample]:
    """Slice vision/text hidden states for one layer using precomputed alignment.

    Args:
        token_alignment: Output of :func:`align_sample_tokens`.
        student_hs: Student hidden-state tuple.
        teacher_hs: Teacher hidden-state tuple.
        student_input: Collated student inputs for this side.
        teacher_input: Collated teacher inputs for this side.
        sample_idx: Batch index.
        semantic_text: De-templated text for hidden slicing.
        teacher_tokenizer: Teacher tokenizer.
        student_tokenizer: Student tokenizer.
        teacher_layer: Teacher layer index to read.
        student_layer: Student layer index to read.

    Returns:
        :class:`AlignedLayerSample` or ``None`` if hidden extraction fails.
    """
    align = token_alignment
    text_hiddens = align_text_hiddens_from_layer(
        teacher_hs[teacher_layer][sample_idx],
        student_hs[student_layer][sample_idx],
        teacher_input["input_ids"][sample_idx],
        student_input["input_ids"][sample_idx],
        teacher_input["attention_mask"][sample_idx],
        student_input["attention_mask"][sample_idx],
        teacher_tokenizer,
        student_tokenizer,
        semantic_text,
        num_vision_teacher=align.num_vision_teacher,
        num_vision_student=align.num_vision_student,
        has_image=True,
    )
    if text_hiddens is None:
        return None
    student_text, teacher_text, _, _ = text_hiddens

    student_visual = extract_vision_hidden_single(
        student_hs, sample_idx, align.num_vision_student, align.num_text,
        is_teacher=False, layer_idx=student_layer,
    )
    teacher_visual = extract_vision_hidden_single(
        teacher_hs, sample_idx, align.num_vision_teacher, align.num_text,
        is_teacher=True, layer_idx=teacher_layer,
    )
    if student_visual.numel() == 0 or teacher_visual.numel() == 0:
        return None

    return AlignedLayerSample(
        student_visual=student_visual,
        teacher_visual=teacher_visual,
        student_text=student_text,
        teacher_text=teacher_text,
        token_alignment=align,
    )


@dataclass(frozen=True)
class _SideSpec:
    """Internal bundle for one query or positive side of the batch."""

    student_hs: Tuple[torch.Tensor, ...]
    teacher_hs: Tuple[torch.Tensor, ...]
    student_image_features: Any
    teacher_image_features: Any
    student_input: Dict
    teacher_input: Dict
    decoded_texts: List[str]


def build_trajectory_batches(
    *,
    batch_size: int,
    sides: Sequence[_SideSpec],
    teacher_tokenizer,
    student_tokenizer,
    align_cfg: TrajectoryAlignConfig,
) -> Tuple[List[TrajectoryBatch], torch.Tensor]:
    """Align tokens once per sample, then extract hiddens for each layer.

    Args:
        batch_size: Number of items along the batch dimension.
        sides: Query and positive side specifications (typically length 2).
        teacher_tokenizer: Teacher tokenizer for span alignment.
        student_tokenizer: Student tokenizer for span alignment.
        align_cfg: Layer indices and patch geometry from training arguments.

    Returns:
        ``(batches, ref_tensor)`` where ``batches`` has one entry per selected
        layer and ``ref_tensor`` is a reference for zero-loss construction.

    Raises:
        ValueError: If no sample or no layer survives alignment / extraction.
    """
    token_alignments: List[SampleTokenAlignment] = []
    sample_refs: List[Tuple[_SideSpec, int, str]] = []

    teacher_layer0, student_layer0 = align_cfg.teacher_layers[0], align_cfg.student_layers[0]
    for side in sides:
        for sample_idx in range(batch_size):
            semantic_text = extract_semantic_text(side.decoded_texts[sample_idx])
            alignment = align_sample_tokens(
                student_hs=side.student_hs,
                teacher_hs=side.teacher_hs,
                student_input=side.student_input,
                teacher_input=side.teacher_input,
                sample_idx=sample_idx,
                student_image_features=side.student_image_features,
                teacher_image_features=side.teacher_image_features,
                semantic_text=semantic_text,
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=student_tokenizer,
                teacher_layer=teacher_layer0,
                student_layer=student_layer0,
                align_cfg=align_cfg,
            )
            if alignment is not None:
                token_alignments.append(alignment)
                sample_refs.append((side, sample_idx, semantic_text))

    if not token_alignments:
        raise ValueError("No valid aligned samples in batch")

    batches: List[TrajectoryBatch] = []
    for teacher_layer, student_layer in zip(align_cfg.teacher_layers, align_cfg.student_layers):
        layer_samples: List[AlignedLayerSample] = []
        for align, (side, sample_idx, semantic_text) in zip(token_alignments, sample_refs):
            sample = extract_layer_sample(
                align,
                student_hs=side.student_hs,
                teacher_hs=side.teacher_hs,
                student_input=side.student_input,
                teacher_input=side.teacher_input,
                sample_idx=sample_idx,
                semantic_text=semantic_text,
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=student_tokenizer,
                teacher_layer=teacher_layer,
                student_layer=student_layer,
            )
            if sample is not None:
                layer_samples.append(sample)
        if layer_samples:
            batches.append(TrajectoryBatch.from_samples(layer_samples))

    if not batches:
        raise ValueError("No valid layer samples after hidden extraction")

    ref = sides[0].student_hs[student_layer0]
    return batches, ref


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss over student query–positive embedding pairs.

    Uses in-batch negatives with optional distributed all-gather (same contract
    as other distillation criteria in this repo). Temperature is read from
    ``distiller.temperature``.
    """

    def __init__(self):
        super().__init__()
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.process_rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

    def _dist_gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, tensor)
        gathered[self.process_rank] = tensor
        return torch.cat(gathered, dim=0)

    def _get_dist_gather_fn(self):
        if self.world_size > 1:
            return self._dist_gather_tensor
        return None

    def forward(
        self,
        distiller,
        student_model,
        student_qry_reps: torch.Tensor,
        student_pos_reps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy contrastive loss on pooled student embeddings.

        Args:
            distiller: :class:`~src.distiller.Distiller` (provides ``temperature``).
            student_model: Student :class:`~src.model.model.MMEBModel`.
            student_qry_reps: Query embeddings ``[B, d]``.
            student_pos_reps: Positive embeddings ``[B, d]``.

        Returns:
            Scalar contrastive loss tensor.
        """
        dist_gather_fn = self._get_dist_gather_fn()
        if dist_gather_fn is not None:
            all_qry_reps = dist_gather_fn(student_qry_reps)
            all_pos_reps = dist_gather_fn(student_pos_reps)
        else:
            all_qry_reps = student_qry_reps
            all_pos_reps = student_pos_reps

        scores = student_model.compute_similarity(all_qry_reps, all_pos_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_pos_reps.size(0))
        return F.cross_entropy(scores / distiller.temperature, target)


# ---------------------------------------------------------------------------
# Top-level criterion
# ---------------------------------------------------------------------------


class TotalLoss(nn.Module):
    """Distillation criterion: contrastive + encode → align → :class:`TrajectoryLoss`.

    Registered in ``criterion_list`` as ``kd_loss_type="trajectory"``. The total
    objective is ``contrastive_loss + w_trajectory_loss * trajectory_loss``.

    Args:
        args: :class:`~src.arguments.TrainingArguments` (trajectory hyperparameters
            and layer mapping).
    """

    def __init__(self, args: TrainingArguments):
        super().__init__()
        self.w_trajectory = args.w_trajectory_loss
        self.align_cfg = TrajectoryAlignConfig.from_args(args)
        self.contrastive = ContrastiveLoss()
        self.trajectory = TrajectoryLoss(TrajectoryLossConfig.from_args(args))
        self._cached_student_tokenizer = None

    def _get_student_tokenizer(self, distiller):
        """Lazy-load and cache the student tokenizer from the distiller."""
        if self._cached_student_tokenizer is None:
            self._cached_student_tokenizer = distiller.get_student_processor().tokenizer
        return self._cached_student_tokenizer

    def forward(self, distiller, input_data, tokenizer=None) -> Dict[str, torch.Tensor]:
        """Compute the total distillation loss for one collated batch.

        Args:
            distiller: :class:`~src.distiller.Distiller` with student/teacher models.
            input_data: Collated batch from :class:`~src.distiller.DistillationCollator`
                with ``student_inputs`` and ``teacher_inputs`` (qry/pos).
            tokenizer: Teacher tokenizer (required).

        Returns:
            Dict with at least ``loss`` (scalar tensor for backward). Includes
            ``contrastive_loss`` and trajectory metrics from ``TOTAL_LOSS_METRIC_KEYS``.
        """
        if tokenizer is None:
            raise ValueError("TotalLoss requires teacher tokenizer")

        encodings = self._encode_all(distiller, input_data, tokenizer)
        contrastive_loss = self.contrastive(
            distiller,
            distiller.student,
            encodings["student_qry_reps"],
            encodings["student_pos_reps"],
        )

        if self.w_trajectory == 0:
            traj_out = _zero_trajectory_kd_output(encodings["ref"])
        else:
            try:
                batches, _ref = build_trajectory_batches(
                    batch_size=encodings["batch_size"],
                    sides=encodings["sides"],
                    teacher_tokenizer=tokenizer,
                    student_tokenizer=self._get_student_tokenizer(distiller),
                    align_cfg=self.align_cfg,
                )
                traj_out = self._run_trajectory(batches)
            except ValueError:
                traj_out = _zero_trajectory_kd_output(encodings["ref"])

        return self._build_output(contrastive_loss, traj_out)

    def _build_output(
        self,
        contrastive_loss: torch.Tensor,
        traj_out: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Combine contrastive and weighted trajectory terms into the return dict."""
        weighted_traj = self.w_trajectory * traj_out["loss"]
        outputs = {key: traj_out[key] for key in TRAJECTORY_KD_METRIC_KEYS if key in traj_out}
        outputs["contrastive_loss"] = contrastive_loss
        outputs["trajectory_loss"] = weighted_traj
        outputs["loss"] = contrastive_loss + weighted_traj
        return outputs

    def _encode_all(self, distiller, input_data, teacher_tokenizer) -> Dict:
        """Forward student and teacher on query/positive inputs."""
        student_model = distiller.student
        teacher_model = distiller.teacher

        s_qry = input_data["student_inputs"]["qry"]
        s_pos = input_data["student_inputs"]["pos"]
        t_qry = input_data["teacher_inputs"]["qry"]
        t_pos = input_data["teacher_inputs"]["pos"]

        with torch.no_grad():
            teacher_model.eval()
            _, t_qry_img, _, t_qry_hs = teacher_model.encode_input(t_qry)
            _, t_pos_img, _, t_pos_hs = teacher_model.encode_input(t_pos)

        s_qry_reps, s_qry_img, _, s_qry_hs = student_model.encode_input(s_qry)
        s_pos_reps, s_pos_img, _, s_pos_hs = student_model.encode_input(s_pos)

        batch_size = s_qry_hs[-1].size(0)
        qry_texts = teacher_tokenizer.batch_decode(t_qry["input_ids"], skip_special_tokens=True)
        pos_texts = teacher_tokenizer.batch_decode(t_pos["input_ids"], skip_special_tokens=True)

        return {
            "student_qry_reps": s_qry_reps,
            "student_pos_reps": s_pos_reps,
            "ref": s_qry_hs[-1],
            "batch_size": batch_size,
            "sides": (
                _SideSpec(s_qry_hs, t_qry_hs, s_qry_img, t_qry_img, s_qry, t_qry, qry_texts),
                _SideSpec(s_pos_hs, t_pos_hs, s_pos_img, t_pos_img, s_pos, t_pos, pos_texts),
            ),
        }

    def _run_trajectory(self, batches: List[TrajectoryBatch]) -> Dict[str, torch.Tensor]:
        """Unpack :class:`TrajectoryBatch` list and call :class:`TrajectoryLoss`."""
        teacher_visual_layers, teacher_text_layers = [], []
        student_visual_layers, student_text_layers = [], []
        av = at = visual_mask = text_mask = None

        for batch in batches:
            teacher_visual_layers.append(batch.teacher_visual)
            teacher_text_layers.append(batch.teacher_text)
            student_visual_layers.append(batch.student_visual)
            student_text_layers.append(batch.student_text)
            av, at = batch.av, batch.at
            visual_mask, text_mask = batch.visual_mask, batch.text_mask

        return self.trajectory(
            teacher_visual_layers,
            teacher_text_layers,
            student_visual_layers,
            student_text_layers,
            av,
            at,
            visual_mask=visual_mask,
            text_mask=text_mask,
        )


# if __name__ == "__main__":
#     import sys
#     from pathlib import Path

#     sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
#     torch.manual_seed(0)

#     b, rv_s, rt_s, rv_t, rt_t, ds, dt, k = 2, 4, 6, 5, 7, 32, 64, 3
#     av = torch.rand(b, rv_s, rv_t)
#     av = av / av.sum(dim=-1, keepdim=True).clamp_min(1e-8)
#     at = torch.rand(b, rt_s, rt_t)
#     at = at / at.sum(dim=-1, keepdim=True).clamp_min(1e-8)

#     cfg = TrajectoryLossConfig(
#         tau=0.07,
#         beta_direct=1.0,
#         beta_cycle=0.5,
#         eps=1e-8,
#         eps_weight=0.05,
#         remove_self_in_cycle=True,
#     )
#     loss_fn = TrajectoryLoss(cfg)
#     out = loss_fn(
#         [torch.randn(b, rv_t, dt) for _ in range(k)],
#         [torch.randn(b, rt_t, dt) for _ in range(k)],
#         [torch.randn(b, rv_s, ds, requires_grad=True) for _ in range(k)],
#         [torch.randn(b, rt_s, ds, requires_grad=True) for _ in range(k)],
#         av,
#         at,
#         visual_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
#         text_mask=torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool),
#     )
#     assert out["loss"].dim() == 0 and not torch.isnan(out["loss"])
#     out["loss"].backward()
#     print("TrajectoryLoss smoke test passed")
