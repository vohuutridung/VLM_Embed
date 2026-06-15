import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_nan_debug_dir: Optional[str] = None
_nan_debug_log_path: Optional[str] = None
_nan_debug_events_dir: Optional[str] = None


def _is_main_process() -> bool:
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def configure_nan_debug_logging(output_dir: str) -> Optional[str]:
    """Create output_dir/nan_debug/ and route NaN debug dumps to that folder."""
    global _nan_debug_dir, _nan_debug_log_path, _nan_debug_events_dir

    if not _is_main_process():
        return None

    _nan_debug_dir = os.path.join(output_dir, "nan_debug")
    _nan_debug_events_dir = os.path.join(_nan_debug_dir, "events")
    os.makedirs(_nan_debug_events_dir, exist_ok=True)

    _nan_debug_log_path = os.path.join(_nan_debug_dir, "nan_debug.log")
    with open(_nan_debug_log_path, "a", encoding="utf-8") as f:
        f.write(
            f"\n===== NaN debug session started {datetime.now().isoformat()} =====\n"
        )

    return _nan_debug_dir


def get_nan_debug_dir() -> Optional[str]:
    return _nan_debug_dir


def _write_nan_debug_files(
    *,
    global_step: int,
    tag: str,
    message: str,
) -> None:
    if not _is_main_process() or _nan_debug_dir is None:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamped_message = f"[{timestamp}] {message}\n"

    if _nan_debug_log_path is not None:
        with open(_nan_debug_log_path, "a", encoding="utf-8") as f:
            f.write(stamped_message)

    if _nan_debug_events_dir is not None:
        event_path = os.path.join(
            _nan_debug_events_dir,
            f"step_{global_step:06d}_{tag}.log",
        )
        with open(event_path, "w", encoding="utf-8") as f:
            f.write(stamped_message)


def tensor_stats(tensor: Optional[torch.Tensor], name: str) -> str:
  """Compact numeric summary for one tensor."""
  if tensor is None:
    return f"{name}=None"
  if not isinstance(tensor, torch.Tensor):
    return f"{name}=<{type(tensor).__name__}>"

  with torch.no_grad():
    t = tensor.detach()
    finite = torch.isfinite(t)
    nan_count = torch.isnan(t).sum().item()
    inf_count = torch.isinf(t).sum().item()
    total = t.numel()

    if finite.any():
      finite_vals = t[finite].float()
      min_v = finite_vals.min().item()
      max_v = finite_vals.max().item()
      mean_v = finite_vals.mean().item()
      if t.dim() >= 1 and t.size(-1) > 1:
        norms = finite_vals.reshape(-1, t.size(-1)).norm(dim=-1)
        zero_norm = (norms < 1e-8).sum().item()
        norm_min = norms.min().item()
        norm_max = norms.max().item()
        norm_part = (
          f" | zero_norm={zero_norm}/{norms.numel()}"
          f" norm=[{norm_min:.2e},{norm_max:.2e}]"
        )
      else:
        norm_part = ""
    else:
      min_v = max_v = mean_v = float("nan")
      norm_part = ""

  shape = tuple(t.shape)
  dtype = str(t.dtype).replace("torch.", "")
  return (
    f"{name}: shape={shape} dtype={dtype}"
    f" nan={nan_count}/{total} inf={inf_count}/{total}"
    f" min={min_v:.4e} max={max_v:.4e} mean={mean_v:.4e}{norm_part}"
  )


def eos_hidden_states(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
  """Hidden states at the EOS / last real token (same indexing as MMEBModel._pooling)."""
  batch_size = last_hidden.shape[0]
  left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
  if left_padding:
    return last_hidden[torch.arange(batch_size, device=last_hidden.device), -1, :]

  max_length = last_hidden.size(1)
  num_padding_tokens = (attention_mask == 0).long().sum(dim=1)
  eos_indices = max_length - num_padding_tokens - 1
  return last_hidden[
    torch.arange(batch_size, device=last_hidden.device),
    eos_indices,
  ]


def attention_stats(attention, name: str) -> str:
  if attention is None:
    return f"{name}=None"
  if isinstance(attention, (list, tuple)):
    if len(attention) == 0:
      return f"{name}=[]"
    last = attention[-1]
    prefix = f"{name}[last/{len(attention) - 1}]"
    if last is None:
      return f"{prefix}=None"
    return tensor_stats(last, prefix)
  return tensor_stats(attention, name)


def module_param_stats(module: nn.Module, prefix: str = "student") -> str:
  nan_params = 0
  inf_params = 0
  total_params = 0
  bad_names = []
  max_abs = 0.0

  for name, param in module.named_parameters():
    if not param.requires_grad:
      continue
    total_params += param.numel()
    with torch.no_grad():
      nan_params += torch.isnan(param).sum().item()
      inf_params += torch.isinf(param).sum().item()
      if param.numel() > 0 and torch.isfinite(param).any():
        param_max = param.detach().float().abs().max().item()
        max_abs = max(max_abs, param_max)
      if torch.isnan(param).any() or torch.isinf(param).any():
        if len(bad_names) < 8:
          bad_names.append(name)

  bad_suffix = f" bad_tensors={bad_names}" if bad_names else ""
  return (
    f"{prefix}_trainable: total={total_params}"
    f" nan={nan_params} inf={inf_params}"
    f" max_abs={max_abs:.4e}{bad_suffix}"
  )


def module_grad_stats(module: nn.Module, prefix: str = "student") -> str:
  nan_grads = 0
  inf_grads = 0
  total_grad_elems = 0
  grad_norm_sq = 0.0
  bad_names = []

  for name, param in module.named_parameters():
    if not param.requires_grad or param.grad is None:
      continue
    grad = param.grad
    total_grad_elems += grad.numel()
    with torch.no_grad():
      nan_grads += torch.isnan(grad).sum().item()
      inf_grads += torch.isinf(grad).sum().item()
      if torch.isfinite(grad).any():
        grad_norm_sq += grad.detach().float().pow(2).sum().item()
      if torch.isnan(grad).any() or torch.isinf(grad).any():
        if len(bad_names) < 8:
          bad_names.append(name)

  grad_norm = grad_norm_sq ** 0.5 if grad_norm_sq > 0 else 0.0
  bad_suffix = f" bad_grads={bad_names}" if bad_names else ""
  return (
    f"{prefix}_grads: elems={total_grad_elems}"
    f" nan={nan_grads} inf={inf_grads}"
    f" l2_norm={grad_norm:.4e}{bad_suffix}"
  )


def log_nonfinite_losses(
    *,
    global_step: int,
    epoch_step: int,
    loss_dict: Dict[str, Any],
    extra_lines: Optional[list] = None,
    tag: str = "NAN_DEBUG",
) -> None:
  if not _is_main_process():
    return

  bad_losses = []
  for key, value in loss_dict.items():
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
      bad_losses.append(key)

  if not bad_losses and not extra_lines:
    return

  if bad_losses:
    header = f"[{tag}] step={global_step} epoch_step={epoch_step} nonfinite_losses={bad_losses}"
  else:
    header = f"[{tag}] step={global_step} epoch_step={epoch_step} grassman_warnings=True"
  lines = [header]
  for key in bad_losses:
    value = loss_dict[key]
    if isinstance(value, torch.Tensor):
      lines.append(f"  {key}={value.detach().float().item()}")
  if extra_lines:
    lines.extend(f"  {line}" for line in extra_lines)

  message = "\n".join(lines)
  logger.warning(message)
  _write_nan_debug_files(global_step=global_step, tag=tag, message=message)


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


def _fmt_graph(name: str, graph: Dict[str, Any]) -> str:
    if graph.get("skip"):
        return f"{name}: SKIP ({graph['skip']})"
    return (
        f"{name}: nodes={graph.get('nodes', 0)} edges={graph.get('edges', 0)} "
        f"k_eff={graph.get('knn_k_effective', 0)}/{graph.get('knn_k_config', 0)} "
        f"eig={graph.get('eigenvectors_used', 0)}/{graph.get('eigenvectors_requested', 0)} "
        f"w=[{graph.get('weight_min', 0):.3e},{graph.get('weight_max', 0):.3e}]"
    )


def format_grassman_debug_lines(grassman_debug: list) -> list:
    """Format per-sample Grassman cluster/graph stats for debug logs."""
    if not grassman_debug:
        return ["grassman: no samples recorded"]

    lines = [f"grassman: num_sample_sides={len(grassman_debug)}"]
    for entry in grassman_debug:
        header = (
            f"  [b{entry.get('batch_idx', '?')}/{entry.get('side', '?')}] "
            f"has_image={entry.get('has_image')} num_text={entry.get('num_text')}"
        )
        lines.append(header)

        vision = entry.get("vision") or {}
        if vision:
            use_cluster = vision.get("use_cluster", True)
            cluster_sizes = vision.get("cluster_sizes", [])
            sizes_str = ",".join(str(s) for s in cluster_sizes[:12])
            if len(cluster_sizes) > 12:
                sizes_str += ",..."
            if use_cluster:
                lines.append(
                    "    vision_cluster: "
                    f"teacher_tokens={vision.get('teacher_tokens', 0)} "
                    f"student_tokens={vision.get('student_tokens', 0)} "
                    f"dbscan_clusters={vision.get('dbscan_clusters', 0)} "
                    f"noise_tokens={vision.get('noise_tokens', 0)} "
                    f"valid_clusters={vision.get('valid_clusters', 0)} "
                    f"sizes=[{sizes_str}] "
                    f"mapped_student_tokens={vision.get('mapped_student_tokens', 0)} "
                    f"graph_nodes_t={vision.get('teacher_graph_nodes', vision.get('teacher_cluster_nodes'))} "
                    f"graph_nodes_s={vision.get('student_graph_nodes', vision.get('student_cluster_nodes'))} "
                    f"loss_valid={vision.get('vision_loss_valid', False)}"
                    + (f" skip={vision.get('skip_reason')}" if vision.get("skip_reason") else "")
                )
            else:
                lines.append(
                    "    vision_all_tokens: "
                    f"teacher_tokens={vision.get('teacher_tokens', 0)} "
                    f"student_tokens={vision.get('student_tokens', 0)} "
                    f"graph_nodes={vision.get('graph_nodes', vision.get('teacher_graph_nodes', 0))} "
                    f"loss_valid={vision.get('vision_loss_valid', False)}"
                    + (f" skip={vision.get('skip_reason')}" if vision.get("skip_reason") else "")
                )
            if vision.get("graph_teacher"):
                lines.append(f"    {_fmt_graph('vision_graph_t', vision['graph_teacher'])}")
            if vision.get("graph_student"):
                lines.append(f"    {_fmt_graph('vision_graph_s', vision['graph_student'])}")

        text = entry.get("text") or {}
        if text:
            if text.get("use_topk", True):
                token_info = f"topk_tokens={text.get('topk_tokens', 0)}"
            else:
                token_info = f"num_tokens={text.get('num_tokens', 0)}"
            lines.append(
                "    text: "
                f"{token_info} "
                f"use_topk={text.get('use_topk', True)} "
                f"loss_valid={text.get('text_loss_valid', False)}"
                + (f" skip={text.get('skip_reason')}" if text.get("skip_reason") else "")
            )
            if text.get("graph_teacher"):
                lines.append(f"    {_fmt_graph('text_graph_t', text['graph_teacher'])}")
            if text.get("graph_student"):
                lines.append(f"    {_fmt_graph('text_graph_s', text['graph_student'])}")

        cross = entry.get("cross") or {}
        if cross:
            lines.append(
                "    cross: "
                f"vision_nodes={cross.get('vision_nodes')} "
                f"text_nodes={cross.get('text_nodes')} "
                f"total_nodes={cross.get('total_nodes')} "
                f"loss_valid={cross.get('cross_loss_valid', False)}"
                + (f" skip={cross.get('skip_reason')}" if cross.get("skip_reason") else "")
            )
            if cross.get("graph_teacher"):
                lines.append(f"    {_fmt_graph('cross_graph_t', cross['graph_teacher'])}")
            if cross.get("graph_student"):
                lines.append(f"    {_fmt_graph('cross_graph_s', cross['graph_student'])}")

        losses = entry.get("losses") or {}
        lines.append(
            "    losses: "
            f"v={losses.get('v', 0):.4f} t={losses.get('t', 0):.4f} "
            f"cross={losses.get('cross', 0):.4f}"
        )

    return lines


def grassman_debug_has_warning(grassman_debug: list) -> bool:
    for entry in grassman_debug:
        vision = entry.get("vision") or {}
        text = entry.get("text") or {}
        cross = entry.get("cross") or {}

        if entry.get("has_image"):
            if vision.get("use_cluster", True):
                if vision.get("valid_clusters", 0) < 2:
                    return True
            if not vision.get("vision_loss_valid", False):
                return True
            for key in ("graph_teacher", "graph_student"):
                graph = vision.get(key) or {}
                nodes = graph.get("nodes", 0)
                if graph and nodes > 0 and nodes < 3:
                    return True

        if entry.get("num_text", 0) >= 2 and not text.get("text_loss_valid", False):
            return True

        if vision.get("vision_loss_valid") and text.get("text_loss_valid"):
            if not cross.get("cross_loss_valid", False):
                return True

    return False


def unwrap_student(distiller: nn.Module) -> nn.Module:
    model = distiller.module if hasattr(distiller, "module") else distiller
    return model.student


def set_debug_step(training_args, global_step: int, epoch_step: int) -> None:
    training_args._debug_global_step = global_step
    training_args._debug_epoch_step = epoch_step


def loss_is_finite(loss: torch.Tensor) -> bool:
    return bool(torch.isfinite(loss).all().item())


def loss_dict_has_nonfinite(loss_dict: Dict[str, Any]) -> bool:
    return any(
        isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
        for value in loss_dict.values()
    )


def _contrastive_logits_line(scores: torch.Tensor, temperature: float) -> str:
    logits = scores / temperature
    if torch.isfinite(logits).any():
        finite_logits = logits[torch.isfinite(logits)]
        return (
            f"contrastive_logits: temperature={temperature}"
            f" min={finite_logits.min().item():.4e}"
            f" max={finite_logits.max().item():.4e}"
        )
    return f"contrastive_logits: temperature={temperature} all_nonfinite=True"


def build_sgd_forward_debug_lines(
    *,
    grassman_debug: list,
    student_qry_input,
    student_pos_input,
    student_qry_reps,
    student_pos_reps,
    teacher_qry_reps,
    teacher_pos_reps,
    student_qry_hidden_states,
    student_pos_hidden_states,
    student_qry_attention,
    student_pos_attention,
    scores,
    rkd_distance_loss,
    rkd_angle_loss,
    temperature,
) -> list:
    qry_eos = eos_hidden_states(
        student_qry_hidden_states[-1],
        student_qry_input["attention_mask"],
    )
    pos_eos = eos_hidden_states(
        student_pos_hidden_states[-1],
        student_pos_input["attention_mask"],
    )
    lines = [
        tensor_stats(student_qry_reps, "student_qry_reps(pooled)"),
        tensor_stats(student_pos_reps, "student_pos_reps(pooled)"),
        tensor_stats(qry_eos, "student_qry_eos_hidden(pre_norm)"),
        tensor_stats(pos_eos, "student_pos_eos_hidden(pre_norm)"),
        tensor_stats(teacher_qry_reps, "teacher_qry_reps(pooled)"),
        tensor_stats(teacher_pos_reps, "teacher_pos_reps(pooled)"),
        tensor_stats(scores, "contrastive_scores"),
        _contrastive_logits_line(scores, temperature),
        tensor_stats(rkd_distance_loss, "rkd_distance_loss"),
        tensor_stats(rkd_angle_loss, "rkd_angle_loss"),
        attention_stats(student_qry_attention, "student_qry_attention"),
        attention_stats(student_pos_attention, "student_pos_attention"),
        tensor_stats(student_qry_hidden_states[-1], "student_qry_hidden[last_layer]"),
        tensor_stats(student_pos_hidden_states[-1], "student_pos_hidden[last_layer]"),
    ]
    lines.extend(format_grassman_debug_lines(grassman_debug))
    return lines


def log_sgd_forward_debug(
    *,
    training_args,
    loss_dict: Dict[str, Any],
    grassman_debug: list,
    student_qry_input,
    student_pos_input,
    student_qry_reps,
    student_pos_reps,
    teacher_qry_reps,
    teacher_pos_reps,
    student_qry_hidden_states,
    student_pos_hidden_states,
    student_qry_attention,
    student_pos_attention,
    scores,
    rkd_distance_loss,
    rkd_angle_loss,
    temperature,
) -> None:
    has_bad_loss = loss_dict_has_nonfinite(loss_dict)
    has_grassman_warning = grassman_debug_has_warning(grassman_debug)
    if not has_bad_loss and not has_grassman_warning:
        return

    global_step = getattr(training_args, "_debug_global_step", -1)
    epoch_step = getattr(training_args, "_debug_epoch_step", -1)
    tag = "SGD_NAN_DEBUG" if has_bad_loss else "SGD_GRASSMAN_DEBUG"
    log_nonfinite_losses(
        global_step=global_step,
        epoch_step=epoch_step,
        loss_dict=loss_dict if has_bad_loss else {},
        extra_lines=build_sgd_forward_debug_lines(
            grassman_debug=grassman_debug,
            student_qry_input=student_qry_input,
            student_pos_input=student_pos_input,
            student_qry_reps=student_qry_reps,
            student_pos_reps=student_pos_reps,
            teacher_qry_reps=teacher_qry_reps,
            teacher_pos_reps=teacher_pos_reps,
            student_qry_hidden_states=student_qry_hidden_states,
            student_pos_hidden_states=student_pos_hidden_states,
            student_qry_attention=student_qry_attention,
            student_pos_attention=student_pos_attention,
            scores=scores,
            rkd_distance_loss=rkd_distance_loss,
            rkd_angle_loss=rkd_angle_loss,
            temperature=temperature,
        ),
        tag=tag,
    )


class TrainNanDebugger:
    """Hooks for NaN/Inf diagnostics during the training loop."""

    def __init__(self, distiller: nn.Module):
        self.student = unwrap_student(distiller)

    def annotate_step(self, training_args, global_step: int, epoch_step: int) -> None:
        set_debug_step(training_args, global_step, epoch_step)

    def before_backward(self, global_step: int, epoch_step: int, outputs: Dict[str, Any]) -> None:
        log_nonfinite_losses(
            global_step=global_step,
            epoch_step=epoch_step,
            loss_dict=outputs,
            extra_lines=[module_param_stats(self.student, prefix="student_params_before_backward")],
            tag="TRAIN_NAN_DEBUG",
        )

    def after_backward(self, global_step: int, epoch_step: int, outputs: Dict[str, Any]) -> None:
        log_nonfinite_losses(
            global_step=global_step,
            epoch_step=epoch_step,
            loss_dict=outputs,
            extra_lines=[module_grad_stats(self.student, prefix="student")],
            tag="TRAIN_NAN_GRAD_DEBUG",
        )

    def clip_gradients(
        self,
        global_step: int,
        epoch_step: int,
        outputs: Dict[str, Any],
        max_grad_norm: float,
    ) -> torch.Tensor:
        grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_grad_norm)
        if not torch.isfinite(grad_norm):
            log_nonfinite_losses(
                global_step=global_step,
                epoch_step=epoch_step,
                loss_dict=outputs,
                extra_lines=[
                    f"clip_grad_norm returned non-finite value: {grad_norm}",
                    module_grad_stats(self.student, prefix="student"),
                    module_param_stats(self.student, prefix="student"),
                ],
                tag="TRAIN_NAN_CLIP_DEBUG",
            )
        return grad_norm

    def after_optimizer_step(self, global_step: int, epoch_step: int, outputs: Dict[str, Any]) -> None:
        log_nonfinite_losses(
            global_step=global_step,
            epoch_step=epoch_step,
            loss_dict=outputs,
            extra_lines=[module_param_stats(self.student, prefix="student_params_after_step")],
            tag="TRAIN_NAN_POST_STEP_DEBUG",
        )


def log_training_output_dirs(train_log_path: str, nan_debug_dir: Optional[str]) -> None:
    if not _is_main_process():
        return
    logger.info(f"Logging to terminal and {train_log_path}")
    if nan_debug_dir:
        logger.info(
            f"NaN debug logs -> {nan_debug_dir}/ "
            f"(nan_debug.log + events/step_*_*.log)"
        )
