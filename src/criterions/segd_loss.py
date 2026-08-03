"""
SEGDLoss — Spectral Knowledge Distillation with Cross-sample Star-Bridge Graph.

Loss composition:
  total = contrastive_loss + kd_weight * spectral_kd_loss

Pipeline:
  Teacher/Student forward (native token hidden states + attentions @ ~80% depth)
    → assemble independent star-bridge graphs (batch-level)
    → signed Laplacian → eigenspace
    → cross-model projection P (FRA visual + char-overlap text)
    → subspace projector KD
  Contrastive reuses student mean-pooled super-node reps (R_q, R_p).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.criterions.sgd_loss import (
    build_paired_text_offsets,
    count_text_tokens_student,
    count_text_tokens_teacher,
    extract_text_hidden_states,
    extract_vision_hidden_states,
    get_batch_text_strings,
    get_text_token_ids,
    strip_vlm_image_markers,
)

logger = logging.getLogger(__name__)

_EPS = 1e-8
_LOBPCG_THRESHOLD = 1500


# ---------------------------------------------------------------------------
# Geometry: text char-overlap + visual FRA
# ---------------------------------------------------------------------------

def char_overlap_matrix(
    spans_t: Sequence[Tuple[int, int]],
    spans_s: Sequence[Tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Character-span overlap O[i, j] between teacher/student subword tokens."""
    n_t, n_s = len(spans_t), len(spans_s)
    if n_t == 0 or n_s == 0:
        return torch.zeros(n_t, n_s, device=device, dtype=dtype)

    t = torch.tensor(spans_t, device=device, dtype=dtype)  # [Nt, 2]
    s = torch.tensor(spans_s, device=device, dtype=dtype)  # [Ns, 2]
    lo = torch.maximum(t[:, None, 0], s[None, :, 0])
    hi = torch.minimum(t[:, None, 1], s[None, :, 1])
    return (hi - lo).clamp(min=0.0)


def axis_overlap_matrix(n_t: int, n_s: int, device: torch.device) -> torch.Tensor:
    """1D closed-form overlap of two uniform partitions of [0, 1]."""
    edges_t = torch.linspace(0, 1, n_t + 1, device=device)
    edges_s = torch.linspace(0, 1, n_s + 1, device=device)
    start_t, end_t = edges_t[:-1], edges_t[1:]
    start_s, end_s = edges_s[:-1], edges_s[1:]
    lo = torch.maximum(start_t[:, None], start_s[None, :])
    hi = torch.minimum(end_t[:, None], end_s[None, :])
    return (hi - lo).clamp(min=0.0)


def fractional_region_alignment(
    h_t: int, w_t: int, h_s: int, w_s: int, device: torch.device,
) -> torch.Tensor:
    """2D FRA overlap O ∈ R^{H_t W_t × H_s W_s} (area of patch intersections)."""
    if min(h_t, w_t, h_s, w_s) <= 0:
        return torch.zeros(h_t * w_t, h_s * w_s, device=device)
    oh = axis_overlap_matrix(h_t, h_s, device)
    ow = axis_overlap_matrix(w_t, w_s, device)
    o = torch.einsum("hi,wj->hwij", oh, ow)
    return o.reshape(h_t * w_t, h_s * w_s)


def infer_spatial_hw(
    num_tokens: int,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    patch_size: Optional[int] = None,
) -> Tuple[int, int]:
    """Infer (H, W) grid for vision tokens."""
    if num_tokens <= 0:
        return 0, 0
    side = int(round(math.sqrt(num_tokens)))
    if side * side == num_tokens:
        return side, side

    if image_width and image_height and patch_size and patch_size > 0:
        w = max(1, int(round(image_width / patch_size)))
        h = max(1, int(round(image_height / patch_size)))
        if h * w == num_tokens:
            return h, w
        aspect = image_width / max(image_height, 1)
        best = (side, max(1, num_tokens // max(side, 1)))
        best_err = abs(best[1] / max(best[0], 1) - aspect)
        for h_try in range(1, num_tokens + 1):
            if num_tokens % h_try != 0:
                continue
            w_try = num_tokens // h_try
            err = abs(w_try / h_try - aspect)
            if err < best_err:
                best, best_err = (h_try, w_try), err
        return best

    best_h = side
    for h_try in range(side, 0, -1):
        if num_tokens % h_try == 0:
            best_h = h_try
            break
    return best_h, num_tokens // best_h


def row_normalize(mat: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    return mat / mat.sum(dim=1, keepdim=True).clamp_min(eps)


# ---------------------------------------------------------------------------
# Attention @ ~80% depth
# ---------------------------------------------------------------------------

def get_target_layer_indices(
    num_layers: int,
    depth_ratio: float = 0.8,
    window: int = 1,
) -> List[int]:
    """Layer indices around depth_ratio (shared by hidden-state and attention paths)."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    center = int(round(depth_ratio * (num_layers - 1)))
    return [i for i in range(center - window, center + window + 1) if 0 <= i < num_layers]


def get_target_layer_hidden_smoothed(
    all_layer_hidden: Sequence[torch.Tensor],
    depth_ratio: float = 0.8,
    window: int = 1,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Mean hidden states over a small window around depth_ratio.

    all_layer_hidden[i]: (B, SeqLen, D)
    Returns smoothed hidden (B, SeqLen, D) and selected layer indices.
    """
    idxs = get_target_layer_indices(len(all_layer_hidden), depth_ratio, window)
    stacked = torch.stack([all_layer_hidden[i] for i in idxs], dim=0)
    return stacked.mean(dim=0), idxs


def get_target_layer_attn_smoothed(
    all_layer_attentions: Sequence[torch.Tensor],
    depth_ratio: float = 0.8,
    window: int = 1,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Mean over heads and a small window of layers around depth_ratio.

    all_layer_attentions[i]: (B, num_heads, N, N) or (B, N, N)
    Returns attn (B, N, N) and selected layer indices.
    """
    if len(all_layer_attentions) == 0:
        raise ValueError("empty attention tuple")
    idxs = get_target_layer_indices(len(all_layer_attentions), depth_ratio, window)
    stacked = []
    for i in idxs:
        a = all_layer_attentions[i]
        if a.dim() == 4:
            a = a.mean(dim=1)
        stacked.append(a)
    return torch.stack(stacked, dim=0).mean(dim=0), idxs


def _fallback_attn_from_embeddings(h: torch.Tensor) -> torch.Tensor:
    """Softmax cosine affinity when model attentions are unavailable."""
    h_n = F.normalize(h.float(), p=2, dim=-1)
    logits = (h_n @ h_n.t()) / math.sqrt(max(h.size(-1), 1))
    return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Graph construction (star-bridge)
# ---------------------------------------------------------------------------

def _cosine_softmax_weight(
    x_i: torch.Tensor, x_candidates: torch.Tensor, tau: float,
) -> torch.Tensor:
    """Softmax(cosine/tau) over candidates; positive weights summing to 1."""
    tau = max(float(tau), _EPS)
    xi_n = F.normalize(x_i.float(), dim=-1)
    xc_n = F.normalize(x_candidates.float(), dim=-1)
    logits = (xc_n @ xi_n) / tau
    return torch.softmax(logits, dim=0)


def _build_attn_topk_index(
    attn: torch.Tensor, mask: torch.Tensor, k: int = 16,
) -> torch.Tensor:
    """Attention values only select neighbor indices (not edge weights)."""
    valid = mask.bool()
    a = attn.float()
    a = a.masked_fill(~valid.unsqueeze(0), float("-inf"))
    a = a.masked_fill(~valid.unsqueeze(1), float("-inf"))
    a = a.clone()
    a.fill_diagonal_(float("-inf"))
    k_eff = min(k, int(valid.sum().item()) - 1)
    if k_eff <= 0:
        return torch.empty(attn.size(0), 0, device=attn.device, dtype=torch.long)
    _, topk_idx = a.topk(k_eff, dim=1)
    return topk_idx


def mean_pool(tokens: torch.Tensor, mask: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    m = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * m).sum(dim=0) / m.sum().clamp_min(eps)


def build_global_index(
    b: int, n_q_list: Sequence[int], n_p_list: Sequence[int],
) -> Tuple[Dict[str, List[int]], List[int], List[int], int]:
    offsets = {"q": [], "p": []}
    cursor = 0
    for i in range(b):
        offsets["q"].append(cursor)
        cursor += int(n_q_list[i])
        offsets["p"].append(cursor)
        cursor += int(n_p_list[i])
    rq_start = cursor
    cursor += b
    rp_start = cursor
    cursor += b
    idx_rq = list(range(rq_start, rq_start + b))
    idx_rp = list(range(rp_start, rp_start + b))
    return offsets, idx_rq, idx_rp, cursor


class _DiffEdgeBuffer:
    """Autograd-safe COO accumulator (values stay as tensors)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.vals: List[torch.Tensor] = []

    def add(self, r: int, c: int, val: torch.Tensor, symmetric: bool = True) -> None:
        v = val.reshape(()).float()
        self.rows.append(r)
        self.cols.append(c)
        self.vals.append(v)
        if symmetric and r != c:
            self.rows.append(c)
            self.cols.append(r)
            self.vals.append(v)

    def to_dense(self, n_total: int) -> torch.Tensor:
        if not self.vals:
            return torch.zeros(n_total, n_total, device=self.device, dtype=torch.float32)
        indices = torch.tensor([self.rows, self.cols], device=self.device, dtype=torch.long)
        values = torch.stack(self.vals)
        with torch.sparse.check_sparse_tensor_invariants(False):
            sp = torch.sparse_coo_tensor(
                indices, values, (n_total, n_total), device=self.device,
            ).coalesce()
        w = sp.to_dense()
        return 0.5 * (w + w.t())


def _intra_cluster_edges(
    edges: _DiffEdgeBuffer,
    attn: torch.Tensor,
    hidden: torch.Tensor,
    mask: torch.Tensor,
    start_idx: int,
    topk: int = 16,
    tau: float = 0.1,
) -> None:
    with torch.no_grad():
        topk_idx = _build_attn_topk_index(attn, mask, k=topk)
    if topk_idx.numel() == 0:
        return
    for i in range(hidden.size(0)):
        if not bool(mask[i]):
            continue
        neighbors = topk_idx[i]
        w = _cosine_softmax_weight(hidden[i], hidden[neighbors], tau)
        for rank, j in enumerate(neighbors.tolist()):
            if not bool(mask[j]):
                continue
            edges.add(start_idx + i, start_idx + j, w[rank], symmetric=False)


def _local_to_global_edges(
    edges: _DiffEdgeBuffer,
    hidden: torch.Tensor,
    mask: torch.Tensor,
    r: torch.Tensor,
    start_idx: int,
    super_idx: int,
    tau: float = 0.1,
) -> None:
    valid_idx = mask.nonzero(as_tuple=True)[0]
    if valid_idx.numel() == 0:
        return
    w = _cosine_softmax_weight(r, hidden[valid_idx], tau)
    for rank, t in enumerate(valid_idx.tolist()):
        edges.add(start_idx + t, super_idx, w[rank], symmetric=True)


def _bridge_edges(
    edges: _DiffEdgeBuffer,
    r_q: torch.Tensor,
    r_p: torch.Tensor,
    idx_rq: Sequence[int],
    idx_rp: Sequence[int],
    k_neg: int = 8,
    temperature: float = 1.0,
    lambda_neg: float = 0.3,
) -> None:
    b = r_q.size(0)
    if b == 0:
        return
    r_q_n = F.normalize(r_q.float(), dim=-1)
    r_p_n = F.normalize(r_p.float(), dim=-1)
    logits = (r_q_n @ r_p_n.t()) / max(temperature, _EPS)

    for i in range(b):
        row = logits[i]
        pos_j = i
        neg_candidates = [j for j in range(b) if j != i]
        if neg_candidates:
            neg_logits = row[neg_candidates]
            k = min(k_neg, len(neg_candidates))
            _, topk_pos = neg_logits.topk(k)
            topk_j = [neg_candidates[p] for p in topk_pos.tolist()]
        else:
            topk_j = []

        candidate_j = [pos_j] + topk_j
        idx = torch.tensor(candidate_j, device=logits.device, dtype=torch.long)
        alpha = torch.softmax(row.index_select(0, idx), dim=0)

        for rank, j in enumerate(candidate_j):
            a = alpha[rank]
            if j == pos_j:
                edges.add(idx_rq[i], idx_rp[j], a, symmetric=True)
            else:
                edges.add(idx_rq[i], idx_rp[j], -lambda_neg * a, symmetric=True)


def assemble_graph(
    matched_q: List[torch.Tensor],
    matched_p: List[torch.Tensor],
    attn_q: List[torch.Tensor],
    attn_p: List[torch.Tensor],
    mask_q: List[torch.Tensor],
    mask_p: List[torch.Tensor],
    topk: int = 16,
    tau_intra: float = 0.1,
    tau_local: float = 0.1,
    k_neg: int = 8,
    bridge_temperature: float = 1.0,
    lambda_neg: float = 0.3,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """
    Build one model's batch star-bridge adjacency (dense via sparse COO).

    Returns W [N,N], N_total, R_q [B,D], R_p [B,D].
    Edge values remain in the autograd graph (no ``.item()``).
    """
    b = len(matched_q)
    assert b == len(matched_p)
    device = matched_q[0].device if b > 0 else torch.device("cpu")

    n_q_list = [t.size(0) for t in matched_q]
    n_p_list = [t.size(0) for t in matched_p]
    offsets, idx_rq, idx_rp, n_total = build_global_index(b, n_q_list, n_p_list)

    edges = _DiffEdgeBuffer(device)
    r_q_all, r_p_all = [], []

    for i in range(b):
        q_start, p_start = offsets["q"][i], offsets["p"][i]
        r_q = mean_pool(matched_q[i], mask_q[i])
        r_p = mean_pool(matched_p[i], mask_p[i])
        r_q_all.append(r_q)
        r_p_all.append(r_p)
        _intra_cluster_edges(
            edges, attn_q[i], matched_q[i], mask_q[i], q_start,
            topk=topk, tau=tau_intra,
        )
        _intra_cluster_edges(
            edges, attn_p[i], matched_p[i], mask_p[i], p_start,
            topk=topk, tau=tau_intra,
        )
        _local_to_global_edges(
            edges, matched_q[i], mask_q[i], r_q, q_start, idx_rq[i], tau=tau_local,
        )
        _local_to_global_edges(
            edges, matched_p[i], mask_p[i], r_p, p_start, idx_rp[i], tau=tau_local,
        )

    r_q = torch.stack(r_q_all, dim=0)
    r_p = torch.stack(r_p_all, dim=0)
    _bridge_edges(
        edges, r_q, r_p, idx_rq, idx_rp,
        k_neg=k_neg, temperature=bridge_temperature, lambda_neg=lambda_neg,
    )
    return edges.to_dense(n_total), n_total, r_q, r_p


# ---------------------------------------------------------------------------
# Signed Laplacian + eigenspace + KD loss
# ---------------------------------------------------------------------------

def build_signed_laplacian(w: torch.Tensor, n_total: int) -> torch.Tensor:
    """Normalized signed Laplacian L = I - D^{-1/2} W D^{-1/2}, D_ii = Σ_j |W_ij|."""
    w_dense = w.to_dense() if w.is_sparse else w
    w_dense = w_dense.to(torch.float32)
    deg = w_dense.abs().sum(dim=1).clamp_min(_EPS)
    deg_inv_sqrt = deg.pow(-0.5)
    w_norm = deg_inv_sqrt.unsqueeze(1) * w_dense * deg_inv_sqrt.unsqueeze(0)
    eye = torch.eye(n_total, device=w_dense.device, dtype=w_dense.dtype)
    return eye - w_norm


def get_eigenspace(
    lap: torch.Tensor,
    n_total: int,
    k: int = 32,
    allow_lobpcg: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Smallest-k eigenpairs of L.
    lobpcg has no autograd — only used when L does not require grad.
    """
    k_use = max(1, min(k, max(n_total - 1, 1)))
    use_lobpcg = (
        allow_lobpcg
        and n_total > _LOBPCG_THRESHOLD
        and not lap.requires_grad
    )
    if use_lobpcg:
        # lobpcg prefers sparse / float32 SPD-ish input
        try:
            eigvals, eigvecs = torch.lobpcg(lap, k=k_use, largest=False)
            return eigvals, eigvecs
        except Exception as exc:  # noqa: BLE001
            logger.warning("lobpcg failed (%s); falling back to eigh", exc)

    eigvals, eigvecs = torch.linalg.eigh(lap)
    return eigvals[:k_use], eigvecs[:, :k_use]


def project_teacher_eigenspace(u_t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """U_t_proj = P^T U_t  → (N_s, k)."""
    if p.is_sparse:
        return torch.sparse.mm(p.transpose(0, 1), u_t)
    return p.t() @ u_t


def spectral_kd_loss(
    u_t: torch.Tensor,
    u_s: torch.Tensor,
    p: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Frobenius distance between subspace projectors after FRA projection."""
    k_use = min(k, u_t.size(1), u_s.size(1))
    if k_use <= 0:
        return u_s.new_zeros(())
    u_t_proj = project_teacher_eigenspace(u_t[:, :k_use].detach(), p)
    us = u_s[:, :k_use]
    pt = u_t_proj @ u_t_proj.t()
    ps = us @ us.t()
    return ((pt - ps) ** 2).sum() / max(ps.size(0), 1)


def bidirectional_infonce_loss(
    r_q: torch.Tensor,
    r_p: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric InfoNCE: 0.5 * (CE(q→p) + CE(p→q)) on L2-normalized reps."""
    r_q = F.normalize(r_q, dim=-1)
    r_p = F.normalize(r_p, dim=-1)
    logits = r_q @ r_p.t() / max(temperature, _EPS)
    labels = torch.arange(r_q.size(0), device=r_q.device, dtype=torch.long)
    loss_q2p = F.cross_entropy(logits, labels)
    loss_p2q = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_q2p + loss_p2q)


# ---------------------------------------------------------------------------
# Token / attention extraction for one encode side
# ---------------------------------------------------------------------------

def _offsets_to_span_list(offsets: Optional[torch.Tensor]) -> List[Tuple[int, int]]:
    if offsets is None or offsets.numel() == 0:
        return []
    return [(int(s.item()), int(e.item())) for s, e in offsets]


def _cluster_seq_indices(
    is_teacher: bool,
    seq_len: int,
    num_vision: int,
    num_text: int,
) -> List[int]:
    """Absolute sequence indices of [vision | text] for one sample."""
    if is_teacher:
        text_start = seq_len - num_text if num_text > 0 else seq_len
        vision_start = text_start - num_vision
        vis = list(range(vision_start, text_start)) if num_vision > 0 else []
        txt = list(range(text_start, seq_len)) if num_text > 0 else []
        return vis + txt
    vis = list(range(0, num_vision)) if num_vision > 0 else []
    txt = list(range(num_vision, num_vision + num_text)) if num_text > 0 else []
    return vis + txt


def _extract_cluster_attn(
    attn_bn: torch.Tensor,
    indices: Sequence[int],
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Slice full-seq attention to cluster nodes; fallback if empty/invalid."""
    n = len(indices)
    if n == 0:
        return tokens.new_zeros(0, 0)
    if attn_bn is None or attn_bn.numel() == 0:
        return _fallback_attn_from_embeddings(tokens)
    idx = torch.tensor(indices, device=attn_bn.device, dtype=torch.long)
    # Guard against seq-length mismatch (e.g. truncated attentions).
    if int(idx.max().item()) >= attn_bn.size(0):
        return _fallback_attn_from_embeddings(tokens)
    sub = attn_bn.index_select(0, idx).index_select(1, idx)
    return sub


def _extract_side_bundle(
    *,
    is_teacher: bool,
    model_input: Dict[str, torch.Tensor],
    hidden_states: Sequence[torch.Tensor],
    attentions: Optional[Sequence[torch.Tensor]],
    image_features: Optional[List[Optional[torch.Tensor]]],
    image_sizes: Optional[Sequence[Tuple[int, int]]],
    text_strings: List[str],
    tokenizer,
    peer_tokenizer,
    peer_input: Dict[str, torch.Tensor],
    patch_size: int,
    depth_ratio: float,
    attn_window: int,
    sample_idx: int,
) -> Optional[Dict[str, Any]]:
    """Native tokens + cluster attention + meta for one (sample, qry|pos) side."""
    input_ids = model_input["input_ids"][sample_idx]
    seq_len = int(input_ids.size(0))

    layer_hidden, layer_idxs = get_target_layer_hidden_smoothed(
        hidden_states, depth_ratio=depth_ratio, window=attn_window,
    )
    hidden_for_extract = [layer_hidden]

    if is_teacher:
        num_text = count_text_tokens_teacher(input_ids)
    else:
        num_text = count_text_tokens_student(input_ids)

    has_image = (
        image_features is not None
        and sample_idx < len(image_features)
        and image_features[sample_idx] is not None
    )
    num_vision = int(image_features[sample_idx].size(0)) if has_image else 0

    img_w = img_h = 0
    if has_image:
        if image_sizes is not None and sample_idx < len(image_sizes):
            img_w, img_h = int(image_sizes[sample_idx][0]), int(image_sizes[sample_idx][1])
        else:
            side = int(math.sqrt(max(num_vision, 1)))
            img_w = img_h = side * patch_size

    # Hidden tokens (last layer) — native, no pooling
    vision = None
    text = None
    if has_image and num_vision > 0:
        vision = extract_vision_hidden_states(
            hidden_for_extract, sample_idx, num_vision, num_text, is_teacher=is_teacher,
        )[-1]
    if num_text > 0:
        text = extract_text_hidden_states(
            hidden_for_extract, sample_idx, num_text, num_vision,
            is_teacher=is_teacher, has_image=has_image,
        )[-1]

    parts = [p for p in (vision, text) if p is not None and p.numel() > 0]
    if not parts:
        return None
    tokens = torch.cat(parts, dim=0)
    mask = torch.ones(tokens.size(0), device=tokens.device, dtype=torch.bool)

    # Attention at ~80% depth, sliced to cluster indices
    indices = _cluster_seq_indices(is_teacher, seq_len, num_vision, num_text)
    if attentions is not None:
        attn_full, attn_layer_idxs = get_target_layer_attn_smoothed(
            attentions, depth_ratio=depth_ratio, window=attn_window,
        )
        attn_sample = attn_full[sample_idx]  # [S, S]
        attn = _extract_cluster_attn(attn_sample, indices, tokens)
        # hidden and attention use the same depth window; log attention center.
        layer_idxs = attn_layer_idxs
    else:
        attn = _fallback_attn_from_embeddings(tokens)

    # Char spans on shared reference text (for cross-model P)
    reference_text = strip_vlm_image_markers(
        text_strings[sample_idx] if sample_idx < len(text_strings) else ""
    )
    char_spans: List[Tuple[int, int]] = []
    if text is not None and num_text > 0:
        own_ids = get_text_token_ids(input_ids, is_teacher=is_teacher)
        peer_ids = get_text_token_ids(
            peer_input["input_ids"][sample_idx], is_teacher=not is_teacher,
        )
        if is_teacher:
            t_off, _ = build_paired_text_offsets(
                tokenizer, peer_tokenizer, own_ids, peer_ids, reference_text, tokens.device,
            )
            char_spans = _offsets_to_span_list(t_off)
        else:
            _, s_off = build_paired_text_offsets(
                peer_tokenizer, tokenizer, peer_ids, own_ids, reference_text, tokens.device,
            )
            char_spans = _offsets_to_span_list(s_off)
        # Length mismatch → drop spans (P text block skipped / fallback later)
        if len(char_spans) != text.size(0):
            char_spans = []

    h, w = (0, 0)
    if has_image and num_vision > 0:
        h, w = infer_spatial_hw(num_vision, img_w, img_h, patch_size)

    return {
        "tokens": tokens,
        "attn": attn,
        "mask": mask,
        "num_vision": num_vision,
        "num_text": int(text.size(0)) if text is not None else 0,
        "H": h,
        "W": w,
        "char_spans": char_spans,
        "attn_layers": layer_idxs,
        "has_image": has_image,
    }


# ---------------------------------------------------------------------------
# Main criterion
# ---------------------------------------------------------------------------

class SEGDLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

        self.args = args
        self.kd_weight = float(getattr(args, "kd_weight", 1.0))

        self.depth_ratio = float(getattr(args, "segd_depth_ratio", 0.8))
        self.attn_window = int(getattr(args, "segd_attn_window", 1))
        self.intra_topk = int(getattr(args, "segd_intra_topk", 16))
        self.tau_intra = float(getattr(args, "segd_tau_intra", 0.1))
        self.tau_local = float(getattr(args, "segd_tau_local", 0.1))
        self.lambda_neg = float(getattr(args, "segd_lambda_neg", 0.3))
        self.k_neg = int(getattr(args, "segd_k_neg", 8))
        self.bridge_temperature = float(getattr(args, "segd_bridge_temperature", 1.0))
        self.k_eigen = int(getattr(args, "segd_k_eigen", getattr(args, "num_eigenvectors", 32)))
        self.use_graph_reps_contrastive = bool(
            getattr(args, "segd_use_graph_reps_contrastive", True)
        )

        self.teacher_patch_size = int(getattr(args, "teacher_patch_size", 28))
        self.student_patch_size = int(getattr(args, "student_patch_size", 64))

        self._student_tokenizer = None

    def _get_student_tokenizer(self, distiller):
        if self._student_tokenizer is None:
            from transformers import AutoTokenizer
            self._student_tokenizer = AutoTokenizer.from_pretrained(
                distiller.model_args.model_name,
                trust_remote_code=True,
            )
        return self._student_tokenizer

    def _dist_gather_tensor(self, t: torch.Tensor) -> torch.Tensor:
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)

    @staticmethod
    def _zero(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros((), device=device, dtype=dtype)

    def _collect_model_batch(
        self,
        *,
        is_teacher: bool,
        qry_input,
        pos_input,
        qry_hidden,
        pos_hidden,
        qry_attn,
        pos_attn,
        qry_img_feats,
        pos_img_feats,
        qry_image_sizes,
        pos_image_sizes,
        qry_texts: List[str],
        pos_texts: List[str],
        tokenizer,
        peer_tokenizer,
        peer_qry_input,
        peer_pos_input,
        patch_size: int,
        batch_size: int,
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor],
        List[torch.Tensor], List[torch.Tensor],
        List[torch.Tensor], List[torch.Tensor],
        List[Dict[str, Any]],
        Dict[str, float],
    ]:
        matched_q, matched_p = [], []
        attn_q, attn_p = [], []
        mask_q, mask_p = [], []
        per_sample_meta: List[Dict[str, Any]] = []
        stats = {
            "vision_nodes_q": 0.0,
            "text_nodes_q": 0.0,
            "vision_nodes_p": 0.0,
            "text_nodes_p": 0.0,
            "attn_layer_center": -1.0,
        }

        for i in range(batch_size):
            q = _extract_side_bundle(
                is_teacher=is_teacher,
                model_input=qry_input,
                hidden_states=qry_hidden,
                attentions=qry_attn,
                image_features=qry_img_feats,
                image_sizes=qry_image_sizes,
                text_strings=qry_texts,
                tokenizer=tokenizer,
                peer_tokenizer=peer_tokenizer,
                peer_input=peer_qry_input,
                patch_size=patch_size,
                depth_ratio=self.depth_ratio,
                attn_window=self.attn_window,
                sample_idx=i,
            )
            p = _extract_side_bundle(
                is_teacher=is_teacher,
                model_input=pos_input,
                hidden_states=pos_hidden,
                attentions=pos_attn,
                image_features=pos_img_feats,
                image_sizes=pos_image_sizes,
                text_strings=pos_texts,
                tokenizer=tokenizer,
                peer_tokenizer=peer_tokenizer,
                peer_input=peer_pos_input,
                patch_size=patch_size,
                depth_ratio=self.depth_ratio,
                attn_window=self.attn_window,
                sample_idx=i,
            )
            if q is None or p is None:
                # Degenerate sample: 1 dummy node so batch indexing stays aligned.
                device = qry_hidden[-1].device
                dtype = qry_hidden[-1].dtype
                dummy = torch.zeros(1, qry_hidden[-1].size(-1), device=device, dtype=dtype)
                dummy_attn = torch.zeros(1, 1, device=device, dtype=torch.float32)
                dummy_mask = torch.ones(1, device=device, dtype=torch.bool)
                if q is None:
                    q = {
                        "tokens": dummy, "attn": dummy_attn, "mask": dummy_mask,
                        "num_vision": 0, "num_text": 1, "H": 0, "W": 0,
                        "char_spans": [], "attn_layers": [], "has_image": False,
                    }
                if p is None:
                    p = {
                        "tokens": dummy.detach() if is_teacher else dummy,
                        "attn": dummy_attn, "mask": dummy_mask,
                        "num_vision": 0, "num_text": 1, "H": 0, "W": 0,
                        "char_spans": [], "attn_layers": [], "has_image": False,
                    }

            matched_q.append(q["tokens"])
            matched_p.append(p["tokens"])
            attn_q.append(q["attn"].float())
            attn_p.append(p["attn"].float())
            mask_q.append(q["mask"])
            mask_p.append(p["mask"])
            per_sample_meta.append({"q": q, "p": p})

            stats["vision_nodes_q"] += float(q["num_vision"])
            stats["text_nodes_q"] += float(q["num_text"])
            stats["vision_nodes_p"] += float(p["num_vision"])
            stats["text_nodes_p"] += float(p["num_text"])
            if q["attn_layers"] and stats["attn_layer_center"] < 0:
                stats["attn_layer_center"] = float(q["attn_layers"][len(q["attn_layers"]) // 2])

        return matched_q, matched_p, attn_q, attn_p, mask_q, mask_p, per_sample_meta, stats

    def _build_batch_meta(
        self,
        teacher_meta: List[Dict[str, Any]],
        student_meta: List[Dict[str, Any]],
        offsets_t: Dict[str, List[int]],
        offsets_s: Dict[str, List[int]],
        idx_rq_t: Sequence[int],
        idx_rp_t: Sequence[int],
        idx_rq_s: Sequence[int],
        idx_rp_s: Sequence[int],
        n_total_t: int,
        n_total_s: int,
    ) -> List[Dict[str, Any]]:
        """
        Per-sample meta for P. Node layout inside each cluster: [vision | text].
        Query/Positive each have their own visual/text blocks; P uses query-side
        geometry for the query cluster and positive-side for the positive cluster.
        """
        batch_meta = []
        b = len(teacher_meta)
        for i in range(b):
            tq, tp = teacher_meta[i]["q"], teacher_meta[i]["p"]
            sq, sp = student_meta[i]["q"], student_meta[i]["p"]

            # Prefer query geometry for FRA (same image policy); fall back to pos.
            h_t = tq["H"] or tp["H"]
            w_t = tq["W"] or tp["W"]
            h_s = sq["H"] or sp["H"]
            w_s = sq["W"] or sp["W"]

            def _side_offsets(cluster_start: int, bundle: Dict[str, Any]) -> Dict[str, Any]:
                n_v = int(bundle["num_vision"])
                n_t = int(bundle["num_text"])
                return {
                    "visual": cluster_start,
                    "text": cluster_start + n_v,
                    "cls": None,
                    "n_vision": n_v,
                    "n_text": n_t,
                }

            # P is built once per sample using query clusters for token matching;
            # positive clusters reuse the same FRA / char-overlap pattern at their offsets.
            t_q_local = _side_offsets(offsets_t["q"][i], tq)
            s_q_local = _side_offsets(offsets_s["q"][i], sq)
            t_p_local = _side_offsets(offsets_t["p"][i], tp)
            s_p_local = _side_offsets(offsets_s["p"][i], sp)

            batch_meta.append({
                # Query-cluster projection block
                "H_t": h_t, "W_t": w_t, "H_s": h_s, "W_s": w_s,
                "teacher_token_char_spans": tq["char_spans"],
                "student_token_char_spans": sq["char_spans"],
                "teacher_offsets": {
                    **t_q_local,
                    "RQ": idx_rq_t[i],
                    "RP": idx_rp_t[i],
                    "end": n_total_t,
                    # also stash positive local offsets for second block
                    "visual_p": t_p_local["visual"],
                    "text_p": t_p_local["text"],
                    "n_vision_p": t_p_local["n_vision"],
                    "n_text_p": t_p_local["n_text"],
                },
                "student_offsets": {
                    **s_q_local,
                    "RQ": idx_rq_s[i],
                    "RP": idx_rp_s[i],
                    "end": n_total_s,
                    "visual_p": s_p_local["visual"],
                    "text_p": s_p_local["text"],
                    "n_vision_p": s_p_local["n_vision"],
                    "n_text_p": s_p_local["n_text"],
                },
                "teacher_pos_char_spans": tp["char_spans"],
                "student_pos_char_spans": sp["char_spans"],
                "H_t_p": tp["H"], "W_t_p": tp["W"],
                "H_s_p": sp["H"], "W_s_p": sp["W"],
            })
        return batch_meta

    def _build_projection_with_pos(
        self,
        batch_meta: List[Dict[str, Any]],
        n_total_t: int,
        n_total_s: int,
        device: torch.device,
    ) -> torch.Tensor:
        """P with both query and positive visual/text blocks + RQ/RP identity."""
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []

        def _add_visual(h_t, w_t, h_s, w_s, t0, s0):
            if min(h_t, w_t, h_s, w_s) <= 0:
                return
            o = fractional_region_alignment(h_t, w_t, h_s, w_s, device)
            o_row = row_normalize(o)
            nz_i, nz_j = o_row.nonzero(as_tuple=True)
            for a, b in zip(nz_i.tolist(), nz_j.tolist()):
                rows.append(t0 + a)
                cols.append(s0 + b)
                vals.append(float(o_row[a, b].item()))

        def _add_text(spans_t, spans_s, t0, s0, n_t_expected, n_s_expected):
            if n_t_expected <= 0 or n_s_expected <= 0:
                return
            if spans_t and spans_s and len(spans_t) == n_t_expected and len(spans_s) == n_s_expected:
                o_text = char_overlap_matrix(spans_t, spans_s, device)
                row_sum = o_text.sum(dim=1)
                empty = row_sum <= 0
                if empty.any():
                    o_text = o_text.clone()
                    o_text[empty] = 1.0 / n_s_expected
                o_row = row_normalize(o_text)
            else:
                # Uniform many-to-many fallback when offsets unavailable
                o_row = torch.full(
                    (n_t_expected, n_s_expected), 1.0 / n_s_expected,
                    device=device, dtype=torch.float32,
                )
            nz_i, nz_j = o_row.nonzero(as_tuple=True)
            for a, b in zip(nz_i.tolist(), nz_j.tolist()):
                rows.append(t0 + a)
                cols.append(s0 + b)
                vals.append(float(o_row[a, b].item()))

        for m in batch_meta:
            t_off, s_off = m["teacher_offsets"], m["student_offsets"]

            # Query cluster
            _add_visual(m["H_t"], m["W_t"], m["H_s"], m["W_s"], t_off["visual"], s_off["visual"])
            _add_text(
                m["teacher_token_char_spans"], m["student_token_char_spans"],
                t_off["text"], s_off["text"], t_off["n_text"], s_off["n_text"],
            )

            # Positive cluster
            _add_visual(
                m["H_t_p"] or m["H_t"], m["W_t_p"] or m["W_t"],
                m["H_s_p"] or m["H_s"], m["W_s_p"] or m["W_s"],
                t_off["visual_p"], s_off["visual_p"],
            )
            _add_text(
                m["teacher_pos_char_spans"], m["student_pos_char_spans"],
                t_off["text_p"], s_off["text_p"],
                t_off["n_text_p"], s_off["n_text_p"],
            )

            rows.extend([int(t_off["RQ"]), int(t_off["RP"])])
            cols.extend([int(s_off["RQ"]), int(s_off["RP"])])
            vals.extend([1.0, 1.0])

        if not vals:
            idx = torch.zeros(2, 0, device=device, dtype=torch.long)
            val = torch.zeros(0, device=device, dtype=torch.float32)
            return torch.sparse_coo_tensor(idx, val, (n_total_t, n_total_s)).coalesce()

        indices = torch.tensor([rows, cols], device=device, dtype=torch.long)
        values = torch.tensor(vals, device=device, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices, values, (n_total_t, n_total_s), device=device,
        ).coalesce()

    def forward(self, distiller, input_data):
        student_model = distiller.student
        teacher_model = distiller.teacher

        student_qry_input = input_data["student_inputs"]["qry"]
        student_pos_input = input_data["student_inputs"]["pos"]
        teacher_qry_input = input_data["teacher_inputs"]["qry"]
        teacher_pos_input = input_data["teacher_inputs"]["pos"]
        qry_image_sizes = input_data.get("qry_image_sizes", None)
        pos_image_sizes = input_data.get("pos_image_sizes", None)

        batch_size = student_qry_input["input_ids"].size(0)
        device = student_qry_input["input_ids"].device

        teacher_tokenizer = distiller.tokenizer
        student_tokenizer = self._get_student_tokenizer(distiller)
        qry_text_strings = get_batch_text_strings(teacher_qry_input, teacher_tokenizer)
        pos_text_strings = get_batch_text_strings(teacher_pos_input, teacher_tokenizer)

        # ----- Forward (need attentions for intra-cluster edges) -----
        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(
                teacher_qry_input, output_attentions=True,
            )
            teacher_pos_output = teacher_model.encode_input(
                teacher_pos_input, output_attentions=True,
            )
            (
                teacher_qry_reps,
                teacher_qry_image_features,
                teacher_qry_attn,
                teacher_qry_hidden_states,
            ) = teacher_qry_output
            (
                teacher_pos_reps,
                teacher_pos_image_features,
                teacher_pos_attn,
                teacher_pos_hidden_states,
            ) = teacher_pos_output

        student_qry_output = student_model.encode_input(
            student_qry_input, output_attentions=True,
        )
        student_pos_output = student_model.encode_input(
            student_pos_input, output_attentions=True,
        )
        (
            student_qry_reps,
            student_qry_image_features,
            student_qry_attn,
            student_qry_hidden_states,
        ) = student_qry_output
        (
            student_pos_reps,
            student_pos_image_features,
            student_pos_attn,
            student_pos_hidden_states,
        ) = student_pos_output

        # ----- Collect native tokens / attentions -----
        (
            t_mq, t_mp, t_aq, t_ap, t_msq, t_msp, t_meta, t_stats,
        ) = self._collect_model_batch(
            is_teacher=True,
            qry_input=teacher_qry_input,
            pos_input=teacher_pos_input,
            qry_hidden=teacher_qry_hidden_states,
            pos_hidden=teacher_pos_hidden_states,
            qry_attn=teacher_qry_attn,
            pos_attn=teacher_pos_attn,
            qry_img_feats=teacher_qry_image_features,
            pos_img_feats=teacher_pos_image_features,
            qry_image_sizes=qry_image_sizes,
            pos_image_sizes=pos_image_sizes,
            qry_texts=qry_text_strings,
            pos_texts=pos_text_strings,
            tokenizer=teacher_tokenizer,
            peer_tokenizer=student_tokenizer,
            peer_qry_input=student_qry_input,
            peer_pos_input=student_pos_input,
            patch_size=self.teacher_patch_size,
            batch_size=batch_size,
        )
        (
            s_mq, s_mp, s_aq, s_ap, s_msq, s_msp, s_meta, s_stats,
        ) = self._collect_model_batch(
            is_teacher=False,
            qry_input=student_qry_input,
            pos_input=student_pos_input,
            qry_hidden=student_qry_hidden_states,
            pos_hidden=student_pos_hidden_states,
            qry_attn=student_qry_attn,
            pos_attn=student_pos_attn,
            qry_img_feats=student_qry_image_features,
            pos_img_feats=student_pos_image_features,
            qry_image_sizes=qry_image_sizes,
            pos_image_sizes=pos_image_sizes,
            qry_texts=qry_text_strings,
            pos_texts=pos_text_strings,
            tokenizer=student_tokenizer,
            peer_tokenizer=teacher_tokenizer,
            peer_qry_input=teacher_qry_input,
            peer_pos_input=teacher_pos_input,
            patch_size=self.student_patch_size,
            batch_size=batch_size,
        )

        # ----- Assemble independent graphs -----
        with torch.no_grad():
            w_t, n_t, _rq_t, _rp_t = assemble_graph(
                t_mq, t_mp, t_aq, t_ap, t_msq, t_msp,
                topk=self.intra_topk,
                tau_intra=self.tau_intra,
                tau_local=self.tau_local,
                k_neg=self.k_neg,
                bridge_temperature=self.bridge_temperature,
                lambda_neg=self.lambda_neg,
            )
        w_s, n_s, rq_s, rp_s = assemble_graph(
            s_mq, s_mp, s_aq, s_ap, s_msq, s_msp,
            topk=self.intra_topk,
            tau_intra=self.tau_intra,
            tau_local=self.tau_local,
            k_neg=self.k_neg,
            bridge_temperature=self.bridge_temperature,
            lambda_neg=self.lambda_neg,
        )

        # Index maps for P (recompute offsets; identical to assemble_graph)
        n_q_t = [t.size(0) for t in t_mq]
        n_p_t = [t.size(0) for t in t_mp]
        n_q_s = [t.size(0) for t in s_mq]
        n_p_s = [t.size(0) for t in s_mp]
        off_t, idx_rq_t, idx_rp_t, n_total_t = build_global_index(batch_size, n_q_t, n_p_t)
        off_s, idx_rq_s, idx_rp_s, n_total_s = build_global_index(batch_size, n_q_s, n_p_s)
        assert n_total_t == n_t and n_total_s == n_s

        batch_meta = self._build_batch_meta(
            t_meta, s_meta, off_t, off_s,
            idx_rq_t, idx_rp_t, idx_rq_s, idx_rp_s,
            n_total_t, n_total_s,
        )

        # ----- Signed Laplacian + eigenspace -----
        with torch.no_grad():
            l_t = build_signed_laplacian(w_t, n_t)
            _, u_t = get_eigenspace(l_t, n_t, k=self.k_eigen, allow_lobpcg=True)
            u_t = u_t.detach()

        l_s = build_signed_laplacian(w_s, n_s)
        _, u_s = get_eigenspace(l_s, n_s, k=self.k_eigen, allow_lobpcg=False)

        p = self._build_projection_with_pos(batch_meta, n_t, n_s, device)
        kd_loss = spectral_kd_loss(u_t, u_s, p, self.k_eigen)
        if not torch.isfinite(kd_loss):
            logger.warning("spectral_kd_loss non-finite; replacing with 0")
            kd_loss = self._zero(device, rq_s.dtype)

        # ----- Contrastive -----
        if self.use_graph_reps_contrastive:
            cq, cp = rq_s, rp_s
        else:
            cq, cp = student_qry_reps, student_pos_reps

        if self.world_size > 1:
            all_q = self._dist_gather_tensor(cq)
            all_p = self._dist_gather_tensor(cp)
        else:
            all_q, all_p = cq, cp

        c_loss = bidirectional_infonce_loss(
            all_q, all_p, temperature=float(distiller.temperature),
        )

        total_loss = c_loss + self.kd_weight * kd_loss
        kd_weighted = self.kd_weight * kd_loss

        def _metric(v: float) -> torch.Tensor:
            return torch.tensor(v, device=device, dtype=torch.float32)

        # Per-side cluster sizes (sum over batch); super-nodes = 2B (RQ + RP).
        n_q_sum_t = float(sum(n_q_t))
        n_p_sum_t = float(sum(n_p_t))
        n_q_sum_s = float(sum(n_q_s))
        n_p_sum_s = float(sum(n_p_s))

        return {
            # ----- loss components -----
            "loss": total_loss,
            "contrastive_loss": c_loss.detach(),
            "segd_loss": kd_loss.detach(),
            "spectral_kd_loss": kd_loss.detach(),
            "kd_weighted": kd_weighted.detach(),
            "kd_weight": _metric(self.kd_weight),
            # ----- graph size (batch-level) -----
            "batch_size": _metric(float(batch_size)),
            "n_total_teacher": _metric(float(n_t)),
            "n_total_student": _metric(float(n_s)),
            "n_supernodes": _metric(float(2 * batch_size)),
            # Teacher nodes
            "t_vision_nodes_qry": _metric(t_stats["vision_nodes_q"]),
            "t_text_nodes_qry": _metric(t_stats["text_nodes_q"]),
            "t_vision_nodes_pos": _metric(t_stats["vision_nodes_p"]),
            "t_text_nodes_pos": _metric(t_stats["text_nodes_p"]),
            "t_cluster_nodes_qry": _metric(n_q_sum_t),
            "t_cluster_nodes_pos": _metric(n_p_sum_t),
            # Student nodes (aliases keep prior wandb keys)
            "batch_vision_nodes_qry": _metric(s_stats["vision_nodes_q"]),
            "batch_text_nodes_qry": _metric(s_stats["text_nodes_q"]),
            "batch_vision_nodes_pos": _metric(s_stats["vision_nodes_p"]),
            "batch_text_nodes_pos": _metric(s_stats["text_nodes_p"]),
            "s_vision_nodes_qry": _metric(s_stats["vision_nodes_q"]),
            "s_text_nodes_qry": _metric(s_stats["text_nodes_q"]),
            "s_vision_nodes_pos": _metric(s_stats["vision_nodes_p"]),
            "s_text_nodes_pos": _metric(s_stats["text_nodes_p"]),
            "s_cluster_nodes_qry": _metric(n_q_sum_s),
            "s_cluster_nodes_pos": _metric(n_p_sum_s),
            # ----- spectral / layer -----
            "segd_attn_layer": _metric(s_stats["attn_layer_center"]),
            "segd_k_eigen": _metric(float(min(self.k_eigen, u_s.size(1)))),
        }
