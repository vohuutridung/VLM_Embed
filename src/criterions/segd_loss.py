"""
SEGDLoss — Multimodal Spectral Eigenspace Distillation (SEKD).

Loss composition:
  total = contrastive_loss
        + kd_weight * segd_loss
        + kd_weight * w_loss_cka * cka_loss

SEKD runs per-sample (query / positive independently). Graphs are built on
native tokens; modality-aware alignment is applied only after eigendecomposition.
Teacher spectral tensors are always detached — gradients flow only through the
student QR / subspace path.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

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
_EIG_EPS = 1e-6
_HEAT_EPS = 1e-8


# ---------------------------------------------------------------------------
# CKA (batch-level)
# ---------------------------------------------------------------------------

class CKALoss(nn.Module):
    """Linear CKA distance: 1 - CKA(SH, TH). Teacher should be detached upstream."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, student_h: torch.Tensor, teacher_h: torch.Tensor) -> torch.Tensor:
        d_s = student_h.size(-1)
        d_t = teacher_h.size(-1)
        sh = student_h.reshape(-1, d_s).to(torch.float64)
        th = teacher_h.reshape(-1, d_t).to(torch.float64)

        sh = sh - sh.mean(0, keepdim=True)
        th = th - th.mean(0, keepdim=True)

        num = torch.norm(sh.t() @ th, p="fro")
        den1 = torch.norm(sh.t() @ sh, p="fro") + self.eps
        den2 = torch.norm(th.t() @ th, p="fro") + self.eps
        return (1.0 - num / torch.sqrt(den1 * den2)).to(student_h.dtype)


# ---------------------------------------------------------------------------
# Token / geometry helpers
# ---------------------------------------------------------------------------

def l2_normalize_tokens(h: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    return h / h.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def pairwise_sq_l2(h: torch.Tensor) -> torch.Tensor:
    """Squared L2 distances for row-normalized (or general) tokens. [N, N]."""
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    norm_sq = (h * h).sum(dim=-1, keepdim=True)
    dist_sq = norm_sq + norm_sq.t() - 2.0 * (h @ h.t())
    return dist_sq.clamp_min(0.0)


def infer_spatial_hw(
    num_tokens: int,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    patch_size: Optional[int] = None,
) -> Tuple[int, int]:
    """Infer (H, W) grid for vision tokens. Prefer perfect square, else aspect."""
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
        # Try nearby factorizations preserving aspect.
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

    # Generic nearest factorization.
    best_h = side
    for h_try in range(side, 0, -1):
        if num_tokens % h_try == 0:
            best_h = h_try
            break
    return best_h, num_tokens // best_h


def build_word_char_spans(text: str) -> List[Tuple[int, int]]:
    """Whitespace / punctuation-aware word spans over `text` (char offsets)."""
    if not text:
        return []
    spans: List[Tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        spans.append((match.start(), match.end()))
    return spans


def build_text_alignment_matrix(
    token_offsets: torch.Tensor,
    word_spans: Sequence[Tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    A[r, i] = |[a_r, b_r) ∩ [s_i, e_i)| / (b_r - a_r)

    Returns [n_shared_words, n_tokens] with empty-overlap rows dropped, or None.
    """
    if token_offsets is None or len(word_spans) == 0 or token_offsets.numel() == 0:
        return None

    offs = token_offsets.to(device=device)
    n_tokens = offs.size(0)
    n_words = len(word_spans)

    t_start = offs[:, 0].to(torch.float64).unsqueeze(0)  # [1, Nt]
    t_end = offs[:, 1].to(torch.float64).unsqueeze(0)

    w_start = torch.tensor([a for a, _ in word_spans], device=device, dtype=torch.float64).unsqueeze(1)
    w_end = torch.tensor([b for _, b in word_spans], device=device, dtype=torch.float64).unsqueeze(1)
    word_len = (w_end - w_start).clamp_min(1.0)

    overlap = (torch.minimum(w_end, t_end) - torch.maximum(w_start, t_start)).clamp_min(0.0)
    valid_tok = (offs[:, 1] > offs[:, 0]).to(torch.float64).unsqueeze(0)
    A = (overlap / word_len) * valid_tok  # [Nw, Nt]

    row_ok = A.sum(dim=1) > 0
    if not bool(row_ok.any()):
        return None
    return A[row_ok].to(dtype=dtype)


def build_vision_alignment_operator(
    num_tokens: int,
    src_h: int,
    src_w: int,
    tgt_h: int,
    tgt_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Dense bilinear resampling matrix A of shape [H0*W0, N] such that
    Z = A @ E reshapes each eigen-column as an H×W map and interpolates to H0×W0.

    Built via basis responses (pure geometry — no learnable params).
    """
    if num_tokens <= 0 or src_h <= 0 or src_w <= 0 or tgt_h <= 0 or tgt_w <= 0:
        return None

    grid_n = src_h * src_w
    if grid_n < num_tokens:
        src_w = int(math.ceil(num_tokens / max(src_h, 1)))
        grid_n = src_h * src_w

    # Columns = source token basis; rows after interpolate = target pixels.
    basis = torch.zeros(grid_n, num_tokens, device=device, dtype=torch.float32)
    n_copy = min(num_tokens, grid_n)
    basis[:n_copy, :n_copy] = torch.eye(n_copy, device=device, dtype=torch.float32)

    maps = basis.t().contiguous().view(num_tokens, 1, src_h, src_w)
    maps = F.interpolate(maps, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
    return maps.view(num_tokens, tgt_h * tgt_w).t().contiguous().to(dtype=dtype)


def build_joint_text_alignment_matrices(
    teacher_offsets: torch.Tensor,
    student_offsets: torch.Tensor,
    word_spans: Sequence[Tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Build paired A_T, A_S over the same shared-word rows.
    A[r, i] = |[a_r,b_r) ∩ [s_i,e_i)| / (b_r - a_r); drop words with no overlap on either side.
    """
    if (
        teacher_offsets is None or student_offsets is None
        or len(word_spans) == 0
        or teacher_offsets.numel() == 0
        or student_offsets.numel() == 0
    ):
        return None, None

    offs_t = teacher_offsets.to(device=device).float()
    offs_s = student_offsets.to(device=device).float()
    w_start = torch.tensor([a for a, _ in word_spans], device=device, dtype=torch.float32)
    w_end = torch.tensor([b for _, b in word_spans], device=device, dtype=torch.float32)
    word_len = (w_end - w_start).clamp_min(1.0).unsqueeze(1)  # [Nw, 1]

    # [Nw, Nt]
    ov_t = (
        torch.minimum(w_end.unsqueeze(1), offs_t[:, 1].unsqueeze(0))
        - torch.maximum(w_start.unsqueeze(1), offs_t[:, 0].unsqueeze(0))
    ).clamp_min(0.0)
    ov_s = (
        torch.minimum(w_end.unsqueeze(1), offs_s[:, 1].unsqueeze(0))
        - torch.maximum(w_start.unsqueeze(1), offs_s[:, 0].unsqueeze(0))
    ).clamp_min(0.0)

    valid_t = (offs_t[:, 1] > offs_t[:, 0]).float().unsqueeze(0)
    valid_s = (offs_s[:, 1] > offs_s[:, 0]).float().unsqueeze(0)
    a_t = (ov_t / word_len) * valid_t
    a_s = (ov_s / word_len) * valid_s

    row_ok = (a_t.sum(dim=1) > 0) & (a_s.sum(dim=1) > 0)
    if not bool(row_ok.any()):
        return None, None
    return a_t[row_ok].to(dtype=dtype), a_s[row_ok].to(dtype=dtype)


# ---------------------------------------------------------------------------
# Graph construction (modality-specific)
# ---------------------------------------------------------------------------

def build_knn_self_tuning_adjacency(
    h_norm: torch.Tensor,
    k_neighbors: int,
    eps: float = _HEAT_EPS,
) -> torch.Tensor:
    """
    Symmetric-union kNN graph with self-tuning heat-kernel weights.
    Discrete neighbor selection is detached; weights remain differentiable w.r.t. h.
    """
    n = h_norm.size(0)
    device, dtype = h_norm.device, h_norm.dtype
    if n < 2:
        return torch.zeros(n, n, device=device, dtype=dtype)

    k = min(max(1, k_neighbors), n - 1)
    h32 = h_norm.float()
    dist_sq = pairwise_sq_l2(h32)

    with torch.no_grad():
        # Exclude self: take k+1 smallest then drop index 0 (self).
        knn_dist_sq, knn_idx = torch.topk(dist_sq, k=k + 1, largest=False, dim=1)
        knn_idx = knn_idx[:, 1:]
        knn_dist_sq = knn_dist_sq[:, 1:]
        sigma = knn_dist_sq[:, -1].clamp_min(eps).sqrt()  # dist to k-th NN

        mask = torch.zeros(n, n, device=device, dtype=torch.bool)
        mask.scatter_(1, knn_idx, True)
        mask = mask | mask.t()
        mask.fill_diagonal_(False)

    sigma_i = sigma.unsqueeze(1)
    sigma_j = sigma.unsqueeze(0)
    denom = (sigma_i * sigma_j).clamp_min(eps)
    W = torch.exp(-dist_sq / denom)
    W = torch.where(mask, W, torch.zeros_like(W))
    W = 0.5 * (W + W.t())
    diag = torch.arange(n, device=device)
    W[diag, diag] = 0.0
    return W.to(dtype=dtype)


def build_bipartite_relu_cosine_adjacency(
    h_v: torch.Tensor,
    h_t: torch.Tensor,
) -> torch.Tensor:
    """
    Bipartite adjacency:
      C_ij = max(0, h_i^v · h_j^t)
      W = [[0, C], [C^T, 0]]
    """
    n_v, n_t = h_v.size(0), h_t.size(0)
    device, dtype = h_v.device, h_v.dtype
    n = n_v + n_t
    if n_v == 0 or n_t == 0:
        return torch.zeros(n, n, device=device, dtype=dtype)

    c = (h_v.float() @ h_t.float().t()).clamp_min(0.0)
    W = torch.zeros(n, n, device=device, dtype=torch.float32)
    W[:n_v, n_v:] = c
    W[n_v:, :n_v] = c.t()
    return W.to(dtype=dtype)


def unnormalized_laplacian(W: torch.Tensor) -> torch.Tensor:
    W32 = W.float()
    deg = W32.sum(dim=1)
    return torch.diag(deg) - W32


# ---------------------------------------------------------------------------
# Spectral analysis + adaptive eigengap dimension
# ---------------------------------------------------------------------------

def _count_nullity(eigenvalues: torch.Tensor, eig_eps: float) -> int:
    return int((eigenvalues <= eig_eps).sum().item())


def select_adaptive_kg(
    teacher_eigenvalues: torch.Tensor,
    teacher_nullity: int,
    k_min: int,
    k_max: int,
) -> int:
    """
    k_g = argmax_r Δ_r where Δ_r = λ_{c+r+1} - λ_{c+r}, r ∈ [k_min, k_max_eff].
    Discrete — no gradient.
    """
    n = teacher_eigenvalues.numel()
    # Need indices c+r and c+r+1 ⇒ r_max ≤ n - c - 2? Actually r can go up to
    # n - c - 1 eigenvectors available; eigengap at r needs λ_{c+r+1} so r ≤ n-c-2.
    max_r_by_spectrum = n - teacher_nullity - 2
    if max_r_by_spectrum < k_min:
        # Fall back to whatever non-null dims remain (at least 1 if possible).
        avail = max(0, n - teacher_nullity - 1)
        return max(1, min(avail, k_max)) if avail > 0 else 0

    k_max_eff = min(k_max, max_r_by_spectrum)
    if k_max_eff < k_min:
        return max(1, k_max_eff)

    with torch.no_grad():
        ev = teacher_eigenvalues.detach()
        best_r = k_min
        best_gap = -1.0
        for r in range(k_min, k_max_eff + 1):
            gap = float(ev[teacher_nullity + r + 1] - ev[teacher_nullity + r])
            if gap > best_gap:
                best_gap = gap
                best_r = r
    return int(best_r)


def eigendecompose_laplacian(
    L: torch.Tensor,
    eig_eps: float = _EIG_EPS,
    jitter: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric eigh on L (+ jitter). Returns (eigenvalues↑, eigenvectors)."""
    n = L.size(0)
    L_j = L.float() + jitter * torch.eye(n, device=L.device, dtype=torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eigh(L_j)
    # Numerical cleanup for tiny negatives.
    eigenvalues = eigenvalues.clamp_min(0.0)
    return eigenvalues, eigenvectors


def extract_eigenmap(
    eigenvectors: torch.Tensor,
    nullity: int,
    k_g: int,
) -> Optional[torch.Tensor]:
    """E = [u_{c+1}, ..., u_{c+k_g}] with shape [N, k_g]."""
    n = eigenvectors.size(0)
    if k_g <= 0:
        return None
    start = nullity
    end = nullity + k_g
    if end > n:
        return None
    return eigenvectors[:, start:end]


def subspace_loss_from_qr(
    z_teacher: torch.Tensor,
    z_student: torch.Tensor,
) -> torch.Tensor:
    """
    Reduced QR on aligned eigenmaps, then principal-angle subspace loss:
      L = 1 - (1/k) ||Q_T^T Q_S||_F^2
    Teacher branch is detached.
    """
    # Match column rank: use min width.
    k = min(z_teacher.size(1), z_student.size(1))
    if k <= 0 or z_teacher.size(0) < 1 or z_student.size(0) < 1:
        return z_student.new_tensor(0.0)

    zt = z_teacher[:, :k].float().detach()
    zs = z_student[:, :k].float()

    # If rows < k, QR still yields Q with min(m,k) cols — clamp k.
    k_eff = min(k, zt.size(0), zs.size(0))
    if k_eff <= 0:
        return zs.new_tensor(0.0)
    zt = zt[:, :k_eff]
    zs = zs[:, :k_eff]

    q_t, _ = torch.linalg.qr(zt, mode="reduced")
    q_s, _ = torch.linalg.qr(zs, mode="reduced")
    q_t = q_t.detach()

    gram = q_t.t() @ q_s
    return (1.0 - (gram.pow(2).sum() / float(k_eff))).to(z_student.dtype)


# ---------------------------------------------------------------------------
# Post-spectral alignment
# ---------------------------------------------------------------------------

def align_vision_eigenmap(
    eigenmap: torch.Tensor,
    src_h: int,
    src_w: int,
    tgt_h: int,
    tgt_w: int,
) -> Optional[torch.Tensor]:
    """Bilinearly resample each eigen-column as an H×W map → [H0*W0, k]."""
    n, k = eigenmap.shape
    if n == 0 or k == 0 or src_h <= 0 or src_w <= 0:
        return None
    if src_h * src_w != n:
        # Pad tokens to rectangular grid.
        grid_n = src_h * src_w
        if grid_n < n:
            src_w = int(math.ceil(n / max(src_h, 1)))
            grid_n = src_h * src_w
        padded = eigenmap.new_zeros(grid_n, k)
        padded[:n] = eigenmap
        eigenmap = padded
        n = grid_n

    # [k, 1, H, W]
    maps = eigenmap.t().contiguous().view(k, 1, src_h, src_w)
    maps = F.interpolate(maps, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
    return maps.view(k, tgt_h * tgt_w).t().contiguous()


def align_text_eigenmap(
    eigenmap: torch.Tensor,
    alignment: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Z = A @ E with A: [n_words, n_tokens], E: [n_tokens, k]."""
    if alignment is None or eigenmap is None:
        return None
    if alignment.size(1) != eigenmap.size(0):
        return None
    if alignment.size(0) == 0:
        return None
    return alignment @ eigenmap


def align_bipartite_eigenmap(
    eigenmap: torch.Tensor,
    n_v: int,
    a_v: Optional[torch.Tensor],
    a_t: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Block-diagonal alignment: Z = blkdiag(A_v, A_t) @ E."""
    if a_v is None or a_t is None or eigenmap is None:
        return None
    n_total = eigenmap.size(0)
    n_t = n_total - n_v
    if n_v <= 0 or n_t <= 0:
        return None
    if a_v.size(1) != n_v or a_t.size(1) != n_t:
        return None

    e_v = eigenmap[:n_v]
    e_t = eigenmap[n_v:]
    z_v = a_v @ e_v
    z_t = a_t @ e_t
    return torch.cat([z_v, z_t], dim=0)


# ---------------------------------------------------------------------------
# Per-graph / per-sample SEKD
# ---------------------------------------------------------------------------

def _spectral_pair_loss(
    h_t: torch.Tensor,
    h_s: torch.Tensor,
    build_w_fn,
    align_fn_t,
    align_fn_s,
    k_min: int,
    k_max: int,
    eig_eps: float,
) -> Optional[torch.Tensor]:
    """
    Shared path for one graph type:
      normalize → W → L → eigh → adaptive k (teacher) → eigenmaps → align → QR loss.
    """
    if h_t is None or h_s is None:
        return None
    if h_t.size(0) < 2 or h_s.size(0) < 2:
        return None

    h_t_n = l2_normalize_tokens(h_t.float())
    h_s_n = l2_normalize_tokens(h_s.float())

    # Teacher spectral (fully detached).
    with torch.no_grad():
        w_t = build_w_fn(h_t_n)
        L_t = unnormalized_laplacian(w_t)
        eval_t, evec_t = eigendecompose_laplacian(L_t, eig_eps=eig_eps)
        c_t = _count_nullity(eval_t, eig_eps)
        k_g = select_adaptive_kg(eval_t, c_t, k_min=k_min, k_max=k_max)
        if k_g <= 0:
            return None
        e_t = extract_eigenmap(evec_t, c_t, k_g)
        if e_t is None:
            return None
        e_t = e_t.detach()

    # Student spectral (grad through W / L / eigh / eigenmap).
    w_s = build_w_fn(h_s_n)
    L_s = unnormalized_laplacian(w_s)
    eval_s, evec_s = eigendecompose_laplacian(L_s, eig_eps=eig_eps)
    with torch.no_grad():
        c_s = _count_nullity(eval_s.detach(), eig_eps)
        # Ensure k_g fits student spectrum.
        max_ks = max(0, evec_s.size(0) - c_s - 0)
        k_g_s = min(k_g, max_ks)
    if k_g_s <= 0:
        return None
    # Re-extract teacher with possibly reduced k for QR compatibility.
    k_use = min(k_g, k_g_s)
    e_t = e_t[:, :k_use]
    e_s = extract_eigenmap(evec_s, c_s, k_use)
    if e_s is None:
        return None

    z_t = align_fn_t(e_t)
    z_s = align_fn_s(e_s)
    if z_t is None or z_s is None:
        return None
    if z_t.size(0) != z_s.size(0) or z_t.size(0) < 1:
        return None

    return subspace_loss_from_qr(z_t, z_s)


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
        self.w_loss_cka = float(getattr(args, "w_loss_cka", 1.0))
        self.w_loss_v = float(getattr(args, "w_loss_v", 1.0))
        self.w_loss_t = float(getattr(args, "w_loss_t", 1.0))
        self.w_loss_cross = float(getattr(args, "w_loss_cross", 1.0))

        self.knn_neighbors = int(getattr(args, "knn_neighbors", 10))
        self.k_min = int(getattr(args, "sekd_k_min", 2))
        self.k_max = int(getattr(args, "sekd_k_max", getattr(args, "num_eigenvectors", 16)))
        self.eig_eps = float(getattr(args, "sekd_eig_eps", _EIG_EPS))
        self.align_grid_h = int(getattr(args, "sekd_align_grid_h", 10))
        self.align_grid_w = int(getattr(args, "sekd_align_grid_w", 10))

        self.teacher_patch_size = int(getattr(args, "teacher_patch_size", 28))
        self.student_patch_size = int(getattr(args, "student_patch_size", 64))

        self.cka = CKALoss(eps=1e-8)
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

    def _sample_sekd_loss(
        self,
        t_vision: Optional[torch.Tensor],
        s_vision: Optional[torch.Tensor],
        t_text: Optional[torch.Tensor],
        s_text: Optional[torch.Tensor],
        t_offsets: Optional[torch.Tensor],
        s_offsets: Optional[torch.Tensor],
        reference_text: str,
        img_w: int,
        img_h: int,
        has_image: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted SEKD loss for one sample across valid graphs.
        Returns (loss, stats).
        """
        device = (
            s_vision.device if s_vision is not None
            else s_text.device if s_text is not None
            else t_vision.device if t_vision is not None
            else t_text.device if t_text is not None
            else torch.device("cpu")
        )
        dtype = (
            s_vision.dtype if s_vision is not None
            else s_text.dtype if s_text is not None
            else torch.float32
        )
        stats = {
            "vision_nodes_t": float(t_vision.size(0)) if t_vision is not None else 0.0,
            "vision_nodes_s": float(s_vision.size(0)) if s_vision is not None else 0.0,
            "text_nodes_t": float(t_text.size(0)) if t_text is not None else 0.0,
            "text_nodes_s": float(s_text.size(0)) if s_text is not None else 0.0,
            "graph_v": 0.0,
            "graph_t": 0.0,
            "graph_vt": 0.0,
        }

        losses: List[torch.Tensor] = []
        weights: List[float] = []

        # --- Shared text alignment operators (post-spectral) ---
        word_spans = build_word_char_spans(reference_text)
        a_t_teacher, a_t_student = build_joint_text_alignment_matrices(
            t_offsets, s_offsets, word_spans, device, dtype,
        )

        # --- Vision spatial shapes ---
        t_hw = s_hw = (0, 0)
        if has_image and t_vision is not None and s_vision is not None:
            t_hw = infer_spatial_hw(
                t_vision.size(0), img_w, img_h, self.teacher_patch_size,
            )
            s_hw = infer_spatial_hw(
                s_vision.size(0), img_w, img_h, self.student_patch_size,
            )

        tgt_h, tgt_w = self.align_grid_h, self.align_grid_w

        # ===== G_v =====
        if (
            has_image
            and t_vision is not None and s_vision is not None
            and t_vision.size(0) >= 2 and s_vision.size(0) >= 2
            and self.w_loss_v > 0
        ):
            th, tw = t_hw
            sh, sw = s_hw

            def build_w_v(h_n):
                return build_knn_self_tuning_adjacency(h_n, self.knn_neighbors)

            loss_v = _spectral_pair_loss(
                t_vision, s_vision, build_w_v,
                align_fn_t=lambda e: align_vision_eigenmap(e, th, tw, tgt_h, tgt_w),
                align_fn_s=lambda e: align_vision_eigenmap(e, sh, sw, tgt_h, tgt_w),
                k_min=self.k_min,
                k_max=self.k_max,
                eig_eps=self.eig_eps,
            )
            if loss_v is not None and torch.isfinite(loss_v):
                losses.append(loss_v)
                weights.append(self.w_loss_v)
                stats["graph_v"] = 1.0

        # ===== G_t =====
        if (
            t_text is not None and s_text is not None
            and t_text.size(0) >= 2 and s_text.size(0) >= 2
            and a_t_teacher is not None and a_t_student is not None
            and self.w_loss_t > 0
        ):
            def build_w_t(h_n):
                return build_knn_self_tuning_adjacency(h_n, self.knn_neighbors)

            loss_t = _spectral_pair_loss(
                t_text, s_text, build_w_t,
                align_fn_t=lambda e: align_text_eigenmap(e, a_t_teacher),
                align_fn_s=lambda e: align_text_eigenmap(e, a_t_student),
                k_min=self.k_min,
                k_max=self.k_max,
                eig_eps=self.eig_eps,
            )
            if loss_t is not None and torch.isfinite(loss_t):
                losses.append(loss_t)
                weights.append(self.w_loss_t)
                stats["graph_t"] = 1.0

        # ===== G_vt =====
        if (
            has_image
            and t_vision is not None and s_vision is not None
            and t_text is not None and s_text is not None
            and t_vision.size(0) >= 1 and s_vision.size(0) >= 1
            and t_text.size(0) >= 1 and s_text.size(0) >= 1
            and a_t_teacher is not None and a_t_student is not None
            and self.w_loss_cross > 0
        ):
            th, tw = t_hw
            sh, sw = s_hw
            # Precompute vision alignment densematrices for block-diag path.
            a_v_t = build_vision_alignment_operator(
                t_vision.size(0), th, tw, tgt_h, tgt_w, device, dtype,
            )
            a_v_s = build_vision_alignment_operator(
                s_vision.size(0), sh, sw, tgt_h, tgt_w, device, dtype,
            )

            if a_v_t is not None and a_v_s is not None:
                t_vt = torch.cat([t_vision, t_text], dim=0)
                s_vt = torch.cat([s_vision, s_text], dim=0)
                loss_vt = self._bipartite_pair_loss(
                    t_vt, s_vt, t_vision.size(0), s_vision.size(0),
                    a_v_t, a_v_s, a_t_teacher, a_t_student,
                )
                if loss_vt is not None and torch.isfinite(loss_vt):
                    losses.append(loss_vt)
                    weights.append(self.w_loss_cross)
                    stats["graph_vt"] = 1.0

        if not losses:
            return self._zero(device, dtype), stats

        w_sum = sum(weights)
        weighted = sum(w * l for w, l in zip(weights, losses)) / max(w_sum, _EPS)
        return weighted, stats

    def _bipartite_pair_loss(
        self,
        h_t: torch.Tensor,
        h_s: torch.Tensor,
        n_v_t: int,
        n_v_s: int,
        a_v_t: torch.Tensor,
        a_v_s: torch.Tensor,
        a_t_t: torch.Tensor,
        a_t_s: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if h_t.size(0) < 3 or h_s.size(0) < 3:
            return None

        h_t_n = l2_normalize_tokens(h_t.float())
        h_s_n = l2_normalize_tokens(h_s.float())

        with torch.no_grad():
            w_t = build_bipartite_relu_cosine_adjacency(h_t_n[:n_v_t], h_t_n[n_v_t:])
            L_t = unnormalized_laplacian(w_t)
            eval_t, evec_t = eigendecompose_laplacian(L_t, eig_eps=self.eig_eps)
            c_t = _count_nullity(eval_t, self.eig_eps)
            k_g = select_adaptive_kg(eval_t, c_t, self.k_min, self.k_max)
            if k_g <= 0:
                return None
            e_t = extract_eigenmap(evec_t, c_t, k_g)
            if e_t is None:
                return None
            e_t = e_t.detach()

        w_s = build_bipartite_relu_cosine_adjacency(h_s_n[:n_v_s], h_s_n[n_v_s:])
        L_s = unnormalized_laplacian(w_s)
        eval_s, evec_s = eigendecompose_laplacian(L_s, eig_eps=self.eig_eps)
        with torch.no_grad():
            c_s = _count_nullity(eval_s.detach(), self.eig_eps)
            k_g_s = min(k_g, max(0, evec_s.size(0) - c_s))
        if k_g_s <= 0:
            return None
        k_use = min(k_g, k_g_s)
        e_t = e_t[:, :k_use]
        e_s = extract_eigenmap(evec_s, c_s, k_use)
        if e_s is None:
            return None

        z_t = align_bipartite_eigenmap(e_t, n_v_t, a_v_t, a_t_t)
        z_s = align_bipartite_eigenmap(e_s, n_v_s, a_v_s, a_t_s)
        if z_t is None or z_s is None:
            return None
        # Aligned dims: vision grid is shared; text words are shared → same row count.
        if z_t.size(0) != z_s.size(0):
            return None
        return subspace_loss_from_qr(z_t, z_s)

    def _extract_sample_modalities(
        self,
        teacher_input,
        student_input,
        text_string: str,
        s_img_feats,
        t_img_feats,
        s_hidden,
        t_hidden,
        image_sizes,
        sample_idx: int,
        teacher_tokenizer,
        student_tokenizer,
    ):
        num_text_t = count_text_tokens_teacher(teacher_input["input_ids"][sample_idx])
        num_text_s = count_text_tokens_student(student_input["input_ids"][sample_idx])
        has_image = (
            s_img_feats is not None
            and sample_idx < len(s_img_feats)
            and s_img_feats[sample_idx] is not None
        )
        num_v_s = int(s_img_feats[sample_idx].size(0)) if has_image else 0
        num_v_t = int(t_img_feats[sample_idx].size(0)) if has_image else 0

        img_w = img_h = 0
        if has_image:
            if image_sizes is not None and sample_idx < len(image_sizes):
                img_w, img_h = image_sizes[sample_idx]
            else:
                side = int(math.sqrt(max(num_v_t, 1)))
                img_w = img_h = side * self.teacher_patch_size

        t_text = s_text = None
        t_offsets = s_offsets = None
        if num_text_t > 0 and num_text_s > 0:
            t_text = extract_text_hidden_states(
                t_hidden, sample_idx, num_text_t, num_v_t,
                is_teacher=True, has_image=has_image,
            )[-1]
            s_text = extract_text_hidden_states(
                s_hidden, sample_idx, num_text_s, num_v_s,
                is_teacher=False, has_image=has_image,
            )[-1]
            reference_text = strip_vlm_image_markers(text_string or "")
            t_ids = get_text_token_ids(teacher_input["input_ids"][sample_idx], is_teacher=True)
            s_ids = get_text_token_ids(student_input["input_ids"][sample_idx], is_teacher=False)
            t_offsets, s_offsets = build_paired_text_offsets(
                teacher_tokenizer, student_tokenizer, t_ids, s_ids,
                reference_text, t_text.device,
            )
        else:
            reference_text = strip_vlm_image_markers(text_string or "")

        t_vision = s_vision = None
        if has_image:
            t_vision = extract_vision_hidden_states(
                t_hidden, sample_idx, num_v_t, num_text_t, is_teacher=True,
            )[-1]
            s_vision = extract_vision_hidden_states(
                s_hidden, sample_idx, num_v_s, num_text_s, is_teacher=False,
            )[-1]

        return {
            "t_vision": t_vision,
            "s_vision": s_vision,
            "t_text": t_text,
            "s_text": s_text,
            "t_offsets": t_offsets,
            "s_offsets": s_offsets,
            "reference_text": reference_text,
            "img_w": img_w,
            "img_h": img_h,
            "has_image": has_image,
        }

    def _side_sekd_loss(
        self,
        batch_size: int,
        teacher_input,
        student_input,
        text_strings: List[str],
        s_img_feats,
        t_img_feats,
        s_hidden,
        t_hidden,
        image_sizes,
        teacher_tokenizer,
        student_tokenizer,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Average SEKD over samples on one side (qry or pos). Always returns a tensor."""
        sample_losses: List[torch.Tensor] = []
        agg = {
            "vision_nodes": 0.0,
            "text_nodes": 0.0,
            "valid_samples": 0.0,
            "graph_v": 0.0,
            "graph_t": 0.0,
            "graph_vt": 0.0,
        }

        for i in range(batch_size):
            mods = self._extract_sample_modalities(
                teacher_input, student_input,
                text_strings[i] if i < len(text_strings) else "",
                s_img_feats, t_img_feats, s_hidden, t_hidden, image_sizes, i,
                teacher_tokenizer, student_tokenizer,
            )
            loss_i, stats_i = self._sample_sekd_loss(**mods)
            # DDP-safe: always append a finite contribution (0 if invalid).
            if stats_i["graph_v"] + stats_i["graph_t"] + stats_i["graph_vt"] > 0:
                sample_losses.append(loss_i)
                agg["valid_samples"] += 1.0
            else:
                sample_losses.append(loss_i * 0.0)

            agg["vision_nodes"] += stats_i["vision_nodes_s"]
            agg["text_nodes"] += stats_i["text_nodes_s"]
            agg["graph_v"] += stats_i["graph_v"]
            agg["graph_t"] += stats_i["graph_t"]
            agg["graph_vt"] += stats_i["graph_vt"]

        if not sample_losses:
            return self._zero(device), agg

        # Mean over samples that had ≥1 valid graph; if none, mean of zeros.
        if agg["valid_samples"] > 0:
            # Masked mean without data-dependent Python branching on rank collectives.
            stacked = torch.stack(sample_losses)
            # valid mask from graphs — rebuild cheaply via loss identity with zeros
            # Use equal weight over batch for DDP stability (invalid = 0 contribution
            # already); scale by batch/valid so magnitude matches per-valid mean.
            loss = stacked.sum() / max(agg["valid_samples"], 1.0)
        else:
            loss = torch.stack(sample_losses).mean()
        return loss, agg

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

        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            (
                teacher_qry_reps,
                teacher_qry_image_features,
                _teacher_qry_attn,
                teacher_qry_hidden_states,
            ) = teacher_qry_output
            (
                teacher_pos_reps,
                teacher_pos_image_features,
                _teacher_pos_attn,
                teacher_pos_hidden_states,
            ) = teacher_pos_output

        student_qry_output = student_model.encode_input(
            student_qry_input, output_attentions=False,
        )
        student_pos_output = student_model.encode_input(
            student_pos_input, output_attentions=False,
        )
        (
            student_qry_reps,
            student_qry_image_features,
            _student_qry_attn,
            student_qry_hidden_states,
        ) = student_qry_output
        (
            student_pos_reps,
            student_pos_image_features,
            _student_pos_attn,
            student_pos_hidden_states,
        ) = student_pos_output

        # ----- Contrastive -----
        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps

        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / distiller.temperature, target)

        # ----- Batch-level CKA on pooled reps -----
        cka_loss = self.cka(student_qry_reps, teacher_qry_reps.detach()) + self.cka(
            student_pos_reps, teacher_pos_reps.detach(),
        )

        # ----- SEKD (per-sample, post-spectral alignment) -----
        segd_qry, stats_qry = self._side_sekd_loss(
            batch_size,
            teacher_qry_input, student_qry_input, qry_text_strings,
            student_qry_image_features, teacher_qry_image_features,
            student_qry_hidden_states, teacher_qry_hidden_states,
            qry_image_sizes, teacher_tokenizer, student_tokenizer, device,
        )
        segd_pos, stats_pos = self._side_sekd_loss(
            batch_size,
            teacher_pos_input, student_pos_input, pos_text_strings,
            student_pos_image_features, teacher_pos_image_features,
            student_pos_hidden_states, teacher_pos_hidden_states,
            pos_image_sizes, teacher_tokenizer, student_tokenizer, device,
        )
        segd_loss = 0.5 * (segd_qry + segd_pos)

        total_loss = (
            contrastive_loss
            + self.kd_weight * segd_loss
            + self.kd_weight * self.w_loss_cka * cka_loss
        )

        def _metric(v: float) -> torch.Tensor:
            return torch.tensor(v, device=device, dtype=torch.float32)

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "cka_loss": cka_loss,
            "segd_loss": segd_loss,
            "segd_loss_qry": segd_qry.detach(),
            "segd_loss_pos": segd_pos.detach(),
            "batch_vision_nodes_qry": _metric(stats_qry["vision_nodes"]),
            "batch_text_nodes_qry": _metric(stats_qry["text_nodes"]),
            "batch_vision_nodes_pos": _metric(stats_pos["vision_nodes"]),
            "batch_text_nodes_pos": _metric(stats_pos["text_nodes"]),
            "sekd_valid_graphs_qry": _metric(
                stats_qry["graph_v"] + stats_qry["graph_t"] + stats_qry["graph_vt"]
            ),
            "sekd_valid_graphs_pos": _metric(
                stats_pos["graph_v"] + stats_pos["graph_t"] + stats_pos["graph_vt"]
            ),
        }
