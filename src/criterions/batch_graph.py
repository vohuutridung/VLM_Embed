import unicodedata
from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

LaplacianType = Literal["unnormalized", "normalized"]
LAPLACIAN_TYPES = ("unnormalized", "normalized")
MASK_FILL_VALUE = -1e4

QWEN_VISION_TOKEN_ID_MIN = 151643
QWEN_VISION_TOKEN_ID_MAX = 151656


def pairwise_sq_dist(x: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance matrix for row vectors in x. Shape: (N, N)."""
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    return x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())


def extract_last_token_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool (B, T, D) hidden states to (B, D) using the last non-padding token."""
    batch_size = hidden_states.size(0)
    left_padding = attention_mask[:, -1].sum() == attention_mask.size(0)
    if left_padding:
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), -1]
    max_length = hidden_states.size(1)
    num_padding = (attention_mask == 0).long().sum(dim=1)
    eos_idx = max_length - num_padding - 1
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), eos_idx]


def build_knn_heat_affinity(
    x: torch.Tensor,
    k: int,
    t: Optional[float] = None,
) -> torch.Tensor:
    """Symmetric kNN affinity matrix with heat-kernel weights. Shape: (N, N)."""
    n = x.size(0)
    if n < 2:
        return torch.zeros(n, n, device=x.device, dtype=x.dtype)

    k = min(k, n - 1)
    sq_dists = pairwise_sq_dist(x)
    sq_dists = sq_dists.clone()
    sq_dists.fill_diagonal_(float("inf"))

    knn_sq_dists, knn_idx = torch.topk(sq_dists, k, largest=False, dim=1)
    knn_idx = knn_idx.detach()

    if t is None:
        t = knn_sq_dists.mean().detach().clamp(min=1e-8)
    else:
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).clamp(min=1e-8)

    weights = torch.exp(-knn_sq_dists / t)
    row_idx = torch.arange(n, device=x.device).unsqueeze(1).expand(-1, k)
    w = torch.zeros(n, n, device=x.device, dtype=x.dtype)
    w[row_idx, knn_idx] = weights
    return torch.maximum(w, w.t())


def laplacian_from_affinity(
    w: torch.Tensor,
    laplacian_type: LaplacianType = "unnormalized",
) -> torch.Tensor:
    """Build graph Laplacian from affinity matrix W.

    unnormalized: L = D - W
    normalized:   L_sym = I - D^{-1/2} W D^{-1/2}
    """
    if laplacian_type not in LAPLACIAN_TYPES:
        raise ValueError(
            f"Unknown laplacian_type={laplacian_type!r}. "
            f"Expected one of {LAPLACIAN_TYPES}."
        )

    n = w.size(0)
    degree = w.sum(dim=1)
    if laplacian_type == "unnormalized":
        return torch.diag(degree) - w

    inv_sqrt_degree = degree.clamp(min=1e-8).pow(-0.5)
    d_inv_sqrt = torch.diag(inv_sqrt_degree)
    identity = torch.eye(n, device=w.device, dtype=w.dtype)
    return identity - d_inv_sqrt @ w @ d_inv_sqrt


def select_num_eigen_by_eigengap(
    eigenvalues: torch.Tensor,
    k_min: int,
    k_max: int,
    skip_trivial: bool = True,
) -> int:
    """Pick the number of eigenvectors from the largest eigengap, clamped to [k_min, k_max].

    Eigenvalues are in ascending order: λ_0 <= λ_1 <= ... <= λ_{n-1}.
    Eigengap at index j is λ_{j+1} - λ_j. If gap at j is largest, use eigenvectors
    1..j (j non-trivial directions when skip_trivial=True).
    """
    n = eigenvalues.numel()
    start = 1 if skip_trivial else 0
    max_available = n - start
    if max_available <= 0:
        return 0

    k_min = max(int(k_min), 1)
    k_max = min(int(k_max), max_available)
    if k_min > k_max:
        return k_max

    gaps = eigenvalues[1:] - eigenvalues[:-1]
    search_lo = max(start, k_min)
    search_hi = min(k_max, n - 1)
    if search_lo > search_hi:
        return k_min

    best_local = torch.argmax(gaps[search_lo : search_hi + 1]).item()
    return search_lo + best_local


def compute_eigenspace_projection_with_k(
    x: torch.Tensor,
    knn_k: int = 5,
    k_min: int = 2,
    k_max: int = 16,
    t: Optional[float] = None,
    skip_trivial: bool = True,
    num_eigen: Optional[int] = None,
    laplacian_type: LaplacianType = "unnormalized",
) -> tuple[torch.Tensor, int]:
    """Like compute_eigenspace_projection but also returns the selected eigenvector count."""
    n = x.size(0)
    if n < 2:
        return torch.zeros(n, n, device=x.device, dtype=x.dtype), 0

    w = build_knn_heat_affinity(x, k=knn_k, t=t)
    laplacian = laplacian_from_affinity(w, laplacian_type=laplacian_type)

    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian.float())
    eigenvectors = eigenvectors.to(dtype=x.dtype)

    start = 1 if skip_trivial else 0
    max_available = n - start
    if max_available <= 0:
        return torch.zeros(n, n, device=x.device, dtype=x.dtype), 0

    if num_eigen is None:
        num_eigen = select_num_eigen_by_eigengap(
            eigenvalues, k_min=k_min, k_max=k_max, skip_trivial=skip_trivial
        )
    else:
        num_eigen = min(int(num_eigen), max_available)

    if num_eigen <= 0:
        return torch.zeros(n, n, device=x.device, dtype=x.dtype), 0

    u = eigenvectors[:, start : start + num_eigen]
    return u @ u.t(), num_eigen


def eigenspace_frobenius_distillation_loss(
    eigenspace_teacher: torch.Tensor,
    eigenspace_student: torch.Tensor,
) -> torch.Tensor:
    """||P_teacher - P_student||_F^2."""
    return torch.linalg.matrix_norm(
        eigenspace_teacher - eigenspace_student,
        ord="fro",
    ) ** 2


def batch_graph_eigenspace_loss(
    teacher_repr: torch.Tensor,
    student_repr: torch.Tensor,
    knn_k: int = 5,
    k_min: int = 2,
    k_max: int = 16,
    t: Optional[float] = None,
    laplacian_type: LaplacianType = "unnormalized",
) -> torch.Tensor:
    """End-to-end batch eigenspace distillation on (N, D) representations."""
    with torch.no_grad():
        eigenspace_teacher, num_eigen = compute_eigenspace_projection_with_k(
            teacher_repr,
            knn_k=knn_k,
            k_min=k_min,
            k_max=k_max,
            t=t,
            laplacian_type=laplacian_type,
        )
    # if you want student to use his own number of eigenvectors, don't set num_eigen
    eigenspace_student, _ = compute_eigenspace_projection_with_k(
        student_repr,
        knn_k=knn_k,
        k_min=k_min,
        k_max=k_max,
        t=t,
        num_eigen=num_eigen,
        laplacian_type=laplacian_type,
    )
    return eigenspace_frobenius_distillation_loss(
        eigenspace_teacher, eigenspace_student
    )


class BatchGraphEigenspaceLoss(nn.Module):
    """Batch-level Laplacian-eigenmap eigenspace distillation loss."""

    def __init__(
        self,
        knn_k: int = 8,
        k_min: int = 2,
        k_max: int = 16,
        t: Optional[float] = None,
        laplacian_type: LaplacianType = "unnormalized",
    ):
        super().__init__()
        self.knn_k = knn_k
        self.k_min = k_min
        self.k_max = k_max
        self.t = t
        self.laplacian_type = laplacian_type

    def forward(
        self,
        teacher_repr: torch.Tensor,
        student_repr: torch.Tensor,
    ) -> torch.Tensor:
        return batch_graph_eigenspace_loss(
            teacher_repr,
            student_repr,
            knn_k=self.knn_k,
            k_min=self.k_min,
            k_max=self.k_max,
            t=self.t,
            laplacian_type=self.laplacian_type,
        )


# ---------------------------------------------------------------------------
# Visual cluster alignment utils (for CMRD — unified cluster representations)
# ---------------------------------------------------------------------------


@dataclass
class VisualClusterAlignment:
    """K unified cluster representations per sample."""

    num_clusters: int
    teacher_repr: torch.Tensor
    student_repr: torch.Tensor
    teacher_cluster_labels: np.ndarray
    student_cluster_mapping: Dict[int, List[int]]
    student_token_to_cluster: List[int]
    teacher_cluster_info: Dict
    student_cluster_info: Dict


def infer_vision_grid(num_tokens: int) -> Optional[Tuple[int, int]]:
    """Return (num_rows, num_cols) with num_rows * num_cols == num_tokens.

    Prefers near-square layouts. Qwen2-VL teachers often emit non-square token counts
    (e.g. 15 = 3x5), so a strict square assumption is invalid.
    """
    if num_tokens <= 0:
        return None

    side = int(round(num_tokens ** 0.5))
    if side * side == num_tokens:
        return side, side

    best: Optional[Tuple[int, int]] = None
    for rows in range(1, int(num_tokens ** 0.5) + 1):
        if num_tokens % rows != 0:
            continue
        cols = num_tokens // rows
        if best is None or abs(rows - cols) < abs(best[0] - best[1]):
            best = (rows, cols)
    return best


def get_patch_coordinates(
    patch_idx: int,
    num_patches_per_row: int,
    patch_size: int,
) -> Tuple[float, float]:
    row = patch_idx // num_patches_per_row
    col = patch_idx % num_patches_per_row
    center_x = col * patch_size + patch_size / 2.0
    center_y = row * patch_size + patch_size / 2.0
    return center_x, center_y


def compute_vision_distance_matrix(
    hidden_states: torch.Tensor,
    num_patches_per_row: int,
    patch_size: int,
    image_width: float,
    image_height: float,
    spatial_weight: float = 0.1,
) -> np.ndarray:
    num_tokens = hidden_states.size(0)
    device = hidden_states.device
    hidden_norm = F.normalize(hidden_states, p=2, dim=-1)
    cosine_distance = 1.0 - hidden_norm @ hidden_norm.T

    token_idx = torch.arange(num_tokens, device=device)
    rows = token_idx // num_patches_per_row
    cols = token_idx % num_patches_per_row
    coords = torch.stack(
        (
            cols.float() * patch_size + patch_size / 2.0,
            rows.float() * patch_size + patch_size / 2.0,
        ),
        dim=-1,
    )

    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    spatial_distance = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    max_dist = torch.sqrt(
        torch.tensor(image_width ** 2 + image_height ** 2, dtype=torch.float, device=device)
    )
    spatial_distance_norm = spatial_distance / max_dist.clamp(min=1e-8)
    total_dist = cosine_distance + spatial_weight * spatial_distance_norm
    return total_dist.detach().cpu().numpy()


def cluster_vision_tokens_dbscan(
    hidden_states: torch.Tensor,
    num_patches_per_row: int,
    patch_size: int,
    image_width: float,
    image_height: float,
    min_cluster_size: int = 3,
    min_samples_dbscan: int = 8,
    spatial_weight: float = 0.1,
    dbscan_eps_percentile: float = 3.0,
) -> np.ndarray:
    num_tokens = hidden_states.size(0)
    if num_tokens < min_cluster_size:
        return np.zeros(num_tokens, dtype=np.int32)

    distance_matrix = compute_vision_distance_matrix(
        hidden_states,
        num_patches_per_row,
        patch_size,
        image_width,
        image_height,
        spatial_weight=spatial_weight,
    )
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    distance_matrix = np.maximum(distance_matrix, 0.0)
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = distance_matrix.astype(np.float64)

    upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    eps = float(np.percentile(upper, dbscan_eps_percentile)) if upper.size else 0.0
    clusterer = DBSCAN(
        eps=eps,
        min_samples=max(1, int(min_samples_dbscan)),
        metric="precomputed",
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)
    if np.all(cluster_labels == -1):
        cluster_labels = np.zeros(num_tokens, dtype=np.int32)
    return cluster_labels


def map_teacher_clusters_to_student(
    cluster_labels: np.ndarray,
    teacher_num_patches_per_row: int,
    teacher_patch_size: int,
    student_num_patches_per_row: int,
    student_patch_size: int,
    original_width: float,
    original_height: float,
    student_resize: int = 1024,
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Map teacher clusters to student tokens (same geometry as span_propose_attn)."""
    num_teacher_tokens = len(cluster_labels)
    num_student_tokens = (student_resize // student_patch_size) ** 2

    student_cluster_mapping: Dict[int, set] = {}
    student_token_to_cluster = [-1] * num_student_tokens

    for teacher_idx in range(num_teacher_tokens):
        cluster_id = int(cluster_labels[teacher_idx])
        if cluster_id == -1:
            continue
        teacher_x, teacher_y = get_patch_coordinates(
            teacher_idx, teacher_num_patches_per_row, teacher_patch_size
        )
        scale_x = student_resize / original_width
        scale_y = student_resize / original_height
        student_x = teacher_x * scale_x
        student_y = teacher_y * scale_y

        student_col = int(student_x // student_patch_size)
        student_row = int(student_y // student_patch_size)
        student_col = min(max(student_col, 0), student_num_patches_per_row - 1)
        student_row = min(max(student_row, 0), student_num_patches_per_row - 1)
        student_idx = student_row * student_num_patches_per_row + student_col

        if cluster_id not in student_cluster_mapping:
            student_cluster_mapping[cluster_id] = set()
        student_cluster_mapping[cluster_id].add(student_idx)
        student_token_to_cluster[student_idx] = cluster_id

    mapping_list = {k: list(v) for k, v in student_cluster_mapping.items()}
    return mapping_list, student_token_to_cluster


def prepare_vision_cluster_info(
    cluster_labels: np.ndarray,
    device: torch.device,
) -> Optional[Dict]:
    cluster_labels = np.asarray(cluster_labels)
    valid_mask = cluster_labels >= 0
    if not np.any(valid_mask):
        return None

    valid_indices = np.where(valid_mask)[0]
    valid_clusters = cluster_labels[valid_mask]
    unique_clusters = np.unique(valid_clusters)
    cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}
    remapped_clusters = np.array([cluster_mapping[c] for c in valid_clusters])

    return {
        "token_indices": torch.tensor(valid_indices, dtype=torch.long, device=device),
        "cluster_ids": torch.tensor(remapped_clusters, dtype=torch.long, device=device),
        "num_clusters": len(unique_clusters),
        "cluster_mapping": cluster_mapping,
        "original_labels": cluster_labels,
    }


def compute_intra_cluster_attention_weights(
    hidden_states: torch.Tensor,
    cluster_info: Optional[Dict],
) -> Optional[torch.Tensor]:
    if cluster_info is None:
        return None

    device = hidden_states.device
    token_indices = cluster_info["token_indices"]
    cluster_ids = cluster_info["cluster_ids"]
    num_clusters = cluster_info["num_clusters"]

    h = hidden_states[token_indices]
    n, d = h.size()
    if n == 0:
        return None

    h_detached = h.detach()
    std = h_detached.std(dim=-1, keepdim=True) + 1e-6
    q = h_detached / std
    k = h_detached / std
    scores = torch.matmul(q, k.T) / (d ** 0.5)

    same_cluster_mask = cluster_ids.unsqueeze(0) == cluster_ids.unsqueeze(1)
    diag_mask = torch.eye(n, device=device, dtype=torch.bool)
    valid_mask = same_cluster_mask & (~diag_mask)
    valid_count_per_row = valid_mask.sum(dim=-1)
    is_singleton = valid_count_per_row == 0

    scores_masked = scores.masked_fill(~valid_mask, float("-inf"))
    attn_weights = F.softmax(scores_masked, dim=-1)
    attn_weights = torch.where(
        torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights
    )
    token_weights = attn_weights.sum(dim=0)
    token_weights = torch.where(is_singleton, torch.ones_like(token_weights), token_weights)

    cluster_weight_sum = torch.zeros(num_clusters, device=device, dtype=token_weights.dtype)
    cluster_weight_sum.scatter_add_(0, cluster_ids, token_weights)
    cluster_weight_sum = cluster_weight_sum.clamp(min=1e-8)
    return token_weights / cluster_weight_sum[cluster_ids]


def compute_weighted_cluster_mean(
    hidden_states: torch.Tensor,
    cluster_info: Optional[Dict],
    token_weights: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if cluster_info is None or token_weights is None:
        return None

    device = hidden_states.device
    token_indices = cluster_info["token_indices"]
    cluster_ids = cluster_info["cluster_ids"]
    num_clusters = cluster_info["num_clusters"]
    d = hidden_states.size(-1)

    h = hidden_states[token_indices]
    h_weighted = h * token_weights.detach().unsqueeze(-1)

    cluster_ids_exp = cluster_ids.unsqueeze(-1).expand(-1, d)
    cluster_sum = torch.zeros(num_clusters, d, device=device, dtype=h.dtype)
    cluster_sum.scatter_add_(0, cluster_ids_exp, h_weighted)

    weight_sum = torch.zeros(num_clusters, device=device, dtype=h.dtype)
    weight_sum.scatter_add_(0, cluster_ids, token_weights.detach())
    weight_sum = weight_sum.clamp(min=1e-6).unsqueeze(-1)
    return cluster_sum / weight_sum


def build_student_cluster_info_from_mapping(
    student_cluster_mapping: Dict[int, List[int]],
    teacher_cluster_info: Dict,
    num_student_tokens: int,
    device: torch.device,
) -> Optional[Dict]:
    cluster_mapping = teacher_cluster_info["cluster_mapping"]
    num_clusters = teacher_cluster_info["num_clusters"]

    token_indices: List[int] = []
    cluster_ids: List[int] = []
    for raw_cluster_id, student_indices in student_cluster_mapping.items():
        if raw_cluster_id not in cluster_mapping:
            continue
        new_id = cluster_mapping[raw_cluster_id]
        for s_idx in student_indices:
            if 0 <= s_idx < num_student_tokens:
                token_indices.append(s_idx)
                cluster_ids.append(new_id)

    if not token_indices:
        return None

    return {
        "token_indices": torch.tensor(token_indices, dtype=torch.long, device=device),
        "cluster_ids": torch.tensor(cluster_ids, dtype=torch.long, device=device),
        "num_clusters": num_clusters,
    }


def extract_vision_hidden_single(
    hidden_states: Tuple[torch.Tensor, ...],
    sample_idx: int,
    num_vision_tokens: int,
    num_text_tokens: int,
    is_teacher: bool,
    layer_idx: int = -1,
) -> torch.Tensor:
    layer_hidden = hidden_states[layer_idx]
    if num_vision_tokens <= 0:
        return layer_hidden.new_zeros(0, layer_hidden.size(-1))

    if is_teacher:
        start_idx = -(num_vision_tokens + num_text_tokens)
        end_idx = -num_text_tokens if num_text_tokens > 0 else None
        return layer_hidden[sample_idx, start_idx:end_idx, :]
    return layer_hidden[sample_idx, :num_vision_tokens, :]


def align_visual_clusters(
    teacher_vision_hidden: torch.Tensor,
    student_vision_hidden: torch.Tensor,
    image_width: float,
    image_height: float,
    teacher_patch_size: int = 28,
    student_patch_size: int = 64,
    student_resize: int = 1024,
    min_cluster_size: int = 3,
    min_samples_dbscan: int = 8,
) -> Optional[VisualClusterAlignment]:
    """
    DBSCAN on teacher -> map to student (span_propose_attn) -> attention-weighted means.

    Returns K cluster vectors on both sides with matching ids 0..K-1.
    """
    if teacher_vision_hidden is None or student_vision_hidden is None:
        return None
    if teacher_vision_hidden.numel() == 0 or student_vision_hidden.numel() == 0:
        return None

    device = teacher_vision_hidden.device
    n_teacher = teacher_vision_hidden.size(0)
    n_student = student_vision_hidden.size(0)

    teacher_grid = infer_vision_grid(n_teacher)
    student_grid = infer_vision_grid(n_student)
    if teacher_grid is None or student_grid is None:
        return None

    teacher_ppr = teacher_grid[1]
    student_ppr = student_grid[1]

    cluster_labels = cluster_vision_tokens_dbscan(
        teacher_vision_hidden,
        teacher_ppr,
        teacher_patch_size,
        image_width,
        image_height,
        min_cluster_size=min_cluster_size,
        min_samples_dbscan=min_samples_dbscan,
    )
    teacher_cluster_info = prepare_vision_cluster_info(cluster_labels, device)
    if teacher_cluster_info is None:
        return None

    student_cluster_mapping, student_token_to_cluster = map_teacher_clusters_to_student(
        cluster_labels,
        teacher_ppr,
        teacher_patch_size,
        student_ppr,
        student_patch_size,
        image_width,
        image_height,
        student_resize=student_resize,
    )
    student_cluster_info = build_student_cluster_info_from_mapping(
        student_cluster_mapping,
        teacher_cluster_info,
        n_student,
        device,
    )
    if student_cluster_info is None:
        return None

    teacher_weights = compute_intra_cluster_attention_weights(
        teacher_vision_hidden, teacher_cluster_info
    )
    student_weights = compute_intra_cluster_attention_weights(
        student_vision_hidden, student_cluster_info
    )
    teacher_repr = compute_weighted_cluster_mean(
        teacher_vision_hidden, teacher_cluster_info, teacher_weights
    )
    student_repr = compute_weighted_cluster_mean(
        student_vision_hidden, student_cluster_info, student_weights
    )
    if teacher_repr is None or student_repr is None:
        return None

    k = teacher_cluster_info["num_clusters"]
    return VisualClusterAlignment(
        num_clusters=k,
        teacher_repr=teacher_repr,
        student_repr=student_repr,
        teacher_cluster_labels=cluster_labels,
        student_cluster_mapping=student_cluster_mapping,
        student_token_to_cluster=student_token_to_cluster,
        teacher_cluster_info=teacher_cluster_info,
        student_cluster_info=student_cluster_info,
    )


# ---------------------------------------------------------------------------
# Text token alignment utils (student-anchored overlap for CMRD)
# ---------------------------------------------------------------------------


def _all_placeholder_strings() -> Set[str]:
    from src.model.processor import VLM_IMAGE_TOKENS, VLM_VIDEO_TOKENS

    return {tok for tok in set(VLM_IMAGE_TOKENS.values()) | set(VLM_VIDEO_TOKENS.values()) if tok}


def normalize_raw_text(text: str) -> str:
    """Unicode NFC normalization without altering whitespace."""
    return unicodedata.normalize("NFC", text)


def extract_semantic_text(prompt: str) -> str:
    """Strip image/video placeholders; preserve whitespace/newlines."""
    text = normalize_raw_text(prompt)
    for placeholder in _all_placeholder_strings():
        text = text.replace(placeholder, "")
    return text


def build_non_semantic_id_set(
    tokenizer,
    extra_ids: Optional[Iterable[int]] = None,
) -> Set[int]:
    """Token ids to exclude from semantic text alignment (special, vision, pad)."""
    from src.model.llava.constants import IMAGE_TOKEN_INDEX

    ids: Set[int] = set(range(QWEN_VISION_TOKEN_ID_MIN, QWEN_VISION_TOKEN_ID_MAX + 1))
    ids.add(IMAGE_TOKEN_INDEX)
    if extra_ids is not None:
        ids.update(int(x) for x in extra_ids)
    if hasattr(tokenizer, "all_special_ids"):
        ids.update(int(x) for x in tokenizer.all_special_ids)
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            ids.add(int(tid))
    return ids


def tokenize_semantic_with_offsets(
    tokenizer,
    semantic_text: str,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Tokenize semantic text; filter (0,0) offsets and keep ids aligned."""
    semantic_text = normalize_raw_text(semantic_text)
    enc = tokenizer(
        semantic_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    sem_ids: List[int] = []
    sem_offsets: List[Tuple[int, int]] = []
    for tid, off in zip(enc["input_ids"], enc["offset_mapping"]):
        start, end = int(off[0]), int(off[1])
        if end > start:
            sem_ids.append(int(tid))
            sem_offsets.append((start, end))
    return sem_ids, sem_offsets


def find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    if not needle or len(needle) > len(haystack):
        return None
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def get_text_token_ids_from_input(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    non_semantic_ids: Set[int],
) -> List[int]:
    return [
        int(tid)
        for tid, mask in zip(input_ids.tolist(), attention_mask.tolist())
        if mask and int(tid) not in non_semantic_ids
    ]


def extract_text_region(
    layer_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_vision_tokens: int,
    is_teacher: bool,
    has_image: bool,
    non_semantic_ids: Set[int],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Slice hidden states and input ids to the text region (same index space).

    Assumption (repo layout): teacher left-pad [pad][vision][text];
    student right-pad [vision][text][pad] when has_image.
    """
    text_ids = get_text_token_ids_from_input(input_ids, attention_mask, non_semantic_ids)
    num_text = len(text_ids)
    if num_text == 0:
        return None

    if has_image:
        if is_teacher:
            text_hidden = layer_hidden[-num_text:, :]
        else:
            end = num_vision_tokens + num_text
            if end > layer_hidden.size(0):
                return None
            text_hidden = layer_hidden[num_vision_tokens:end, :]
    else:
        if is_teacher:
            text_hidden = layer_hidden[-num_text:, :]
        else:
            text_hidden = layer_hidden[:num_text, :]

    if text_hidden.size(0) != num_text:
        return None

    text_region_input_ids = torch.tensor(
        text_ids, device=layer_hidden.device, dtype=torch.long
    )
    return text_hidden, text_region_input_ids


def align_semantic_tokens_to_hidden(
    text_region_input_ids: torch.Tensor,
    text_region_hidden: torch.Tensor,
    tokenizer,
    semantic_text: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Match standalone semantic tokenization to text-region hidden states."""
    sem_ids, sem_offsets = tokenize_semantic_with_offsets(tokenizer, semantic_text)
    if not sem_ids:
        return None

    region_ids = text_region_input_ids.tolist()
    if len(region_ids) != text_region_hidden.size(0):
        return None

    start = find_subsequence(region_ids, sem_ids)
    if start is None:
        return None

    idxs = list(range(start, start + len(sem_ids)))
    h_sem = text_region_hidden[idxs]
    offsets = torch.tensor(sem_offsets, device=text_region_hidden.device, dtype=torch.long)
    return h_sem, offsets


def build_student_anchored_overlap_matrix(
    student_offsets: torch.Tensor,
    teacher_offsets: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build row-stochastic overlap matrix A (N_S, N_T) and valid student row mask."""
    s0 = student_offsets[:, 0:1].float()
    s1 = student_offsets[:, 1:2].float()
    t0 = teacher_offsets[:, 0].unsqueeze(0).float()
    t1 = teacher_offsets[:, 1].unsqueeze(0).float()

    overlap = (torch.minimum(s1, t1) - torch.maximum(s0, t0)).clamp(min=0)
    row_sum = overlap.sum(dim=1)
    valid = row_sum > 0
    a = overlap / row_sum.clamp(min=1e-8).unsqueeze(1)
    a = a * valid.unsqueeze(1).float()
    if dtype is not None:
        a = a.to(dtype=dtype)
    return a, valid


@dataclass
class TextTokenAlignment:
    """Student-anchored text alignment for one sample."""

    num_student_tokens: int
    num_teacher_tokens: int
    overlap_matrix: torch.Tensor
    valid_student: torch.Tensor
    student_repr: torch.Tensor
    teacher_repr: torch.Tensor
    teacher_repr_on_student: torch.Tensor


def align_text_tokens(
    teacher_layer_hidden: torch.Tensor,
    student_layer_hidden: torch.Tensor,
    teacher_input_ids: torch.Tensor,
    student_input_ids: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    student_attention_mask: torch.Tensor,
    teacher_tokenizer,
    student_tokenizer,
    semantic_text: str,
    num_vision_teacher: int = 0,
    num_vision_student: int = 0,
    has_image: bool = False,
) -> Optional[TextTokenAlignment]:
    """
    Align teacher text tokens to student token space via character-span overlap.

    Returns student anchors H_S (N_S, d_S) and resampled teacher H_T_to_S (N_S, d_T).
    """
    semantic_text = normalize_raw_text(semantic_text)
    teacher_non_sem = build_non_semantic_id_set(teacher_tokenizer)
    student_non_sem = build_non_semantic_id_set(student_tokenizer)

    teacher_region = extract_text_region(
        teacher_layer_hidden,
        teacher_input_ids,
        teacher_attention_mask,
        num_vision_teacher,
        is_teacher=True,
        has_image=has_image,
        non_semantic_ids=teacher_non_sem,
    )
    student_region = extract_text_region(
        student_layer_hidden,
        student_input_ids,
        student_attention_mask,
        num_vision_student,
        is_teacher=False,
        has_image=has_image,
        non_semantic_ids=student_non_sem,
    )
    if teacher_region is None or student_region is None:
        return None

    teacher_text_hidden, teacher_text_ids = teacher_region
    student_text_hidden, student_text_ids = student_region

    teacher_aligned = align_semantic_tokens_to_hidden(
        teacher_text_ids,
        teacher_text_hidden,
        teacher_tokenizer,
        semantic_text,
    )
    student_aligned = align_semantic_tokens_to_hidden(
        student_text_ids,
        student_text_hidden,
        student_tokenizer,
        semantic_text,
    )
    if teacher_aligned is None or student_aligned is None:
        return None

    h_teacher, off_teacher = teacher_aligned
    h_student, off_student = student_aligned

    overlap_matrix, valid_student = build_student_anchored_overlap_matrix(
        off_student, off_teacher, dtype=h_teacher.dtype
    )
    if not valid_student.any():
        return None

    h_teacher = h_teacher.detach()
    teacher_repr_on_student = overlap_matrix @ h_teacher

    return TextTokenAlignment(
        num_student_tokens=h_student.size(0),
        num_teacher_tokens=h_teacher.size(0),
        overlap_matrix=overlap_matrix,
        valid_student=valid_student,
        student_repr=h_student,
        teacher_repr=h_teacher,
        teacher_repr_on_student=teacher_repr_on_student,
    )


def align_text_tokens_from_hidden_states(
    teacher_hidden_states: Tuple[torch.Tensor, ...],
    student_hidden_states: Tuple[torch.Tensor, ...],
    teacher_input_ids: torch.Tensor,
    student_input_ids: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    student_attention_mask: torch.Tensor,
    teacher_tokenizer,
    student_tokenizer,
    semantic_text: str,
    sample_idx: int,
    num_vision_teacher: int = 0,
    num_vision_student: int = 0,
    has_image: bool = False,
    layer_idx: int = -1,
) -> Optional[TextTokenAlignment]:
    """Convenience wrapper using a single layer from hidden state tuples."""
    teacher_h = teacher_hidden_states[layer_idx][sample_idx]
    student_h = student_hidden_states[layer_idx][sample_idx]
    return align_text_tokens(
        teacher_h,
        student_h,
        teacher_input_ids[sample_idx],
        student_input_ids[sample_idx],
        teacher_attention_mask[sample_idx],
        student_attention_mask[sample_idx],
        teacher_tokenizer,
        student_tokenizer,
        semantic_text,
        num_vision_teacher=num_vision_teacher,
        num_vision_student=num_vision_student,
        has_image=has_image,
    )


# ---------------------------------------------------------------------------
# CMRD loss core
# ---------------------------------------------------------------------------


def rowwise_kl(
    p_teacher: torch.Tensor,
    p_student: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    p_teacher = p_teacher.clamp_min(eps)
    p_student = p_student.clamp_min(eps)
    return (p_teacher * (p_teacher.log() - p_student.log())).sum(dim=-1)


def masked_mean(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if mask is None:
        return x.mean()
    mask_f = mask.to(dtype=x.dtype)
    return (x * mask_f).sum() / (mask_f.sum() + eps)


def weighted_masked_mean(
    x: torch.Tensor,
    weight: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if mask is not None:
        weight = weight * mask.to(dtype=weight.dtype)
    return (x * weight).sum() / (weight.sum() + eps)


def _distribution_entropy_weights(
    probs: torch.Tensor,
    num_keys: int,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
    max_entropy = math.log(num_keys) if num_keys > 1 else 1.0
    weights = (1.0 - entropy / max_entropy).clamp(min=0.0, max=1.0)
    if mask is not None:
        weights = weights * mask.to(dtype=weights.dtype)
    return weights


def _masked_conditional_probs(
    cosine: torch.Tensor,
    temperature: float,
    key_mask: Optional[torch.Tensor],
    dim: int = -1,
) -> torch.Tensor:
    logits = cosine / temperature
    if key_mask is not None:
        if dim == -1:
            logits = logits.masked_fill(~key_mask.unsqueeze(-2), MASK_FILL_VALUE)
        else:
            logits = logits.masked_fill(~key_mask.unsqueeze(-1), MASK_FILL_VALUE)
    return F.softmax(logits, dim=dim)


def cmrd_loss(
    student_visual: torch.Tensor,
    student_text: torch.Tensor,
    teacher_visual: torch.Tensor,
    teacher_text: torch.Tensor,
    temperature: float = 0.07,
    eta: float = 0.5,
    eps: float = 1e-8,
    visual_mask: Optional[torch.Tensor] = None,
    text_mask: Optional[torch.Tensor] = None,
    return_dict: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Cross-Modal Relational Distillation loss (direct KL + cycle KL)."""
    if student_visual.dim() != 3 or student_text.dim() != 3:
        raise ValueError("student_visual and student_text must be rank-3 tensors")
    if teacher_visual.shape != student_visual.shape[:2] + (teacher_visual.size(-1),):
        raise ValueError(
            f"teacher_visual shape {teacher_visual.shape} incompatible with "
            f"student_visual batch/seq {student_visual.shape[:2]}"
        )
    if teacher_text.shape != student_text.shape[:2] + (teacher_text.size(-1),):
        raise ValueError(
            f"teacher_text shape {teacher_text.shape} incompatible with "
            f"student_text batch/seq {student_text.shape[:2]}"
        )
    if student_visual.size(0) != student_text.size(0):
        raise ValueError(
            f"student_visual batch size {student_visual.size(0)} != "
            f"student_text batch size {student_text.size(0)}"
        )

    batch_size, num_visual, num_text = student_visual.shape[0], student_visual.shape[1], student_text.shape[1]
    if visual_mask is not None and visual_mask.shape != (batch_size, num_visual):
        raise ValueError(f"visual_mask shape {visual_mask.shape} != {(batch_size, num_visual)}")
    if text_mask is not None and text_mask.shape != (batch_size, num_text):
        raise ValueError(f"text_mask shape {text_mask.shape} != {(batch_size, num_text)}")

    student_visual = F.normalize(student_visual, dim=-1)
    student_text = F.normalize(student_text, dim=-1)
    teacher_visual = F.normalize(teacher_visual, dim=-1)
    teacher_text = F.normalize(teacher_text, dim=-1)

    c_s = torch.matmul(student_visual, student_text.transpose(-1, -2))
    c_t = torch.matmul(teacher_visual, teacher_text.transpose(-1, -2))

    p_s_vt = _masked_conditional_probs(c_s, temperature, text_mask, dim=-1)
    p_t_vt = _masked_conditional_probs(c_t, temperature, text_mask, dim=-1)
    c_s_tv = c_s.transpose(-1, -2)
    c_t_tv = c_t.transpose(-1, -2)
    p_s_tv = _masked_conditional_probs(c_s_tv, temperature, visual_mask, dim=-1)
    p_t_tv = _masked_conditional_probs(c_t_tv, temperature, visual_mask, dim=-1)

    p_t_vt = p_t_vt.detach()
    p_t_tv = p_t_tv.detach()

    a_v = _distribution_entropy_weights(p_t_vt, num_text, visual_mask).detach()
    a_t = _distribution_entropy_weights(p_t_tv, num_visual, text_mask).detach()

    kl_vt = rowwise_kl(p_t_vt, p_s_vt, eps=eps)
    kl_tv = rowwise_kl(p_t_tv, p_s_tv, eps=eps)
    l_vt = weighted_masked_mean(kl_vt, a_v, visual_mask, eps=eps)
    l_tv = weighted_masked_mean(kl_tv, a_t, text_mask, eps=eps)
    l_direct = l_vt + l_tv

    p_s_vtv = torch.matmul(p_s_vt, p_s_tv)
    p_t_vtv = torch.matmul(p_t_vt, p_t_tv)
    p_s_tvt = torch.matmul(p_s_tv, p_s_vt)
    p_t_tvt = torch.matmul(p_t_tv, p_t_vt)

    p_t_vtv = p_t_vtv.detach()
    p_t_tvt = p_t_tvt.detach()

    kl_vtv = rowwise_kl(p_t_vtv, p_s_vtv, eps=eps)
    kl_tvt = rowwise_kl(p_t_tvt, p_s_tvt, eps=eps)
    l_vtv = masked_mean(kl_vtv, visual_mask, eps=eps)
    l_tvt = masked_mean(kl_tvt, text_mask, eps=eps)
    l_cycle = l_vtv + l_tvt

    loss = l_direct + eta * l_cycle
    if not return_dict:
        return loss

    return {
        "loss": loss,
        "L_direct": l_direct.detach(),
        "L_cycle": l_cycle.detach(),
        "L_vt": l_vt.detach(),
        "L_tv": l_tv.detach(),
        "L_vtv": l_vtv.detach(),
        "L_tvt": l_tvt.detach(),
        "avg_entropy_weight_v": a_v.mean().detach(),
        "avg_entropy_weight_t": a_t.mean().detach(),
    }


def _count_text_tokens(input_ids: torch.Tensor) -> int:
    return int(((input_ids < QWEN_VISION_TOKEN_ID_MIN) | (input_ids > QWEN_VISION_TOKEN_ID_MAX)).sum().item())


def _infer_image_size(num_vision_tokens: int, patch_size: int) -> Tuple[float, float]:
    grid = infer_vision_grid(num_vision_tokens)
    if grid is None:
        side = int(round(num_vision_tokens ** 0.5))
        size = float(side * patch_size)
        return size, size
    rows, cols = grid
    return float(cols * patch_size), float(rows * patch_size)


def _zero_cmrd_output(ref: torch.Tensor) -> Dict[str, torch.Tensor]:
    zero = ref.sum() * 0.0
    return {
        "cmrd_loss": zero,
        "L_direct": zero,
        "L_cycle": zero,
        "L_vt": zero,
        "L_tv": zero,
        "L_vtv": zero,
        "L_tvt": zero,
        "avg_entropy_weight_v": zero,
        "avg_entropy_weight_t": zero,
    }


def _zero_batch_output(ref: torch.Tensor) -> Dict[str, torch.Tensor]:
    zero = ref.sum() * 0.0
    return {"batch_level_loss": zero}


class CMRDCriterion(nn.Module):
    """Cross-modal relational distillation on aligned visual clusters and text tokens."""

    def __init__(self, args):
        super().__init__()
        self.eta = args.cmrd_eta
        self.temperature = args.cmrd_temperature
        self.teacher_patch_size = getattr(args, "teacher_patch_size", 28)
        self.student_patch_size = getattr(args, "student_patch_size", 64)
        self.student_resize = getattr(args, "student_resize", 1024)
        self.min_samples_dbscan = getattr(args, "min_samples_dbscan_teacher", 2)

    def _align_single(
        self,
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
    ) -> Optional[Dict[str, torch.Tensor]]:
        has_image = (
            student_image_features is not None
            and sample_idx < len(student_image_features)
            and student_image_features[sample_idx] is not None
        )
        if not has_image:
            return None

        num_vision_student = student_image_features[sample_idx].size(0)
        num_vision_teacher = teacher_image_features[sample_idx].size(0)
        num_text = _count_text_tokens(teacher_input["input_ids"][sample_idx])
        if num_text <= 0:
            return None

        img_w, img_h = _infer_image_size(num_vision_teacher, self.teacher_patch_size)

        student_vision = extract_vision_hidden_single(
            student_hs, sample_idx, num_vision_student, num_text, is_teacher=False
        )
        teacher_vision = extract_vision_hidden_single(
            teacher_hs, sample_idx, num_vision_teacher, num_text, is_teacher=True
        )

        visual_align = align_visual_clusters(
            teacher_vision,
            student_vision,
            img_w,
            img_h,
            teacher_patch_size=self.teacher_patch_size,
            student_patch_size=self.student_patch_size,
            student_resize=self.student_resize,
            min_samples_dbscan=self.min_samples_dbscan,
        )
        text_align = align_text_tokens_from_hidden_states(
            teacher_hs,
            student_hs,
            teacher_input["input_ids"],
            student_input["input_ids"],
            teacher_input["attention_mask"],
            student_input["attention_mask"],
            teacher_tokenizer,
            student_tokenizer,
            semantic_text,
            sample_idx=sample_idx,
            num_vision_teacher=num_vision_teacher,
            num_vision_student=num_vision_student,
            has_image=True,
        )
        if visual_align is None or text_align is None:
            return None
        if visual_align.num_clusters <= 0 or text_align.num_student_tokens <= 0:
            return None

        num_clusters = visual_align.num_clusters
        return {
            "student_visual": visual_align.student_repr,
            "teacher_visual": visual_align.teacher_repr,
            "student_text": text_align.student_repr,
            "teacher_text": text_align.teacher_repr_on_student,
            "visual_mask": torch.ones(num_clusters, dtype=torch.bool, device=student_vision.device),
            "text_mask": text_align.valid_student,
        }

    def _pad_batch(
        self,
        samples: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, ...]:
        max_rv = max(s["student_visual"].size(0) for s in samples)
        max_rt = max(s["student_text"].size(0) for s in samples)
        device = samples[0]["student_visual"].device
        ds = samples[0]["student_visual"].size(-1)
        dt = samples[0]["teacher_visual"].size(-1)

        student_visual = torch.zeros(len(samples), max_rv, ds, device=device, dtype=samples[0]["student_visual"].dtype)
        teacher_visual = torch.zeros(len(samples), max_rv, dt, device=device, dtype=samples[0]["teacher_visual"].dtype)
        student_text = torch.zeros(len(samples), max_rt, ds, device=device, dtype=samples[0]["student_text"].dtype)
        teacher_text = torch.zeros(len(samples), max_rt, dt, device=device, dtype=samples[0]["teacher_text"].dtype)
        visual_mask = torch.zeros(len(samples), max_rv, dtype=torch.bool, device=device)
        text_mask = torch.zeros(len(samples), max_rt, dtype=torch.bool, device=device)

        for i, sample in enumerate(samples):
            rv = sample["student_visual"].size(0)
            rt = sample["student_text"].size(0)
            student_visual[i, :rv] = sample["student_visual"]
            teacher_visual[i, :rv] = sample["teacher_visual"]
            student_text[i, :rt] = sample["student_text"]
            teacher_text[i, :rt] = sample["teacher_text"]
            visual_mask[i, :rv] = sample["visual_mask"]
            text_mask[i, :rt] = sample["text_mask"]

        return student_visual, student_text, teacher_visual, teacher_text, visual_mask, text_mask

    def forward(
        self,
        *,
        student_qry_hs,
        student_pos_hs,
        teacher_qry_hs,
        teacher_pos_hs,
        student_qry_image_features,
        student_pos_image_features,
        teacher_qry_image_features,
        teacher_pos_image_features,
        student_input_qry,
        student_input_pos,
        teacher_input_qry,
        teacher_input_pos,
        teacher_tokenizer,
        student_tokenizer,
    ) -> Dict[str, torch.Tensor]:
        batch_size = student_qry_hs[-1].size(0)
        qry_texts = teacher_tokenizer.batch_decode(
            teacher_input_qry["input_ids"], skip_special_tokens=True
        )
        pos_texts = teacher_tokenizer.batch_decode(
            teacher_input_pos["input_ids"], skip_special_tokens=True
        )

        aligned_samples: List[Dict[str, torch.Tensor]] = []
        side_specs = (
            (student_qry_hs, teacher_qry_hs, student_qry_image_features, teacher_qry_image_features,
             student_input_qry, teacher_input_qry, qry_texts),
            (student_pos_hs, teacher_pos_hs, student_pos_image_features, teacher_pos_image_features,
             student_input_pos, teacher_input_pos, pos_texts),
        )
        for i in range(batch_size):
            for student_hs, teacher_hs, student_img, teacher_img, student_in, teacher_in, texts in side_specs:
                sample = self._align_single(
                    student_hs=student_hs,
                    teacher_hs=teacher_hs,
                    student_input=student_in,
                    teacher_input=teacher_in,
                    sample_idx=i,
                    student_image_features=student_img,
                    teacher_image_features=teacher_img,
                    semantic_text=extract_semantic_text(texts[i]),
                    teacher_tokenizer=teacher_tokenizer,
                    student_tokenizer=student_tokenizer,
                )
                if sample is not None:
                    aligned_samples.append(sample)

        if not aligned_samples:
            return _zero_cmrd_output(student_qry_hs[-1])

        (
            student_visual,
            student_text,
            teacher_visual,
            teacher_text,
            visual_mask,
            text_mask,
        ) = self._pad_batch(aligned_samples)

        out = cmrd_loss(
            student_visual,
            student_text,
            teacher_visual,
            teacher_text,
            temperature=self.temperature,
            eta=self.eta,
            visual_mask=visual_mask,
            text_mask=text_mask,
            return_dict=True,
        )
        return {
            "cmrd_loss": out["loss"],
            "L_direct": out["L_direct"],
            "L_cycle": out["L_cycle"],
            "L_vt": out["L_vt"],
            "L_tv": out["L_tv"],
            "L_vtv": out["L_vtv"],
            "L_tvt": out["L_tvt"],
            "avg_entropy_weight_v": out["avg_entropy_weight_v"],
            "avg_entropy_weight_t": out["avg_entropy_weight_t"],
        }


def compute_contrastive_loss(
    distiller,
    student_model,
    student_qry_reps: torch.Tensor,
    student_pos_reps: torch.Tensor,
    dist_gather_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if dist_gather_fn is not None:
        all_student_qry_reps = dist_gather_fn(student_qry_reps)
        all_student_pos_reps = dist_gather_fn(student_pos_reps)
    else:
        all_student_qry_reps = student_qry_reps
        all_student_pos_reps = student_pos_reps

    scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
    scores = scores.view(all_student_qry_reps.size(0), -1)
    target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
    target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
    return F.cross_entropy(scores / distiller.temperature, target)


class BatchGraphEigenspaceCriterion(nn.Module):
    """Batch-level Laplacian-eigenmap eigenspace distillation."""

    def __init__(self, args):
        super().__init__()
        self.batch_graph_loss_fn = BatchGraphEigenspaceLoss(
            knn_k=args.batch_graph_k,
            k_min=args.batch_graph_k_min,
            k_max=args.batch_graph_k_max,
            t=args.batch_graph_heat_t,
            laplacian_type=args.batch_graph_laplacian_type,
        )

    def forward(
        self,
        *,
        student_qry_hs,
        student_pos_hs,
        teacher_qry_hs,
        teacher_pos_hs,
        student_input_qry,
        student_input_pos,
        teacher_input_qry,
        teacher_input_pos,
        dist_gather_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        teacher_qry_repr = extract_last_token_hidden_states(
            teacher_qry_hs[-1], teacher_input_qry["attention_mask"]
        )
        teacher_pos_repr = extract_last_token_hidden_states(
            teacher_pos_hs[-1], teacher_input_pos["attention_mask"]
        )
        student_qry_repr = extract_last_token_hidden_states(
            student_qry_hs[-1], student_input_qry["attention_mask"]
        )
        student_pos_repr = extract_last_token_hidden_states(
            student_pos_hs[-1], student_input_pos["attention_mask"]
        )

        teacher_repr = torch.cat([teacher_qry_repr, teacher_pos_repr], dim=0)
        student_repr = torch.cat([student_qry_repr, student_pos_repr], dim=0)

        if dist_gather_fn is not None:
            teacher_repr = dist_gather_fn(teacher_repr)
            student_repr = dist_gather_fn(student_repr)

        batch_level_loss = self.batch_graph_loss_fn(teacher_repr, student_repr)
        return {"batch_level_loss": batch_level_loss}


class TotalLossCriterion(nn.Module):
    """Orchestrator: contrastive + batch eigenspace + CMRD distillation."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.w_loss_batch = args.w_loss_batch
        self.w_cmrd_loss = args.w_cmrd_loss
        self.batch_graph = BatchGraphEigenspaceCriterion(args)
        self.cmrd = CMRDCriterion(args)
        self._student_tokenizer = None
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.process_rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

    def _dist_gather_tensor(self, t: torch.Tensor) -> torch.Tensor:
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        torch.distributed.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)

    def _get_dist_gather_fn(self):
        if self.world_size > 1:
            return self._dist_gather_tensor
        return None

    def _get_student_tokenizer(self, distiller):
        if self._student_tokenizer is None:
            self._student_tokenizer = distiller.get_student_processor().tokenizer
        return self._student_tokenizer

    def forward(self, distiller, input_data, tokenizer=None):
        if tokenizer is None:
            raise ValueError("TotalLossCriterion requires teacher tokenizer")

        student_model = distiller.student
        teacher_model = distiller.teacher
        student_input_qry = input_data["student_inputs"]["qry"]
        student_input_pos = input_data["student_inputs"]["pos"]
        teacher_input_qry = input_data["teacher_inputs"]["qry"]
        teacher_input_pos = input_data["teacher_inputs"]["pos"]

        use_batch = self.w_loss_batch != 0
        use_cmrd = self.w_cmrd_loss != 0
        use_teacher = use_batch or use_cmrd

        if use_teacher:
            with torch.no_grad():
                teacher_model.eval()
                _, teacher_qry_image_features, _, teacher_qry_hs = teacher_model.encode_input(
                    teacher_input_qry
                )
                _, teacher_pos_image_features, _, teacher_pos_hs = teacher_model.encode_input(
                    teacher_input_pos
                )
        else:
            teacher_qry_image_features = teacher_pos_image_features = None
            teacher_qry_hs = teacher_pos_hs = None

        student_qry_reps, student_qry_image_features, _, student_qry_hs = student_model.encode_input(
            student_input_qry
        )
        student_pos_reps, student_pos_image_features, _, student_pos_hs = student_model.encode_input(
            student_input_pos
        )

        dist_gather_fn = self._get_dist_gather_fn()
        contrastive_loss = compute_contrastive_loss(
            distiller,
            student_model,
            student_qry_reps,
            student_pos_reps,
            dist_gather_fn,
        )

        if use_batch:
            batch_out = self.batch_graph(
                student_qry_hs=student_qry_hs,
                student_pos_hs=student_pos_hs,
                teacher_qry_hs=teacher_qry_hs,
                teacher_pos_hs=teacher_pos_hs,
                student_input_qry=student_input_qry,
                student_input_pos=student_input_pos,
                teacher_input_qry=teacher_input_qry,
                teacher_input_pos=teacher_input_pos,
                dist_gather_fn=dist_gather_fn,
            )
        else:
            batch_out = _zero_batch_output(student_qry_hs[-1])

        if use_cmrd:
            cmrd_out = self.cmrd(
                student_qry_hs=student_qry_hs,
                student_pos_hs=student_pos_hs,
                teacher_qry_hs=teacher_qry_hs,
                teacher_pos_hs=teacher_pos_hs,
                student_qry_image_features=student_qry_image_features,
                student_pos_image_features=student_pos_image_features,
                teacher_qry_image_features=teacher_qry_image_features,
                teacher_pos_image_features=teacher_pos_image_features,
                student_input_qry=student_input_qry,
                student_input_pos=student_input_pos,
                teacher_input_qry=teacher_input_qry,
                teacher_input_pos=teacher_input_pos,
                teacher_tokenizer=tokenizer,
                student_tokenizer=self._get_student_tokenizer(distiller),
            )
        else:
            cmrd_out = _zero_cmrd_output(student_qry_hs[-1])

        total_loss = (
            contrastive_loss
            + self.w_loss_batch * batch_out["batch_level_loss"]
            + self.w_cmrd_loss * cmrd_out["cmrd_loss"]
        )

        outputs = {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
        }
        if use_batch:
            outputs["batch_level_loss"] = batch_out["batch_level_loss"]
        if use_cmrd:
            outputs.update(cmrd_out)
        return outputs


if __name__ == "__main__":
    torch.manual_seed(0)
    b, rv, rt, ds, dt = 2, 4, 6, 32, 64
    student_visual = torch.randn(b, rv, ds, requires_grad=True)
    student_text = torch.randn(b, rt, ds, requires_grad=True)
    teacher_visual = torch.randn(b, rv, dt)
    teacher_text = torch.randn(b, rt, dt)
    visual_mask = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool
    )
    text_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool
    )

    out = cmrd_loss(
        student_visual,
        student_text,
        teacher_visual,
        teacher_text,
        visual_mask=visual_mask,
        text_mask=text_mask,
        return_dict=True,
    )
    assert out["loss"].dim() == 0
    assert not torch.isnan(out["loss"])
    out["loss"].backward()
    assert student_visual.grad is not None
    assert student_text.grad is not None
    assert student_visual.grad.abs().sum() > 0
    assert student_text.grad.abs().sum() > 0
    print("cmrd_loss smoke test passed:", {k: float(v) for k, v in out.items() if k != "loss"})
    print("total loss:", float(out["loss"]))
