from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LaplacianType = Literal["unnormalized", "normalized"]
LAPLACIAN_TYPES = ("unnormalized", "normalized")


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


def compute_eigenspace_projection(
    x: torch.Tensor,
    knn_k: int = 5,
    k_min: int = 2,
    k_max: int = 16,
    t: Optional[float] = None,
    skip_trivial: bool = True,
    num_eigen: Optional[int] = None,
    laplacian_type: LaplacianType = "unnormalized",
) -> torch.Tensor:
    """Laplacian eigenmap projection P = U U^T. Shape: (N, N).

    When num_eigen is None, the number of eigenvectors is chosen by eigengap in [k_min, k_max].
    When num_eigen is set, that fixed count is used (clamped to the feasible range).
    """
    n = x.size(0)
    if n < 2:
        return torch.zeros(n, n, device=x.device, dtype=x.dtype)

    w = build_knn_heat_affinity(x, k=knn_k, t=t)
    laplacian = laplacian_from_affinity(w, laplacian_type=laplacian_type)

    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian.float())
    eigenvectors = eigenvectors.to(dtype=x.dtype)

    start = 1 if skip_trivial else 0
    max_available = n - start
    if max_available <= 0:
        return torch.zeros(n, n, device=x.device, dtype=x.dtype)

    if num_eigen is None:
        num_eigen = select_num_eigen_by_eigengap(
            eigenvalues, k_min=k_min, k_max=k_max, skip_trivial=skip_trivial
        )
    else:
        num_eigen = min(int(num_eigen), max_available)

    if num_eigen <= 0:
        return torch.zeros(n, n, device=x.device, dtype=x.dtype)

    u = eigenvectors[:, start : start + num_eigen]
    return u @ u.t()


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


class BatchGraphDistillationCriterion(nn.Module):
    """Contrastive loss + batch-level Laplacian eigenspace distillation."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.w_loss_batch = args.w_loss_batch
        self.batch_graph_loss_fn = BatchGraphEigenspaceLoss(
            knn_k=args.batch_graph_k,
            k_min=args.batch_graph_k_min,
            k_max=args.batch_graph_k_max,
            t=args.batch_graph_heat_t,
            laplacian_type=args.batch_graph_laplacian_type,
        )
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

    def forward(self, distiller, input_data):
        student_model = distiller.student
        teacher_model = distiller.teacher

        student_input_qry = input_data["student_inputs"]["qry"]
        student_input_pos = input_data["student_inputs"]["pos"]
        teacher_input_qry = input_data["teacher_inputs"]["qry"]
        teacher_input_pos = input_data["teacher_inputs"]["pos"]

        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_reps, _, _, teacher_qry_hs = teacher_model.encode_input(
                teacher_input_qry
            )
            teacher_pos_reps, _, _, teacher_pos_hs = teacher_model.encode_input(
                teacher_input_pos
            )

        student_qry_reps, _, _, student_qry_hs = student_model.encode_input(
            student_input_qry
        )
        student_pos_reps, _, _, student_pos_hs = student_model.encode_input(
            student_input_pos
        )

        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps

        scores = student_model.compute_similarity(
            all_student_qry_reps, all_student_pos_reps
        )
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (
            all_student_qry_reps.size(0) // all_student_pos_reps.size(0)
        )
        contrastive_loss = F.cross_entropy(scores / distiller.temperature, target)

        # Batch-level eigenspace distillation
        teacher_qry_repr = extract_last_token_hidden_states(
            teacher_qry_hs[-1], teacher_input_qry["attention_mask"]
        ) # (B, D)
        teacher_pos_repr = extract_last_token_hidden_states(
            teacher_pos_hs[-1], teacher_input_pos["attention_mask"]
        ) # (B, D)
        student_qry_repr = extract_last_token_hidden_states(
            student_qry_hs[-1], student_input_qry["attention_mask"]
        ) # (B, D)
        student_pos_repr = extract_last_token_hidden_states(
            student_pos_hs[-1], student_input_pos["attention_mask"]
        ) # (B, D)

        teacher_repr = torch.cat([teacher_qry_repr, teacher_pos_repr], dim=0) # (2B, D)
        student_repr = torch.cat([student_qry_repr, student_pos_repr], dim=0) # (2B, D)

        if self.world_size > 1:
            teacher_repr = self._dist_gather_tensor(teacher_repr)
            student_repr = self._dist_gather_tensor(student_repr)

        batch_level_loss = self.batch_graph_loss_fn(teacher_repr, student_repr)

        total_loss = (
            contrastive_loss + self.w_loss_batch * batch_level_loss
        )
        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "batch_level_loss": batch_level_loss,
        }
