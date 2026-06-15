import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from sklearn.cluster import DBSCAN

from src.nan_debug import (
    log_sgd_forward_debug,
    summarize_weight_graph,
)

logger = logging.getLogger(__name__)


# ====== Vision Clustering Functions ======

def get_patch_coordinates(patch_idx, num_patch_per_row, patch_size):
    """Tinh tọa độ center của patch trên ảnh"""
    row = patch_idx // num_patch_per_row
    col = patch_idx % num_patch_per_row
    center_x = col * patch_size + patch_size / 2
    center_y = row * patch_size + patch_size / 2
    return center_x, center_y

def compute_vision_distance_matrix(hidden_states, num_pathches_per_row, patch_size, 
                                   image_width, image_height, spatial_weight=0.15):
    # tính distance matrix cho hdbscan
    num_tokens = hidden_states.size(0)
    device = hidden_states.device
    hidden_norm = F.normalize(hidden_states, p=2, dim=-1)
    sim_matrix = hidden_norm @ hidden_norm.T  # (num_tokens, num_tokens)
    cosine_distance = 1 - sim_matrix  # (num_tokens, num_tokens)
    coords = []
    for i in range(num_tokens):
        x, y = get_patch_coordinates(i, num_pathches_per_row, patch_size)
        coords.append([x,y])
    coords = torch.tensor(coords, dtype=torch.float, device=device)  # (num_tokens, 2)
    
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (num_tokens, num_tokens, 2)
    spatial_distance = torch.sqrt((diff **2).sum(dim=-1) + 1e-8)  # (num_tokens, num_tokens)
    max_dist = torch.sqrt(torch.tensor(image_width **2 + image_height **2, dtype=torch.float, device=device))
    spatial_distance_norm = spatial_distance / max_dist  # normalize to [0,1]
    
    total_dist = cosine_distance + spatial_weight * spatial_distance_norm
    return total_dist.cpu().numpy()

def cluster_vision_tokens_hdbscan(hidden_states, num_patches_per_row, patch_size, image_width, image_height,
                                  min_cluster_size=3, min_samples_dbscan=8):
    """Phân cụm vision tokens bằng HDBSCAN"""
    
    if hidden_states.size(0) < min_cluster_size:
        return np.zeros(hidden_states.size(0), dtype=np.int32)
    
    distance_matrix = compute_vision_distance_matrix(
        hidden_states, num_patches_per_row, patch_size,
        image_width, image_height, spatial_weight=0.1
    )
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    distance_matrix = np.maximum(distance_matrix, 0)
    np.fill_diagonal(distance_matrix, 0)
    
    distance_matrix = distance_matrix.astype(np.float64)
    
    # Use DBSCAN here, uncomment to switch back to HDBSCAN if needed
    D = distance_matrix.copy()
    D = D[np.triu_indices_from(D, k=1)]
    eps = np.percentile(D, 3)
    
    clusterer = DBSCAN(
        eps=eps,
        min_samples=max(1, int(min_samples_dbscan)),
        metric="precomputed"
    )
    # End of DBSCAN
    
    # clusterer = hdbscan.HDBSCAN(
    #     min_cluster_size=min_cluster_size, 
    #     metric='precomputed',
    #     allow_single_cluster=True,
    #     approx_min_span_tree=True,
    # )
    cluster_labels = clusterer.fit_predict(distance_matrix)
    if np.all(cluster_labels == -1):
        cluster_labels = np.zeros(hidden_states.size(0), dtype=np.int32)
    return cluster_labels


def map_teacher_clusters_to_student(cluster_labels, 
                                    teacher_num_patches_per_row, teacher_patch_size, 
                                    student_num_patches_per_row, student_patch_size,
                                    original_width, original_height,
                                    student_resize=1024):
    """Map cluster labels từ teacher sang student dựa trên vị trí patch"""
    num_teacher_tokens = len(cluster_labels)
    num_student_tokens = (student_resize // student_patch_size) ** 2
    
    student_cluster_mapping = {}
    student_token_to_cluster = [-1] * num_student_tokens
    for teacher_idx in range(num_teacher_tokens):
        cluster_id = int(cluster_labels[teacher_idx])
        if cluster_id == -1:
            continue
        teacher_x, teacher_y = get_patch_coordinates(
            teacher_idx, teacher_num_patches_per_row, teacher_patch_size
        )
        
        # Scale về ảnh resize của student
        scale_x = student_resize / original_width
        scale_y = student_resize / original_height
        student_x = teacher_x * scale_x
        student_y = teacher_y * scale_y
        
        student_col = int(student_x // student_patch_size)
        student_row = int(student_y // student_patch_size)
        
        # Clamp để đảm bảo trong range
        student_col = min(max(student_col, 0), student_num_patches_per_row - 1)
        student_row = min(max(student_row, 0), student_num_patches_per_row - 1)
        
        student_idx = student_row * student_num_patches_per_row + student_col
        
        if cluster_id not in student_cluster_mapping:
            student_cluster_mapping[cluster_id] = set()
        student_cluster_mapping[cluster_id].add(student_idx)
        student_token_to_cluster[student_idx] = cluster_id
        
    for cluster_id in student_cluster_mapping:
        student_cluster_mapping[cluster_id] = list(student_cluster_mapping[cluster_id])
        
    return student_cluster_mapping, student_token_to_cluster


def map_teacher_tokens_to_student(
    num_teacher_tokens,
    teacher_num_patches_per_row,
    teacher_patch_size,
    student_num_patches_per_row,
    student_patch_size,
    original_width,
    original_height,
    num_student_tokens,
    student_resize=1024,
):
    """Map each teacher vision token to a spatially corresponding student token index."""
    student_indices = []
    for teacher_idx in range(num_teacher_tokens):
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
        student_idx = min(max(student_idx, 0), num_student_tokens - 1)
        student_indices.append(student_idx)
    return student_indices


def prepare_vision_cluster_info(cluster_labels, device):
    """Chuẩn bị thông tin cluster cho vision tokens"""
    cluster_labels = np.array(cluster_labels)
    
    valid_mask = cluster_labels >= 0
    if not np.any(valid_mask):
        return None
    
    valid_indices = np.where(valid_mask)[0]
    valid_clusters = cluster_labels[valid_mask]
    
    # Reindex clusters từ 0
    
    unique_clusters = np.unique(valid_clusters)
    cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}
    remapped_clusters = np.array([cluster_mapping[c] for c in valid_clusters])
    
    return {
        'token_indices': torch.tensor(valid_indices, dtype=torch.long, device=device),
        'cluster_ids': torch.tensor(remapped_clusters, dtype=torch.long, device=device),
        'num_clusters': len(unique_clusters),
        'cluster_mapping': cluster_mapping,
        'original_labels': cluster_labels
    }

def extract_text_hidden_states(hidden_states, sample_idx, num_text_tokens, num_vision_tokens, 
                                is_teacher=False, has_image=True):
    """
    Trích xuất text hidden states từ hidden_states.
    
    Args:
        hidden_states: List of (B, SeqLen, D) hoặc single tensor
        sample_idx: index của sample trong batch
        num_text_tokens: số lượng text tokens
        num_vision_tokens: số lượng vision tokens
        is_teacher: True nếu là teacher (left padding), False nếu là student (right padding)
        has_image: True nếu sample có image
    
    Returns:
        List of (num_text_tokens, D) cho mỗi layer
    """
    text_hidden_list = []
    
    for layer_hidden in hidden_states:
        if has_image:
            if is_teacher:
                # Teacher: left padding, format: [padding] [vision] [text]
                # Text tokens ở cuối
                text_hidden = layer_hidden[sample_idx, -num_text_tokens:, :]
            else:
                # Student: right padding, format: [vision] [text] [padding]
                # Vision ở đầu, text tiếp theo
                text_hidden = layer_hidden[sample_idx, num_vision_tokens:(num_vision_tokens + num_text_tokens), :]
        else:
            if is_teacher:
                # Teacher không có image: [padding] [text]
                text_hidden = layer_hidden[sample_idx, -num_text_tokens:, :]
            else:
                # Student không có image: [text] [padding]
                text_hidden = layer_hidden[sample_idx, :num_text_tokens, :]
        
        text_hidden_list.append(text_hidden)
    
    return text_hidden_list

def extract_vision_hidden_states(hidden_states, sample_idx, num_vision_tokens, num_text_tokens, 
                                 is_teacher=False):
    """Trích xuất vision hidden states từ hidden states."""
    
    vision_hidden_list = []
    for layer_hidden in hidden_states:
        if is_teacher:
            # Teacher: left padding, format: [padding] [vision] [text]
            # Vision nằm ở vị trí: -(num_vision_tokens + num_text_tokens) đến -num_text_tokens
            start_idx = -(num_vision_tokens + num_text_tokens)
            end_idx = -num_text_tokens if num_text_tokens > 0 else None
            vision_hidden = layer_hidden[sample_idx, start_idx:end_idx, :]
        else:
            # Student: right padding, format: [vision] [text] [padding]
            # Vision ở đầu
            vision_hidden = layer_hidden[sample_idx, :num_vision_tokens, :]
        
        vision_hidden_list.append(vision_hidden)
    
    return vision_hidden_list

def extract_attention_for_sample(attention_states, sample_idx, num_vision_tokens, num_text_tokens, is_teacher=True):
    """Trích xuất attention matrix cho một sample"""
    attention_list = []
    for layer_attn in attention_states:
        if layer_attn is None:
            attention_list.append(None)
            continue
        
        if len(layer_attn.shape) == 4:
            # (B, NumHeads, SeqLen, SeqLen)
            attn = layer_attn[sample_idx].mean(dim=0)  # (SeqLen, SeqLen)
        else:
            # (B, SeqLen, SeqLen)
            attn = layer_attn[sample_idx]  # (SeqLen, SeqLen)
        
        if is_teacher:
            # Teacher: [padding] [vision] [text]
            # Text tokens: cuối cùng num_text_tokens
            # Vision tokens: từ -(num_vision + num_text) đến -num_text
            text_start = -num_text_tokens if num_text_tokens > 0 else attn.size(0)
            vision_start = -(num_vision_tokens + num_text_tokens)
            vision_end = -num_text_tokens if num_text_tokens > 0 else None
            
            # Attention từ text đến vision: attn[text_rows, vision_cols]
            if num_text_tokens > 0:
                text_to_vision_attn = attn[text_start:, vision_start:vision_end]  # (num_text, num_vision)
            else:
                text_to_vision_attn = None
        else:
            # Student: [vision] [text] [padding]
            # Vision: 0 đến num_vision
            # Text: num_vision đến num_vision + num_text
            text_start = num_vision_tokens
            text_end = num_vision_tokens + num_text_tokens
            
            text_to_vision_attn = attn[text_start:text_end, :num_vision_tokens]  # (num_text, num_vision)
        
        attention_list.append(text_to_vision_attn)
    
    return attention_list

# ========= Attention-Weighted Functions =========

def compute_intra_cluster_attention_weights(hidden_states, cluster_info):
    """Tính attention weights cho các token trong mỗi cluster dựa trên self-attention giữa các token trong cluster đó"""
    if cluster_info is None:
        return None
    
    device = hidden_states.device
    token_indices = cluster_info['token_indices']
    cluster_ids = cluster_info.get('cluster_ids', cluster_info.get('span_ids'))
    num_clusters = cluster_info.get('num_clusters', cluster_info.get('num_spans'))
    
    # Get hidden states of tokens in clusters
    H = hidden_states[token_indices]  # (N, D)
    N = H.size(0)
    D = H.size(1)
    
    if N == 0:
        return None
    
    # Normalize hidden states
    H_detached = H.detach()
    std = H_detached.std(dim=-1, keepdim=True) + 1e-6
    Q = H_detached / std
    K = H_detached / std
    
    # Calculate attention scores (N, N)
    scores = torch.matmul(Q, K.T) / (D ** 0.5)
    
    # Create mask, only keep scores within the same cluster
    # cluster_ids: (N,)
    same_cluster_mask = cluster_ids.unsqueeze(0) == cluster_ids.unsqueeze(1)  # (N, N)
    
    # Mask diagonal (do not attention to itself)
    diag_mask = torch.eye(N, device=device, dtype=torch.bool)
    
    # Tạo combined mask
    valid_mask = same_cluster_mask & (~diag_mask)
    
    # Đếm số tokens hợp lệ cho mỗi row
    valid_count_per_row = valid_mask.sum(dim=-1)  # (N,)
    
    # Xác định singleton tokens (không có token khác cùng cluster)
    is_singleton = valid_count_per_row == 0  # (N,)
    
    # Apply mask với -inf cho invalid positions
    scores_masked = scores.masked_fill(~valid_mask, float('-inf'))
    
    # Softmax để có attention weights
    # Với singleton tokens, softmax của all -inf sẽ cho NaN
    attn_weights = F.softmax(scores_masked, dim=-1)  # (N, N)
    
    # Xử lý NaN cho singleton tokens - KHÔNG dùng inplace operation
    # Thay vì attn_weights[nan_mask] = 0.0, dùng torch.where
    nan_mask = torch.isnan(attn_weights)
    attn_weights = torch.where(nan_mask, torch.zeros_like(attn_weights), attn_weights)
    
    # Token weight = tổng attention mà token nhận được từ các token khác cùng cluster
    token_weights = attn_weights.sum(dim=0)  # (N,)
    
    # Cho singleton token weight = 1
    # KHÔNG dùng inplace: token_weights[is_singleton] = 1.0
    token_weights = torch.where(is_singleton, torch.ones_like(token_weights), token_weights)
    
    # Normalize weights trong mỗi cluster để tổng = 1
    cluster_weight_sum = torch.zeros(num_clusters, device=device, dtype=token_weights.dtype)
    cluster_weight_sum.scatter_add_(0, cluster_ids, token_weights)
    cluster_weight_sum = cluster_weight_sum.clamp(min=1e-8)
    
    # Gather để lấy tổng weight của cluster tương ứng cho mỗi token
    token_cluster_sum = cluster_weight_sum[cluster_ids]  # (N,)
    
    # Normalize
    normalized_weights = token_weights / token_cluster_sum  # (N,)
    
    return normalized_weights

def compute_weighted_cluster_mean(hidden_states, cluster_info, token_weights):
    """Calculate weighted cluster means given token weights"""
    
    if cluster_info is None or token_weights is None:
        return None
    
    device = hidden_states.device
    token_indices = cluster_info['token_indices']
    cluster_ids = cluster_info.get('cluster_ids', cluster_info.get('span_ids'))
    num_clusters = cluster_info.get('num_clusters', cluster_info.get('num_spans'))
    D = hidden_states.size(-1)
    
    # Get hidden states of tokens in clusters
    H = hidden_states[token_indices]  # (N, D)
    H_detached = H.detach()
    
    weights_detached = token_weights.detach()
    
    # Apply token weights
    H_weighted = H_detached * weights_detached.unsqueeze(-1)  # (N, D)
    
    # Scatter add to sum weighted hidden states per cluster
    cluster_ids_expanded = cluster_ids.unsqueeze(-1).expand(-1, D)
    cluster_sum = torch.zeros(num_clusters, D, device=device, dtype=H.dtype)
    cluster_sum.scatter_add_(0, cluster_ids_expanded, H_weighted)
    
    # Calculate weighted for each cluster
    weight_sum = torch.zeros(num_clusters, device=device, dtype=H.dtype)
    weight_sum.scatter_add_(0, cluster_ids, token_weights)
    weight_sum = weight_sum.clamp(min=1e-6).unsqueeze(-1)
    
    cluster_mean = cluster_sum / weight_sum  # (num_clusters, D)
    return cluster_mean
    


# ====== Grassman loss helpers ======

def select_topk_text_tokens_by_last_token_cosine(text_hidden, ratio):
    """Chọn top-k text tokens theo cosine similarity với last text token (trên teacher)."""
    num_text = text_hidden.size(0)
    k = max(1, int(ratio * num_text))
    k = min(k, num_text)

    last_token = text_hidden[-1]  # (D,)
    text_norm = F.normalize(text_hidden, p=2, dim=-1)
    last_norm = F.normalize(last_token.unsqueeze(0), p=2, dim=-1)
    cos_scores = (text_norm * last_norm).sum(dim=-1)  # (num_text,)

    _, topk_indices = torch.topk(cos_scores, k)
    return topk_indices


def compute_pairwise_sq_distances(features):
    norm_sq = (features ** 2).sum(dim=-1, keepdim=True)
    dist_sq = norm_sq + norm_sq.t() - 2.0 * features @ features.t()
    return dist_sq.clamp(min=0.0)


def build_knn_weight_matrix(features, k_neighbors):
    n = features.size(0)
    if n < 2:
        return torch.zeros(n, n, device=features.device, dtype=features.dtype)

    k_neighbors = min(k_neighbors, n - 1)
    dist_sq = compute_pairwise_sq_distances(features)

    nonzero_dists = dist_sq[dist_sq > 0]
    sigma = nonzero_dists.median().item() if nonzero_dists.numel() > 0 else 1.0
    if sigma < 1e-8:
        sigma = 1.0

    W = torch.zeros_like(dist_sq)
    for i in range(n):
        _, knn_idx = torch.topk(dist_sq[i], k_neighbors + 1, largest=False)
        knn_idx = knn_idx[1:]
        W[i, knn_idx] = torch.exp(-dist_sq[i, knn_idx] / sigma)

    W = (W + W.t()) / 2.0
    diag_idx = torch.arange(n, device=features.device)
    W[diag_idx, diag_idx] = 0.0
    return W


def build_bipartite_weight_matrix(features_v, features_t):
    n_v, n_t = features_v.size(0), features_t.size(0)
    n_total = n_v + n_t
    device, dtype = features_v.device, features_v.dtype

    if n_v == 0 or n_t == 0:
        return torch.zeros(n_total, n_total, device=device, dtype=dtype)

    cross_dist_sq = compute_pairwise_sq_distances(torch.cat([features_v, features_t], dim=0))
    cross_dist = cross_dist_sq[:n_v, n_v:]

    nonzero_dists = cross_dist[cross_dist > 0]
    sigma = nonzero_dists.median().item() if nonzero_dists.numel() > 0 else 1.0
    if sigma < 1e-8:
        sigma = 1.0

    cross_weights = torch.exp(-cross_dist / sigma)
    W = torch.zeros(n_total, n_total, device=device, dtype=dtype)
    W[:n_v, n_v:] = cross_weights
    W[n_v:, :n_v] = cross_weights.t()
    return W


def compute_laplacian_eigenspace(W, num_eigenvectors, laplacian_type="unnormalized"):
    n = W.size(0)
    if n < 3:
        return torch.eye(n, device=W.device, dtype=W.dtype)

    k_eig = min(num_eigenvectors, n - 1)
    if k_eig < 1:
        return torch.eye(n, device=W.device, dtype=W.dtype)

    D = W.sum(dim=1)
    if laplacian_type == "normalized":
        D_inv_sqrt = (D + 1e-10).rsqrt().clamp(max=1e8)
        L = torch.diag(D) - W
        D_inv_sqrt_mat = torch.diag(D_inv_sqrt)
        L = D_inv_sqrt_mat @ L @ D_inv_sqrt_mat
    else:
        L = torch.diag(D) - W

    try:
        _, eigenvectors = torch.linalg.eigh(L.to(torch.float64))
    except Exception:
        return torch.eye(n, device=W.device, dtype=W.dtype)

    U = eigenvectors[:, 1:1 + k_eig].to(W.dtype) # (n, k_eig), eigenmap
    return U @ U.T # (n, n), eigenspace


def compute_grassman_loss(espace_teacher, espace_student):
    if espace_teacher is None or espace_student is None:
        device = (
            espace_teacher.device if espace_teacher is not None
            else espace_student.device if espace_student is not None
            else torch.device('cpu')
        )
        return torch.tensor(0.0, device=device)
    if espace_teacher.shape != espace_student.shape:
        return torch.tensor(0.0, device=espace_teacher.device)
    return ((espace_teacher.detach() - espace_student) ** 2).sum()



class CKALoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, SH, TH):
        # SH: student hidden states, TH: teacher hidden states
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(torch.float64)
        TH = TH.view(-1, dT).to(torch.float64)

        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)

        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps

        return 1 - num / torch.sqrt(den1 * den2)

class SGDLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

        self.args = args
        self.teacher_patch_size = getattr(args, 'teacher_patch_size', 28)
        self.student_patch_size = getattr(args, 'student_patch_size', 64)
        self.student_resize = getattr(args, 'student_resize', 1024)
        self.grassman_vision_use_cluster = getattr(args, 'grassman_vision_use_cluster', True)
        self.grassman_text_use_topk = getattr(args, 'grassman_text_use_topk', True)
        self.topk_text_ratio = getattr(args, 'topk_text_ratio', 0.3)
        self.knn_neighbors = getattr(args, 'knn_neighbors', 10)
        self.num_eigenvectors = getattr(args, 'num_eigenvectors', 16)
        self.laplacian_type = getattr(args, 'laplacian_type', 'unnormalized')
        self.kd_weight = getattr(args, 'kd_weight', 1.0)
        self.w_loss_v = getattr(args, 'w_loss_v', 1.0)
        self.w_loss_t = getattr(args, 'w_loss_t', 1.0)
        self.w_loss_cross = getattr(args, 'w_loss_cross', 1.0)
        self.w_loss_batch = getattr(args, 'w_loss_batch', 1.0)

    def _dist_gather_tensor(self, t):
        """Gather tensor từ tất cả các process"""
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def _compute_sample_grassman_loss(self, s_text_hidden, t_text_hidden,
                                      s_vision_hidden, t_vision_hidden,
                                      num_text, has_image, original_width, original_height,
                                      batch_idx=0, side="qry"):
        """Tính loss Grassman cho một sample trong batch"""
        device = (
            s_text_hidden.device if s_text_hidden is not None
            else s_vision_hidden.device if s_vision_hidden is not None
            else t_text_hidden.device if t_text_hidden is not None
            else t_vision_hidden.device if t_vision_hidden is not None
            else torch.device('cpu')
        )
        loss_v = torch.tensor(0.0, device=device) # loss cho vision tokens
        loss_t = torch.tensor(0.0, device=device) # loss cho text tokens
        loss_cross = torch.tensor(0.0, device=device) # loss cho cross-modal tokens
        valid_v = valid_t = valid_cross = 0
        h_t_v = h_s_v = h_t_t = h_s_t = None # hidden states cho vision tokens, text tokens của teacher và student

        debug = {
            "batch_idx": batch_idx,
            "side": side,
            "has_image": bool(has_image),
            "num_text": int(num_text),
            "vision": {},
            "text": {},
            "cross": {},
            "losses": {"v": 0.0, "t": 0.0, "cross": 0.0},
        }
        vision_dbg = debug["vision"]
        text_dbg = debug["text"]
        cross_dbg = debug["cross"]

        # ===== Vision tokens =====
        if has_image and t_vision_hidden is not None and s_vision_hidden is not None:
            num_teacher_tokens = t_vision_hidden.size(0)
            num_student_tokens = s_vision_hidden.size(0)
            vision_dbg["teacher_tokens"] = int(num_teacher_tokens)
            vision_dbg["student_tokens"] = int(num_student_tokens)
            vision_dbg["use_cluster"] = bool(self.grassman_vision_use_cluster)

            if num_teacher_tokens >= 2:
                if self.grassman_vision_use_cluster:
                    teacher_patches_per_row = int(np.sqrt(num_teacher_tokens))
                    cluster_labels = cluster_vision_tokens_hdbscan(
                        t_vision_hidden,
                        teacher_patches_per_row, self.teacher_patch_size,
                        original_width, original_height,
                        min_cluster_size=6,
                        min_samples_dbscan=max(1, int(getattr(self.args, 'min_samples_dbscan_teacher', 8))),
                    )
                    labels_np = np.array(cluster_labels)
                    unique_labels = [int(c) for c in np.unique(labels_np) if c >= 0]
                    vision_dbg["dbscan_clusters"] = len(unique_labels)
                    vision_dbg["noise_tokens"] = int((labels_np < 0).sum())
                    vision_dbg["cluster_sizes"] = [
                        int((labels_np == c).sum()) for c in unique_labels
                    ]

                    cluster_info = prepare_vision_cluster_info(cluster_labels, device)
                    vision_dbg["valid_clusters"] = (
                        int(cluster_info["num_clusters"]) if cluster_info is not None else 0
                    )

                    if cluster_info is None:
                        vision_dbg["skip_reason"] = "no_valid_cluster_labels"
                    elif cluster_info['num_clusters'] < 2:
                        vision_dbg["skip_reason"] = "valid_clusters_lt_2"

                    if cluster_info is not None and cluster_info['num_clusters'] >= 2:
                        t_weights = compute_intra_cluster_attention_weights(t_vision_hidden, cluster_info)
                        if t_weights is not None:
                            h_t_v = compute_weighted_cluster_mean(t_vision_hidden, cluster_info, t_weights)
                        vision_dbg["teacher_graph_nodes"] = (
                            int(h_t_v.size(0)) if h_t_v is not None else None
                        )

                        student_mapping, _ = map_teacher_clusters_to_student(
                            cluster_labels,
                            teacher_patches_per_row, self.teacher_patch_size,
                            int(np.sqrt(num_student_tokens)) if num_student_tokens > 0 else 0,
                            self.student_patch_size,
                            original_width, original_height,
                            self.student_resize,
                        )

                        mapped_tokens = 0
                        if student_mapping:
                            s_token_indices_list, s_cluster_ids_list = [], []
                            for cluster_id, student_indices in student_mapping.items():
                                for s_idx in student_indices:
                                    if s_idx < num_student_tokens:
                                        s_token_indices_list.append(s_idx)
                                        s_cluster_ids_list.append(cluster_id)
                                        mapped_tokens += 1
                            vision_dbg["mapped_student_tokens"] = mapped_tokens

                            if s_token_indices_list:
                                student_cluster_info = {
                                    'token_indices': torch.tensor(s_token_indices_list, dtype=torch.long, device=device),
                                    'cluster_ids': torch.tensor(s_cluster_ids_list, dtype=torch.long, device=device),
                                    'num_clusters': cluster_info['num_clusters'],
                                }
                                s_weights = compute_intra_cluster_attention_weights(s_vision_hidden, student_cluster_info)
                                if s_weights is not None:
                                    h_s_v = compute_weighted_cluster_mean(s_vision_hidden, student_cluster_info, s_weights)
                            else:
                                vision_dbg["skip_reason"] = "no_mapped_student_tokens"
                        else:
                            vision_dbg["skip_reason"] = "empty_student_cluster_mapping"

                        vision_dbg["student_graph_nodes"] = (
                            int(h_s_v.size(0)) if h_s_v is not None else None
                        )
                else:
                    teacher_patches_per_row = int(np.sqrt(num_teacher_tokens))
                    student_patches_per_row = int(np.sqrt(num_student_tokens)) if num_student_tokens > 0 else 0
                    if student_patches_per_row <= 0:
                        vision_dbg["skip_reason"] = "zero_student_vision_tokens"
                    else:
                        student_indices = map_teacher_tokens_to_student(
                            num_teacher_tokens,
                            teacher_patches_per_row,
                            self.teacher_patch_size,
                            student_patches_per_row,
                            self.student_patch_size,
                            original_width,
                            original_height,
                            num_student_tokens,
                            self.student_resize,
                        )
                        student_idx_tensor = torch.tensor(student_indices, dtype=torch.long, device=device)
                        h_t_v = t_vision_hidden
                        h_s_v = s_vision_hidden[student_idx_tensor]
                        vision_dbg["graph_nodes"] = int(h_t_v.size(0))
                        vision_dbg["teacher_graph_nodes"] = int(h_t_v.size(0))
                        vision_dbg["student_graph_nodes"] = int(h_s_v.size(0))

                if (h_t_v is not None and h_s_v is not None
                        and h_t_v.size(0) == h_s_v.size(0) and h_t_v.size(0) >= 2):
                    W_t = build_knn_weight_matrix(h_t_v, self.knn_neighbors)
                    W_s = build_knn_weight_matrix(h_s_v, self.knn_neighbors)
                    vision_dbg["graph_teacher"] = summarize_weight_graph(
                        W_t,
                        knn_neighbors=self.knn_neighbors,
                        num_eigenvectors=self.num_eigenvectors,
                        laplacian_type=self.laplacian_type,
                    )
                    vision_dbg["graph_student"] = summarize_weight_graph(
                        W_s,
                        knn_neighbors=self.knn_neighbors,
                        num_eigenvectors=self.num_eigenvectors,
                        laplacian_type=self.laplacian_type,
                    )
                    espace_t = compute_laplacian_eigenspace(W_t, self.num_eigenvectors, self.laplacian_type)
                    espace_s = compute_laplacian_eigenspace(W_s, self.num_eigenvectors, self.laplacian_type)
                    loss_v = compute_grassman_loss(espace_t, espace_s)
                    valid_v = 1
                    vision_dbg["vision_loss_valid"] = True
                elif h_t_v is None or h_s_v is None:
                    if "skip_reason" not in vision_dbg:
                        vision_dbg["skip_reason"] = "missing_vision_representations"
                elif h_t_v.size(0) != h_s_v.size(0):
                    vision_dbg["skip_reason"] = "teacher_student_node_count_mismatch"
                elif h_t_v.size(0) < 2:
                    vision_dbg["skip_reason"] = "vision_nodes_lt_2"
            else:
                vision_dbg["skip_reason"] = "teacher_vision_tokens_lt_2"
        elif has_image:
            vision_dbg["skip_reason"] = "missing_vision_hidden_states"

        vision_dbg["vision_loss_valid"] = bool(valid_v)

        # ===== Text tokens =====
        if num_text > 0 and t_text_hidden is not None and s_text_hidden is not None:
            text_dbg["use_topk"] = bool(self.grassman_text_use_topk)
            if self.grassman_text_use_topk:
                topk_indices = select_topk_text_tokens_by_last_token_cosine(t_text_hidden, self.topk_text_ratio)
                h_t_t = t_text_hidden[topk_indices]
                h_s_t = s_text_hidden[topk_indices]
                text_dbg["topk_tokens"] = int(h_t_t.size(0))
            else:
                h_t_t = t_text_hidden
                h_s_t = s_text_hidden
                text_dbg["num_tokens"] = int(h_t_t.size(0))

            if h_t_t.size(0) >= 2:
                W_t = build_knn_weight_matrix(h_t_t, self.knn_neighbors)
                W_s = build_knn_weight_matrix(h_s_t, self.knn_neighbors)
                text_dbg["graph_teacher"] = summarize_weight_graph(
                    W_t,
                    knn_neighbors=self.knn_neighbors,
                    num_eigenvectors=self.num_eigenvectors,
                    laplacian_type=self.laplacian_type,
                )
                text_dbg["graph_student"] = summarize_weight_graph(
                    W_s,
                    knn_neighbors=self.knn_neighbors,
                    num_eigenvectors=self.num_eigenvectors,
                    laplacian_type=self.laplacian_type,
                )
                espace_t = compute_laplacian_eigenspace(W_t, self.num_eigenvectors, self.laplacian_type)
                espace_s = compute_laplacian_eigenspace(W_s, self.num_eigenvectors, self.laplacian_type)
                loss_t = compute_grassman_loss(espace_t, espace_s)
                valid_t = 1
                text_dbg["text_loss_valid"] = True
            else:
                text_dbg["skip_reason"] = "text_tokens_lt_2"
        elif num_text > 0:
            text_dbg["skip_reason"] = "missing_text_hidden_states"
        text_dbg["text_loss_valid"] = bool(valid_t)

        # ===== Cross-modal tokens =====
        cross_dbg["vision_nodes"] = int(h_t_v.size(0)) if h_t_v is not None else None
        cross_dbg["text_nodes"] = int(h_t_t.size(0)) if h_t_t is not None else None
        if (valid_v and valid_t and h_t_v is not None and h_s_v is not None
                and h_t_t is not None and h_s_t is not None
                and h_t_v.size(0) == h_s_v.size(0)
                and h_t_t.size(0) == h_s_t.size(0)):
            n_total = h_t_v.size(0) + h_t_t.size(0)
            cross_dbg["total_nodes"] = int(n_total)
            if n_total >= 3:
                W_t_cross = build_bipartite_weight_matrix(h_t_v, h_t_t)
                W_s_cross = build_bipartite_weight_matrix(h_s_v, h_s_t)
                cross_dbg["graph_teacher"] = summarize_weight_graph(
                    W_t_cross,
                    knn_neighbors=self.knn_neighbors,
                    num_eigenvectors=self.num_eigenvectors,
                    laplacian_type=self.laplacian_type,
                )
                cross_dbg["graph_student"] = summarize_weight_graph(
                    W_s_cross,
                    knn_neighbors=self.knn_neighbors,
                    num_eigenvectors=self.num_eigenvectors,
                    laplacian_type=self.laplacian_type,
                )
                espace_t = compute_laplacian_eigenspace(W_t_cross, self.num_eigenvectors, self.laplacian_type)
                espace_s = compute_laplacian_eigenspace(W_s_cross, self.num_eigenvectors, self.laplacian_type)
                loss_cross = compute_grassman_loss(espace_t, espace_s)
                valid_cross = 1
                cross_dbg["cross_loss_valid"] = True
            else:
                cross_dbg["skip_reason"] = "total_nodes_lt_3"
        else:
            if not valid_v or not valid_t:
                cross_dbg["skip_reason"] = "vision_or_text_loss_invalid"
            elif h_t_v is None or h_t_t is None:
                cross_dbg["skip_reason"] = "missing_modal_representations"
            else:
                cross_dbg["skip_reason"] = "vision_text_node_count_mismatch"
        cross_dbg["cross_loss_valid"] = bool(valid_cross)

        debug["losses"] = {
            "v": float(loss_v.detach().item()) if torch.isfinite(loss_v) else float("nan"),
            "t": float(loss_t.detach().item()) if torch.isfinite(loss_t) else float("nan"),
            "cross": float(loss_cross.detach().item()) if torch.isfinite(loss_cross) else float("nan"),
        }

        return loss_v, loss_t, loss_cross, valid_v, valid_t, valid_cross, debug


    @staticmethod
    def _num_vision_tokens(image_features, batch_idx):
        if image_features is None or batch_idx >= len(image_features):
            return 0
        feats = image_features[batch_idx]
        return feats.size(0) if feats is not None else 0

    @staticmethod
    def _build_text_token_mask(seq_len, num_text, num_vision, is_teacher, device):
        """Build a mask aligned with hidden/attention sequence length (post vision merge)."""
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        num_text = int(num_text)
        num_vision = int(num_vision)
        if num_text <= 0:
            return mask
        if is_teacher:
            text_start = max(0, seq_len - num_text)
        else:
            text_start = min(num_vision, seq_len)
        text_end = min(text_start + num_text, seq_len)
        mask[text_start:text_end] = True
        return mask

    def _compute_batch_level_loss(self, input_data, 
                                 teacher_qry_attention, teacher_pos_attention,
                                 student_qry_attention, student_pos_attention,
                                 teacher_qry_hidden_states, teacher_pos_hidden_states,
                                 student_qry_hidden_states, student_pos_hidden_states,
                                 num_text_qry_tokens, num_text_pos_tokens,
                                 student_qry_image_features, student_pos_image_features,
                                 teacher_qry_image_features, teacher_pos_image_features):
        # tính batch-level loss trên 1 batch
        device = input_data['student_inputs']['qry']['input_ids'].device
        batch_size = input_data['student_inputs']['qry']['input_ids'].size(0)
        cka_fn_loss = CKALoss(eps=1e-8).to(device)

        t_qry_atten = teacher_qry_attention[-1].mean(dim=1)
        t_pos_atten = teacher_pos_attention[-1].mean(dim=1)
        s_qry_atten = student_qry_attention[-1].mean(dim=1)
        s_pos_atten = student_pos_attention[-1].mean(dim=1)

        t_qry_importance = t_qry_atten.sum(dim=1)
        t_pos_importance = t_pos_atten.sum(dim=1)
        s_qry_importance = s_qry_atten.sum(dim=1)
        s_pos_importance = s_pos_atten.sum(dim=1)

        t_qry_hidden = teacher_qry_hidden_states[-1]
        t_pos_hidden = teacher_pos_hidden_states[-1]
        s_qry_hidden = student_qry_hidden_states[-1]
        s_pos_hidden = student_pos_hidden_states[-1]

        t_qry_reps, t_pos_reps = [], []
        s_qry_reps, s_pos_reps = [], []

        for i in range(batch_size):
            t_qry_mask = self._build_text_token_mask(
                t_qry_hidden[i].size(0),
                num_text_qry_tokens[i].item(),
                self._num_vision_tokens(teacher_qry_image_features, i),
                is_teacher=True,
                device=device,
            )
            t_pos_mask = self._build_text_token_mask(
                t_pos_hidden[i].size(0),
                num_text_pos_tokens[i].item(),
                self._num_vision_tokens(teacher_pos_image_features, i),
                is_teacher=True,
                device=device,
            )
            s_qry_mask = self._build_text_token_mask(
                s_qry_hidden[i].size(0),
                num_text_qry_tokens[i].item(),
                self._num_vision_tokens(student_qry_image_features, i),
                is_teacher=False,
                device=device,
            )
            s_pos_mask = self._build_text_token_mask(
                s_pos_hidden[i].size(0),
                num_text_pos_tokens[i].item(),
                self._num_vision_tokens(student_pos_image_features, i),
                is_teacher=False,
                device=device,
            )

            t_qry_w = t_qry_importance[i] * t_qry_mask.float()
            t_pos_w = t_pos_importance[i] * t_pos_mask.float()
            s_qry_w = s_qry_importance[i] * s_qry_mask.float()
            s_pos_w = s_pos_importance[i] * s_pos_mask.float()

            if t_qry_w.sum() > 0: t_qry_w = t_qry_w / t_qry_w.sum()
            if t_pos_w.sum() > 0: t_pos_w = t_pos_w / t_pos_w.sum()
            if s_qry_w.sum() > 0: s_qry_w = s_qry_w / s_qry_w.sum()
            if s_pos_w.sum() > 0: s_pos_w = s_pos_w / s_pos_w.sum()

            t_qry_rep = (t_qry_hidden[i] * t_qry_w.unsqueeze(-1)).sum(dim=0)
            t_pos_rep = (t_pos_hidden[i] * t_pos_w.unsqueeze(-1)).sum(dim=0)
            s_qry_rep = (s_qry_hidden[i] * s_qry_w.unsqueeze(-1)).sum(dim=0)
            s_pos_rep = (s_pos_hidden[i] * s_pos_w.unsqueeze(-1)).sum(dim=0)

            t_qry_reps.append(t_qry_rep)
            t_pos_reps.append(t_pos_rep)
            s_qry_reps.append(s_qry_rep)
            s_pos_reps.append(s_pos_rep)

        t_qry_reps = torch.stack(t_qry_reps)
        t_pos_reps = torch.stack(t_pos_reps)
        s_qry_reps = torch.stack(s_qry_reps)
        s_pos_reps = torch.stack(s_pos_reps)

        loss_qry = cka_fn_loss(s_qry_reps, t_qry_reps)
        loss_pos = cka_fn_loss(s_pos_reps, t_pos_reps)

        return (loss_qry + loss_pos) / 2

    def _compute_token_level_loss(self, batch_size, device,
                                 num_text_qry_tokens, num_text_pos_tokens,
                                 student_qry_image_features, teacher_qry_image_features,
                                 student_pos_image_features, teacher_pos_image_features,
                                 student_qry_hidden_states, teacher_qry_hidden_states,
                                 student_pos_hidden_states, teacher_pos_hidden_states,
                                 qry_image_sizes, pos_image_sizes):
        # tính token-level loss trên 1 batch
        total_loss_v = total_loss_t = total_loss_cross = 0.0
        valid_vision_samples = valid_text_samples = valid_cross_modal_samples = 0
        grassman_debug = []

        for i in range(batch_size):
            # for sample in [query, positive]
            for side, side_args in (
                ("qry", (num_text_qry_tokens[i].item(), student_qry_image_features, teacher_qry_image_features,
                 student_qry_hidden_states, teacher_qry_hidden_states, qry_image_sizes)),
                ("pos", (num_text_pos_tokens[i].item(), student_pos_image_features, teacher_pos_image_features,
                 student_pos_hidden_states, teacher_pos_hidden_states, pos_image_sizes)),
            ):
                num_text, s_img_feats, t_img_feats, s_hidden, t_hidden, image_sizes = side_args

                has_image = (s_img_feats is not None and i < len(s_img_feats) and s_img_feats[i] is not None)
                num_vision_student = s_img_feats[i].size(0) if has_image else 0
                num_vision_teacher = t_img_feats[i].size(0) if has_image else 0
                img_w = img_h = 0
                if has_image:
                    if image_sizes is not None and i < len(image_sizes):
                        img_w, img_h = image_sizes[i]
                    else:
                        patches_per_row = int(np.sqrt(num_vision_teacher))
                        img_w = img_h = patches_per_row * self.teacher_patch_size

                s_text_last = extract_text_hidden_states(
                    s_hidden, i, num_text, num_vision_student, is_teacher=False, has_image=has_image,
                )[-1] if num_text > 0 else None
                t_text_last = extract_text_hidden_states(
                    t_hidden, i, num_text, num_vision_teacher, is_teacher=True, has_image=has_image,
                )[-1] if num_text > 0 else None

                s_vision_last = t_vision_last = None
                if has_image:
                    s_vision_last = extract_vision_hidden_states(
                        s_hidden, i, num_vision_student, num_text, is_teacher=False,
                    )[-1]
                    t_vision_last = extract_vision_hidden_states(
                        t_hidden, i, num_vision_teacher, num_text, is_teacher=True,
                    )[-1]

                lv, lt, lc, vv, vt, vc, sample_debug = self._compute_sample_grassman_loss(
                    s_text_last, t_text_last, s_vision_last, t_vision_last,
                    num_text, has_image, img_w, img_h,
                    batch_idx=i,
                    side=side,
                )
                grassman_debug.append(sample_debug)
                total_loss_v += lv
                total_loss_t += lt
                total_loss_cross += lc
                valid_vision_samples += vv
                valid_text_samples += vt
                valid_cross_modal_samples += vc

        token_level_loss_v = total_loss_v / valid_vision_samples if valid_vision_samples > 0 else torch.tensor(0.0, device=device)
        token_level_loss_t = total_loss_t / valid_text_samples if valid_text_samples > 0 else torch.tensor(0.0, device=device)
        token_level_loss_cross = total_loss_cross / valid_cross_modal_samples if valid_cross_modal_samples > 0 else torch.tensor(0.0, device=device)
        # token_level_loss = lambda_v * loss_v + lambda_t * loss_t + lambda_cross * loss_cross
        token_level_loss = (
            self.w_loss_v * token_level_loss_v
            + self.w_loss_t * token_level_loss_t
            + self.w_loss_cross * token_level_loss_cross
        )
        return (
            token_level_loss,
            token_level_loss_v,
            token_level_loss_t,
            token_level_loss_cross,
            grassman_debug,
        )

    def forward(self, distiller, input_data):
        student_model = distiller.student
        teacher_model = distiller.teacher

        student_qry_input = input_data['student_inputs']['qry']
        student_pos_input = input_data['student_inputs']['pos']
        teacher_qry_input = input_data['teacher_inputs']['qry']
        teacher_pos_input = input_data['teacher_inputs']['pos']
        qry_image_sizes = input_data.get('qry_image_sizes', None)
        pos_image_sizes = input_data.get('pos_image_sizes', None)

        # Đếm số text tokens (loại bỏ image tokens)
        # Giả sử image token IDs nằm trong khoảng [151643, 151656]
        num_text_qry_tokens = ((teacher_qry_input['input_ids'] < 151643) | (teacher_qry_input['input_ids'] > 151656)).sum(dim=1)
        num_text_pos_tokens = ((teacher_pos_input['input_ids'] < 151643) | (teacher_pos_input['input_ids'] > 151656)).sum(dim=1)

        batch_size = student_qry_input['input_ids'].size(0)
        device = student_qry_input['input_ids'].device

        # Forward teacher
        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            teacher_qry_reps, teacher_qry_image_features, teacher_qry_attention, teacher_qry_hidden_states = teacher_qry_output
            teacher_pos_reps, teacher_pos_image_features, teacher_pos_attention, teacher_pos_hidden_states = teacher_pos_output

        # Forward student
        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        # Contrastive loss
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

        # RKD loss (distance loss + angle loss)
        rkd_distance_loss = self.compute_distance_loss(
            student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps
        )
        rkd_angle_loss = self.compute_angle_loss(
            student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps
        )
        rkd_loss = (rkd_distance_loss + rkd_angle_loss) / 2.0

        # Token-level loss
        token_level_loss, token_level_loss_v, token_level_loss_t, token_level_loss_cross, grassman_debug = (
            self._compute_token_level_loss(
            batch_size, device,
            num_text_qry_tokens, num_text_pos_tokens,
            student_qry_image_features, teacher_qry_image_features,
            student_pos_image_features, teacher_pos_image_features,
            student_qry_hidden_states, teacher_qry_hidden_states,
            student_pos_hidden_states, teacher_pos_hidden_states,
            qry_image_sizes, pos_image_sizes
        ))
        
        # Batch-level loss
        batch_level_loss = self._compute_batch_level_loss(
            input_data, 
            teacher_qry_attention, teacher_pos_attention,
            student_qry_attention, student_pos_attention,
            teacher_qry_hidden_states, teacher_pos_hidden_states,
            student_qry_hidden_states, student_pos_hidden_states,
            num_text_qry_tokens, num_text_pos_tokens,
            student_qry_image_features, student_pos_image_features,
            teacher_qry_image_features, teacher_pos_image_features,
        )

        total_loss = (
            contrastive_loss
            + (self.kd_weight / 10.0) * rkd_loss
            + self.kd_weight * token_level_loss
            + self.kd_weight * self.w_loss_batch * batch_level_loss
        )

        loss_dict = {
            'loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'rkd_loss': rkd_loss,
            'batch_level_loss': batch_level_loss,
            'token_level_loss': token_level_loss,
            'token_level_loss_v': token_level_loss_v,
            'token_level_loss_t': token_level_loss_t,
            'token_level_loss_cross': token_level_loss_cross,
        }
        log_sgd_forward_debug(
            training_args=self.args,
            loss_dict=loss_dict,
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
            temperature=distiller.temperature,
        )
        return loss_dict

    def pairwise_distance(self, x):
        norm = (x**2).sum(dim=1, keepdim=True)
        dist = norm + norm.t() - 2.0 * torch.mm(x, x.t())
        return dist
    
    def compute_distance_loss(self, student_qry, student_pos, teacher_qry, teacher_pos):
        
        student_repr = torch.cat([student_qry, student_pos], dim=0)
        teacher_repr = torch.cat([teacher_qry, teacher_pos], dim=0)
        
        dist_student = self.pairwise_distance(student_repr)
        dist_teacher = self.pairwise_distance(teacher_repr)
        
        mask = torch.triu(torch.ones_like(dist_student), diagonal=1).bool()
        dist_student = dist_student[mask]
        dist_teacher = dist_teacher[mask]
        
        mean_td = dist_teacher.mean().detach() + 1e-8
        mean_sd = dist_student.mean().detach() + 1e-8
        
        dist_student = dist_student / mean_sd
        dist_teacher = dist_teacher / mean_td
        
        diff = dist_student - dist_teacher
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * (abs_diff ** 2)
        linear = abs_diff - 0.5
        
        loss = torch.where(abs_diff < 1.0, quadratic, linear)
        loss = loss.mean()
        return loss
    
    def angle_potentials(self, x):
        n = x.size(0)
        diffs = x.unsqueeze(0) - x.unsqueeze(1)
        norms = torch.norm(diffs, dim=-1, keepdim=True) + 1e-8
        e = diffs / norms
        
        cos_angles = torch.einsum('ijd,kjd->ijk', e, e)
        return cos_angles
    
    def compute_angle_loss(self, student_qry, student_pos, teacher_qry, teacher_pos):
        
        student_repr = torch.cat([student_qry, student_pos], dim=0)
        teacher_repr = torch.cat([teacher_qry, teacher_pos], dim=0)
        
        psi_student = self.angle_potentials(student_repr)
        psi_teacher = self.angle_potentials(teacher_repr)
        
        n = psi_student.size(0)
        mask = torch.ones((n, n, n), dtype=torch.bool, device=psi_student.device)
        idx = torch.arange(n, device=psi_student.device)
        mask[idx, idx, :] = 0
        mask[idx, :, idx] = 0
        mask[:, idx, idx] = 0
        
        psi_teacher = psi_teacher[mask]
        psi_student = psi_student[mask]
        
        diff = psi_student - psi_teacher
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * (abs_diff ** 2)
        linear = abs_diff - 0.5
        loss = torch.where(abs_diff < 1.0, quadratic, linear)
        loss = loss.mean()
        return loss