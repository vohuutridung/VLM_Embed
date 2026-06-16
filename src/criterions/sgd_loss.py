import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from sklearn.cluster import DBSCAN

from src.nan_debug import log_sgd_forward_debug
from src.sgd_debug import (
    GraphConfig,
    ModalSpectralOutcome,
    SGDSpectralDebugSession,
    build_batch_side_debug_entry,
    build_cross_modal_debug,
    build_sgd_loss_dict,
    build_text_modal_debug,
    build_vision_modal_debug,
    new_sample_extraction_debug,
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


# ====== Text token alignment (character-offset mapping, span_propose-style) ======

IMAGE_TOKEN_ID_MIN = 151643
IMAGE_TOKEN_ID_MAX = 151656
STUDENT_IMAGE_TOKEN_INDEX = -200

VLM_IMAGE_MARKER_STRINGS = (
    "<|image_1|>", "<image>", "<|image_pad|>", "<|video_pad|>",
)


def count_text_tokens_teacher(input_ids_row):
    """Count non-image text tokens in teacher input_ids."""
    mask = (input_ids_row < IMAGE_TOKEN_ID_MIN) | (input_ids_row > IMAGE_TOKEN_ID_MAX)
    return int(mask.sum().item())


def count_text_tokens_student(input_ids_row):
    """Count non-image text tokens in student input_ids (exclude vision placeholder)."""
    mask = (input_ids_row < IMAGE_TOKEN_ID_MIN) | (input_ids_row > IMAGE_TOKEN_ID_MAX)
    mask = mask & (input_ids_row != STUDENT_IMAGE_TOKEN_INDEX)
    return int(mask.sum().item())


def strip_vlm_image_markers(text):
    """Remove VLM image placeholders so offset tokenization targets the text segment only."""
    if not text:
        return text
    for marker in VLM_IMAGE_MARKER_STRINGS:
        text = text.replace(marker, "")
    return text


def get_batch_text_strings(teacher_input, teacher_tokenizer):
    """Raw text per sample (shared semantics for teacher/student)."""
    texts = teacher_input.get("texts")
    if texts is not None:
        if isinstance(texts, (list, tuple)):
            return list(texts)
        return [texts]
    return teacher_tokenizer.batch_decode(
        teacher_input["input_ids"], skip_special_tokens=True,
    )


def get_text_token_mask(input_ids_row, is_teacher):
    """Boolean mask of text tokens in input_ids (exclude vision/image placeholders)."""
    mask = (input_ids_row < IMAGE_TOKEN_ID_MIN) | (input_ids_row > IMAGE_TOKEN_ID_MAX)
    if not is_teacher:
        mask = mask & (input_ids_row != STUDENT_IMAGE_TOKEN_INDEX)
    return mask


def get_text_token_ids(input_ids_row, is_teacher):
    """Text token IDs in sequence order (same order as extracted text hidden states)."""
    return input_ids_row[get_text_token_mask(input_ids_row, is_teacher)]


def _offsets_match_text_token_ids(tokenizer, text, text_token_ids):
    """Return offset tensor if tokenizing `text` reproduces `text_token_ids` exactly."""
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    enc_ids = encoding["input_ids"].squeeze(0)
    offsets = encoding["offset_mapping"].squeeze(0)
    text_token_ids_cpu = text_token_ids.detach().cpu()
    if enc_ids.numel() != text_token_ids_cpu.numel():
        return None
    if not torch.equal(enc_ids, text_token_ids_cpu):
        return None
    return offsets


def build_text_offsets_for_hidden_segment(tokenizer, text_token_ids, reference_text, device):
    """
    Character offsets for one side's text hidden tokens (strict token-id match).
    Prefer `reference_text` so teacher/student share the same char coordinate system.
    """
    if text_token_ids.numel() == 0:
        return None

    candidate_texts = []
    if reference_text:
        candidate_texts.append(reference_text)
    for skip_special in (False, True):
        decoded = tokenizer.decode(
            text_token_ids.tolist(),
            skip_special_tokens=skip_special,
            clean_up_tokenization_spaces=False,
        )
        if decoded and decoded not in candidate_texts:
            candidate_texts.append(decoded)

    for text in candidate_texts:
        offsets = _offsets_match_text_token_ids(tokenizer, text, text_token_ids)
        if offsets is not None:
            return offsets.to(device)
    return None


def build_paired_text_offsets(
    teacher_tokenizer,
    student_tokenizer,
    teacher_text_ids,
    student_text_ids,
    reference_text,
    device,
):
    """
    Build teacher/student offsets from the same candidate text string so overlap
    is computed in a shared character coordinate system.
    """
    candidate_texts = []
    if reference_text:
        candidate_texts.append(reference_text)

    for skip_special in (False, True):
        t_decoded = teacher_tokenizer.decode(
            teacher_text_ids.tolist(),
            skip_special_tokens=skip_special,
            clean_up_tokenization_spaces=False,
        )
        if t_decoded and t_decoded not in candidate_texts:
            candidate_texts.append(t_decoded)

    for text in candidate_texts:
        t_offsets = _offsets_match_text_token_ids(teacher_tokenizer, text, teacher_text_ids)
        s_offsets = _offsets_match_text_token_ids(student_tokenizer, text, student_text_ids)
        if t_offsets is not None and s_offsets is not None:
            return t_offsets.to(device), s_offsets.to(device)
    return None, None


def align_student_to_teacher_by_offsets(
    t_text_hidden,
    s_text_hidden,
    teacher_offsets,
    student_offsets,
):
    """
    Align student text hidden states to teacher text tokens via char-span overlap.

    Each teacher token i is paired with a weighted sum of all student tokens whose
    character spans overlap with teacher token i (weights proportional to overlap length).

    Returns:
        t_aligned: [M, D_t] — one row per teacher token with valid overlap
        s_aligned: [M, D_s] — weighted student representation for the same teacher tokens
    """
    debug = {
        "teacher_text_tokens": int(t_text_hidden.size(0)) if t_text_hidden is not None else 0,
        "student_text_tokens": int(s_text_hidden.size(0)) if s_text_hidden is not None else 0,
        "mapped_teacher_tokens": 0,
        "student_tokens_used": 0,
        "skip_reason": None,
    }

    if t_text_hidden is None or s_text_hidden is None:
        debug["skip_reason"] = "missing_text_hidden_states"
        return None, None, debug

    if teacher_offsets is None or student_offsets is None:
        debug["skip_reason"] = "missing_offsets"
        return None, None, debug

    if teacher_offsets.size(0) != t_text_hidden.size(0):
        debug["skip_reason"] = "teacher_offset_hidden_length_mismatch"
        return None, None, debug

    if student_offsets.size(0) != s_text_hidden.size(0):
        debug["skip_reason"] = "student_offset_hidden_length_mismatch"
        return None, None, debug

    device = t_text_hidden.device
    t_offsets = teacher_offsets.to(device)
    s_offsets = student_offsets.to(device)

    t_start = t_offsets[:, 0].long().unsqueeze(1)   # [Nt, 1]
    t_end = t_offsets[:, 1].long().unsqueeze(1)     # [Nt, 1]
    s_start = s_offsets[:, 0].long().unsqueeze(0)   # [1, Ns]
    s_end = s_offsets[:, 1].long().unsqueeze(0)     # [1, Ns]

    overlap = torch.clamp(
        torch.minimum(t_end, s_end) - torch.maximum(t_start, s_start),
        min=0,
    ).float()  # [Nt, Ns]

    valid_t = t_offsets[:, 1] > t_offsets[:, 0]
    valid_s = s_offsets[:, 1] > s_offsets[:, 0]
    overlap = overlap * valid_t.unsqueeze(1).float() * valid_s.unsqueeze(0).float()

    denom = overlap.sum(dim=1)  # [Nt]
    valid_teacher = denom > 0

    if not valid_teacher.any():
        debug["skip_reason"] = "no_character_overlap_pairs"
        return None, None, debug

    weights = overlap[valid_teacher] / denom[valid_teacher].unsqueeze(1).clamp(min=1e-8)
    s_aligned = weights.to(s_text_hidden.dtype) @ s_text_hidden
    t_aligned = t_text_hidden[valid_teacher]

    debug["mapped_teacher_tokens"] = int(valid_teacher.sum().item())
    debug["student_tokens_used"] = int((overlap[valid_teacher].sum(dim=0) > 0).sum().item())

    return t_aligned, s_aligned, debug


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
    features_fp32 = features.float()
    dist_sq = compute_pairwise_sq_distances(features_fp32)

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
    return W.to(features.dtype)


def build_bipartite_weight_matrix(features_v, features_t, k_neighbors):
    """Build a sparse bipartite graph: each vision/text node connects to k nearest nodes on the other side."""
    n_v, n_t = features_v.size(0), features_t.size(0)
    n_total = n_v + n_t
    device, dtype = features_v.device, features_v.dtype

    if n_v == 0 or n_t == 0:
        return torch.zeros(n_total, n_total, device=device, dtype=dtype)

    cross_dist_sq = compute_pairwise_sq_distances(torch.cat([features_v, features_t], dim=0))
    cross_dist_sq = cross_dist_sq[:n_v, n_v:]

    nonzero_dists = cross_dist_sq[cross_dist_sq > 0]
    sigma = nonzero_dists.median().item() if nonzero_dists.numel() > 0 else 1.0
    if sigma < 1e-8:
        sigma = 1.0

    W_cross = torch.zeros(n_v, n_t, device=device, dtype=dtype)

    k_v_to_t = min(k_neighbors, n_t)
    for i in range(n_v):
        _, knn_idx = torch.topk(cross_dist_sq[i], k_v_to_t, largest=False)
        W_cross[i, knn_idx] = torch.exp(-cross_dist_sq[i, knn_idx] / sigma)

    k_t_to_v = min(k_neighbors, n_v)
    for j in range(n_t):
        _, knn_idx = torch.topk(cross_dist_sq[:, j], k_t_to_v, largest=False)
        W_cross[knn_idx, j] = torch.exp(-cross_dist_sq[knn_idx, j] / sigma)

    W = torch.zeros(n_total, n_total, device=device, dtype=dtype)
    W[:n_v, n_v:] = W_cross
    W[n_v:, :n_v] = W_cross.t()
    return W


def compute_laplacian_eigenspace(W, num_eigenvectors, laplacian_type="unnormalized"):
    n = W.size(0)
    if n < 3:
        return torch.eye(n, device=W.device, dtype=W.dtype)

    k_eig = min(num_eigenvectors, n - 1)
    if k_eig < 1:
        return torch.eye(n, device=W.device, dtype=W.dtype)

    # Laplacian in fp32 with diagonal jitter for stable eigh backward.
    W_fp32 = W.float()
    D = W_fp32.sum(dim=1)
    if laplacian_type == "normalized":
        D_inv_sqrt = (D + 1e-10).rsqrt().clamp(max=1e8)
        L = torch.diag(D) - W_fp32
        D_inv_sqrt_mat = torch.diag(D_inv_sqrt)
        L = D_inv_sqrt_mat @ L @ D_inv_sqrt_mat
    else:
        L = torch.diag(D) - W_fp32

    jitter = 1e-4
    L = L + jitter * torch.eye(n, device=W.device, dtype=torch.float32)

    try:
        _, eigenvectors = torch.linalg.eigh(L)
    except Exception:
        return torch.eye(n, device=W.device, dtype=W.dtype)

    U = eigenvectors[:, 1:1 + k_eig].to(W.dtype)
    return U @ U.T


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


def local_cross_affinity_loss(
    teacher_v,
    student_v,
    teacher_t,
    student_t,
    temperature=0.1,
):
    """Per-sample vision-text affinity KL distillation (teacher distributions as targets)."""
    if teacher_v.size(0) < 2 or teacher_t.size(0) < 2:
        return teacher_v.new_tensor(0.0)

    teacher_v = F.normalize(teacher_v.float(), dim=-1)
    teacher_t = F.normalize(teacher_t.float(), dim=-1)
    student_v = F.normalize(student_v.float(), dim=-1)
    student_t = F.normalize(student_t.float(), dim=-1)

    a_t = teacher_v @ teacher_t.T / temperature
    a_s = student_v @ student_t.T / temperature

    p_t_v2t = F.softmax(a_t, dim=-1).detach()
    log_p_s_v2t = F.log_softmax(a_s, dim=-1)
    loss_v2t = F.kl_div(log_p_s_v2t, p_t_v2t, reduction="batchmean")

    p_t_t2v = F.softmax(a_t.T, dim=-1).detach()
    log_p_s_t2v = F.log_softmax(a_s.T, dim=-1)
    loss_t2v = F.kl_div(log_p_s_t2v, p_t_t2v, reduction="batchmean")

    return 0.5 * (loss_v2t + loss_t2v)


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
        self.w_loss_local_cross = getattr(args, 'w_loss_local_cross', 0.2)
        self.local_cross_temperature = getattr(args, 'local_cross_temperature', 0.1)
        self._student_tokenizer = None

    def _get_student_tokenizer(self, distiller):
        if self._student_tokenizer is None:
            from transformers import AutoTokenizer
            self._student_tokenizer = AutoTokenizer.from_pretrained(
                distiller.model_args.model_name,
                trust_remote_code=True,
            )
        return self._student_tokenizer

    def _dist_gather_tensor(self, t):
        """Gather tensor từ tất cả các process"""
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def _graph_config(self):
        return GraphConfig(
            knn_neighbors=self.knn_neighbors,
            num_eigenvectors=self.num_eigenvectors,
            laplacian_type=self.laplacian_type,
        )

    @staticmethod
    def _zero_loss(device):
        return torch.tensor(0.0, device=device)

    def _compute_knn_spectral_outcome(self, h_teacher, h_student, min_nodes):
        device = h_teacher.device
        if h_teacher.size(0) < min_nodes:
            return ModalSpectralOutcome(
                loss=self._zero_loss(device),
                num_nodes=int(h_teacher.size(0)),
                valid=False,
            )

        w_teacher = build_knn_weight_matrix(h_teacher, self.knn_neighbors)
        w_student = build_knn_weight_matrix(h_student, self.knn_neighbors)
        espace_teacher = compute_laplacian_eigenspace(
            w_teacher, self.num_eigenvectors, self.laplacian_type,
        )
        espace_student = compute_laplacian_eigenspace(
            w_student, self.num_eigenvectors, self.laplacian_type,
        )
        return ModalSpectralOutcome(
            loss=compute_grassman_loss(espace_teacher, espace_student),
            num_nodes=int(h_teacher.size(0)),
            valid=True,
            w_teacher=w_teacher,
            w_student=w_student,
        )

    def _compute_batch_vision_spectral_outcome(self, t_v_list, s_v_list, device):
        if not t_v_list:
            return ModalSpectralOutcome(
                loss=self._zero_loss(device),
                skip_reason="no_vision_reps",
            )

        h_teacher = torch.cat(t_v_list, dim=0)
        h_student = torch.cat(s_v_list, dim=0)
        if h_teacher.size(0) < 2:
            return ModalSpectralOutcome(
                loss=self._zero_loss(device),
                num_nodes=int(h_teacher.size(0)),
                skip_reason="batch_vision_nodes_lt_2",
            )
        return self._compute_knn_spectral_outcome(h_teacher, h_student, min_nodes=2)

    def _compute_batch_text_spectral_outcome(self, t_t_list, s_t_list, device):
        if not t_t_list:
            return ModalSpectralOutcome(
                loss=self._zero_loss(device),
                skip_reason="no_text_reps",
            )

        h_teacher = torch.cat(t_t_list, dim=0)
        h_student = torch.cat(s_t_list, dim=0)
        if h_teacher.size(0) < 2:
            return ModalSpectralOutcome(
                loss=self._zero_loss(device),
                num_nodes=int(h_teacher.size(0)),
                skip_reason="batch_text_nodes_lt_2",
            )
        return self._compute_knn_spectral_outcome(h_teacher, h_student, min_nodes=2)

    def _compute_batch_cross_spectral_outcome(self, t_v_list, s_v_list, t_t_list, s_t_list, device):
        if not (t_v_list and t_t_list):
            return ModalSpectralOutcome(
                loss=self._zero_loss(device),
                skip_reason="missing_modal_representations",
            )

        h_t_v = torch.cat(t_v_list, dim=0)
        h_s_v = torch.cat(s_v_list, dim=0)
        h_t_t = torch.cat(t_t_list, dim=0)
        h_s_t = torch.cat(s_t_list, dim=0)
        vision_nodes = int(h_t_v.size(0))
        text_nodes = int(h_t_t.size(0))
        total_nodes = vision_nodes + text_nodes
        if total_nodes < 3:
            return ModalSpectralOutcome(
                loss=self._zero_loss(device),
                vision_nodes=vision_nodes,
                text_nodes=text_nodes,
                total_nodes=total_nodes,
                skip_reason="total_nodes_lt_3",
            )

        w_teacher = build_bipartite_weight_matrix(h_t_v, h_t_t, self.knn_neighbors)
        w_student = build_bipartite_weight_matrix(h_s_v, h_s_t, self.knn_neighbors)
        espace_teacher = compute_laplacian_eigenspace(
            w_teacher, self.num_eigenvectors, self.laplacian_type,
        )
        espace_student = compute_laplacian_eigenspace(
            w_student, self.num_eigenvectors, self.laplacian_type,
        )
        return ModalSpectralOutcome(
            loss=compute_grassman_loss(espace_teacher, espace_student),
            vision_nodes=vision_nodes,
            text_nodes=text_nodes,
            total_nodes=total_nodes,
            valid=True,
            w_teacher=w_teacher,
            w_student=w_student,
        )

    def _collect_side_batch_representations(
        self, batch_size, side, side_args, debug_session,
    ):
        (
            teacher_input, student_input, text_strings,
            s_img_feats, t_img_feats, s_hidden, t_hidden, image_sizes,
            teacher_tokenizer, student_tokenizer,
        ) = side_args
        t_v_list, s_v_list, t_t_list, s_t_list = [], [], [], []
        local_cross_losses = []

        for i in range(batch_size):
            num_text_teacher = count_text_tokens_teacher(teacher_input["input_ids"][i])
            num_text_student = count_text_tokens_student(student_input["input_ids"][i])
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
                s_hidden, i, num_text_student, num_vision_student,
                is_teacher=False, has_image=has_image,
            )[-1] if num_text_student > 0 else None
            t_text_last = extract_text_hidden_states(
                t_hidden, i, num_text_teacher, num_vision_teacher,
                is_teacher=True, has_image=has_image,
            )[-1] if num_text_teacher > 0 else None

            s_text_aligned = t_text_aligned = None
            text_align_dbg = {}
            if num_text_teacher > 0 and num_text_student > 0 and t_text_last is not None and s_text_last is not None:
                reference_text = strip_vlm_image_markers(text_strings[i] if i < len(text_strings) else "")
                device = t_text_last.device
                t_text_ids = get_text_token_ids(teacher_input["input_ids"][i], is_teacher=True)
                s_text_ids = get_text_token_ids(student_input["input_ids"][i], is_teacher=False)
                t_offsets, s_offsets = build_paired_text_offsets(
                    teacher_tokenizer,
                    student_tokenizer,
                    t_text_ids,
                    s_text_ids,
                    reference_text,
                    device,
                )
                if t_offsets is None or s_offsets is None:
                    text_align_dbg = {
                        "teacher_text_tokens": int(t_text_last.size(0)),
                        "student_text_tokens": int(s_text_last.size(0)),
                        "mapped_teacher_tokens": 0,
                        "student_tokens_used": 0,
                        "skip_reason": "offset_token_id_mismatch",
                    }
                else:
                    t_text_aligned, s_text_aligned, text_align_dbg = align_student_to_teacher_by_offsets(
                        t_text_last, s_text_last, t_offsets, s_offsets,
                    )

            s_vision_last = t_vision_last = None
            if has_image:
                s_vision_last = extract_vision_hidden_states(
                    s_hidden, i, num_vision_student, num_text_student, is_teacher=False,
                )[-1]
                t_vision_last = extract_vision_hidden_states(
                    t_hidden, i, num_vision_teacher, num_text_teacher, is_teacher=True,
                )[-1]

            h_t_v, h_s_v, h_t_t, h_s_t, sample_debug = self._extract_sample_representations(
                s_text_aligned, t_text_aligned,
                s_vision_last, t_vision_last,
                num_text_teacher, has_image, img_w, img_h,
                batch_idx=i,
                side=side,
                text_align_debug=text_align_dbg,
            )
            debug_session.maybe_record_sample_warning(sample_debug)

            if h_t_v is not None and h_s_v is not None and h_t_v.size(0) == h_s_v.size(0):
                t_v_list.append(h_t_v)
                s_v_list.append(h_s_v)
            if h_t_t is not None and h_s_t is not None and h_t_t.size(0) == h_s_t.size(0):
                t_t_list.append(h_t_t)
                s_t_list.append(h_s_t)

            if (
                h_t_v is not None and h_s_v is not None
                and h_t_t is not None and h_s_t is not None
                and h_t_v.size(0) >= 2 and h_t_t.size(0) >= 2
                and h_t_v.size(0) == h_s_v.size(0)
                and h_t_t.size(0) == h_s_t.size(0)
            ):
                local_cross_losses.append(
                    local_cross_affinity_loss(
                        h_t_v, h_s_v, h_t_t, h_s_t,
                        temperature=self.local_cross_temperature,
                    )
                )

        return t_v_list, s_v_list, t_t_list, s_t_list, local_cross_losses

    def _extract_sample_representations(self, s_text_hidden, t_text_hidden,
                                        s_vision_hidden, t_vision_hidden,
                                        num_text, has_image, original_width, original_height,
                                        batch_idx=0, side="qry",
                                        text_align_debug=None):
        """Extract per-sample vision cluster reps and topk text reps for batch-level graphs."""
        device = (
            s_text_hidden.device if s_text_hidden is not None
            else s_vision_hidden.device if s_vision_hidden is not None
            else t_text_hidden.device if t_text_hidden is not None
            else t_vision_hidden.device if t_vision_hidden is not None
            else torch.device('cpu')
        )
        h_t_v = h_s_v = h_t_t = h_s_t = None

        debug = new_sample_extraction_debug(batch_idx, side, has_image, num_text)
        vision_dbg = debug["vision"]
        text_dbg = debug["text"]

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

                if h_t_v is None or h_s_v is None:
                    if "skip_reason" not in vision_dbg:
                        vision_dbg["skip_reason"] = "missing_vision_representations"
                elif h_t_v.size(0) != h_s_v.size(0):
                    vision_dbg["skip_reason"] = "teacher_student_node_count_mismatch"
                    h_t_v = h_s_v = None
            else:
                vision_dbg["skip_reason"] = "teacher_vision_tokens_lt_2"
        elif has_image:
            vision_dbg["skip_reason"] = "missing_vision_hidden_states"

        vision_dbg["vision_reps_valid"] = (
            h_t_v is not None and h_s_v is not None and h_t_v.size(0) == h_s_v.size(0)
        )

        # ===== Text tokens (pre-aligned teacher↔student via character offsets) =====
        if text_align_debug:
            text_dbg.update(text_align_debug)
        if num_text > 0 and t_text_hidden is not None and s_text_hidden is not None:
            text_dbg["use_topk"] = bool(self.grassman_text_use_topk)
            if self.grassman_text_use_topk:
                topk_indices = select_topk_text_tokens_by_last_token_cosine(
                    t_text_hidden, self.topk_text_ratio,
                )
                h_t_t = t_text_hidden[topk_indices]
                h_s_t = s_text_hidden[topk_indices]
                text_dbg["topk_tokens"] = int(h_t_t.size(0))
            else:
                h_t_t = t_text_hidden
                h_s_t = s_text_hidden
                text_dbg["num_tokens"] = int(h_t_t.size(0))

            if h_t_t.size(0) != h_s_t.size(0):
                text_dbg["skip_reason"] = "teacher_student_text_count_mismatch"
                h_t_t = h_s_t = None
        elif num_text > 0:
            if "skip_reason" not in text_dbg:
                text_dbg["skip_reason"] = "missing_aligned_text_hidden_states"

        text_dbg["text_reps_valid"] = (
            h_t_t is not None and h_s_t is not None and h_t_t.size(0) == h_s_t.size(0)
        )

        return h_t_v, h_s_v, h_t_t, h_s_t, debug

    def _compute_side_batch_spectral_loss(self, device, batch_size, side, side_args, debug_session):
        """Build batch-level v-v, t-t, v-t graphs for one side (qry or pos) and compute Grassman loss."""
        graph_cfg = self._graph_config()
        t_v_list, s_v_list, t_t_list, s_t_list, local_cross_losses = (
            self._collect_side_batch_representations(
                batch_size, side, side_args, debug_session,
            )
        )

        vision_outcome = self._compute_batch_vision_spectral_outcome(t_v_list, s_v_list, device)
        text_outcome = self._compute_batch_text_spectral_outcome(t_t_list, s_t_list, device)
        cross_outcome = self._compute_batch_cross_spectral_outcome(
            t_v_list, s_v_list, t_t_list, s_t_list, device,
        )

        batch_debug = build_batch_side_debug_entry(
            side,
            vision_outcome.loss,
            text_outcome.loss,
            cross_outcome.loss,
            build_vision_modal_debug(vision_outcome, graph_cfg),
            build_text_modal_debug(text_outcome, graph_cfg),
            build_cross_modal_debug(cross_outcome, graph_cfg),
        )
        debug_session.record_batch_side(side, batch_debug)

        side_loss = (
            self.w_loss_v * vision_outcome.loss
            + self.w_loss_t * text_outcome.loss
            + self.w_loss_cross * cross_outcome.loss
        )
        local_cross_loss = self._average_losses(local_cross_losses, device)
        side_spectral_valid = (
            vision_outcome.valid or text_outcome.valid or cross_outcome.valid
        )
        side_local_cross_valid = bool(local_cross_losses)
        return (
            side_loss,
            vision_outcome.loss,
            text_outcome.loss,
            cross_outcome.loss,
            local_cross_loss,
            side_spectral_valid,
            vision_outcome.valid,
            text_outcome.valid,
            cross_outcome.valid,
            side_local_cross_valid,
        )

    @staticmethod
    def _average_losses(losses, device):
        if not losses:
            return SGDLoss._zero_loss(device)
        return sum(losses) / len(losses)

    def _compute_batch_spectral_loss(
        self, batch_size, device,
        teacher_qry_input, student_qry_input, qry_text_strings,
        teacher_pos_input, student_pos_input, pos_text_strings,
        student_qry_image_features, teacher_qry_image_features,
        student_pos_image_features, teacher_pos_image_features,
        student_qry_hidden_states, teacher_qry_hidden_states,
        student_pos_hidden_states, teacher_pos_hidden_states,
        qry_image_sizes, pos_image_sizes,
        teacher_tokenizer, student_tokenizer,
    ):
        debug_session = SGDSpectralDebugSession()
        side_losses = []
        side_loss_v = []
        side_loss_t = []
        side_loss_cross = []
        side_local_cross_losses = []

        for side, side_args in (
            ("qry", (
                teacher_qry_input, student_qry_input, qry_text_strings,
                student_qry_image_features, teacher_qry_image_features,
                student_qry_hidden_states, teacher_qry_hidden_states, qry_image_sizes,
                teacher_tokenizer, student_tokenizer,
            )),
            ("pos", (
                teacher_pos_input, student_pos_input, pos_text_strings,
                student_pos_image_features, teacher_pos_image_features,
                student_pos_hidden_states, teacher_pos_hidden_states, pos_image_sizes,
                teacher_tokenizer, student_tokenizer,
            )),
        ):
            (
                side_loss,
                loss_v,
                loss_t,
                loss_cross,
                loss_local_cross,
                side_spectral_valid,
                v_valid,
                t_valid,
                cross_valid,
                side_local_cross_valid,
            ) = self._compute_side_batch_spectral_loss(
                device, batch_size, side, side_args, debug_session,
            )
            if side_spectral_valid:
                side_losses.append(side_loss)
            if v_valid:
                side_loss_v.append(loss_v)
            if t_valid:
                side_loss_t.append(loss_t)
            if cross_valid:
                side_loss_cross.append(loss_cross)
            if side_local_cross_valid:
                side_local_cross_losses.append(loss_local_cross)

        return (
            self._average_losses(side_losses, device),
            self._average_losses(side_loss_v, device),
            self._average_losses(side_loss_t, device),
            self._average_losses(side_loss_cross, device),
            self._average_losses(side_local_cross_losses, device),
            debug_session,
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

        batch_size = student_qry_input['input_ids'].size(0)
        device = student_qry_input['input_ids'].device

        teacher_tokenizer = distiller.tokenizer
        student_tokenizer = self._get_student_tokenizer(distiller)
        qry_text_strings = get_batch_text_strings(teacher_qry_input, teacher_tokenizer)
        pos_text_strings = get_batch_text_strings(teacher_pos_input, teacher_tokenizer)

        # Forward teacher
        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            teacher_qry_reps, teacher_qry_image_features, teacher_qry_attention, teacher_qry_hidden_states = teacher_qry_output
            teacher_pos_reps, teacher_pos_image_features, teacher_pos_attention, teacher_pos_hidden_states = teacher_pos_output

        # Forward student (no attention tensors — only needed for NaN debug dumps)
        student_qry_output = student_model.encode_input(student_qry_input, output_attentions=False)
        student_pos_output = student_model.encode_input(student_pos_input, output_attentions=False)
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

        # Unified batch spectral loss
        (
            spectral_loss,
            spectral_loss_v,
            spectral_loss_t,
            spectral_loss_cross,
            local_cross_loss,
            debug_session,
        ) = self._compute_batch_spectral_loss(
            batch_size, device,
            teacher_qry_input, student_qry_input, qry_text_strings,
            teacher_pos_input, student_pos_input, pos_text_strings,
            student_qry_image_features, teacher_qry_image_features,
            student_pos_image_features, teacher_pos_image_features,
            student_qry_hidden_states, teacher_qry_hidden_states,
            student_pos_hidden_states, teacher_pos_hidden_states,
            qry_image_sizes, pos_image_sizes,
            teacher_tokenizer, student_tokenizer,
        )

        total_loss = (
            contrastive_loss
            + (self.kd_weight / 10.0) * rkd_loss
            + self.kd_weight * spectral_loss
            + self.kd_weight * self.w_loss_local_cross * local_cross_loss
        )

        loss_dict = build_sgd_loss_dict(
            device,
            total_loss,
            contrastive_loss,
            rkd_loss,
            spectral_loss,
            spectral_loss_v,
            spectral_loss_t,
            spectral_loss_cross,
            local_cross_loss,
            debug_session.batch_stats,
        )
        log_sgd_forward_debug(
            training_args=self.args,
            loss_dict=loss_dict,
            grassman_debug=debug_session.entries,
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