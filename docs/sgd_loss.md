# SGDLoss — Tổng quan loss

Tài liệu theo dõi cấu trúc loss hiện tại của [`src/criterions/sgd_loss.py`](../src/criterions/sgd_loss.py).  
Cập nhật lần cuối: sau khi gộp `batch_level` + `token_level` thành **unified spectral loss**.

---

## Công thức tổng

```
loss = contrastive_loss
     + (kd_weight / 10) * rkd_loss
     + kd_weight * spectral_loss
```

| Thành phần | Trọng số mặc định | Mô tả ngắn |
|------------|-------------------|------------|
| `contrastive_loss` | 1.0 (implicit) | InfoNCE trên embedding pooled query ↔ positive |
| `rkd_loss` | `kd_weight / 10` | Relational KD (distance + angle) trên rep pooled |
| `spectral_loss` | `kd_weight` | Grassman / Laplacian spectral KD ở batch level |

---

## 1. Contrastive loss

- **Input:** `student_qry_reps`, `student_pos_reps` (sau `encode_input`, có gather multi-GPU).
- **Cách tính:** `CrossEntropyLoss` trên ma trận similarity / `temperature`.
- **Mục tiêu:** Học alignment retrieval — query khớp positive trong batch (và across GPUs nếu DDP).

**Metric log:** `contrastive_loss`

---

## 2. RKD loss (Relational Knowledge Distillation)

Gồm 2 phần, average:

| Sub-loss | Hàm | Ý nghĩa |
|----------|-----|----------|
| `rkd_distance_loss` | `compute_distance_loss` | Huber trên pairwise distance giữa các sample (qry+pos), student vs teacher |
| `rkd_angle_loss` | `compute_angle_loss` | Huber trên góc (cosine) giữa các cặp vector, student vs teacher |

```
rkd_loss = (rkd_distance_loss + rkd_angle_loss) / 2
```

- **Input:** pooled reps của student và teacher (qry + pos concat).
- **Trọng số trong total:** `kd_weight / 10` (mặc định `kd_weight=1` → hệ số 0.1).

**Metric log:** `rkd_loss`  
*(Chi tiết `rkd_distance_loss`, `rkd_angle_loss` chỉ xuất hiện trong NaN debug dump, không log W&B mỗi step.)*

---

## 3. Spectral loss (unified batch-level Grassman KD)

Thay thế loss cũ:
- ~~`token_level_loss`~~ (Grassman per-sample)
- ~~`batch_level_loss`~~ (CKA trên pooled reps)

### Luồng tính

```mermaid
flowchart TB
    subgraph per_sample [Per sample]
        V[Cluster vision teacher → reps]
        T[TopK text tokens → reps]
    end
    subgraph batch [Batch level - riêng qry và pos]
        CAT[Concat tất cả vision + text reps]
        VV[v-v graph kNN]
        TT[t-t graph kNN]
        VT[v-t bipartite kNN]
        G[Grassman loss trên Laplacian eigenspace]
    end
    per_sample --> CAT
    CAT --> VV --> G
    CAT --> TT --> G
    CAT --> VT --> G
```

1. **Per sample:** cluster vision (DBSCAN) + weighted cluster mean; chọn topk text (cosine với last token).
2. **Batch (qry / pos tách riêng):** concat reps → build 3 đồ thị teacher & student.
3. **Grassman loss:** `||Π_teacher − Π_student||²_F` trên eigenspace Laplacian.
4. **Average** loss giữa side `qry` và `pos`.

### Ba thành phần spectral

| Key | Đồ thị | Điều kiện tối thiểu |
|-----|--------|---------------------|
| `spectral_loss_v` | Vision–vision (kNN trên cluster reps batch) | ≥ 2 vision nodes / side |
| `spectral_loss_t` | Text–text (kNN trên topk text reps batch) | ≥ 2 text nodes / side |
| `spectral_loss_cross` | Vision–text bipartite (kNN 2 chiều, full batch) | ≥ 3 nodes tổng (v+t) / side |

```
spectral_loss_side = w_loss_v * L_v + w_loss_t * L_t + w_loss_cross * L_cross
spectral_loss = mean(spectral_loss_qry, spectral_loss_pos)
```

### Hyperparameters liên quan (`arguments.py`)

| Arg | Default | Ý nghĩa |
|-----|---------|---------|
| `kd_weight` | 1.0 | Nhân cho `spectral_loss` (và `/10` cho RKD) |
| `w_loss_v` | 1.0 | Trọng số vision spectral |
| `w_loss_t` | 1.0 | Trọng số text spectral |
| `w_loss_cross` | 1.0 | Trọng số cross-modal spectral |
| `grassman_vision_use_cluster` | false | Cluster vision vs dùng toàn bộ tokens |
| `grassman_text_use_topk` | false | TopK text vs toàn bộ text tokens |
| `topk_text_ratio` | 0.8 | Tỷ lệ topk text |
| `knn_neighbors` | 10 | k cho v-v, t-t, v-t |
| `num_eigenvectors` | 16 | Số eigenvector Laplacian (không tính v₀) |
| `laplacian_type` | `unnormalized` | `unnormalized` hoặc `normalized` |

---

## loss_dict — keys trả về từ `forward()`

### Loss chính

| Key | Backprop? | Ghi chú |
|-----|-----------|---------|
| `loss` | ✓ | Tổng weighted |
| `contrastive_loss` | ✓ | |
| `rkd_loss` | ✓ | |
| `spectral_loss` | ✓ | Combined weighted qry+pos |
| `spectral_loss_v` | ✓* | *Qua `spectral_loss`; log riêng để monitor |
| `spectral_loss_t` | ✓* | |
| `spectral_loss_cross` | ✓* | |

### Metrics theo dõi (không phải loss term riêng)

| Key | Ý nghĩa |
|-----|---------|
| `batch_vision_nodes_qry` | Số đỉnh vision trong đồ thị batch (query) |
| `batch_text_nodes_qry` | Số đỉnh text trong đồ thị batch (query) |
| `batch_vision_nodes_pos` | Số đỉnh vision (positive) |
| `batch_text_nodes_pos` | Số đỉnh text (positive) |

Các metric này log qua `train.log` / W&B mỗi `logging_steps` (xem `KD_LOSS_METRIC_KEYS["sgd_loss"]` trong `main.py`).

---

## Debug (không phải loss)

| Module | Vai trò |
|--------|---------|
| [`src/sgd_debug.py`](../src/sgd_debug.py) | Thu thập & format debug spectral (batch nodes, graph stats) |
| [`src/nan_debug.py`](../src/nan_debug.py) | Ghi file khi NaN hoặc grassman warning |

**Đường dẫn file debug:** `{output_dir}/nan_debug/`  
- `nan_debug.log` — append  
- `events/step_{NNNNNN}_SGD_GRASSMAN_DEBUG.log` — per event  

Chỉ ghi khi loss non-finite hoặc spectral graph / extraction có warning.

---

## File liên quan

| File | Nội dung |
|------|----------|
| `src/criterions/sgd_loss.py` | `SGDLoss` — tính loss |
| `src/sgd_debug.py` | Debug session, `build_sgd_loss_dict`, format grassman |
| `src/nan_debug.py` | `log_sgd_forward_debug` → ghi file |
| `src/arguments.py` | CLI hyperparameters |
| `main.py` | `KD_LOSS_METRIC_KEYS`, training loop |
| `scripts/cls/train_SGD_fastvlm.sh` | Script train mẫu |

---

## Lịch sử thay đổi (tóm tắt)

| Trước | Hiện tại |
|-------|----------|
| `token_level_loss` (Grassman per-sample) | Gộp vào `spectral_loss` (batch-level) |
| `batch_level_loss` (CKA pooled) | **Đã xóa** |
| `w_loss_batch` | **Đã xóa** |
| Log cluster per-sample | Log `batch_*_nodes` + debug file khi warning |
