# SGDLoss — Tổng quan loss

Tài liệu theo dõi cấu trúc loss hiện tại của [`src/criterions/sgd_loss.py`](../src/criterions/sgd_loss.py).  
Cập nhật lần cuối: **token-level Grassman spectral** (per sample) + **batch-level CKA** + local cross.

---

## Công thức tổng

```
loss = contrastive_loss
     + (kd_weight / 10) * rkd_loss
     + kd_weight * token_level_loss
     + kd_weight * w_loss_batch * batch_level_loss
     + kd_weight * w_loss_local_cross * local_cross_loss
```

| Thành phần | Trọng số mặc định | Mô tả ngắn |
|------------|-------------------|------------|
| `contrastive_loss` | 1.0 (implicit) | InfoNCE trên embedding pooled query ↔ positive |
| `rkd_loss` | `kd_weight / 10` | Relational KD (distance + angle) trên rep pooled |
| `token_level_loss` | `kd_weight` | Grassman spectral KD **trong từng sample** (v-v, t-t, v-t) |
| `batch_level_loss` | `kd_weight * w_loss_batch` | CKA trên 1 vector đại diện / sample (attention pool text) |
| `local_cross_loss` | `kd_weight * w_loss_local_cross` | KL distillation phân phối vision↔text affinity trong từng sample |

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

## 3. Token-level spectral loss (Grassman KD per sample)

Mỗi cặp `(batch_idx, side)` với `side ∈ {qry, pos}`:

1. **Extract reps:** spatial vision mapping + char-span text align + optional top-k (vision/text)
2. **Build graphs trong sample:**
   - v-v: kNN trên vision nodes (`Nv ≥ 2`)
   - t-t: kNN trên text nodes (`Nt ≥ 2`)
   - v-t: bipartite kNN (`Nv + Nt ≥ 3`)
3. **Grassman loss** trên Laplacian eigenspace teacher vs student
4. **Average** qua các sample-side hợp lệ (riêng cho v, t, cross)

| Key | Đồ thị | Điều kiện |
|-----|--------|-----------|
| `token_level_loss_v` | Vision–vision trong 1 sample | `Nv ≥ 2` |
| `token_level_loss_t` | Text–text trong 1 sample | `Nt ≥ 2` |
| `token_level_loss_cross` | Vision–text bipartite trong 1 sample | `Nv + Nt ≥ 3` |

```
token_level_loss = w_loss_v * L_v + w_loss_t * L_t + w_loss_cross * L_cross
```

(L_v, L_t, L_cross là mean over valid sample-side entries.)

### Hyperparameters token-level (graph + top-k)

| Arg | Default | Ý nghĩa |
|-----|---------|---------|
| `w_loss_v` / `w_loss_t` / `w_loss_cross` | 1.0 | Trọng số v-v / t-t / v-t trong `token_level_loss` |
| `grassman_vision_use_topk` | true | Bật top-k vision sau spatial map |
| `topk_vision_ratio` | 0.8 | \(k_v = \max(1, \lfloor ratio \cdot M_v \rfloor)\) trên mapped vision patches |
| `grassman_text_use_topk` | false | Bật top-k text sau char-span align |
| `topk_text_ratio` | 0.8 | \(k_t = \max(1, \lfloor ratio \cdot M_t \rfloor)\) trên aligned text tokens |
| `knn_neighbors` | 10 | k cho v-v, t-t, v-t |
| `num_eigenvectors` | 16 | Số eigenvector Laplacian |
| `laplacian_type` | `unnormalized` | Loại Laplacian |

`kd_weight` scale toàn bộ KD (token-level, batch CKA, local cross, RKD `/10`) — xem § công thức tổng.

---

## 4. Batch-level CKA loss

**1 vector đại diện / sample / side** từ attention-weighted pool trên **text hidden** (layer cuối):

- Importance = `sum(attention)` theo sequence (mean heads)
- Mask chỉ text tokens (teacher: cuối seq; student: sau vision)
- Weight normalize → weighted sum hidden → rep `[D]`
- Stack batch → `CKA(s_reps, t_reps)` cho qry và pos, average

**Metric log:** `batch_level_loss`  
**Arg:** `w_loss_batch` (default `1.0`) — nhân thêm sau `kd_weight`.

**Lưu ý:** Student forward cần `output_attentions=True`.

---

## 5. Local cross-modal affinity loss

Bổ sung **local grounding trong từng sample** — teacher gán text token / patch ảnh nào quan trọng với nhau; student học cùng phân phối quan hệ mà không cần khớp trực tiếp hidden dimension.

### Input (sau extraction per sample)

| Tensor | Shape | Nguồn |
|--------|-------|-------|
| `V_T`, `V_S` | `[Nv, D_t]`, `[Nv, D_s]` | Vision patch reps (teacher anchor → spatial overlap map sang student) |
| `T_T`, `T_S` | `[Nt, D_t]`, `[Nt, D_s]` | Top-k text tokens đã align (cùng số hàng teacher/student) |

`Nv`, `Nt` phải khớp giữa teacher và student. `D_t` và `D_s` **không** cần giống nhau.

### Công thức

```
A_T = cos(V_T, T_T) / τ          # [Nv, Nt]
A_S = cos(V_S, T_S) / τ

P_T^{v→t} = softmax(A_T, dim=text)     P_S^{v→t} = softmax(A_S, dim=text)
P_T^{t→v} = softmax(A_T^T, dim=vision)   P_S^{t→v} = softmax(A_S^T, dim=vision)

L_{v→t} = KL(P_T^{v→t} || P_S^{v→t})
L_{t→v} = KL(P_T^{t→v} || P_S^{t→v})

local_cross_loss_sample = 0.5 * (L_{v→t} + L_{t→v})
```

- Teacher distribution `.detach()` — chỉ student nhận gradient.
- Sample bị bỏ nếu `Nv < 2` hoặc `Nt < 2`.
- Average qua các sample hợp lệ trong batch, rồi average `qry` và `pos`.

### Hyperparameters

| Arg | Default | Gợi ý tuning |
|-----|---------|--------------|
| `w_loss_local_cross` | 0.2 | Tăng 0.5 nếu loss quá nhỏ; giảm 0.05–0.1 nếu dao động |
| `local_cross_temperature` | 0.1 | Tăng 0.2 nếu training không ổn định |

**Metric log:** `local_cross_loss`

---

## 5. Text mapping (teacher → student)

Dùng cho cả spectral text nodes và local cross loss. Không ghép index thô `s[i] ↔ t[i]`.

### Pipeline

1. **Đếm text tokens** riêng teacher/student (loại pad/special; student thêm loại `IMAGE_TOKEN_INDEX`).
2. **Cắt hidden** layer cuối:
   - Teacher `[pad][vision][text]` → `hidden[-Nt:]`
   - Student `[vision][text][pad]` → `hidden[Nv:Nv+Ns]`
3. **Build offsets** (`build_paired_text_offsets`):
   - `reference_text` = raw text đã strip image markers
   - Thử cùng candidate string cho cả hai tokenizer
   - **Strict:** `tokenizer(text)` phải reproduce đúng `text_token_ids` từ `input_ids`
   - Cả teacher và student phải thành công trên **cùng** candidate → hệ tọa độ ký tự chung
4. **Weighted align** (`align_student_to_teacher_by_offsets`):
   - Ma trận overlap `[Nt, Ns]` = độ dài char-span giao nhau
   - Mỗi teacher token `i`: `s_aligned[i] = Σ_j w_{ij} * s_hidden[j]`, `w_{ij} ∝ overlap[i,j]`
   - `t_aligned[i] = t_hidden[i]`
5. **Top-k** (`select_topk_tokens_by_last_token_cosine`) trên tensor **đã align** nếu `grassman_text_use_topk=True`; ratio = `topk_text_ratio`.

### Skip reasons (text)

| `skip_reason` | Khi nào |
|---------------|---------|
| `offset_token_id_mismatch` | Không build được offset khớp `input_ids` |
| `teacher_offset_hidden_length_mismatch` | `len(offsets_t) ≠ Nt` |
| `student_offset_hidden_length_mismatch` | `len(offsets_s) ≠ Ns` |
| `no_character_overlap_pairs` | Ma trận overlap toàn 0 |
| `missing_aligned_text_hidden_states` | Align thất bại |

---

## 6. Vision mapping (teacher → student)

Spatial-only mapping — **không cluster**. Teacher patch là anchor; student patches được gom theo overlap bbox trong không gian ảnh đã scale.

### Pipeline

1. **Patch bboxes** (`get_patch_bboxes`): mỗi teacher/student patch → `[x0, y0, x1, y1]` trên lưới patch.
2. **Scale teacher bbox** sang không gian student (`student_resize / original_width|height`).
3. **Weighted align** (`align_student_vision_to_teacher_spatial`):
   - Ma trận overlap `[Nt, Ns]` = diện tích giao nhau bbox 2D
   - Mỗi teacher patch `i`: `s_aligned[i] = Σ_j w_{ij} * s_hidden[j]`, `w_{ij} ∝ overlap[i,j]`
   - `t_aligned[i] = t_hidden[i]`
4. **Top-k** nếu `grassman_vision_use_topk=True`; ratio = `topk_vision_ratio`; cùng indices cho teacher và student.

Output: `h_t_v`, `h_s_v` cùng số node `Nv` — dùng cho spectral (cấu trúc quan hệ) và local cross, không so trực tiếp hidden khác chiều.

### Skip reasons (vision)

| `skip_reason` | Khi nào |
|---------------|---------|
| `no_spatial_overlap_pairs` | Ma trận overlap toàn 0 |
| `mapped_teacher_tokens_lt_2` | Sau align còn < 2 teacher patches |
| `vision_nodes_lt_2_after_topk` | Sau top-k còn < 2 nodes |
| `teacher_vision_tokens_lt_2` | Teacher có < 2 vision tokens |

---

## loss_dict — keys trả về từ `forward()`

### Loss chính

| Key | Backprop? | Ghi chú |
|-----|-----------|---------|
| `loss` | ✓ | Tổng weighted |
| `contrastive_loss` | ✓ | |
| `rkd_loss` | ✓ | |
| `token_level_loss` | ✓ | Combined v/t/cross (weighted) |
| `token_level_loss_v` | ✓* | *Qua `token_level_loss`; monitor |
| `token_level_loss_t` | ✓* | |
| `token_level_loss_cross` | ✓* | |
| `batch_level_loss` | ✓ | CKA attention-pooled reps |
| `local_cross_loss` | ✓ | Per-sample affinity KL |

### Metrics theo dõi (số node sau extraction, trung bình sample-side trong batch)

| Key | Ý nghĩa |
|-----|---------|
| `avg_vision_nodes` | Trung bình số vision nodes / sample-side (qry+pos) |
| `avg_text_nodes` | Trung bình số text nodes / sample-side |
| `avg_vision_nodes_qry` / `avg_vision_nodes_pos` | Tách theo side |
| `avg_text_nodes_qry` / `avg_text_nodes_pos` | Tách theo side |

Các metric này log qua `KD_LOSS_METRIC_KEYS["sgd_loss"]` trong `main.py`.

---

## Debug (không phải loss)

| Module | Vai trò |
|--------|---------|
| [`src/sgd_debug.py`](../src/sgd_debug.py) | Per-sample spectral debug, `build_sgd_loss_dict` |
| [`src/nan_debug.py`](../src/nan_debug.py) | Ghi file khi NaN hoặc grassman warning |

**Đường dẫn file debug:** `{output_dir}/nan_debug/`  
- `nan_debug.log` — append  
- `events/step_{NNNNNN}_SGD_GRASSMAN_DEBUG.log` — per event  

Chỉ ghi khi loss non-finite hoặc spectral graph / extraction có warning.

---

## File liên quan

| File | Nội dung |
|------|----------|
| `src/criterions/sgd_loss.py` | `SGDLoss`, `local_cross_affinity_loss`, text/vision mapping |
| `src/sgd_debug.py` | Debug session, `build_sgd_loss_dict`, format grassman |
| `src/nan_debug.py` | `log_sgd_forward_debug` → ghi file |
| `src/arguments.py` | CLI hyperparameters |
| `main.py` | `KD_LOSS_METRIC_KEYS`, training loop |
| `scripts/cls/train_SGD_fastvlm.sh` | Script train mẫu (`W_LOSS_LOCAL_CROSS`, `LOCAL_CROSS_TEMPERATURE`) |

---

## Lịch sử thay đổi (tóm tắt)

| Trước | Hiện tại |
|-------|----------|
| `spectral_loss` (batch-concat Grassman) | **`token_level_loss`** (Grassman per sample) |
| (đã xóa) `batch_level_loss` | **Khôi phục** CKA attention pool |
| `w_loss_batch` | **Khôi phục** (default `1.0`) |
| Vision cluster DBSCAN | **Spatial bbox overlap** mapping |
| Text map index-i | **Weighted char-span overlap** |
| — | **`local_cross_loss`** per-sample affinity KL |
