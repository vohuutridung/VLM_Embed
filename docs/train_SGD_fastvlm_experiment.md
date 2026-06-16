# Kịch bản thử nghiệm `train_SGD_fastvlm.sh`

Tài liệu này mô tả chi tiết kịch bản chạy của script [`scripts/cls/train_SGD_fastvlm.sh`](../scripts/cls/train_SGD_fastvlm.sh), gồm:
- Teacher / Student model dùng gì
- Nguồn dữ liệu & cấu hình train
- Công thức và cách tính các hàm loss trong `SGDLoss`
- Loss tổng thể và các metric log

Chi tiết kiến trúc loss: [`docs/sgd_loss.md`](sgd_loss.md).

---

## 1) Mục tiêu thử nghiệm

Chạy một vòng train ngắn để kiểm tra:
- Pipeline distillation chạy được với `kd_loss_type="sgd_loss"`
- Unified **batch-level spectral loss** hoạt động (vision/text/cross graphs)
- **Local cross-modal affinity loss** hoạt động (per-sample vision↔text KL)
- Text mapping weighted char-span overlap teacher→student
- Logging/W&B có các key loss mới và các metric số đỉnh batch
- Debug dump (nếu có warning) được ghi ra file

---

## 2) Teacher & Student model

Trong script:
- **Student**: `--model_name "apple/FastVLM-0.5B"`
- **Teacher**: `--teacher_model_name "raghavlite/B3_Qwen2_2B"`

Các backbone/pooling liên quan:
- Student backbone: `--model_backbone "llava_qwen2"`, `--pooling "eos"`
- Teacher backbone: `--teacher_backbone "qwen2_vl"`, `--teacher_pooling "eos"`

LoRA:
- Student: `--lora True`, `--lora_r $LORA_R`, `--lora_alpha $LORA_A`
- Teacher: `--teacher_lora True`, `--teacher_lora_r 8`

---

## 3) Nguồn dữ liệu & cách lấy batch

Dataset:
- `--dataset_name "TIGER-Lab/MMEB-train"`
- `--subset_name ...` (mặc định trong script: `ImageNet_1K`; có thể bật `USE_FULLSET=true` để dùng nhiều subset)
- `--dataset_split "original"`
- `--image_dir "vlm2vec_train/MMEB-train"`
- `--percent_data 0.05` (chỉ dùng 5% dữ liệu để test nhanh)

Batching/epochs:
- `--per_device_train_batch_size $BATCH_SIZE` (mặc định 16)
- `--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS` (mặc định 1)
- `--num_train_epochs 1`

---

## 4) Các hyperparameter của SGD loss

Script đặt các tham số chính:

```bash
KD_WEIGHT=0.05
W_LOSS_V=1.0
W_LOSS_T=0.7
W_LOSS_CROSS=1.0
W_LOSS_LOCAL_CROSS=0.2
LOCAL_CROSS_TEMPERATURE=0.1
```

CLI tương ứng:
- `--kd_loss_type "sgd_loss"`
- `--kd_weight $KD_WEIGHT`
- `--w_loss_v $W_LOSS_V`
- `--w_loss_t $W_LOSS_T`
- `--w_loss_cross $W_LOSS_CROSS`
- `--w_loss_local_cross $W_LOSS_LOCAL_CROSS`
- `--local_cross_temperature $LOCAL_CROSS_TEMPERATURE`

Thiết lập extraction & graph:
- `--grassman_vision_use_cluster $GRASSMAN_VISION_USE_CLUSTER`
- `--grassman_text_use_topk $GRASSMAN_TEXT_USE_TOPK`
- `--topk_text_ratio $TOPK_TEXT_RATIO`
- `--knn_neighbors $KNN_NEIGHBORS`
- `--num_eigenvectors $NUM_EIGENVECTORS`
- `--laplacian_type "$LAPLACIAN_TYPE"`

Lưu ý:
- `knn_neighbors` áp dụng cho **v-v**, **t-t**, và cả **v-t bipartite** (kNN hai chiều).
- `w_loss_batch` **đã bị loại bỏ** (batch-level CKA không còn).

### 4.1 Bảng hyperparameters liên quan đến loss (chi tiết)

Nguồn truth: [`src/criterions/sgd_loss.py`](../src/criterions/sgd_loss.py) và [`src/arguments.py`](../src/arguments.py).

#### A) Hyperparameters chung (tổng loss)

| Hyperparameter | CLI arg | Default (`arguments.py`) | Script set | Ảnh hưởng |
|---|---:|---:|---:|---|
| `kd_weight` | `--kd_weight` | `1.0` | `0.05` | Scale cho `spectral_loss`, `local_cross_loss`; scale nhỏ cho `rkd_loss` (chia 10) |

#### B) Trọng số của spectral loss

| Hyperparameter | CLI arg | Default (`arguments.py`) | Script set | Ảnh hưởng |
|---|---:|---:|---:|---|
| `w_loss_v` | `--w_loss_v` | `1.0` | `1.0` | Trọng số `spectral_loss_v` |
| `w_loss_t` | `--w_loss_t` | `1.0` | `0.7` | Trọng số `spectral_loss_t` |
| `w_loss_cross` | `--w_loss_cross` | `1.0` | `1.0` | Trọng số `spectral_loss_cross` |

#### C) Local cross-modal affinity loss

| Hyperparameter | CLI arg | Default (`arguments.py`) | Script set | Ảnh hưởng |
|---|---:|---:|---:|---|
| `w_loss_local_cross` | `--w_loss_local_cross` | `0.2` | `0.2` | Nhân thêm sau `kd_weight` cho `local_cross_loss` |
| `local_cross_temperature` | `--local_cross_temperature` | `0.1` | `0.1` | Temperature softmax trong ma trận affinity v-t |

Gợi ý tuning:
- Loss quá nhỏ → tăng `w_loss_local_cross` lên `0.5`
- Training dao động → giảm `w_loss_local_cross` xuống `0.05`–`0.1`, hoặc tăng `local_cross_temperature` lên `0.2`

#### D) Extraction (tạo đỉnh) cho spectral graphs & local cross

| Hyperparameter | CLI arg | Default (`arguments.py`) | Script set | Ảnh hưởng |
|---|---:|---:|---:|---|
| `grassman_vision_use_cluster` | `--grassman_vision_use_cluster` | `false` | `True` | Vision nodes = cluster reps (teacher DBSCAN + spatial mapping) |
| `grassman_text_use_topk` | `--grassman_text_use_topk` | `false` | `True` | Text nodes = top-k tokens (sau align) thay vì all text tokens |
| `topk_text_ratio` | `--topk_text_ratio` | `0.8` | `0.8` | \(k = \max(1, \lfloor ratio \cdot M \rfloor)\) trên tensor text đã align |
| `min_samples_dbscan_teacher` | `--min_samples_dbscan_teacher` | `2` | *(không set)* | Ảnh hưởng DBSCAN clustering cho vision (teacher) |

#### E) Graph construction + spectral embedding

| Hyperparameter | CLI arg | Default (`arguments.py`) | Script set | Ảnh hưởng |
|---|---:|---:|---:|---|
| `knn_neighbors` | `--knn_neighbors` | `10` | `10` | kNN cho v-v, t-t; và bipartite kNN cho v-t (2 chiều) |
| `num_eigenvectors` | `--num_eigenvectors` | `16` | `16` | Số eigenvectors dùng để tạo eigenspace (bỏ eigenvector \(v_0\)) |
| `laplacian_type` | `--laplacian_type` | `"unnormalized"` | `"unnormalized"` | Loại Laplacian: unnormalized / normalized |

#### F) Temperature (ảnh hưởng contrastive)

| Hyperparameter | Nguồn | Ảnh hưởng |
|---|---|---|
| `distiller.temperature` | distiller/training args | `contrastive_loss = CE(scores / temperature, target)` |

---

## 5) Loss trong `SGDLoss` gồm những gì?

Code: [`src/criterions/sgd_loss.py`](../src/criterions/sgd_loss.py)

Trong `forward()` hiện tại có **4 nhóm loss chính**:

### 5.1 Contrastive loss (InfoNCE)

Tính từ pooled reps của **student**:
- `student_qry_reps`, `student_pos_reps`

Similarity:
- `scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)`
- `scores / temperature` vào `CrossEntropyLoss`

Kết quả:
- `contrastive_loss`

**Công thức (mức khái niệm):**

Gọi \(s_{ij}\) là similarity giữa query \(i\) và pos \(j\), \(T\) là temperature:

\[
\text{contrastive\_loss}
= \frac{1}{B}\sum_{i=1}^{B}
\mathrm{CE}\Big(\frac{s_{i,:}}{T},\; y_i\Big)
\]

Trong code, \(y_i\) là index của positive đúng trong batch (có điều chỉnh khi gather DDP).

### 5.2 RKD loss (Relational KD)

RKD được tính giữa pooled reps của student và teacher:
- `compute_distance_loss(student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps)`
- `compute_angle_loss(student_qry_reps, student_pos_reps, teacher_qry_reps, teacher_pos_reps)`

Kết hợp:

```text
rkd_loss = (rkd_distance_loss + rkd_angle_loss) / 2
```

Trong total loss, RKD được scale bởi `(kd_weight / 10)`.

**Chi tiết `compute_distance_loss`:**

- Input: concat reps \([qry; pos]\) của student và teacher.
- Tính pairwise distances (trên batch concat), lấy upper-triangular (không tính diagonal).
- Normalize bởi mean distance (teacher & student).
- Dùng Huber (threshold 1.0):

\[
\Delta = d_S - d_T,\quad
L(\Delta)=
\begin{cases}
0.5\Delta^2 & |\Delta|<1 \\\\
|\Delta|-0.5 & \text{otherwise}
\end{cases}
\]

`compute_distance_loss` là mean của \(L(\Delta)\).

**Chi tiết `compute_angle_loss`:**

- Input: concat reps \([qry; pos]\)
- Với mọi triple hợp lệ (không trùng index), tính cosine giữa các hướng sai khác (unit vectors) và so student vs teacher.
- Dùng Huber như trên, rồi mean.

### 5.3 Unified batch-level spectral loss (`spectral_loss`)

Đây là phần thay thế toàn bộ `token_level_loss` và `batch_level_loss` cũ.

Luồng tổng quát (mỗi side `qry` và `pos` tính riêng, rồi average):

1. **Per-sample extraction**
   - Vision: cluster teacher vision tokens (DBSCAN) → weighted cluster mean; map spatial sang student
   - Text: **map teacher→student bằng char-span overlap có trọng số** (`align_student_to_teacher_by_offsets`), rồi top-k trên tensor đã align (cùng indices hai phía)
2. **Đẩy lên batch level**
   - Concat tất cả vision reps của batch → tập đỉnh vision batch
   - Concat tất cả text reps của batch → tập đỉnh text batch
3. **Xây đồ thị & spectral KD**
   - v-v graph: kNN trên vision batch reps
   - t-t graph: kNN trên text batch reps
   - v-t graph: bipartite kNN (kNN hai chiều giữa 2 phía)
   - Từ weight matrix \(W\) → Laplacian → eigenvectors → projection matrix → Grassman loss

Ba thành phần được log riêng:
- `spectral_loss_v` (v-v)
- `spectral_loss_t` (t-t)
- `spectral_loss_cross` (v-t)

Kết hợp theo trọng số:

```text
spectral_loss_side = w_loss_v * spectral_loss_v
                  + w_loss_t * spectral_loss_t
                  + w_loss_cross * spectral_loss_cross

spectral_loss = mean(spectral_loss_qry, spectral_loss_pos)
```

#### 5.3.1 Text mapping (tóm tắt)

1. Build paired character offsets (`build_paired_text_offsets`) — cùng candidate text cho teacher & student, strict khớp `input_ids`
2. Weighted align: mỗi teacher token = tổ hợp có trọng số các student tokens overlap theo ký tự
3. Top-k cosine với last teacher token trên tensor đã align

Sample bị skip nếu offset không khớp (ví dụ chat template lệch `reference_text`).

#### 5.3.2 Xây weight matrix \(W\)

- **v-v / t-t (kNN graph):** với mỗi node \(i\), nối tới \(k\) neighbor gần nhất theo khoảng cách bình phương. Trọng số kiểu RBF:

\[
W_{ij}=\exp\Big(-\frac{\|x_i-x_j\|^2}{\sigma}\Big)
\]

\(\sigma\) lấy theo median của các distance khác 0 (theo code).

- **v-t (bipartite kNN 2 chiều):**
  - mỗi vision node nối tới top-k text gần nhất
  - mỗi text node nối tới top-k vision gần nhất
  - lấy union cạnh từ 2 chiều, rồi đối xứng hoá vào ma trận \(W\) kích thước \((n_v+n_t)\times(n_v+n_t)\).

#### 5.3.3 Laplacian và eigenspace

Gọi \(D\) là degree vector (tổng trọng số theo hàng), \(L\) là Laplacian:
- unnormalized: \(L = \mathrm{diag}(D) - W\)
- normalized: \(L = D^{-1/2}(\mathrm{diag}(D)-W)D^{-1/2}\)

Giải eigen decomposition \(L = V\Lambda V^\top\). Bỏ eigenvector đầu tiên \(v_0\), lấy \(k\) vector tiếp theo:

\[
U = V[:, 1:1+k]
\]

Eigenspace projection matrix:

\[
\Pi = U U^\top
\]

#### 5.3.4 Grassman loss (spectral KD)

\[
\text{grassman}(T,S) = \|\Pi_T - \Pi_S\|_F^2
\]

Trong code, \(\Pi_T\) được detach (teacher không backprop).

### 5.4 Local cross-modal affinity loss (`local_cross_loss`)

Bổ sung **local grounding trong từng sample** — không cần hidden dimension teacher/student giống nhau.

Sau extraction (cùng vision cluster reps + top-k text tokens như §5.3), mỗi sample hợp lệ (`Nv ≥ 2`, `Nt ≥ 2`):

```text
A_T = cos(V_T, T_T) / τ
A_S = cos(V_S, T_S) / τ

L_{v→t} = KL( softmax(A_T, dim=text)  || softmax(A_S, dim=text)  )   # teacher detach
L_{t→v} = KL( softmax(A_T^T, dim=vision) || softmax(A_S^T, dim=vision) )

local_cross_loss_sample = 0.5 * (L_{v→t} + L_{t→v})
```

Average qua sample hợp lệ trong batch, rồi average `qry` và `pos`.

Hàm: `local_cross_affinity_loss()` trong `sgd_loss.py`.  
Temperature \(\tau\) = `local_cross_temperature` (script: `0.1`).

---

## 6) Loss tổng thể

Theo `SGDLoss.forward()`:

```text
total_loss = contrastive_loss
           + (kd_weight / 10) * rkd_loss
           + kd_weight * spectral_loss
           + kd_weight * w_loss_local_cross * local_cross_loss
```

Với giá trị script mặc định (`KD_WEIGHT=0.05`, `W_LOSS_LOCAL_CROSS=0.2`):
- Hệ số thực tế của `local_cross_loss` = `0.05 * 0.2 = 0.01`

`loss_dict["loss"]` chính là `total_loss`.

---

## 7) Metrics log trong train/W&B

Các key chính cho `kd_loss_type="sgd_loss"` được định nghĩa trong:
- [`main.py`](../main.py) → `KD_LOSS_METRIC_KEYS["sgd_loss"]`

Bao gồm:
- `loss`, `contrastive_loss`, `rkd_loss`
- `spectral_loss`, `spectral_loss_v`, `spectral_loss_t`, `spectral_loss_cross`
- `local_cross_loss`
- `batch_vision_nodes_qry`, `batch_text_nodes_qry`, `batch_vision_nodes_pos`, `batch_text_nodes_pos`

---

## 8) Debug dump ghi ra file ở đâu?

Spectral debug không ghi trực tiếp trong `SGDLoss`, mà đi qua:
- [`src/sgd_debug.py`](../src/sgd_debug.py): thu thập/format debug
- [`src/nan_debug.py`](../src/nan_debug.py): `log_sgd_forward_debug()` ghi file

Đường dẫn:
- `{output_dir}/nan_debug/nan_debug.log` (append)
- `{output_dir}/nan_debug/events/step_XXXXXX_SGD_GRASSMAN_DEBUG.log` (mỗi step 1 file nếu có warning)
- `{output_dir}/nan_debug/events/step_XXXXXX_SGD_NAN_DEBUG.log` (nếu có NaN/Inf)

Chỉ ghi khi:
- loss non-finite **hoặc**
- spectral debug có warning (ví dụ: đủ đỉnh mà graph/loss không hợp lệ, hoặc extraction fail — kể cả text offset mismatch)

---

## 9) Checklist chạy thử

1. Chạy script:
   - `bash scripts/cls/train_SGD_fastvlm.sh`
2. Kiểm tra log console / `training/$EXP_NAME/train.log`:
   - Có `train/spectral_loss*`
   - Có `train/local_cross_loss`
   - Có `train/batch_vision_nodes_*`, `train/batch_text_nodes_*`
3. Nếu có warning:
   - Kiểm tra `training/$EXP_NAME/nan_debug/` có `nan_debug.log` và file trong `events/`
4. Nếu `local_cross_loss` ≈ 0 liên tục:
   - Kiểm tra debug sample skip (`offset_token_id_mismatch`, `no_character_overlap_pairs`)
   - Kiểm tra đủ vision clusters (`Nv ≥ 2`) và text top-k (`Nt ≥ 2`)
