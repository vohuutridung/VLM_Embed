"""Teacher/student token-count alignment for visual and text modalities.

Visual: 2D patch-box overlap in normalized image coords ``[0, 1]``.
Text: 1D character-span overlap on shared semantic content.

Both alignment matrices are student-anchored and row-stochastic.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

import torch

QWEN_VISION_TOKEN_ID_MIN = 151643
QWEN_VISION_TOKEN_ID_MAX = 151656
IMAGE_TOKEN_INDEX = -200

_DEFAULT_PLACEHOLDER_STRINGS = frozenset({
    "<|image_1|>",
    "<image>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<video>",
})


@dataclass
class VisualPreprocessMeta:
    """Processed canvas size used to build visual-token boxes."""

    canvas_width: float
    canvas_height: float
    original_width: Optional[float] = None
    original_height: Optional[float] = None

    @property
    def scale_x(self) -> float:
        if self.original_width is None or self.original_width <= 0:
            return 1.0
        return self.canvas_width / self.original_width

    @property
    def scale_y(self) -> float:
        if self.original_height is None or self.original_height <= 0:
            return 1.0
        return self.canvas_height / self.original_height


@dataclass
class VisualTokenAlignment:
    """Student-anchored visual alignment for one image/sample."""

    num_student_tokens: int
    num_teacher_tokens: int
    overlap_matrix: torch.Tensor
    valid_student: torch.Tensor
    student_boxes: torch.Tensor
    teacher_boxes: torch.Tensor


def infer_vision_grid(num_tokens: int) -> Optional[Tuple[int, int]]:
    """Return ``(num_rows, num_cols)`` with ``num_rows * num_cols == num_tokens``."""
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


def build_patch_boxes_grid(
    grid_rows: int,
    grid_cols: int,
    patch_size: float,
    canvas_width: float,
    canvas_height: float,
    *,
    normalized: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build axis-aligned patch boxes for a dense ``grid_rows x grid_cols`` layout.

    Returns:
        Tensor of shape ``(grid_rows * grid_cols, 4)`` as ``[x0, y0, x1, y1]``.
    """
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError(f"grid_rows and grid_cols must be positive, got {(grid_rows, grid_cols)}")

    token_idx = torch.arange(grid_rows * grid_cols, device=device, dtype=torch.long)
    rows = token_idx // grid_cols
    cols = token_idx % grid_cols

    x0 = cols.to(dtype) * patch_size
    y0 = rows.to(dtype) * patch_size
    x1 = x0 + patch_size
    y1 = y0 + patch_size
    boxes = torch.stack([x0, y0, x1, y1], dim=-1)

    if normalized:
        boxes = normalize_boxes_to_canvas(boxes, canvas_width, canvas_height)
    return boxes


def build_visual_token_boxes(
    num_tokens: int,
    patch_size: float,
    meta: VisualPreprocessMeta,
    *,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    normalized: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build visual-token boxes from token count and preprocessing metadata."""
    if grid_rows is None or grid_cols is None:
        grid = infer_vision_grid(num_tokens)
        if grid is None:
            raise ValueError(f"Cannot infer vision grid for num_tokens={num_tokens}")
        grid_rows, grid_cols = grid
        if grid_rows * grid_cols != num_tokens:
            raise ValueError(
                f"Inferred grid {(grid_rows, grid_cols)} does not match num_tokens={num_tokens}"
            )

    return build_patch_boxes_grid(
        grid_rows,
        grid_cols,
        patch_size,
        meta.canvas_width,
        meta.canvas_height,
        normalized=normalized,
        device=device,
        dtype=dtype,
    )


def normalize_boxes_to_canvas(
    boxes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
) -> torch.Tensor:
    """Map pixel-space boxes to normalized ``[0, 1]`` canvas coordinates."""
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError(f"canvas size must be positive, got {(canvas_width, canvas_height)}")

    scale = boxes.new_tensor([canvas_width, canvas_height, canvas_width, canvas_height])
    return (boxes / scale).clamp(0.0, 1.0)


def map_boxes_to_original_normalized(
    boxes_normalized: torch.Tensor,
    meta: VisualPreprocessMeta,
) -> torch.Tensor:
    """Map canvas-normalized boxes to original-image normalized coordinates.

    Assumes independent axis scaling from original image to processed canvas
    (typical resize without letterboxing on each axis).
    """
    if meta.original_width is None or meta.original_height is None:
        return boxes_normalized

    sx = meta.scale_x
    sy = meta.scale_y
    if sx <= 0 or sy <= 0:
        return boxes_normalized

    x0 = boxes_normalized[:, 0] * meta.canvas_width / meta.original_width
    y0 = boxes_normalized[:, 1] * meta.canvas_height / meta.original_height
    x1 = boxes_normalized[:, 2] * meta.canvas_width / meta.original_width
    y1 = boxes_normalized[:, 3] * meta.canvas_height / meta.original_height
    return torch.stack([x0, y0, x1, y1], dim=-1).clamp(0.0, 1.0)


def box_intersection_area(
    student_boxes: torch.Tensor,
    teacher_boxes: torch.Tensor,
) -> torch.Tensor:
    """Pairwise intersection area between student and teacher boxes.

    Args:
        student_boxes: ``(R_v^S, 4)``
        teacher_boxes: ``(R_v^T, 4)``

    Returns:
        Overlap matrix ``(R_v^S, R_v^T)``.
    """
    if student_boxes.dim() != 2 or student_boxes.size(-1) != 4:
        raise ValueError(f"student_boxes must be (N, 4), got {tuple(student_boxes.shape)}")
    if teacher_boxes.dim() != 2 or teacher_boxes.size(-1) != 4:
        raise ValueError(f"teacher_boxes must be (M, 4), got {tuple(teacher_boxes.shape)}")

    s0 = student_boxes[:, None, :2]
    s1 = student_boxes[:, None, 2:]
    t0 = teacher_boxes[None, :, :2]
    t1 = teacher_boxes[None, :, 2:]

    wh = (torch.minimum(s1, t1) - torch.maximum(s0, t0)).clamp(min=0.0)
    return wh[..., 0] * wh[..., 1]


def build_visual_alignment_matrix(
    student_boxes: torch.Tensor,
    teacher_boxes: torch.Tensor,
    *,
    student_valid_mask: Optional[torch.Tensor] = None,
    teacher_valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build row-stochastic visual alignment matrix ``A_v`` and valid student rows.

    ``A_v[a, i] = O_{a,i} / (sum_k O_{a,k} + eps)`` where ``O`` is intersection area.
    Rows with zero total overlap are marked invalid.
    """
    overlap = box_intersection_area(student_boxes.float(), teacher_boxes.float())

    if teacher_valid_mask is not None:
        if teacher_valid_mask.shape != (teacher_boxes.size(0),):
            raise ValueError(
                f"teacher_valid_mask shape {tuple(teacher_valid_mask.shape)} "
                f"!= ({teacher_boxes.size(0)},)"
            )
        overlap = overlap * teacher_valid_mask.to(dtype=overlap.dtype).unsqueeze(0)

    row_sum = overlap.sum(dim=1)
    valid = row_sum > 0
    if student_valid_mask is not None:
        if student_valid_mask.shape != (student_boxes.size(0),):
            raise ValueError(
                f"student_valid_mask shape {tuple(student_valid_mask.shape)} "
                f"!= ({student_boxes.size(0)},)"
            )
        valid = valid & student_valid_mask.to(dtype=torch.bool)

    a_v = overlap / row_sum.clamp(min=eps).unsqueeze(1)
    a_v = a_v * valid.unsqueeze(1).to(dtype=a_v.dtype)

    if dtype is not None:
        a_v = a_v.to(dtype=dtype)
    return a_v, valid


def align_visual_tokens(
    student_boxes: torch.Tensor,
    teacher_boxes: torch.Tensor,
    *,
    student_valid_mask: Optional[torch.Tensor] = None,
    teacher_valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    dtype: Optional[torch.dtype] = None,
) -> VisualTokenAlignment:
    """End-to-end visual alignment from normalized student/teacher boxes."""
    a_v, valid_student = build_visual_alignment_matrix(
        student_boxes,
        teacher_boxes,
        student_valid_mask=student_valid_mask,
        teacher_valid_mask=teacher_valid_mask,
        eps=eps,
        dtype=dtype,
    )
    return VisualTokenAlignment(
        num_student_tokens=student_boxes.size(0),
        num_teacher_tokens=teacher_boxes.size(0),
        overlap_matrix=a_v,
        valid_student=valid_student,
        student_boxes=student_boxes,
        teacher_boxes=teacher_boxes,
    )


def align_visual_tokens_from_grids(
    *,
    num_student_tokens: int,
    num_teacher_tokens: int,
    student_patch_size: float,
    teacher_patch_size: float,
    student_meta: VisualPreprocessMeta,
    teacher_meta: VisualPreprocessMeta,
    student_grid: Optional[Tuple[int, int]] = None,
    teacher_grid: Optional[Tuple[int, int]] = None,
    map_to_original: bool = True,
    student_valid_mask: Optional[torch.Tensor] = None,
    teacher_valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> VisualTokenAlignment:
    """Build boxes from grid metadata, optionally map to original image, then align."""
    s_rows, s_cols = student_grid or infer_vision_grid(num_student_tokens) or (0, 0)
    t_rows, t_cols = teacher_grid or infer_vision_grid(num_teacher_tokens) or (0, 0)

    student_boxes = build_patch_boxes_grid(
        s_rows,
        s_cols,
        student_patch_size,
        student_meta.canvas_width,
        student_meta.canvas_height,
        normalized=True,
        device=device,
    )
    teacher_boxes = build_patch_boxes_grid(
        t_rows,
        t_cols,
        teacher_patch_size,
        teacher_meta.canvas_width,
        teacher_meta.canvas_height,
        normalized=True,
        device=device,
    )

    if map_to_original:
        student_boxes = map_boxes_to_original_normalized(student_boxes, student_meta)
        teacher_boxes = map_boxes_to_original_normalized(teacher_boxes, teacher_meta)

    return align_visual_tokens(
        student_boxes,
        teacher_boxes,
        student_valid_mask=student_valid_mask,
        teacher_valid_mask=teacher_valid_mask,
        eps=eps,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Text token alignment (student-anchored character-span overlap)
# ---------------------------------------------------------------------------


@dataclass
class TextTokenAlignment:
    """Student-anchored text alignment for one sample."""

    num_student_tokens: int
    num_teacher_tokens: int
    overlap_matrix: torch.Tensor
    valid_student: torch.Tensor
    student_offsets: torch.Tensor
    teacher_offsets: torch.Tensor


def _all_placeholder_strings() -> Set[str]:
    return set(_DEFAULT_PLACEHOLDER_STRINGS)


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
    *,
    exclude_eos: bool = True,
) -> Set[int]:
    """Token ids to exclude from semantic text alignment (special, vision, pad)."""
    ids: Set[int] = set(range(QWEN_VISION_TOKEN_ID_MIN, QWEN_VISION_TOKEN_ID_MAX + 1))
    ids.add(IMAGE_TOKEN_INDEX)
    if extra_ids is not None:
        ids.update(int(x) for x in extra_ids)
    if hasattr(tokenizer, "all_special_ids"):
        ids.update(int(x) for x in tokenizer.all_special_ids)
    for attr in ("pad_token_id", "bos_token_id", "unk_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            ids.add(int(tid))
    if exclude_eos:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            ids.add(int(eos_id))
    return ids


def tokenize_semantic_with_offsets(
    tokenizer,
    semantic_text: str,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Tokenize semantic text; filter ``(0, 0)`` offsets and keep ids aligned."""
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


def filter_semantic_token_offsets(
    token_ids: List[int],
    offsets: List[Tuple[int, int]],
    exclude_ids: Set[int],
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Drop tokens whose ids appear in ``exclude_ids``."""
    kept_ids: List[int] = []
    kept_offsets: List[Tuple[int, int]] = []
    for tid, off in zip(token_ids, offsets):
        if int(tid) in exclude_ids:
            continue
        kept_ids.append(int(tid))
        kept_offsets.append(off)
    return kept_ids, kept_offsets


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


def semantic_text_to_offsets(
    tokenizer,
    semantic_text: str,
    *,
    exclude_eos: bool = True,
    extra_exclude_ids: Optional[Iterable[int]] = None,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Tokenize shared semantic content and return ``(N, 2)`` character offsets."""
    semantic_text = extract_semantic_text(semantic_text)
    sem_ids, sem_offsets = tokenize_semantic_with_offsets(tokenizer, semantic_text)
    if not sem_offsets:
        return None

    exclude_ids = build_non_semantic_id_set(
        tokenizer,
        extra_ids=extra_exclude_ids,
        exclude_eos=exclude_eos,
    )
    _, sem_offsets = filter_semantic_token_offsets(sem_ids, sem_offsets, exclude_ids)
    if not sem_offsets:
        return None

    return torch.tensor(sem_offsets, device=device, dtype=torch.long)


def extract_semantic_offsets_from_input(
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    semantic_text: str,
    *,
    exclude_eos: bool = True,
    extra_exclude_ids: Optional[Iterable[int]] = None,
) -> Optional[torch.Tensor]:
    """Locate shared semantic span inside a templated input and return its offsets.

    Offsets refer to the normalized semantic text character space, not template tokens.
    """
    semantic_text = extract_semantic_text(semantic_text)
    non_sem = build_non_semantic_id_set(
        tokenizer,
        extra_ids=extra_exclude_ids,
        exclude_eos=exclude_eos,
    )
    text_ids = get_text_token_ids_from_input(input_ids, attention_mask, non_sem)
    sem_ids, sem_offsets = tokenize_semantic_with_offsets(tokenizer, semantic_text)
    sem_ids, sem_offsets = filter_semantic_token_offsets(sem_ids, sem_offsets, non_sem)
    if not sem_ids or not sem_offsets:
        return None

    start = find_subsequence(text_ids, sem_ids)
    if start is None:
        return None

    return torch.tensor(sem_offsets, device=input_ids.device, dtype=torch.long)


def span_overlap_length(
    student_offsets: torch.Tensor,
    teacher_offsets: torch.Tensor,
) -> torch.Tensor:
    """Pairwise 1D span-overlap lengths between student and teacher text tokens.

    Args:
        student_offsets: ``(R_t^S, 2)`` as ``[start, end)``
        teacher_offsets: ``(R_t^T, 2)`` as ``[start, end)``

    Returns:
        Overlap matrix ``(R_t^S, R_t^T)``.
    """
    if student_offsets.dim() != 2 or student_offsets.size(-1) != 2:
        raise ValueError(f"student_offsets must be (N, 2), got {tuple(student_offsets.shape)}")
    if teacher_offsets.dim() != 2 or teacher_offsets.size(-1) != 2:
        raise ValueError(f"teacher_offsets must be (M, 2), got {tuple(teacher_offsets.shape)}")

    s0 = student_offsets[:, 0:1].float()
    s1 = student_offsets[:, 1:2].float()
    t0 = teacher_offsets[:, 0].unsqueeze(0).float()
    t1 = teacher_offsets[:, 1].unsqueeze(0).float()
    return (torch.minimum(s1, t1) - torch.maximum(s0, t0)).clamp(min=0.0)


def build_text_alignment_matrix(
    student_offsets: torch.Tensor,
    teacher_offsets: torch.Tensor,
    *,
    student_valid_mask: Optional[torch.Tensor] = None,
    teacher_valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build row-stochastic text alignment matrix ``A_t`` and valid student rows.

    ``A_t[b, j] = O_{b,j} / (sum_k O_{b,k} + eps)`` where ``O`` is span-overlap length.
    Rows with zero total overlap are marked invalid.
    """
    overlap = span_overlap_length(student_offsets, teacher_offsets)

    if teacher_valid_mask is not None:
        if teacher_valid_mask.shape != (teacher_offsets.size(0),):
            raise ValueError(
                f"teacher_valid_mask shape {tuple(teacher_valid_mask.shape)} "
                f"!= ({teacher_offsets.size(0)},)"
            )
        overlap = overlap * teacher_valid_mask.to(dtype=overlap.dtype).unsqueeze(0)

    row_sum = overlap.sum(dim=1)
    valid = row_sum > 0
    if student_valid_mask is not None:
        if student_valid_mask.shape != (student_offsets.size(0),):
            raise ValueError(
                f"student_valid_mask shape {tuple(student_valid_mask.shape)} "
                f"!= ({student_offsets.size(0)},)"
            )
        valid = valid & student_valid_mask.to(dtype=torch.bool)

    a_t = overlap / row_sum.clamp(min=eps).unsqueeze(1)
    a_t = a_t * valid.unsqueeze(1).to(dtype=a_t.dtype)

    if dtype is not None:
        a_t = a_t.to(dtype=dtype)
    return a_t, valid


def align_text_tokens(
    student_offsets: torch.Tensor,
    teacher_offsets: torch.Tensor,
    *,
    student_valid_mask: Optional[torch.Tensor] = None,
    teacher_valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    dtype: Optional[torch.dtype] = None,
) -> TextTokenAlignment:
    """End-to-end text alignment from student/teacher character-span offsets."""
    a_t, valid_student = build_text_alignment_matrix(
        student_offsets,
        teacher_offsets,
        student_valid_mask=student_valid_mask,
        teacher_valid_mask=teacher_valid_mask,
        eps=eps,
        dtype=dtype,
    )
    return TextTokenAlignment(
        num_student_tokens=student_offsets.size(0),
        num_teacher_tokens=teacher_offsets.size(0),
        overlap_matrix=a_t,
        valid_student=valid_student,
        student_offsets=student_offsets,
        teacher_offsets=teacher_offsets,
    )


def align_text_tokens_from_semantic(
    teacher_tokenizer,
    student_tokenizer,
    semantic_text: str,
    *,
    exclude_eos: bool = True,
    extra_teacher_exclude_ids: Optional[Iterable[int]] = None,
    extra_student_exclude_ids: Optional[Iterable[int]] = None,
    student_valid_mask: Optional[torch.Tensor] = None,
    teacher_valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Optional[TextTokenAlignment]:
    """Align teacher/student text tokens on shared semantic content only.

    Tokenizes ``semantic_text`` independently for each tokenizer so different
    prompt templates do not affect ``A_t`` as long as the content span matches.
    """
    teacher_offsets = semantic_text_to_offsets(
        teacher_tokenizer,
        semantic_text,
        exclude_eos=exclude_eos,
        extra_exclude_ids=extra_teacher_exclude_ids,
        device=device,
    )
    student_offsets = semantic_text_to_offsets(
        student_tokenizer,
        semantic_text,
        exclude_eos=exclude_eos,
        extra_exclude_ids=extra_student_exclude_ids,
        device=device,
    )
    if teacher_offsets is None or student_offsets is None:
        return None

    alignment = align_text_tokens(
        student_offsets,
        teacher_offsets,
        student_valid_mask=student_valid_mask,
        teacher_valid_mask=teacher_valid_mask,
        eps=eps,
        dtype=dtype,
    )
    if not alignment.valid_student.any():
        return None
    return alignment


def align_text_tokens_from_input(
    teacher_tokenizer,
    student_tokenizer,
    teacher_input_ids: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    student_input_ids: torch.Tensor,
    student_attention_mask: torch.Tensor,
    semantic_text: str,
    *,
    exclude_eos: bool = True,
    extra_teacher_exclude_ids: Optional[Iterable[int]] = None,
    extra_student_exclude_ids: Optional[Iterable[int]] = None,
    student_valid_mask: Optional[torch.Tensor] = None,
    teacher_valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    dtype: Optional[torch.dtype] = None,
) -> Optional[TextTokenAlignment]:
    """Align text tokens by locating the shared semantic span inside each input."""
    teacher_offsets = extract_semantic_offsets_from_input(
        teacher_tokenizer,
        teacher_input_ids,
        teacher_attention_mask,
        semantic_text,
        exclude_eos=exclude_eos,
        extra_exclude_ids=extra_teacher_exclude_ids,
    )
    student_offsets = extract_semantic_offsets_from_input(
        student_tokenizer,
        student_input_ids,
        student_attention_mask,
        semantic_text,
        exclude_eos=exclude_eos,
        extra_exclude_ids=extra_student_exclude_ids,
    )
    if teacher_offsets is None or student_offsets is None:
        return None

    alignment = align_text_tokens(
        student_offsets,
        teacher_offsets,
        student_valid_mask=student_valid_mask,
        teacher_valid_mask=teacher_valid_mask,
        eps=eps,
        dtype=dtype,
    )
    if not alignment.valid_student.any():
        return None
    return alignment


# ---------------------------------------------------------------------------
# Hidden-state extraction (Qwen2-VL layout: teacher [pad|vision|text], student [vision|text|pad])
# ---------------------------------------------------------------------------


def count_text_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
) -> int:
    """Count non-vision semantic text tokens in one sample."""
    non_semantic_ids = build_non_semantic_id_set(tokenizer)
    return len(get_text_token_ids_from_input(input_ids, attention_mask, non_semantic_ids))


def infer_image_size_from_vision_tokens(
    num_vision_tokens: int,
    patch_size: float,
) -> Tuple[float, float]:
    """Infer original image (width, height) from teacher vision token count."""
    grid = infer_vision_grid(num_vision_tokens)
    if grid is None:
        side = int(round(num_vision_tokens ** 0.5))
        size = float(side * patch_size)
        return size, size
    rows, cols = grid
    return float(cols * patch_size), float(rows * patch_size)


def extract_vision_hidden_single(
    hidden_states: Tuple[torch.Tensor, ...],
    sample_idx: int,
    num_vision_tokens: int,
    num_text_tokens: int,
    is_teacher: bool,
    layer_idx: int = -1,
) -> torch.Tensor:
    """Slice vision-token hidden states from one layer for a single sample."""
    layer_hidden = hidden_states[layer_idx]
    if num_vision_tokens <= 0:
        return layer_hidden.new_zeros(0, layer_hidden.size(-1))

    if is_teacher:
        start_idx = -(num_vision_tokens + num_text_tokens)
        end_idx = -num_text_tokens if num_text_tokens > 0 else None
        return layer_hidden[sample_idx, start_idx:end_idx, :]
    return layer_hidden[sample_idx, :num_vision_tokens, :]


def extract_text_region(
    layer_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    num_vision_tokens: int,
    is_teacher: bool,
    has_image: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Slice hidden states and ids to the text region (same index space)."""
    non_semantic_ids = build_non_semantic_id_set(tokenizer)
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
        text_hidden = layer_hidden[-num_text:, :] if is_teacher else layer_hidden[:num_text, :]

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
    semantic_text = extract_semantic_text(semantic_text)
    sem_ids, _ = tokenize_semantic_with_offsets(tokenizer, semantic_text)
    if not sem_ids:
        return None

    region_ids = text_region_input_ids.tolist()
    if len(region_ids) != text_region_hidden.size(0):
        return None

    start = find_subsequence(region_ids, sem_ids)
    if start is None:
        return None

    end = start + len(sem_ids)
    return text_region_hidden[start:end, :], text_region_input_ids[start:end]


def align_text_hiddens_from_layer(
    teacher_layer_hidden: torch.Tensor,
    student_layer_hidden: torch.Tensor,
    teacher_input_ids: torch.Tensor,
    student_input_ids: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    student_attention_mask: torch.Tensor,
    teacher_tokenizer,
    student_tokenizer,
    semantic_text: str,
    num_vision_teacher: int,
    num_vision_student: int,
    has_image: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return student/teacher text hiddens plus ``A_t`` and valid student rows."""
    text_align = align_text_tokens_from_input(
        teacher_tokenizer,
        student_tokenizer,
        teacher_input_ids,
        teacher_attention_mask,
        student_input_ids,
        student_attention_mask,
        semantic_text,
        dtype=student_layer_hidden.dtype,
    )
    if text_align is None or not text_align.valid_student.any():
        return None

    teacher_region = extract_text_region(
        teacher_layer_hidden,
        teacher_input_ids,
        teacher_attention_mask,
        teacher_tokenizer,
        num_vision_teacher,
        is_teacher=True,
        has_image=has_image,
    )
    student_region = extract_text_region(
        student_layer_hidden,
        student_input_ids,
        student_attention_mask,
        student_tokenizer,
        num_vision_student,
        is_teacher=False,
        has_image=has_image,
    )
    if teacher_region is None or student_region is None:
        return None

    teacher_aligned = align_semantic_tokens_to_hidden(
        teacher_region[1], teacher_region[0], teacher_tokenizer, semantic_text
    )
    student_aligned = align_semantic_tokens_to_hidden(
        student_region[1], student_region[0], student_tokenizer, semantic_text
    )
    if teacher_aligned is None or student_aligned is None:
        return None

    student_text, _ = student_aligned
    teacher_text, _ = teacher_aligned
    if (
        student_text.size(0) != text_align.num_student_tokens
        or teacher_text.size(0) != text_align.num_teacher_tokens
    ):
        return None

    return student_text, teacher_text, text_align.overlap_matrix, text_align.valid_student
