#!/bin/bash
# =========================================================================
# Evaluate student BASE (apple/FastVLM-0.5B) on MMEB-eval classification subsets.
#
# Prerequisites (from repo root):
#   wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
#   unzip images.zip -d eval_images/
#
# Usage:
#   bash scripts/cls/eval_base.sh
#   bash scripts/cls/eval_base.sh [OUTPUT_DIR] [SUBSET] [with_prefix|no_prefix]
#
# Examples:
#   bash scripts/cls/eval_base.sh
#   bash scripts/cls/eval_base.sh eval_outputs/FastVLM_base_cls "ImageNet-1K"
#   bash scripts/cls/eval_base.sh eval_outputs/FastVLM_base_cls all with_prefix
#   bash scripts/cls/eval_base.sh "" ImageNet-1K no_prefix   # default no-prefix output dir
#   bash scripts/cls/eval_base.sh eval_outputs/my_run ImageNet-1K no_prefix
#
# PREFIX mode (arg 3):
#   with_prefix  — add "Represent the class label: " to target text (MMEB default)
#   no_prefix    — encode raw class names only
# =========================================================================

set -euo pipefail

cd "$(dirname "$0")/../.."

NUM_GPUS_PER_NODE=1
BATCH_SIZE=16
BASE_MODEL="apple/FastVLM-0.5B"

PREFIX_MODE="${3:-with_prefix}"
SUBSET_ARG="${2:-ImageNet-1K}"
USE_FULLSET=false

if [ "$PREFIX_MODE" != "with_prefix" ] && [ "$PREFIX_MODE" != "no_prefix" ]; then
    echo "Error: arg3 must be 'with_prefix' or 'no_prefix', got: ${PREFIX_MODE}"
    exit 1
fi

if [ -n "${1:-}" ]; then
    OUTPUT_DIR="$1"
elif [ "$PREFIX_MODE" = "no_prefix" ]; then
    OUTPUT_DIR="eval_outputs/FastVLM-0.5B_base_cls_no_prefix"
else
    OUTPUT_DIR="eval_outputs/FastVLM-0.5B_base_cls"
fi

if [ -x "vlm/bin/python" ]; then
    PYTHON="vlm/bin/python"
else
    PYTHON="python3"
fi

if [ ! -d "eval_images" ] || [ -z "$(ls -A eval_images 2>/dev/null)" ]; then
    echo "Error: eval_images/ is missing or empty."
    echo "Download eval images first:"
    echo "  bash scripts/cls/download_evaldata.sh"
    exit 1
fi

if [ "$SUBSET_ARG" = "all" ]; then
    if [ "$USE_FULLSET" = true ]; then
        SUBSETS=("ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211")
    else
        SUBSETS=("ImageNet-1K")
    fi
    echo "Evaluating with FULL dataset set."
else
    # shellcheck disable=SC2206
    SUBSETS=($SUBSET_ARG)
    echo "Evaluating subset(s): ${SUBSETS[*]}"
fi

echo "========================================================="
echo "Evaluating student BASE model"
echo "Model:       ${BASE_MODEL}"
echo "Output:      ${OUTPUT_DIR}"
echo "Subset:      ${SUBSETS[*]}"
echo "Prefix mode: ${PREFIX_MODE}"
echo "========================================================="

EVAL_ARGS=(
    --model_name "${BASE_MODEL}"
    --encode_output_path "${OUTPUT_DIR}"
    --dataset_name "TIGER-Lab/MMEB-eval"
    --subset_name "${SUBSETS[@]}"
    --dataset_split "test"
    --per_device_eval_batch_size "${BATCH_SIZE}"
    --image_dir "eval_images/"
    --image_resolution "low"
    --pooling "eos"
    --model_backbone "llava_qwen2"
    --normalize True
    --bf16
    --lora False
)

if [ "$PREFIX_MODE" = "with_prefix" ]; then
    EVAL_ARGS+=(--tgt_prefix_mod)
fi

if [ "${NUM_GPUS_PER_NODE}" -gt 1 ]; then
    echo "Using multi-GPU mode with accelerate"
    accelerate launch --multi_gpu --num_processes="${NUM_GPUS_PER_NODE}" eval_mmeb.py "${EVAL_ARGS[@]}"
else
    echo "Using single GPU mode"
    "${PYTHON}" eval_mmeb.py "${EVAL_ARGS[@]}"
fi

echo "========================================================="
echo "Evaluation completed"
echo "Scores: ${OUTPUT_DIR}/*_score.json"
echo "========================================================="
