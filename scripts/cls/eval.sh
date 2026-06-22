#!/bin/bash

# =========================================================================
# Classification evaluation on MMEB-eval
# Run from repo root:
#   bash scripts/cls/eval.sh
#   bash scripts/cls/eval.sh [MODEL_PATH] [OUTPUT_DIR] [SUBSET]
#   bash scripts/cls/eval.sh training/RKD/checkpoint-final eval_outputs/RKD all
# SUBSET: space-separated subset names, or "all" for full MMEB-eval list
# =========================================================================

# GPU per node
NUM_GPUS_PER_NODE=1
BATCH_SIZE=16

MODEL_PATH="${1:-training/${EXP_NAME}/checkpoint-final}"
OUTPUT_DIR="${2:-eval_outputs/${EXP_NAME}}"
USE_FULLSET=false

echo "========================================================="
echo "Starting Evaluation"
echo "Model:  ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Subset: ${SUBSET}"
echo "========================================================="

if [ "$USE_FULLSET" == true ]; then
    SUBSETS=("ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211")
    echo "Evaluating with FULL dataset set."
else
    SUBSETS=("ImageNet-1K")
    echo "Evaluating with SINGLE dataset (ImageNet-1K)."
fi

echo "Detected ${NUM_GPUS_PER_NODE} GPU(s)"

EVAL_ARGS=(
    --model_name "${MODEL_PATH}"
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
    --tgt_prefix_mod
    --lora True
)

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using multi-GPU mode with accelerate"
    accelerate launch --multi_gpu --num_processes="${NUM_GPUS}" eval_mmeb.py "${EVAL_ARGS[@]}"
else
    echo "Using single GPU mode"
    python3 eval_mmeb.py "${EVAL_ARGS[@]}"
fi

echo "========================================================="
echo "Evaluation Completed"
echo "Results saved in ${OUTPUT_DIR}"
echo "========================================================="