#!/bin/bash

# =========================================================================
# Classification evaluation on MMEB-eval
# Run from repo root:
#   bash scripts/cls/eval.sh
#   bash scripts/cls/eval.sh [MODEL_PATH] [OUTPUT_DIR] [SUBSET]
#   bash scripts/cls/eval.sh training/RKD/checkpoint-final eval_outputs/RKD all
# SUBSET: space-separated subset names, or "all" for full MMEB-eval list
# =========================================================================

LORA_R=64
LORA_A=128
BATCH_SIZE=16
EXP_NAME="SGD_FastVLM_full_cls_r${LORA_R}_bs${BATCH_SIZE}"

MODEL_PATH="${1:-training/${EXP_NAME}/checkpoint-final}"
OUTPUT_DIR="${2:-eval_outputs/${EXP_NAME}}"
SUBSET="${3:-ImageNet-1K N24News HatefulMemes VOC2007 SUN397}"
# SUBSET="OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W"

echo "========================================================="
echo "Starting Evaluation"
echo "Model:  ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Subset: ${SUBSET}"
echo "========================================================="

if [ "$SUBSET" == "all" ]; then
    SUBSETS=("Wiki-SS-NQ" "VisDial" "CIRR" "VisualNews_t2i" "VisualNews_i2t" "MSCOCO_t2i" "MSCOCO_i2t" "NIGHTS" "WebQA" "OVEN" "FashionIQ" "EDIS" "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W" "ScienceQA" "GQA" "TextVQA" "VizWiz" "ImageNet-1K" "HatefulMemes" "SUN397" "N24News" "VOC2007" "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211" "MSCOCO" "RefCOCO" "RefCOCO-Matching" "Visual7W-Pointing")
else
    IFS=' ' read -r -a SUBSETS <<< "$SUBSET"
fi

NUM_GPUS=1
echo "Detected ${NUM_GPUS} GPU(s)"

EVAL_ARGS=(
    --model_name "${MODEL_PATH}"
    --encode_output_path "${OUTPUT_DIR}"
    --dataset_name "TIGER-Lab/MMEB-eval"
    --subset_name "${SUBSETS[@]}"
    --dataset_split "test"
    --per_device_eval_batch_size 128
    --image_dir "eval_images/"
    --image_resolution "low"
    --pooling "eos"
    --model_backbone "llava_qwen2"
    --normalize True
    --bf16
    --tgt_prefix_mod
    --lora True
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_A}"
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
