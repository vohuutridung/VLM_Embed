#!/bin/bash

# GPU per node
NUM_GPUS_PER_NODE=1
LORA_R=32
LORA_A=64
# Star-bridge batch graph is O((B·N_tok)^2); start small then scale
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
NUM_TRAIN_EPOCHS=1
PERCENT_DATA=1.0

KD_WEIGHT=1.0

# Spectral KD / Star-Bridge hyperparameters
SEGD_DEPTH_RATIO=0.8
SEGD_ATTN_WINDOW=1
SEGD_INTRA_TOPK=16
SEGD_LAMBDA_NEG=0.3
SEGD_K_NEG=8
SEGD_BRIDGE_TEMPERATURE=1.0
SEGD_K_EIGEN=32

TRAIN_SCRIPT="main.py"
EXP_NAME="SEGD_FastVLM_cls_r${LORA_R}_bs${BATCH_SIZE}_starbridge"
USE_FULLSET=false

echo "========================================================="
echo "Starting SEGD Training (Spectral KD + Star-Bridge)"
echo "========================================================="

if [ "$USE_FULLSET" = true ]; then
    SUBSETS=("ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397")
    echo "Training with FULL dataset set."
else
    SUBSETS=("ImageNet_1K")
    echo "Training with SINGLE dataset (ImageNet_1K)."
fi

torchrun --standalone --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name "apple/FastVLM-0.5B" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --teacher_lora True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_A \
    --teacher_lora_r 8 \
    --teacher_pooling "mean" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "mean" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data $PERCENT_DATA \
    --output_dir "training/$EXP_NAME" \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate 1e-4 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --bf16 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --kd_loss_type "segd_loss" \
    --kd_weight $KD_WEIGHT \
    --segd_depth_ratio $SEGD_DEPTH_RATIO \
    --segd_attn_window $SEGD_ATTN_WINDOW \
    --segd_intra_topk $SEGD_INTRA_TOPK \
    --segd_lambda_neg $SEGD_LAMBDA_NEG \
    --segd_k_neg $SEGD_K_NEG \
    --segd_bridge_temperature $SEGD_BRIDGE_TEMPERATURE \
    --segd_k_eigen $SEGD_K_EIGEN \
    --segd_use_graph_reps_contrastive False \
    --teacher_patch_size 28 \
    --student_patch_size 64 \
    --image_resolution "low" \
    --report_to "wandb" \
    --run_name "$EXP_NAME"

echo "========================================================="
echo "Training Completed"
echo "Results saved in training/$EXP_NAME"
echo "========================================================="
