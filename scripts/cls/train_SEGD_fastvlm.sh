#!/bin/bash

# GPU per node
NUM_GPUS_PER_NODE=1
LORA_R=32
LORA_A=64
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
NUM_TRAIN_EPOCHS=1
PERCENT_DATA=1.0

KD_WEIGHT=1.0
W_LOSS_CKA=1.0
# CKA global embedding: "mean" | "last" | "eos"
CKA_POOLING="last"
W_LOSS_V=1.0
W_LOSS_T=0.7
W_LOSS_CROSS=1.0
# Semantic Grounding Distillation — direct KL on G_vt vision–text affinity
W_LOSS_GROUNDING=0.5
SEKD_GROUNDING_TEMP=0.1
# Grounding weight warmup: 15% tổng optimizer steps (tính trong main.py từ max_train_steps)
W_LOSS_GROUNDING_WARMUP_RATIO=0.15

# SEKD hyperparameters
SEKD_K_MIN=2
SEKD_K_MAX=16
SEKD_EIG_EPS=1e-6
SEKD_ALIGN_GRID_H=10
SEKD_ALIGN_GRID_W=10
KNN_NEIGHBORS=10

TRAIN_SCRIPT="main.py"
EXP_NAME="SEGD_FastVLM_cls_r${LORA_R}_bs${BATCH_SIZE}_cka${CKA_POOLING}"
USE_FULLSET=false

echo "========================================================="
echo "Starting SEGD Training (CKA pooling: ${CKA_POOLING})"
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
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
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
    --w_loss_cka $W_LOSS_CKA \
    --cka_pooling "$CKA_POOLING" \
    --w_loss_v $W_LOSS_V \
    --w_loss_t $W_LOSS_T \
    --w_loss_cross $W_LOSS_CROSS \
    --w_loss_grounding $W_LOSS_GROUNDING \
    --w_loss_grounding_warmup_ratio $W_LOSS_GROUNDING_WARMUP_RATIO \
    --sekd_grounding_temp $SEKD_GROUNDING_TEMP \
    --knn_neighbors $KNN_NEIGHBORS \
    --sekd_k_min $SEKD_K_MIN \
    --sekd_k_max $SEKD_K_MAX \
    --sekd_eig_eps $SEKD_EIG_EPS \
    --sekd_align_grid_h $SEKD_ALIGN_GRID_H \
    --sekd_align_grid_w $SEKD_ALIGN_GRID_W \
    --image_resolution "low" \
    --report_to "wandb" \
    --run_name "$EXP_NAME"

echo "========================================================="
echo "Training Completed"
echo "Results saved in training/$EXP_NAME"
echo "========================================================="
