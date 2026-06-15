#!/bin/bash

# GPU per node
NUM_GPUS_PER_NODE=1
LORA_R=32
LORA_A=64
BATCH_SIZE=8

KD_WEIGHT=1.0
W_LOSS_V=1.0
W_LOSS_T=1.0
W_LOSS_CROSS=1.0

# Spectral loss (unified batch-level Grassman KD)
GRASSMAN_VISION_USE_CLUSTER=True
GRASSMAN_TEXT_USE_TOPK=True
TOPK_TEXT_RATIO=0.8
KNN_NEIGHBORS=10
NUM_EIGENVECTORS=16
LAPLACIAN_TYPE="unnormalized"

# Configs
TRAIN_SCRIPT="main.py"
EXP_NAME="SGD_FastVLM_full_cls_r${LORA_R}_bs${BATCH_SIZE}"
USE_FULLSET=false

echo "========================================================="
echo "Starting Training"
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
    --percent_data 0.05 \
    --output_dir "training/$EXP_NAME" \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
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
    --kd_loss_type "sgd_loss" \
    --kd_weight $KD_WEIGHT \
    --w_loss_v $W_LOSS_V \
    --w_loss_t $W_LOSS_T \
    --w_loss_cross $W_LOSS_CROSS \
    --grassman_vision_use_cluster $GRASSMAN_VISION_USE_CLUSTER \
    --grassman_text_use_topk $GRASSMAN_TEXT_USE_TOPK \
    --topk_text_ratio $TOPK_TEXT_RATIO \
    --knn_neighbors $KNN_NEIGHBORS \
    --num_eigenvectors $NUM_EIGENVECTORS \
    --laplacian_type "$LAPLACIAN_TYPE" \
    --image_resolution "low" \
    --report_to "wandb" \
    --run_name "$EXP_NAME"

echo "========================================================="
echo "Training Completed"
echo "Results saved in training/$EXP_NAME"
echo "========================================================="