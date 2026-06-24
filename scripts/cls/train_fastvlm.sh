#!/bin/bash

# GPU per node
NUM_GPUS_PER_NODE=1
LORA_R=32
LORA_A=64
BATCH_SIZE=12

# Configs
TRAIN_SCRIPT="main.py"
EXP_NAME="FastVLM_cls_entropy_r${LORA_R}_bs${BATCH_SIZE}_traj4"
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
    --student_hidden_dim 896 \
    --teacher_hidden_dim 1536 \
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
    --kd_loss_type "trajectory" \
    --teacher_layer_mapping 0 22 25 28 \
    --student_layer_mapping 0 18 21 24 \
    --teacher_patch_size 28 \
    --student_patch_size 64 \
    --student_resize 1024 \
    --image_resolution "low" \
    --report_to "wandb" \
    --run_name "$EXP_NAME"

echo "========================================================="
echo "Training Completed"
echo "Results saved in training/$EXP_NAME"
echo "========================================================="
