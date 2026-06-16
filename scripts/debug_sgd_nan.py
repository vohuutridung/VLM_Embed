#!/usr/bin/env python3
"""Isolate which SGD loss term produces NaN gradients on one batch."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.distiller import Distiller, DistillationDataset, DistillationCollator
from src.criterions.sgd_loss import SGDLoss
from src.nan_debug import module_grad_stats, unwrap_student


def grad_has_nan(student) -> bool:
    for p in student.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True
    return False


def test_loss_term(name, loss, student):
    student.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        return f"{name}: forward loss non-finite ({loss.item()})"
    loss.backward(retain_graph=True)
    if grad_has_nan(student):
        return f"{name}: NaN/Inf gradients"
    grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        return f"{name}: non-finite grad_norm={grad_norm}"
    return f"{name}: OK (loss={loss.detach().float().item():.4f}, grad_norm={float(grad_norm):.4f})"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = ModelArguments(
        model_name="apple/FastVLM-0.5B",
        teacher_model_name="raghavlite/B3_Qwen2_2B",
        lora=True,
        teacher_lora=True,
        lora_r=32,
        lora_alpha=64,
        teacher_lora_r=8,
        teacher_pooling="eos",
        teacher_backbone="qwen2_vl",
        model_backbone="llava_qwen2",
        pooling="eos",
        normalize=True,
        teacher_normalize=True,
    )
    data_args = DataArguments(
        dataset_name="TIGER-Lab/MMEB-train",
        subset_name=["ImageNet_1K"],
        dataset_split="original",
        image_dir="vlm2vec_train/MMEB-train",
        percent_data=0.05,
        image_resolution="low",
    )
    training_args = TrainingArguments(
        output_dir="training/debug_sgd_nan",
        per_device_train_batch_size=8,
        kd_loss_type="sgd_loss",
        kd_weight=1.0,
        w_loss_v=1.0,
        w_loss_t=1.0,
        w_loss_cross=1.0,
        grassman_vision_use_cluster=True,
        grassman_text_use_topk=True,
        topk_text_ratio=0.8,
        knn_neighbors=10,
        num_eigenvectors=16,
        laplacian_type="unnormalized",
        bf16=True,
        seed=42,
    )

    print("Loading distiller...")
    distiller = Distiller(model_args, training_args)
    distiller = distiller.to(device)
    distiller.train()

    dataset = DistillationDataset(data_args, model_args)
    collator = DistillationCollator(
        student_processor=distiller.get_student_processor(),
        teacher_processor=distiller.get_teacher_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collator, drop_last=True)
    batch = next(iter(loader))

    def to_device(obj, dev):
        if isinstance(obj, torch.Tensor):
            return obj.to(dev)
        if isinstance(obj, dict):
            return {k: to_device(v, dev) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(to_device(v, dev) for v in obj)
        return obj

    batch = to_device(batch, device)

    criterion = SGDLoss(training_args).to(device)
    student = unwrap_student(distiller)

    print("Forward pass...")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
        outputs = criterion(distiller, batch)

    print("\n=== Forward loss values ===")
    for k in ["loss", "contrastive_loss", "rkd_loss", "spectral_loss", "spectral_loss_v", "spectral_loss_t", "spectral_loss_cross"]:
        v = outputs[k]
        print(f"  {k}: {v.detach().float().item()}")

    print("\n=== Per-term backward test ===")
    for name, key in [
        ("contrastive", "contrastive_loss"),
        ("rkd", "rkd_loss"),
        ("spectral", "spectral_loss"),
        ("spectral_v", "spectral_loss_v"),
        ("spectral_t", "spectral_loss_t"),
        ("spectral_cross", "spectral_loss_cross"),
        ("total", "loss"),
    ]:
        print(test_loss_term(name, outputs[key], student))

    print("\n=== Full backward ===")
    student.zero_grad(set_to_none=True)
    outputs["loss"].backward()
    print(module_grad_stats(student, prefix="student"))


if __name__ == "__main__":
    main()
