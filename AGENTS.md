# AGENTS.md

## Project Overview

VLMEmbed — knowledge distillation framework for training compact multimodal embedding models (students) from larger teacher models. Built on top of VLM2Vec and B3 codebases.

## Setup

```bash
python -m venv vlm
source vlm/bin/activate
pip install -r requirements.txt
python fix_lib.py   # patches a Qwen2-VL image processor bug in transformers
```

**`fix_lib.py` is mandatory after install.** It comments out lines 140-143 in `transformers/models/qwen2_vl/image_processing_qwen2_vl.py` inside the venv. Without this, Qwen2-VL models will crash on image preprocessing.

## Data

Training images must be at `vlm2vec_train/MMEB-train/images/`. Download via:

```bash
bash download_traindata.sh
bash download_traindata_2.sh
```

Eval images go in `eval_images/`:

```bash
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
unzip images.zip -d eval_images/
```

## Training

Three training entrypoints with different distributed setups:

- `train.py` — standard HF Trainer (single GPU or DDP via accelerate)
- `train_distillation.py` — DeepSpeed-based distillation (`deepspeed` launcher)
- `train_distill_ddp.py` — pure torch DDP distillation (`torchrun` launcher)
- `train_distill_no_deepspeed.py` — single-GPU or torchrun distillation (no DeepSpeed)

All shell scripts in `scripts/` use `torchrun` (calls `train_distill_no_deepspeed.py` or `train_distill_ddp.py`), except `scripts/train_distill.sh` which uses `deepspeed` (calls `train_distillation.py`).

Key distillation arguments (via `--kd_loss_type`):

- `contrastive_rkd` — RKD distance+angle distillation
- `proposal_dtw` — proposal loss with DTW alignment + projector
- `span_propose`, `span_propose_attn`, `span_propose_attn_only_phrase` — span-based losses
- `em_kd`, `em_kd_llava_ov`, `emo_loss`, `universal_logit` — other KD variants

## Evaluation

```bash
bash eval.sh          # single dataset (MSCOCO by default)
bash eval_all.sh      # SLURM array job over all MMEB subsets
```

`eval_mmeb.py` is the evaluation entrypoint. It deletes `__pycache__` dirs on startup. Model checkpoint paths in `eval.sh` and `eval_2.sh` are hardcoded — update before running.

## Architecture

- `src/arguments.py` — all CLI args (`ModelArguments`, `DataArguments`, `TrainingArguments`, `MTEBArguments`)
- `src/model/model.py` — `MMEBModel`, the core biencoder wrapper
- `src/model/processor.py` — backbone routing (`model_backbone` arg selects VLM type), tokenizer/processor loading, `MODEL2BACKBONE` and `backbone2model` maps
- `src/distiller.py` — `Distiller`, `DistillationDataset`, `DistillationCollator`
- `src/criterions/` — loss functions registered in `criterion_list` dict, selected by `--kd_loss_type`
- `src/loss.py` — contrastive losses (`SimpleContrastiveLoss`, `DistributedContrastiveLoss`, etc.)
- `src/trainer.py` — `MMEBTrainer` and `GradCacheLateProcessTrainer`
- `src/data/` — dataset (`mmeb_dataset`), collator, and loader logic
- `src/model/vlm_backbone/` — VLM backbone implementations (Qwen2-VL, LLaVA, Phi3V, InternVL3, ColPali, GME, etc.)
- `src/model/llava/` — LLaVA-specific model code and FastVLM processor
- `config/` — DeepSpeed configs and projector configs

## Supported `--model_backbone` Values

`phi3_v`, `llava_next`, `llava_onevision`, `qwen2_vl`, `qwen2_5_vl`, `qwen2_vl_tokenselection`, `qwen2_5_vl_tokenselection`, `internvl_chat`, `internvideo2`, `gme`, `lamra`, `colpali`, `llava_qwen2`

## Conventions

- LoRA is enabled via `--lora True` (not a flag, boolean arg). Same for `--teacher_lora True`.
- `--pooling eos` and `--normalize True` are standard for embedding training.
- Image resolution controlled by `--image_resolution low` (uses processor-side resize).
- Projector configs in `config/projector_config*.json` are used when `--projector_config_path` is passed.
- `--tgt_prefix_mod` flips the text prefix for the positive target in evaluation.
- Output dirs are always `training/<experiment_name>`.
- `push_to_hub.py` contains a hardcoded HF token — do not commit changes to it.
