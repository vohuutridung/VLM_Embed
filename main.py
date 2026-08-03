import json
import os
import sys
import logging
import time
from datetime import timedelta
from typing import Dict, Optional, Tuple

# --- FIX 1: Disable Tokenizer Parallelism/OMP to prevent Deadlock ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, random_split, SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

import wandb
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_scheduler,
)

from src.distiller import Distiller, DistillationCollator, DistillationDataset
from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.criterions import build_criterion
from src.nan_debug import (
    TrainNanDebugger,
    configure_nan_debug_logging,
    get_nan_debug_dir,
    grads_are_finite,
    log_training_output_dirs,
    loss_is_finite,
)

logger = logging.getLogger(__name__)

def is_sgd_loss(training_args: TrainingArguments) -> bool:
    return training_args.kd_loss_type == "sgd_loss"


def is_segd_loss(training_args: TrainingArguments) -> bool:
    return training_args.kd_loss_type == "segd_loss"


def resolve_grounding_warmup_steps(
    training_args: TrainingArguments,
    max_train_steps: int,
) -> int:
    """Explicit warmup_steps wins; else ratio × total optimizer steps."""
    steps = int(training_args.w_loss_grounding_warmup_steps)
    if steps > 0:
        return steps
    ratio = float(getattr(training_args, "w_loss_grounding_warmup_ratio", 0.0))
    if ratio <= 0 or max_train_steps <= 0:
        return 0
    return max(1, int(max_train_steps * ratio))


# Full metric keys for logger / wandb (subset names without train/ prefix).
KD_LOSS_METRIC_KEYS: Dict[str, Tuple[str, ...]] = {
    "sgd_loss": (
        "loss",
        "contrastive_loss",
        "rkd_loss",
        "spectral_loss",
        "spectral_loss_v",
        "spectral_loss_t",
        "spectral_loss_cross",
        "local_cross_loss",
        "batch_vision_nodes_qry",
        "batch_text_nodes_qry",
        "batch_vision_nodes_pos",
        "batch_text_nodes_pos",
    ),
    "segd_loss": (
        "loss",
        "contrastive_loss",
        "segd_loss",
        "spectral_kd_loss",
        "kd_weighted",
        "kd_weight",
        "batch_size",
        "n_total_teacher",
        "n_total_student",
        "n_supernodes",
        "t_vision_nodes_qry",
        "t_text_nodes_qry",
        "t_vision_nodes_pos",
        "t_text_nodes_pos",
        "t_cluster_nodes_qry",
        "t_cluster_nodes_pos",
        "s_vision_nodes_qry",
        "s_text_nodes_qry",
        "s_vision_nodes_pos",
        "s_text_nodes_pos",
        "s_cluster_nodes_qry",
        "s_cluster_nodes_pos",
        "batch_vision_nodes_qry",
        "batch_text_nodes_qry",
        "batch_vision_nodes_pos",
        "batch_text_nodes_pos",
        "segd_attn_layer",
        "segd_k_eigen",
    ),
    "span_propose": (
        "loss",
        "contrastive_loss",
        "span_loss",
        "text_span_loss",
        "vision_cluster_loss",
        "cross_modal_loss",
        "kd_loss_rkd",
    ),
    "span_propose_attn": (
        "loss",
        "contrastive_loss",
        "span_loss",
        "text_span_loss",
        "vision_cluster_loss",
        "cross_modal_loss",
        "kd_loss_rkd",
    ),
    "span_propose_attn_only_phrase": (
        "loss",
        "contrastive_loss",
        "span_loss",
        "text_span_loss",
        "vision_cluster_loss",
        "cross_modal_loss",
        "kd_loss_rkd",
    ),
    "proposal_dtw": (
        "loss",
        "contrastive_loss",
        "kd_loss",
        "kd_loss_rkd",
        "kd_loss_dtw",
        "ot_loss",
    ),
    "proposal_proj": (
        "loss",
        "contrastive_loss",
        "kd_loss",
        "attn_loss",
        "kd_loss_mse",
    ),
    "contrastive_rkd": ("loss", "contrastive_loss", "kd_loss"),
    "emo_loss": ("loss", "contrastive_loss", "kd_loss", "ot_loss"),
    "em_kd": ("loss", "contrastive_loss", "kd_loss"),
    "em_kd_llava_ov": ("loss", "contrastive_loss", "kd_loss"),
    "universal_logit": ("loss", "contrastive_loss", "kd_loss"),
}


def use_wandb(training_args: TrainingArguments) -> bool:
    if not is_main_process():
        return False
    report_to = training_args.report_to
    if report_to is None:
        return False
    if isinstance(report_to, str):
        return report_to == "wandb" or "wandb" in report_to
    return "wandb" in report_to


def init_wandb(
    training_args: TrainingArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
) -> None:
    """Initialize W&B (cloud only, no terminal console capture)."""
    api_key = training_args.wandb_api_key or os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)
    project = (
        getattr(training_args, "project_name", None)
        or os.getenv("WANDB_PROJECT")
        or "vlm_distillation_segd_nothing"
    )
    run = wandb.init(
        project=project,
        name=training_args.run_name or f"run-{int(time.time())}",
        config={
            "model_args": vars(model_args),
            "data_args": vars(data_args),
            "training_args": {
                k: v for k, v in vars(training_args).items()
                if k not in ("wandb_api_key", "distributed_state", "__cached__setup_devices", "deepspeed_plugin")
            },
        },
        settings=wandb.Settings(console="off"),
        reinit=True,
    )
    # Smoke-check that history upload works (visible immediately on the run page).
    wandb.log({"train/wandb_ready": 1}, step=0)
    logger.info(
        "W&B initialized (metrics only; console output disabled). "
        f"project={project} run={run.name} id={run.id} url={run.url}"
    )


def log_wandb_metrics(metrics: dict, step: int) -> None:
    """Log metrics to the active W&B run; warn instead of failing training."""
    if wandb.run is None:
        logger.warning(f"wandb.log skipped at step={step}: no active wandb.run")
        return
    try:
        # Ensure JSON-serializable floats only
        clean = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                clean[k] = float(v.detach().cpu().item())
            elif isinstance(v, (float, int)):
                clean[k] = float(v)
            else:
                continue
        wandb.log(clean, step=step)
    except Exception as exc:
        logger.warning(f"wandb.log failed at step={step}: {exc}")


def configure_student_params(distiller: Distiller, training_args: TrainingArguments) -> None:
    """Enable mm_projector, disable lm_head, cast trainable params to bf16."""
    for n, p in distiller.student.named_parameters():
        if "mm_projector" in n or "multi_modal_projector" in n:
            p.requires_grad = True
        if "lm_head" in n:
            p.requires_grad = False
        if p.requires_grad and training_args.bf16:
            p.data = p.data.to(torch.bfloat16)


def build_lr_scheduler(optimizer, training_args: TrainingArguments, max_train_steps: int):
    warmup_steps = int(training_args.warmup_ratio * max_train_steps)
    scheduler_type = training_args.lr_scheduler_type
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )
    if scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )


def collect_train_metrics(outputs: dict, lr_scheduler, epoch: float) -> dict:
    """Collect training metrics from outputs."""
    metrics = {
        "train/loss": outputs["loss"].item(),
        "train/lr": lr_scheduler.get_last_lr()[0],
        "train/epoch": epoch,
    }
    for k, v in outputs.items():
        if k != "loss" and isinstance(v, torch.Tensor):
            metrics[f"train/{k}"] = v.item()
    return metrics


def format_tqdm_postfix(outputs: dict, lr_scheduler) -> dict:
    """Basic realtime fields for tqdm: total loss and learning rate only."""
    return {
        "loss": f"{outputs['loss'].item():.4f}",
        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
    }


def format_detailed_metrics_log(
    global_step: int, metrics: dict, kd_loss_type: str = ""
) -> str:
    """Full loss breakdown for logger (printed every logging_steps)."""
    meta_parts = [f"step={global_step}"]
    if "train/epoch" in metrics:
        meta_parts.append(f"epoch={metrics['train/epoch']:.4f}")
    if "train/lr" in metrics:
        meta_parts.append(f"lr={metrics['train/lr']:.2e}")

    loss_parts = []
    ordered_keys = []
    if kd_loss_type in KD_LOSS_METRIC_KEYS:
        for short in KD_LOSS_METRIC_KEYS[kd_loss_type]:
            full = f"train/{short}"
            if full in metrics:
                ordered_keys.append(full)
    for key in sorted(metrics.keys()):
        if key.startswith("train/") and key not in ordered_keys:
            if key.replace("train/", "") in ("lr", "epoch"):
                continue
            ordered_keys.append(key)

    for key in ordered_keys:
        short = key.replace("train/", "")
        value = metrics[key]
        if isinstance(value, float):
            loss_parts.append(f"{short}={value:.4f}")
        else:
            loss_parts.append(f"{short}={value}")

    return " | ".join(meta_parts) + " || " + " | ".join(loss_parts)


def collect_eval_metrics(outputs: dict) -> Dict[str, float]:
    metrics = {"eval/loss": outputs["loss"].item()}
    for k, v in outputs.items():
        if k != "loss" and isinstance(v, torch.Tensor):
            metrics[f"eval/{k}"] = v.item()
    return metrics


def format_eval_log_line(global_step: int, epoch: float, metrics: dict) -> str:
    parts = [f"step={global_step}", f"epoch={epoch:.4f}"]
    for key in sorted(metrics):
        short = key.replace("eval/", "")
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f"{short}={value:.4f}")
        else:
            parts.append(f"{short}={value}")
    return "eval | " + " | ".join(parts)


def setup_logging(training_args: TrainingArguments, output_dir: str) -> Optional[str]:
    """Configure root logging: terminal (rank 0) + train.log file."""
    log_level = logging.INFO
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    datefmt = "%m/%d/%Y %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=datefmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    log_path = None
    if is_main_process():
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(log_level)
        root.addHandler(stream_handler)

        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "train.log")
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.ERROR)
        root.addHandler(stream_handler)

    logger.setLevel(log_level)

    import transformers
    transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    return log_path


def ddp_setup() -> None:
    if not dist.is_initialized() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))

def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def to_device(obj, device):
    if obj is None:
        return None
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [to_device(v, device) for v in obj]
        return tuple(result) if isinstance(obj, tuple) else result
    elif hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj

def download_artifacts(model_args: ModelArguments):
    logger.info("  [Pre-load] Starting artifact download...")
    if model_args.model_name:
        try:
            AutoConfig.from_pretrained(model_args.model_name)
            AutoTokenizer.from_pretrained(model_args.model_name)
            AutoProcessor.from_pretrained(model_args.model_name)
        except Exception as e:
            logger.warning(f"  [Pre-load] Warning: Some artifacts failed to pre-download: {e}")
    logger.info("  [Pre-load] Finished.")

# --- CHANGED: Updated save_checkpoint to handle specific folder names (for best model) ---
def save_checkpoint(
    output_dir: str,
    epoch: int,
    distiller: nn.Module,
    model_args: ModelArguments,
    step: Optional[int] = None,
    folder_name: Optional[str] = None,
) -> None:
    """Saves model checkpoint. Supports custom folder name for best model."""
    if not is_main_process():
        return

    if folder_name:
        ckpt_dir = os.path.join(output_dir, folder_name)
    elif step is not None:
         ckpt_dir = os.path.join(output_dir, f"checkpoint-step-{step}")
    else:
        ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
         
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"Saving checkpoint to {ckpt_dir}...")

    # Unwrap DDP model
    model_to_save = distiller.module if hasattr(distiller, "module") else distiller
    student = model_to_save.student
    
    # Save encoder/adapter
    if hasattr(student, "peft_config"):
        student.save_pretrained(ckpt_dir)
        logger.info("Saved LoRA adapter model.")
    else:
        if hasattr(student, "encoder"):
            student.encoder.save_pretrained(ckpt_dir)
        else:
            try:
                student.save_pretrained(ckpt_dir)
            except:
                torch.save(student.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
        logger.info("Saved student model.")
    
    # Save Projector
    projector_dir = os.path.join(ckpt_dir, "mm_projector.pth")
    try:
        projector_weights = None
        if hasattr(student, "encoder") and hasattr(student.encoder, "model"):
             if hasattr(student.encoder.model, "multi_modal_projector"):
                 projector_weights = student.encoder.model.multi_modal_projector.state_dict()
             elif hasattr(student.encoder.model, "model") and hasattr(student.encoder.model.model, "mm_projector"):
                 projector_weights = student.encoder.model.model.mm_projector.state_dict()
        
        if projector_weights is not None:
            torch.save(projector_weights, projector_dir)
    except AttributeError:
        pass

    # Save tokenizer and config
    try:
        if model_args.model_name:
            AutoTokenizer.from_pretrained(model_args.model_name).save_pretrained(ckpt_dir)
            AutoConfig.from_pretrained(model_args.model_name).save_pretrained(ckpt_dir)
            try:
                AutoProcessor.from_pretrained(model_args.model_name).save_pretrained(ckpt_dir)
            except Exception:
                pass
    except Exception:
        pass

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Computes average validation metrics over the eval dataloader."""
    model.eval()
    metric_sums: Dict[str, float] = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Validating",
            disable=not is_main_process(),
            mininterval=1.0,
            leave=False,
        ):
            batch = to_device(batch, device)
            outputs = model(criterion, batch)
            if not isinstance(outputs, dict):
                outputs = {"loss": outputs}
            batch_metrics = collect_eval_metrics(outputs)
            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value
            num_batches += 1

    if num_batches == 0:
        return {"eval/loss": 0.0}

    avg_metrics = {key: value / num_batches for key, value in metric_sums.items()}

    if dist.is_initialized():
        for key in avg_metrics:
            metric_tensor = torch.tensor(avg_metrics[key], device=device)
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
            avg_metrics[key] = metric_tensor.item() / dist.get_world_size()

    model.train()
    return avg_metrics


def run_validation(
    distiller: nn.Module,
    eval_dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    global_step: int,
    epoch: float,
    training_args: TrainingArguments,
    best_val_loss: float,
    model_args: ModelArguments,
    use_wandb_logging: bool,
) -> float:
    """Run validation, log metrics, and save best checkpoint if improved."""
    eval_metrics = evaluate(distiller, eval_dataloader, criterion, device)
    val_loss = eval_metrics.get("eval/loss", float("inf"))

    if is_main_process():
        logger.info(format_eval_log_line(global_step, epoch, eval_metrics))
        if use_wandb_logging:
            log_wandb_metrics(eval_metrics, global_step)

        if val_loss < best_val_loss:
            logger.info(
                f"New best validation model (loss: {val_loss:.4f} < {best_val_loss:.4f}) "
                f"-> saving checkpoint-best"
            )
            best_val_loss = val_loss
            save_checkpoint(
                training_args.output_dir,
                epoch=int(epoch),
                distiller=distiller,
                model_args=model_args,
                step=global_step,
                folder_name="checkpoint-best",
            )
        else:
            logger.info(
                f"Validation loss {val_loss:.4f} did not improve best {best_val_loss:.4f}"
            )

    if dist.is_initialized():
        dist.barrier()

    return best_val_loss

def main():
    ddp_setup() # Initialize DDP for distributed training

    # Parse arguments by HfArgumentParser
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    output_dir = training_args.output_dir or "."
    setup_logging(training_args, output_dir)
    nan_debug_dir = configure_nan_debug_logging(output_dir)
    use_sgd_loss = is_sgd_loss(training_args)
    use_segd_loss = is_segd_loss(training_args)
    wandb_enabled = use_wandb(training_args)

    if wandb_enabled:
        init_wandb(training_args, model_args, data_args)

    train_log_path = os.path.join(output_dir, "train.log")
    log_training_output_dirs(train_log_path, nan_debug_dir)

    # Artifact Sync
    logger.info("Handling artifact downloading...") # Download artifacts from Hugging Face
    if dist.is_initialized():
        if is_main_process():
            download_artifacts(model_args)
        dist.barrier()
    else:
        download_artifacts(model_args)

    # --- Dataset Splitting Logic ---
    logger.info("Preparing dataset...")
    full_dataset = DistillationDataset(data_args, model_args) # Load dataset
    
    train_dataset = full_dataset
    eval_dataset = None
    
    if data_args.val_split_ratio > 0: # Split dataset into training and validation sets
        val_size = int(len(full_dataset) * data_args.val_split_ratio)
        train_size = len(full_dataset) - val_size
        logger.info(f"Splitting dataset: {train_size} training, {val_size} validation.")
        
        # Use a fixed generator for reproducibility across ranks
        generator = torch.Generator().manual_seed(42)
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    else:
        logger.info("No validation split ratio provided. Using full dataset for training.")

    # Samplers
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        # For eval, we don't necessarily need to shuffle, but we need DistributedSampler 
        # to split data across GPUs to speed up eval
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if eval_dataset else None
    else:
        train_sampler = None
        eval_sampler = SequentialSampler(eval_dataset) if eval_dataset else None

    # Load Model
    logger.info("Loading Distiller model...") # Load Distiller model
    distiller = Distiller(model_args, training_args)

    # Collator
    collator = DistillationCollator(
        student_processor=distiller.get_student_processor(),
        teacher_processor=distiller.get_teacher_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collator,
        drop_last=True,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=True,
    )
    
    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size or training_args.per_device_train_batch_size,
            sampler=eval_sampler,
            shuffle=False,
            collate_fn=collator,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=True
        )

    # Optimizer setup
    logger.info("Setting up optimizer...")
    configure_student_params(distiller, training_args)

    optimizer = AdamW(
        distiller.student.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    if (
        model_args.projector_config_path is not None
        and not use_sgd_loss
        and not use_segd_loss
        and hasattr(distiller, "add_optimizer_param_group")
    ):
        optimizer = distiller.add_optimizer_param_group(optimizer)

    # Move to device and wrap DDP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dist.is_initialized():
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    
    distiller = distiller.to(device)
    find_unused = use_sgd_loss or use_segd_loss or model_args.projector_config_path is not None
    if dist.is_initialized():
        distiller = DDP(
            distiller,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            find_unused_parameters=find_unused,
        )

    # Scheduler
    num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = build_lr_scheduler(optimizer, training_args, max_train_steps)

    criterion = build_criterion(training_args).to(device)
    if hasattr(criterion, "configure_grounding_warmup"):
        grounding_warmup_steps = resolve_grounding_warmup_steps(
            training_args, max_train_steps
        )
        criterion.configure_grounding_warmup(grounding_warmup_steps)
    nan_debugger = TrainNanDebugger(distiller)

    # Training Stats
    if is_main_process():
        logger.info("***** Running training *****")
        logger.info(f"  KD loss type = {training_args.kd_loss_type}")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Eval examples = {len(eval_dataset) if eval_dataset else 0}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        if hasattr(criterion, "w_loss_grounding_warmup_steps"):
            logger.info(
                f"  Grounding warmup steps = {criterion.w_loss_grounding_warmup_steps}"
            )
        logger.info(f"  Val split ratio = {data_args.val_split_ratio}")
        logger.info(f"  Eval step = {training_args.eval_steps}")
        logger.info(f"  Output dir = {training_args.output_dir}")
        logger.info(f"  W&B enabled = {wandb_enabled}")

    global_step = 0
    best_val_loss = float('inf') # Track best loss
    last_eval_step = -1

    for epoch in range(int(training_args.num_train_epochs)):
        if dist.is_initialized():
            train_dataloader.sampler.set_epoch(epoch)
        
        distiller.train()
        epoch_iterator = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not is_main_process(),
            mininterval=1.0,
        )
        
        for step, batch in enumerate(epoch_iterator):
            batch = to_device(batch, device)
            step_num = global_step + 1
            epoch_step = step + 1
            nan_debugger.annotate_step(training_args, step_num, epoch_step)

            if hasattr(criterion, "set_training_step"):
                criterion.set_training_step(global_step)

            outputs = distiller(criterion, batch)
            loss = outputs["loss"] / training_args.gradient_accumulation_steps
            finite_loss = loss_is_finite(loss)

            if not finite_loss:
                nan_debugger.before_backward(step_num, epoch_step, outputs)
            loss.backward()
            if not finite_loss:
                nan_debugger.after_backward(step_num, epoch_step, outputs)

            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                grad_norm = None
                if training_args.max_grad_norm is not None and training_args.max_grad_norm > 0:
                    grad_norm = nan_debugger.clip_gradients(
                        step_num, epoch_step, outputs, training_args.max_grad_norm
                    )

                grad_finite = (
                    torch.isfinite(grad_norm)
                    if grad_norm is not None
                    else grads_are_finite(distiller.student)
                )
                should_step = finite_loss and grad_finite

                if should_step:
                    optimizer.step()
                elif is_main_process():
                    logger.warning(
                        f"Skipping optimizer step at global_step={global_step + 1} "
                        f"(finite_loss={finite_loss}, grad_finite={grad_finite})"
                    )

                if not finite_loss:
                    nan_debugger.after_optimizer_step(step_num, epoch_step, outputs)
                if should_step:
                    lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main_process():
                    epoch_iterator.set_postfix(
                        **format_tqdm_postfix(outputs, lr_scheduler),
                        refresh=False,
                    )

                if is_main_process() and global_step % training_args.logging_steps == 0:
                    metrics = collect_train_metrics(
                        outputs,
                        lr_scheduler,
                        epoch + (step + 1) / len(train_dataloader),
                    )
                    detail_line = format_detailed_metrics_log(
                        global_step, metrics, training_args.kd_loss_type
                    )
                    logger.info(detail_line)
                    if wandb_enabled:
                        log_wandb_metrics(metrics, global_step)

                # Periodic validation
                eval_steps = training_args.eval_steps or 0
                if (
                    eval_dataloader is not None
                    and eval_steps > 0
                    and global_step % eval_steps == 0
                ):
                    best_val_loss = run_validation(
                        distiller,
                        eval_dataloader,
                        criterion,
                        device,
                        global_step,
                        epoch + (step + 1) / len(train_dataloader),
                        training_args,
                        best_val_loss,
                        model_args,
                        wandb_enabled,
                    )
                    last_eval_step = global_step

        # End-of-epoch validation (when val split is enabled)
        if eval_dataloader is not None and last_eval_step != global_step:
            epoch_progress = epoch + 1
            best_val_loss = run_validation(
                distiller,
                eval_dataloader,
                criterion,
                device,
                global_step,
                epoch_progress,
                training_args,
                best_val_loss,
                model_args,
                wandb_enabled,
            )

        # End of epoch Saving
        save_checkpoint(training_args.output_dir, epoch + 1, distiller, model_args)
        if is_main_process():
            logger.info(f"Epoch {epoch + 1}/{training_args.num_train_epochs} completed.")

        if dist.is_initialized():
            dist.barrier()

    if is_main_process():
        logger.info("Training completed.")
        if eval_dataloader is not None:
            if best_val_loss < float("inf"):
                logger.info(f"Best validation loss: {best_val_loss:.4f} (checkpoint-best)")
            else:
                logger.info("Validation was run but no improvement over initial best loss.")
        logger.info(f"Full log saved to {train_log_path}")
        nan_debug_dir = get_nan_debug_dir()
        if nan_debug_dir:
            logger.info(f"NaN debug logs saved to {nan_debug_dir}/")
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()

    # Final Save
    save_checkpoint(training_args.output_dir, int(training_args.num_train_epochs), distiller, model_args, folder_name="checkpoint-final")
    if dist.is_initialized():
        dist.barrier()  # <--- Crucial: Wait for Rank 0 to finish saving!
    # =========================================================================
    
    if wandb_enabled:
        wandb.finish()

    cleanup_ddp()

if __name__ == "__main__":
    main()