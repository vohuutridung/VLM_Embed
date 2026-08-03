#!/bin/bash
# VLM_Embed — project commands (SEGD Star-Bridge).
#
# Usage:
#   source project_commands.sh && setup_env && download_train_cls
#   ./project_commands.sh segd_graph_audit
#   ./project_commands.sh train_segd_fastvlm
#   ./project_commands.sh help
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

_activate() { source vlm/bin/activate; }

# =============================================================================
# 1. Environment
# =============================================================================
setup_env() {
  uv python install 3.11 2>/dev/null || true
  python3 -m venv vlm
  _activate
  pip install -r requirements.txt
  python fix_lib.py
}

# =============================================================================
# 2. Download data (Hugging Face)
# =============================================================================
download_train_cls() {
  _activate
  mkdir -p vlm2vec_train/MMEB-train/images
  pip install -q hf_transfer 2>/dev/null || true
  export HF_HUB_ENABLE_HF_TRANSFER=1
  for subset in ImageNet_1K N24News HatefulMemes VOC2007 SUN397; do
    echo "Downloading ${subset}..."
    hf download TIGER-Lab/MMEB-train "images_zip/${subset}.zip" --repo-type dataset --local-dir /tmp/mmeb_cls
    unzip -o "/tmp/mmeb_cls/images_zip/${subset}.zip" -d vlm2vec_train/MMEB-train/images/
  done
}

download_eval_images() {
  _activate
  pip install -q hf_transfer 2>/dev/null || true
  export HF_HUB_ENABLE_HF_TRANSFER=1
  hf download TIGER-Lab/MMEB-eval images.zip --repo-type dataset --local-dir .
  mkdir -p eval_images
  unzip -o images.zip -d eval_images/
}

# =============================================================================
# 3. SEGD Star-Bridge graph audit
#    Builds a mini-batch star-bridge graph (teacher) with current edge-weight
#    design: attention → topology, cosine+softmax → affinity.
# =============================================================================
_segd_audit_py() {
  _activate
  export SEGD_AUDIT_ROOT="$ROOT"
  export SEGD_AUDIT_MODE="${SEGD_AUDIT_MODE:-run}"
  export SEGD_AUDIT_OUTPUT="${SEGD_AUDIT_OUTPUT:-results/segd_graph_audit.json}"
  export SEGD_AUDIT_BATCH="${SEGD_AUDIT_BATCH:-16}"
  export SEGD_AUDIT_INTRA_TOPK="${SEGD_AUDIT_INTRA_TOPK:-16}"
  export SEGD_AUDIT_TAU_INTRA="${SEGD_AUDIT_TAU_INTRA:-0.1}"
  export SEGD_AUDIT_TAU_LOCAL="${SEGD_AUDIT_TAU_LOCAL:-0.1}"
  export SEGD_AUDIT_K_NEG="${SEGD_AUDIT_K_NEG:-8}"
  export SEGD_AUDIT_BRIDGE_TEMP="${SEGD_AUDIT_BRIDGE_TEMP:-1.0}"
  export SEGD_AUDIT_LAMBDA_NEG="${SEGD_AUDIT_LAMBDA_NEG:-0.3}"
  export SEGD_AUDIT_K_EIGEN="${SEGD_AUDIT_K_EIGEN:-32}"
  export SEGD_AUDIT_DEPTH="${SEGD_AUDIT_DEPTH:-0.8}"
  export SEGD_AUDIT_WINDOW="${SEGD_AUDIT_WINDOW:-1}"
  python <<'PY'
import json, os, sys
from statistics import mean
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from PIL import Image

ROOT = os.environ["SEGD_AUDIT_ROOT"]
sys.path.insert(0, ROOT)

from src.arguments import DataArguments, ModelArguments
from src.criterions.segd_loss import (
    _extract_side_bundle,
    assemble_graph,
    build_signed_laplacian,
    get_eigenspace,
)
from src.data.dataset.mmeb_dataset import process_image
from src.model.model import MMEBModel
from src.model.processor import PHI3V, QWEN2_VL, VLM_IMAGE_TOKENS, load_processor, process_vlm_inputs_fns


def _dev(inp, device):
    o = dict(inp)
    for k in ("input_ids", "attention_mask"):
        if k in o and isinstance(o[k], torch.Tensor):
            o[k] = o[k].to(device)
    pv = o.get("pixel_values")
    if isinstance(pv, list) and pv and pv[0] is not None:
        o["pixel_values"] = torch.cat([p.to(device) for p in pv], 0)
    elif isinstance(pv, torch.Tensor):
        o["pixel_values"] = pv.to(device)
    ig = o.get("image_grid_thw")
    if isinstance(ig, list) and ig and ig[0] is not None:
        o["image_grid_thw"] = torch.cat(
            [g.to(device) if isinstance(g, torch.Tensor) else torch.tensor(g, device=device) for g in ig], 0
        )
    elif isinstance(ig, torch.Tensor):
        o["image_grid_thw"] = ig.to(device)
    return o


def _print_summary(s: Dict[str, Any]) -> None:
    print("\n=== SEGD Star-Bridge graph audit ===")
    g = s["graph"]
    print(f"  batch_size={g['batch_size']}  n_total={g['n_total']}  n_supernodes={g['n_supernodes']}")
    print(f"  nodes qry: vision={g['vision_nodes_qry']:.0f} text={g['text_nodes_qry']:.0f} "
          f"(cluster={g['cluster_nodes_qry']:.0f})")
    print(f"  nodes pos: vision={g['vision_nodes_pos']:.0f} text={g['text_nodes_pos']:.0f} "
          f"(cluster={g['cluster_nodes_pos']:.0f})")
    e = s["edges"]
    print(f"  edges: nnz={e['nnz']}  pos={e['n_pos']}  neg={e['n_neg']}  "
          f"|w|_mean={e['abs_w_mean']:.4f}  w_pos_mean={e['w_pos_mean']:.4f}  w_neg_mean={e['w_neg_mean']:.4f}")
    print(f"  checks: symmetric={e['is_symmetric']}  no_nan={e['finite']}  "
          f"has_neg_bridge={e['n_neg'] > 0}")
    sp = s["spectrum"]
    print(f"  signed Laplacian: λ_min={sp['eig_min']:.4e}  λ_max={sp['eig_max']:.4e}  "
          f"k_eigen={sp['k_eigen']}  λ_k={sp['eig_k']:.4e}")
    print(f"  attn_layer_center={s['attn_layer_center']}")
    print("  edge weights = cosine softmax (attention only selects intra top-k)\n")


mode = os.environ.get("SEGD_AUDIT_MODE", "run")
out_path = os.environ.get("SEGD_AUDIT_OUTPUT", "results/segd_graph_audit.json")
if mode == "summarize":
    with open(out_path) as f:
        data = json.load(f)
    _print_summary(data["summary"])
    sys.exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

cfg = dict(
    teacher_model="raghavlite/B3_Qwen2_2B",
    teacher_backbone="qwen2_vl",
    dataset="TIGER-Lab/MMEB-train",
    subset="ImageNet_1K",
    split="original",
    image_dir="vlm2vec_train/MMEB-train",
    image_resolution="low",
    batch_size=int(os.environ.get("SEGD_AUDIT_BATCH", 16)),
    depth_ratio=float(os.environ.get("SEGD_AUDIT_DEPTH", 0.8)),
    attn_window=int(os.environ.get("SEGD_AUDIT_WINDOW", 1)),
    intra_topk=int(os.environ.get("SEGD_AUDIT_INTRA_TOPK", 16)),
    tau_intra=float(os.environ.get("SEGD_AUDIT_TAU_INTRA", 0.1)),
    tau_local=float(os.environ.get("SEGD_AUDIT_TAU_LOCAL", 0.1)),
    k_neg=int(os.environ.get("SEGD_AUDIT_K_NEG", 8)),
    bridge_temperature=float(os.environ.get("SEGD_AUDIT_BRIDGE_TEMP", 1.0)),
    lambda_neg=float(os.environ.get("SEGD_AUDIT_LAMBDA_NEG", 0.3)),
    k_eigen=int(os.environ.get("SEGD_AUDIT_K_EIGEN", 32)),
    patch_size=28,
)

ma = ModelArguments(
    model_name=cfg["teacher_model"],
    model_backbone=cfg["teacher_backbone"],
    lora=True, lora_r=8, pooling="mean", normalize=True,
)
da = DataArguments(image_dir=cfg["image_dir"], image_resolution=cfg["image_resolution"])
proc = load_processor(ma, da)
pfn = process_vlm_inputs_fns[cfg["teacher_backbone"]]
print(f"Loading teacher on {device}...")
teacher = MMEBModel.load(ma, is_trainable=False).to(device).eval()
tokenizer = proc.tokenizer if hasattr(proc, "tokenizer") else proc

ds = load_dataset(cfg["dataset"], cfg["subset"], split=cfg["split"])
B = min(cfg["batch_size"], len(ds))


def _load_side(row, text_key, image_key):
    text = row[text_key].replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[QWEN2_VL])
    img = None
    p = row.get(image_key) or ""
    if p:
        full = os.path.join(cfg["image_dir"], p)
        if os.path.isfile(full):
            img = process_image(Image.open(full), cfg["image_resolution"])
    return text, img


qry_texts, qry_imgs, pos_texts, pos_imgs = [], [], [], []
for i in range(B):
    row = ds[i]
    qt, qi = _load_side(row, "qry", "qry_image_path")
    pt, pi = _load_side(row, "pos_text", "pos_image_path")
    qry_texts.append(qt); qry_imgs.append(qi)
    pos_texts.append(pt); pos_imgs.append(pi)

qry_inp = _dev(pfn({"text": qry_texts, "images": qry_imgs}, proc), device)
pos_inp = _dev(pfn({"text": pos_texts, "images": pos_imgs}, proc), device)

with torch.no_grad():
    _, qry_feats, qry_attn, qry_hid = teacher.encode_input(qry_inp, output_attentions=True)
    _, pos_feats, pos_attn, pos_hid = teacher.encode_input(pos_inp, output_attentions=True)

matched_q, matched_p, attn_q, attn_p, mask_q, mask_p = [], [], [], [], [], []
stats = {
    "vision_nodes_q": 0.0, "text_nodes_q": 0.0,
    "vision_nodes_p": 0.0, "text_nodes_p": 0.0,
    "attn_layer_center": -1.0,
}
raw_texts_q = qry_texts
raw_texts_p = pos_texts

for i in range(B):
    q = _extract_side_bundle(
        is_teacher=True, model_input=qry_inp, hidden_states=qry_hid,
        attentions=qry_attn, image_features=qry_feats, image_sizes=None,
        text_strings=raw_texts_q, tokenizer=tokenizer, peer_tokenizer=tokenizer,
        peer_input=qry_inp, patch_size=cfg["patch_size"],
        depth_ratio=cfg["depth_ratio"], attn_window=cfg["attn_window"], sample_idx=i,
    )
    p = _extract_side_bundle(
        is_teacher=True, model_input=pos_inp, hidden_states=pos_hid,
        attentions=pos_attn, image_features=pos_feats, image_sizes=None,
        text_strings=raw_texts_p, tokenizer=tokenizer, peer_tokenizer=tokenizer,
        peer_input=pos_inp, patch_size=cfg["patch_size"],
        depth_ratio=cfg["depth_ratio"], attn_window=cfg["attn_window"], sample_idx=i,
    )
    if q is None or p is None:
        print(f"skip sample {i}: extract failed")
        continue
    matched_q.append(q["tokens"]); matched_p.append(p["tokens"])
    attn_q.append(q["attn"].float()); attn_p.append(p["attn"].float())
    mask_q.append(q["mask"]); mask_p.append(p["mask"])
    stats["vision_nodes_q"] += float(q["num_vision"])
    stats["text_nodes_q"] += float(q["num_text"])
    stats["vision_nodes_p"] += float(p["num_vision"])
    stats["text_nodes_p"] += float(p["num_text"])
    if q["attn_layers"] and stats["attn_layer_center"] < 0:
        stats["attn_layer_center"] = float(q["attn_layers"][len(q["attn_layers"]) // 2])

if not matched_q:
    raise RuntimeError("No valid samples extracted for star-bridge audit")

B_eff = len(matched_q)
with torch.no_grad():
    W, n_total, R_q, R_p = assemble_graph(
        matched_q, matched_p, attn_q, attn_p, mask_q, mask_p,
        topk=cfg["intra_topk"],
        tau_intra=cfg["tau_intra"],
        tau_local=cfg["tau_local"],
        k_neg=cfg["k_neg"],
        bridge_temperature=cfg["bridge_temperature"],
        lambda_neg=cfg["lambda_neg"],
    )
    L = build_signed_laplacian(W, n_total)
    evals, U = get_eigenspace(L, n_total, k=cfg["k_eigen"], allow_lobpcg=True)

# Edge stats
sym_err = float((W - W.t()).abs().max().item())
finite = bool(torch.isfinite(W).all().item())
pos_mask = W > 0
neg_mask = W < 0
nnz = int((W.abs() > 0).sum().item())
n_pos = int(pos_mask.sum().item())
n_neg = int(neg_mask.sum().item())
abs_w = W.abs()
abs_w_mean = float(abs_w[abs_w > 0].mean().item()) if nnz else 0.0
w_pos_mean = float(W[pos_mask].mean().item()) if n_pos else 0.0
w_neg_mean = float(W[neg_mask].mean().item()) if n_neg else 0.0

k_use = int(U.size(1))
eig_vals = evals[:k_use].detach().cpu().tolist() if evals.numel() >= k_use else evals.detach().cpu().tolist()

summary = {
    "graph": {
        "batch_size": B_eff,
        "n_total": int(n_total),
        "n_supernodes": 2 * B_eff,
        "vision_nodes_qry": stats["vision_nodes_q"],
        "text_nodes_qry": stats["text_nodes_q"],
        "vision_nodes_pos": stats["vision_nodes_p"],
        "text_nodes_pos": stats["text_nodes_p"],
        "cluster_nodes_qry": stats["vision_nodes_q"] + stats["text_nodes_q"],
        "cluster_nodes_pos": stats["vision_nodes_p"] + stats["text_nodes_p"],
    },
    "edges": {
        "nnz": nnz,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "abs_w_mean": abs_w_mean,
        "w_pos_mean": w_pos_mean,
        "w_neg_mean": w_neg_mean,
        "is_symmetric": sym_err < 1e-5,
        "sym_err": sym_err,
        "finite": finite,
    },
    "spectrum": {
        "k_eigen": k_use,
        "eig_min": float(evals[0].item()) if evals.numel() else float("nan"),
        "eig_max": float(evals[-1].item()) if evals.numel() else float("nan"),
        "eig_k": float(evals[k_use - 1].item()) if evals.numel() >= k_use else float("nan"),
        "eig_first_k": eig_vals,
    },
    "attn_layer_center": stats["attn_layer_center"],
    "R_q_norm_mean": float(R_q.float().norm(dim=-1).mean().item()),
    "R_p_norm_mean": float(R_p.float().norm(dim=-1).mean().item()),
}

payload = {"config": cfg, "summary": summary}
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote {out_path}")
_print_summary(summary)
PY
}

segd_graph_audit() {
  SEGD_AUDIT_MODE=run
  SEGD_AUDIT_BATCH="${BATCH_SIZE:-16}"
  SEGD_AUDIT_OUTPUT="${OUTPUT:-results/segd_graph_audit.json}"
  SEGD_AUDIT_INTRA_TOPK="${INTRA_TOPK:-16}"
  SEGD_AUDIT_TAU_INTRA="${TAU_INTRA:-0.1}"
  SEGD_AUDIT_TAU_LOCAL="${TAU_LOCAL:-0.1}"
  SEGD_AUDIT_K_NEG="${K_NEG:-8}"
  SEGD_AUDIT_BRIDGE_TEMP="${BRIDGE_TEMP:-1.0}"
  SEGD_AUDIT_LAMBDA_NEG="${LAMBDA_NEG:-0.3}"
  SEGD_AUDIT_K_EIGEN="${K_EIGEN:-32}"
  _segd_audit_py
}

segd_graph_audit_summarize() {
  SEGD_AUDIT_MODE=summarize
  SEGD_AUDIT_OUTPUT="${1:-results/segd_graph_audit.json}"
  _segd_audit_py
}

# Back-compat aliases (old SEKD k_g audit removed)
kg_audit() {
  echo "[deprecated] kg_audit → segd_graph_audit (Star-Bridge)" >&2
  segd_graph_audit "$@"
}
kg_audit_summarize() {
  echo "[deprecated] kg_audit_summarize → segd_graph_audit_summarize" >&2
  segd_graph_audit_summarize "$@"
}

# =============================================================================
# 4. Training
# =============================================================================
train_segd_fastvlm() {
  _activate
  bash scripts/cls/train_SEGD_fastvlm.sh
}

train_segd_smoke() {
  # 1-GPU smoke: small batch, few steps, report_to=none
  _activate
  local BS="${BATCH_SIZE:-2}"
  local STEPS="${MAX_STEPS:-3}"
  local EXP="SEGD_smoke_bs${BS}"
  python -u main.py \
    --model_name "apple/FastVLM-0.5B" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True --teacher_lora True \
    --lora_r 32 --lora_alpha 64 --teacher_lora_r 8 \
    --teacher_pooling mean --pooling mean \
    --teacher_backbone qwen2_vl --model_backbone llava_qwen2 \
    --dataset_name TIGER-Lab/MMEB-train --subset_name ImageNet_1K \
    --dataset_split original --image_dir vlm2vec_train/MMEB-train \
    --percent_data 0.001 \
    --output_dir "training/$EXP" \
    --per_device_train_batch_size "$BS" \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 --max_steps "$STEPS" --bf16 \
    --logging_steps 1 --save_strategy no --seed 42 \
    --normalize True --teacher_normalize True \
    --kd_loss_type segd_loss --kd_weight 1.0 \
    --segd_depth_ratio 0.8 --segd_attn_window 1 \
    --segd_intra_topk 16 --segd_tau_intra 0.1 --segd_tau_local 0.1 \
    --segd_lambda_neg 0.3 --segd_k_neg 8 \
    --segd_bridge_temperature 1.0 --segd_k_eigen 8 \
    --segd_use_graph_reps_contrastive False \
    --teacher_patch_size 28 --student_patch_size 64 \
    --image_resolution low --report_to none --run_name "$EXP"
}

train_sgd_fastvlm() {
  _activate
  bash scripts/cls/train_SGD_fastvlm.sh
}

# =============================================================================
# 5. Eval
# =============================================================================
eval_checkpoint() {
  _activate
  bash scripts/cls/eval.sh
}

# =============================================================================
# CLI dispatcher
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
_cmd="${1:-}"
case "$_cmd" in
  setup_env)                 setup_env ;;
  download_train_cls)        download_train_cls ;;
  download_eval_images)      download_eval_images ;;
  segd_graph_audit)          shift; segd_graph_audit "$@" ;;
  segd_graph_audit_summarize) shift; segd_graph_audit_summarize "$@" ;;
  kg_audit)                  shift; kg_audit "$@" ;;
  kg_audit_summarize)        shift; kg_audit_summarize "$@" ;;
  train_segd_fastvlm)        train_segd_fastvlm ;;
  train_segd_smoke)          train_segd_smoke ;;
  train_sgd_fastvlm)         train_sgd_fastvlm ;;
  eval_checkpoint)           eval_checkpoint ;;
  ""|help)
    cat <<EOF
Usage:
  source project_commands.sh && <function>
  ./project_commands.sh <command>

Commands:
  setup_env                   Create venv, install deps, run fix_lib.py
  download_train_cls          HF download all 5 cls subsets
  download_eval_images        HF download MMEB-eval images
  segd_graph_audit            Audit Star-Bridge graph (nodes / edges / spectrum)
  segd_graph_audit_summarize  Print summary from JSON (optional path)
  train_segd_fastvlm          SEGD train (scripts/cls/train_SEGD_fastvlm.sh, wandb)
  train_segd_smoke            Quick SEGD smoke (B=2, few steps, report_to=none)
  train_sgd_fastvlm           SGD distillation training
  eval_checkpoint             Run cls eval script

Env for segd_graph_audit:
  BATCH_SIZE=16 OUTPUT=results/segd_graph_audit.json
  INTRA_TOPK=16 TAU_INTRA=0.1 TAU_LOCAL=0.1
  K_NEG=8 BRIDGE_TEMP=1.0 LAMBDA_NEG=0.3 K_EIGEN=32

Env for train_segd_smoke:
  BATCH_SIZE=2 MAX_STEPS=3
EOF
    ;;
  *)
    echo "Unknown command: $_cmd (try: ./project_commands.sh help)" >&2
    exit 1
    ;;
esac
fi
