#!/bin/bash
# VLM_Embed — all project commands in one file.
# No wandb, no external upload. Data/models via Hugging Face download only.
#
# Usage:
#   source project_commands.sh && setup_env && download_train_cls && kg_audit
#   ./project_commands.sh kg_audit
#   ./project_commands.sh train_segd_fastvlm
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
# 3. k_g spectral audit (Python embedded — logic from segd_loss.py)
# =============================================================================
_kg_audit_py() {
  _activate
  export KG_ROOT="$ROOT"
  export KG_MODE="${KG_MODE:-run}"
  export KG_OUTPUT="${KG_OUTPUT:-results/kg_audit.json}"
  export KG_NUM_SAMPLES="${KG_NUM_SAMPLES:-32}"
  python <<'PY'
import json, math, os, sys
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple
import torch
from datasets import load_dataset
from PIL import Image

ROOT = os.environ["KG_ROOT"]
sys.path.insert(0, ROOT)
from src.arguments import DataArguments, ModelArguments
from src.data.dataset.mmeb_dataset import process_image
from src.model.model import MMEBModel
from src.model.processor import PHI3V, QWEN2_VL, VLM_IMAGE_TOKENS, load_processor, process_vlm_inputs_fns

_EPS, _HEAT = 1e-6, 1e-8
_IMG_LO, _IMG_HI = 151643, 151656

def _l2(h): return h / h.norm(p=2, dim=-1, keepdim=True).clamp_min(_EPS)
def _d2(h):
    n2 = (h*h).sum(-1, keepdim=True)
    return (n2 + n2.t() - 2*(h@h.t())).clamp_min(0.)
def _knn(h, k):
    n = h.size(0)
    if n < 2: return torch.zeros(n, n, device=h.device, dtype=h.dtype)
    k = min(max(1,k), n-1); h32 = h.float(); d2 = _d2(h32)
    with torch.no_grad():
        kd, ki = torch.topk(d2, k=k+1, largest=False, dim=1)
        ki, kd = ki[:,1:], kd[:,1:]
        sig = kd[:,-1].clamp_min(_HEAT).sqrt()
        m = torch.zeros(n,n,device=h.device,dtype=torch.bool)
        m.scatter_(1, ki, True); m = m|m.t(); m.fill_diagonal_(False)
    den = (sig.unsqueeze(1)*sig.unsqueeze(0)).clamp_min(_HEAT)
    w = torch.exp(-d2/den); w = torch.where(m, w, torch.zeros_like(w))
    w = 0.5*(w+w.t()); w.fill_diagonal_(0.); return w.to(dtype=h.dtype)
def _bip(hv, ht):
    c = (hv.float()@ht.float().t()).clamp_min(0.)
    nv, nt = c.shape; w = torch.zeros(nv+nt, nv+nt, device=c.device)
    w[:nv, nv:] = c; w[nv:, :nv] = c.t(); return w
def _lap(w): d = w.float().sum(1); return torch.diag(d) - w.float()
def _null(ev, eps): return int((ev <= eps).sum().item())
def _sel(ev, c, kmin, kmax):
    n = ev.numel(); mx = n - c - 1
    if mx < kmin:
        a = max(0, n-c); return max(1, min(a, kmax)) if a > 0 else 0
    ke = min(kmax, mx)
    if ke < kmin: return max(1, ke)
    bm, bg = kmin, -1.
    with torch.no_grad():
        for m in range(kmin, ke+1):
            g = float(ev[c+m]-ev[c+m-1])
            if g > bg: bg, bm = g, m
    return int(bm)
def _kg(h, knn, kmin, kmax, eps):
    n = h.size(0)
    if n < 2: return 0.
    ev,_ = torch.linalg.eigh(_lap(_knn(_l2(h.float()), knn)) + 1e-4*torch.eye(n, device=h.device))
    ev = ev.clamp_min(0.); c = _null(ev, eps); return float(_sel(ev, c, kmin, kmax))
def _kg_vt(hv, ht, kmin, kmax, eps):
    if hv.size(0)<1 or ht.size(0)<1: return 0.
    nv = hv.size(0); h = torch.cat([_l2(hv.float()), _l2(ht.float())], 0); n = h.size(0)
    ev,_ = torch.linalg.eigh(_lap(_bip(h[:nv], h[nv:])) + 1e-4*torch.eye(n, device=h.device))
    ev = ev.clamp_min(0.); c = _null(ev, eps); return float(_sel(ev, c, kmin, kmax))
def _nt(ids): return int(((ids<_IMG_LO)|(ids>_IMG_HI)).sum().item())
def _tok(hidden, n_v, n_t):
    h = hidden[-1][0]
    if n_t <= 0: return None, None
    tt = h[-n_t:,:]
    if n_v <= 0: return None, tt
    return h[-(n_v+n_t):-n_t,:], tt
def _dev(inp, device):
    o = dict(inp)
    for k in ("input_ids","attention_mask"):
        if k in o and isinstance(o[k], torch.Tensor): o[k] = o[k].to(device)
    pv = o.get("pixel_values")
    if isinstance(pv, list) and pv and pv[0] is not None:
        o["pixel_values"] = torch.cat([p.to(device) for p in pv], 0)
    ig = o.get("image_grid_thw")
    if isinstance(ig, list) and ig and ig[0] is not None:
        o["image_grid_thw"] = torch.cat([g.to(device) if isinstance(g,torch.Tensor) else torch.tensor(g,device=device) for g in ig], 0)
    return o
def _corr(xs, ys):
    if len(xs)<2: return None
    mx, my = mean(xs), mean(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    return num/den if den>1e-12 else None
def _stats(rows, k, nk):
    v = [r[k] for r in rows if r.get(k,0)>0]; nd = [r[nk] for r in rows if r.get(k,0)>0]
    if not v: return {"mean":0.,"std":0.,"n":0,"mean_nodes":0.,"corr_nodes_k_g":None}
    return {"mean":mean(v),"std":pstdev(v) if len(v)>1 else 0.,"n":len(v),"mean_nodes":mean(nd),"corr_nodes_k_g":_corr(nd,v)}
def _print_summary(s):
    print("\n=== k_g audit summary ===")
    for g in ("v","t","vt"):
        x = s[f"k_g_{g}"]; c = x.get("corr_nodes_k_g")
        cs = f"{c:+.3f}" if c is not None else "n/a"
        print(f"  G_{g}: mean k_g={x['mean']:.2f} ± {x['std']:.2f}  (n={x['n']}, mean_nodes={x['mean_nodes']:.1f}, corr(nodes,k_g)={cs})")
    print(f"  overall_mean_k_g={s['overall_mean_k_g']:.2f}")
    print("  (corr < 0 on G_t → k_g may be capped by few text tokens)\n")

mode = os.environ.get("KG_MODE", "run")
out_path = os.environ.get("KG_OUTPUT", "results/kg_audit.json")
if mode == "summarize":
    with open(out_path) as f: data = json.load(f)
    _print_summary(data["summary"]); sys.exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
cfg = dict(teacher_model="raghavlite/B3_Qwen2_2B", teacher_backbone="qwen2_vl",
    dataset="TIGER-Lab/MMEB-train", subset="ImageNet_1K", split="original",
    image_dir="vlm2vec_train/MMEB-train", image_resolution="low",
    num_samples=int(os.environ.get("KG_NUM_SAMPLES", 32)),
    knn=10, kmin=2, kmax=16, eig_eps=1e-6)
ma = ModelArguments(model_name=cfg["teacher_model"], model_backbone=cfg["teacher_backbone"],
    lora=True, lora_r=8, pooling="eos", normalize=True)
da = DataArguments(image_dir=cfg["image_dir"], image_resolution=cfg["image_resolution"])
proc = load_processor(ma, da); pfn = process_vlm_inputs_fns[cfg["teacher_backbone"]]
print(f"Loading teacher on {device}...")
teacher = MMEBModel.load(ma, is_trainable=False).to(device).eval()
ds = load_dataset(cfg["dataset"], cfg["subset"], split=cfg["split"])
n = min(cfg["num_samples"], len(ds)); rows = []
for i in range(n):
    row = ds[i]
    for side, tk, ik in (("qry","qry","qry_image_path"),("pos","pos_text","pos_image_path")):
        text = row[tk].replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[QWEN2_VL])
        img = None
        p = row.get(ik) or ""
        if p:
            full = os.path.join(cfg["image_dir"], p)
            if os.path.isfile(full): img = process_image(Image.open(full), cfg["image_resolution"])
        inp = _dev(pfn({"text":[text],"images":[img]}, proc), device)
        with torch.no_grad():
            _, feats, _, hid = teacher.encode_input(inp, output_attentions=False)
        nv = int(feats[0].size(0)) if feats and feats[0] is not None else 0
        nt = _nt(inp["input_ids"][0]); hv, ht = _tok(hid, nv, nt)
        rec = {"idx":i,"side":side,"n_v":nv,"n_t":nt,"k_g_v":0.,"k_g_t":0.,"k_g_vt":0.,"n_vt":0.}
        kw = dict(knn=cfg["knn"], kmin=cfg["kmin"], kmax=cfg["kmax"], eps=cfg["eig_eps"])
        if hv is not None and hv.size(0)>=2: rec["k_g_v"] = _kg(hv, **kw)
        if ht is not None and ht.size(0)>=2: rec["k_g_t"] = _kg(ht, **kw)
        if hv is not None and ht is not None and hv.size(0)>=1 and ht.size(0)>=1:
            rec["k_g_vt"] = _kg_vt(hv, ht, cfg["kmin"], cfg["kmax"], cfg["eig_eps"])
            rec["n_vt"] = float(hv.size(0)+ht.size(0))
        rows.append(rec)
summary = {
    "k_g_v": _stats(rows,"k_g_v","n_v"), "k_g_t": _stats(rows,"k_g_t","n_t"),
    "k_g_vt": _stats(rows,"k_g_vt","n_vt"),
    "overall_mean_k_g": mean([r[x] for r in rows for x in ("k_g_v","k_g_t","k_g_vt") if r[x]>0]) if rows else 0.,
}
with open(out_path,"w") as f: json.dump({"config":cfg,"summary":summary,"samples":rows}, f, indent=2)
print(f"Wrote {out_path} ({len(rows)} records)"); _print_summary(summary)
PY
}

kg_audit() {
  KG_MODE=run
  KG_NUM_SAMPLES="${NUM_SAMPLES:-32}"
  KG_OUTPUT="${OUTPUT:-results/kg_audit.json}"
  _kg_audit_py
}

kg_audit_summarize() {
  KG_MODE=summarize
  KG_OUTPUT="${1:-results/kg_audit.json}"
  _kg_audit_py
}

# =============================================================================
# 4. Training
# =============================================================================
train_segd_fastvlm() {
  _activate
  bash scripts/cls/train_SEGD_fastvlm.sh
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
# CLI dispatcher (direct execution only — sourcing just defines functions)
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
_cmd="${1:-}"
case "$_cmd" in
  setup_env)           setup_env ;;
  download_train_cls)  download_train_cls ;;
  download_eval_images) download_eval_images ;;
  kg_audit)            shift; kg_audit "$@" ;;
  kg_audit_summarize)  shift; kg_audit_summarize "$@" ;;
  train_segd_fastvlm)  train_segd_fastvlm ;;
  train_sgd_fastvlm)   train_sgd_fastvlm ;;
  eval_checkpoint)     eval_checkpoint ;;
  ""|help)
    cat <<EOF
Usage:
  source project_commands.sh && <function>
  ./project_commands.sh <command>

Commands:
  setup_env            Create venv, install deps, run fix_lib.py
  download_train_cls   HF download all 5 cls subsets (ImageNet_1K, N24News, ...)
  download_eval_images HF download MMEB-eval images
  kg_audit             Audit k_g per graph type (v/t/vt) → results/kg_audit.json
  kg_audit_summarize   Print summary from JSON (optional path arg)
  train_segd_fastvlm   SEGD distillation training (report_to=none)
  train_sgd_fastvlm    SGD distillation training
  eval_checkpoint      Run cls eval script

Env vars for kg_audit: NUM_SAMPLES=32 OUTPUT=results/kg_audit.json
EOF
    ;;
  *)
    echo "Unknown command: $_cmd (try: ./project_commands.sh help)" >&2
    exit 1
    ;;
esac
fi
