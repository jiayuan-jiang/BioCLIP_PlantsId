#!/usr/bin/env bash
# Stage 1 pod 引导 —— 环境 + eval 数据 + BioCLIP2 权重预热。
# 在 pod 上: bash stage1/pod_bootstrap.sh  （幂等，可重跑）
# 之后按 doc/spec/stage1-pod-runbook.md 跑 Phase 1/2/3。
set -euo pipefail

REPO="${REPO:-/workspace/BioCLIP}"
EVAL="${EVAL:-/workspace/eval_data}"
# 载入 .env（HF_TOKEN 等）
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN && echo "HF_TOKEN 已载入 (${HF_TOKEN:0:3}...)"

echo "=== [1/4] 依赖 ==="
pip install -q --break-system-packages \
  open_clip_torch pillow numpy huggingface_hub hf_transfer pyarrow pandas \
  webdataset imagehash tqdm || pip install -q \
  open_clip_torch pillow numpy huggingface_hub hf_transfer pyarrow pandas webdataset imagehash tqdm
python -c "import torch,open_clip,webdataset,imagehash,pyarrow,pandas; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'open_clip',open_clip.__version__)"

echo "=== [2/4] BioCLIP2 权重预热（HF cache）==="
python - <<'PY'
import open_clip
m,_,_ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
print("bioclip-2 OK, visual blocks:", len(m.visual.transformer.resblocks))
PY

echo "=== [3/4] eval 数据（S0 镜像，Phase 2/3 训练期评测要用）==="
# 该镜像里 huggingface-cli 已废弃 -> 用 hf download
mkdir -p "$EVAL"
P2I="$REPO/eval_harness/_parquet_to_imagefolder.py"   # 位置参数: <which> <src> <out> [filter]
HF="hf download"
${HF_TOKEN:+export HF_TOKEN=$HF_TOKEN}
# Pl@ntNet-300K test（长尾关键 gate）
if [ ! -d "$EVAL/plantnet-300k/images/test" ]; then
  $HF mikehemberger/plantnet300K --repo-type dataset \
    --include "data/test-*" --include "data/test/*" --local-dir "$EVAL/_pn_raw"
  python "$P2I" plantnet300k "$EVAL/_pn_raw" "$EVAL"
fi
# rare-species（诊断列；用 none 过滤 = 与 S0 一致的 400 物种集，核对 n≈9924）
if [ ! -d "$EVAL/rare-species" ]; then
  $HF imageomics/rare-species --repo-type dataset --local-dir "$EVAL/_rs_raw"
  python "$P2I" rare_species "$EVAL/_rs_raw" "$EVAL" none
fi
# imagenetv2（诊断列）
if [ ! -d "$EVAL/imagenetv2" ]; then
  $HF vaishaal/ImageNetV2 --repo-type dataset \
    --local-dir "$EVAL/imagenetv2" --include "imagenetv2-matched-frequency*"
  wget -qO "$EVAL/imagenetv2/imagenet_class_index.json" \
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fi

echo "=== [4/4] 冒烟: eval harness 能发现基准 ==="
cd "$REPO"
python -c "from eval_harness import benchmarks as B; import json; \
print([ (b.name,len(b.image_paths),len(b.classnames)) for b in B.discover('$EVAL',['plantnet300k','rare_species','imagenetv2'])])"

echo
echo "✅ bootstrap 完成。下一步: doc/spec/stage1-pod-runbook.md 的 Phase 1"
