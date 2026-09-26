#!/usr/bin/env bash
# Stage 1 v2 端到端自驱动 —— pod 上 setsid 后台跑，跑完自己停机。
#   Phase1(24片，若已在跑则等它) -> M0 主训练 -> 消融矩阵 -> WiSE-FT -> 自停机。
#   /workspace/STOP 文件存在则任何阶段前中断（外部急停）。
#   硬看门狗：脚本启动 WATCHDOG_H 小时后无条件停机。
# 用法: POD_ID=xxx setsid bash stage1/run_v2.sh > /workspace/v2.log 2>&1 &
set -uo pipefail
cd /workspace/BioCLIP
set -a; . ./.env 2>/dev/null; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2

POD_ID="${POD_ID:-}"
WATCHDOG_H="${WATCHDOG_H:-17}"
R=/workspace/runs/stage1v2
D=/workspace/data/stage1v2
EVAL=/workspace/eval_data
mkdir -p "$R"
NB="--data-root $D --eval-data-root $EVAL --workers 24 --device cuda --grad-ckpt 0 --batch-size 384"
PROTO="--proto-ce-lambda 0.15 --proto-logit-scale 30 --proto-logit-adjust 1"
SB=/workspace/BioCLIP/eval_harness/scoreboard.csv

stop_pod() {
  echo "=== [$(date -u +%T)] stop_pod($POD_ID) ==="
  [ -z "$POD_ID" ] || [ -z "${RUNPODCTL_KEY:-}" ] && { echo "  no POD_ID/key, skip"; return; }
  # 1) REST v1
  curl -s -m 20 -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
    -H "Authorization: Bearer $RUNPODCTL_KEY" -H "Content-Type: application/json"; echo
  sleep 3
  # 2) GraphQL 兜底
  curl -s -m 20 "https://api.runpod.io/graphql?api_key=$RUNPODCTL_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation{podStop(input:{podId:\\\"$POD_ID\\\"}){id desiredStatus}}\"}"; echo
}
# 硬看门狗
( sleep $((WATCHDOG_H*3600)); echo "=== WATCHDOG $WATCHDOG_H h -> stop ==="; touch /workspace/STOP; sleep 60; stop_pod ) &
WD=$!

ck() { [ -f /workspace/STOP ] && { echo "[$(date -u +%T)] STOP -> halt+stop_pod"; stop_pod; kill $WD 2>/dev/null; exit 0; }; }
run() { echo "=== [$(date -u +%T)] BEGIN $1 ==="; shift; "$@"; local rc=$?; echo "=== [$(date -u +%T)] END rc=$rc ==="; return $rc; }

# ---------- Phase 1: 24 分片（若外部已在跑，等它出 index.parquet）----------
ck
if [ ! -f "$D/index.parquet" ]; then
  if pgrep -f "stage1.prepare --shards 0-23" >/dev/null; then
    echo "[$(date -u +%T)] Phase1 已在外部运行，等待 index.parquet…"
    while [ ! -f "$D/index.parquet" ]; do ck; sleep 60; done
  else
    run "Phase1 24-shard" python -m stage1.prepare --shards 0-23 --jobs 24 --out "$D" \
        --eval-roots "$EVAL" --eval-phash-cache /workspace/data/_eval_phash.npy \
        --device cuda --text-batch 2048
  fi
fi
touch "$R/PHASE1_DONE"
echo "[$(date -u +%T)] Phase1 ready: $(python -c "import pandas as pd;print(len(pd.read_parquet('$D/index.parquet')))" 2>/dev/null) rows"

# ---------- M0: 主训练（纯 locked-text InfoNCE，无 protoCE）----------
ck
if [ ! -f "$R/M0/DONE" ]; then
  mkdir -p "$R/M0"
  run "M0 main" python -m stage1.train --run-tag v2-M0 $NB --out "$R/M0" \
      --shards 0-23 --epochs 3 --max-steps ${M0_STEPS:-16800} --eval-every 1500 --patience 4 \
      --lora-r 32 --unfreeze-depth 4 --memory-k 32000 --lr-lora 1e-4
  touch "$R/M0/DONE"
fi

# ---------- 消融（8 片，2200 步 ≈ 1 epoch）。M0 基线(纯InfoNCE, unfreeze4, rank32) ----------
ABL="--shards 0-7 --epochs 1 --max-steps ${ABL_STEPS:-2200} --eval-every 2200 --patience 2 --lr-lora 1e-4"
BASE="--unfreeze-depth 4 --lora-r 32"                       # M0 同口径基线
declare -a RUNS=(
  "abl-unfreeze0|--unfreeze-depth 0 --lora-r 32"
  "abl-unfreeze2|--unfreeze-depth 2 --lora-r 32"
  "abl-unfreeze4|$BASE"
  "abl-proto015|$BASE --proto-ce-lambda 0.15 --proto-logit-scale 30 --proto-logit-adjust 1"
  "abl-rank16|--unfreeze-depth 4 --lora-r 16"
  "abl-rank64|--unfreeze-depth 4 --lora-r 64"
)
for entry in "${RUNS[@]}"; do
  ck
  tag="${entry%%|*}"; extra="${entry#*|}"
  [ -f "$R/$tag/DONE" ] && { echo "skip $tag"; continue; }
  mkdir -p "$R/$tag"
  run "$tag" python -m stage1.train --run-tag "v2-$tag" $NB --out "$R/$tag" $ABL $extra
  touch "$R/$tag/DONE"
done
touch "$R/ABL_DONE"

# ---------- WiSE-FT: 对 M0 best.pt 扫 α ----------
ck
if [ -f "$R/M0/best.pt" ] && [ ! -f "$R/wiseft/DONE" ]; then
  mkdir -p "$R/wiseft"
  run "WiSE-FT M0" python -m stage1.wise_ft --ckpt "$R/M0/best.pt" \
      --alphas 0.05 0.1 0.15 0.2 0.3 0.5 --eval-data-root "$EVAL" \
      --out "$R/wiseft" --save-best "$R/BioCLIP2-Plant-v2.pt" \
      --scoreboard "$SB" --lora-r 32 --unfreeze-depth 4 --device cuda
  touch "$R/wiseft/DONE"
fi

echo "=== [$(date -u +%T)] ALL DONE ==="
touch "$R/ALL_DONE"
kill $WD 2>/dev/null
stop_pod
