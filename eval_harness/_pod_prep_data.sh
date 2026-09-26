#!/bin/bash
# 在 pod 上一次性拉齐 + 整理 S0 数据。产出 /workspace/eval_data/{plantnet-300k,rare-species,imagenetv2}
set -e
cd /workspace/eval_data 2>/dev/null || { mkdir -p /workspace/eval_data && cd /workspace/eval_data; }

echo "=== [1/5] downloads (HF) ==="
hf download mikehemberger/plantnet300K --repo-type dataset \
   --include "data/test-*" --include "data/test/*" --local-dir pn300k-hf &
P1=$!
hf download imageomics/rare-species --repo-type dataset --local-dir rare-species-raw &
P2=$!
hf download vaishaal/ImageNetV2 --repo-type dataset \
   --include "imagenetv2-matched-frequency*" --local-dir imagenetv2-raw &
P3=$!
wget -q -O imagenetv2_class_index.json \
   "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json" &
P4=$!
wait $P1 $P2 $P3 $P4
echo "downloads done"

echo "=== [2/5] plantnet-300k parquet -> imagefolder ==="
python3 /workspace/_parquet_to_imagefolder.py plantnet300k /workspace/eval_data/pn300k-hf /workspace/eval_data

echo "=== [3/5] rare-species parquet -> imagefolder (all kingdoms) ==="
python3 /workspace/_parquet_to_imagefolder.py rare_species /workspace/eval_data/rare-species-raw /workspace/eval_data all

echo "=== [4/5] imagenetv2 extract ==="
mkdir -p imagenetv2
tar xf imagenetv2-raw/imagenetv2-matched-frequency.tar.gz -C imagenetv2 --no-same-owner
cp imagenetv2_class_index.json imagenetv2/imagenet_class_index.json

echo "=== [5/5] summary ==="
echo "plantnet-300k test imgs: $(find plantnet-300k/images/test -type f 2>/dev/null | wc -l)  classes: $(find plantnet-300k/images/test -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
echo "rare-species imgs:       $(find rare-species -name '*.jpg' 2>/dev/null | wc -l)  classes: $(find rare-species -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
echo "imagenetv2 imgs:         $(find imagenetv2 -type f -name '*.jpeg' 2>/dev/null | wc -l)"
df -h /workspace | tail -1
echo "PREP_DONE"
