# S0 on RunPod — runbook

本地已出 `plant_head5k` 的 BioCLIP 2 地板（top1 0.8775，简单区）。
剩下在 pod 上跑：**Pl@ntNet-300K 长尾（关键）** + rare_species + 通用留存 + **DALIP ckpt**。

---

## 1. 起 pod

- 模板：**RunPod PyTorch 2.x**（CUDA）
- GPU：RTX 4090 24GB（~$0.44/hr）足够；S0 全基准约 15–30 min
- 卷：40–60 GB（Pl@ntNet-300K 解压 ~40 GB）

## 2. 代码上 pod

本地：
```bash
cd ~/Documents/GDS/NO/BioCLIP
runpodctl send eval_harness
# 可选：把本地 plant_head5k 也带上（~1–2 GB）
runpodctl send data/test_images
```
pod 上（`runpodctl receive <code>`），放成：
```
/workspace/BioCLIP/eval_harness/...
/workspace/eval_data/test_images/...        # 若带了 plant_head5k
```
或直接 `git clone` 你的仓库。

## 3. 环境

```bash
pip install open_clip_torch pillow numpy huggingface_hub
mkdir -p /workspace/eval_data && cd /workspace/eval_data
```

## 4. 数据

### Pl@ntNet-300K（必须 — 长尾关键基准）
```bash
# Zenodo record 5645731，文件 plantnet_300K.zip（~31 GB）
# 链接以 github.com/plantnet/PlantNet-300K 的 README 为准
wget -c "https://zenodo.org/record/5645731/files/plantnet_300K.zip"
unzip -q plantnet_300K.zip
# 整理成 loader 期望的布局：
mkdir -p plantnet-300k
mv plantnet_300K/images       plantnet-300k/           2>/dev/null || mv images       plantnet-300k/
mv plantnet_300K/plantnet300K_species_id_2_name.json  plantnet-300k/ 2>/dev/null || \
   mv plantnet300K_species_id_2_name.json plantnet-300k/
# 校验：应存在 plantnet-300k/images/test/<species_id>/*.jpg
ls plantnet-300k/images/test | head -3
```

### rare_species（推荐）
```bash
huggingface-cli download imageomics/rare-species --repo-type dataset \
    --local-dir rare-species-raw
# 需整成 rare-species/<Genus species>/*.jpg
# 若下载出来是 metadata.csv + 图片平铺，用一次性脚本按学名分目录；
# 若已是 ImageFolder，直接: mv rare-species-raw rare-species
```

### imagenetv2（通用留存，ungated，可选）
```bash
huggingface-cli download vaishaal/ImageNetV2 --repo-type dataset \
    --local-dir imagenetv2 --include "imagenetv2-matched-frequency*"
# 解出后应为 imagenetv2/<0..999>/*.jpeg
wget -O imagenetv2/imagenet_class_index.json \
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
```

### nabirds（可选，非植物生物留存）
放成 `nabirds/images/<class_id>/*.jpg` + `classes.txt image_class_labels.txt images.txt`，或纯 ImageFolder。

> loader 对缺失基准自动跳过，最少有 `plantnet300k` 就能跑。

## 5. DALIP checkpoint

百度网盘（pwd `vbtk`）→ 下到本地 → `runpodctl send dalip.pt` → pod `/workspace/ckpt/dalip.pt`。
（或 pod 上装 `bypy` 拉。）ViT-B/16、OpenCLIP 结构。

## 6. 跑

```bash
cd /workspace/BioCLIP        # 含 eval_harness/

# BioCLIP 2 — 全部基准
python -m eval_harness.zeroshot_eval \
    --model hf-hub:imageomics/bioclip-2 \
    --data-root /workspace/eval_data --benchmarks all \
    --batch-size 512 --workers 16 \
    --out eval_harness/scoreboard.csv

# DALIP ckpt（文本原型缓存按 model_id 分开，不冲突）
python -m eval_harness.zeroshot_eval \
    --arch ViT-B-16 --pretrained /workspace/ckpt/dalip.pt --model-id dalip-vitb16 \
    --data-root /workspace/eval_data --benchmarks all \
    --batch-size 512 --workers 16 \
    --out eval_harness/scoreboard.csv

# 可选：BioCLIP 2.5 Huge 外部对照
python -m eval_harness.zeroshot_eval \
    --model hf-hub:imageomics/bioclip-2 --model-id bioclip2 ...   # 换成 2.5 的 hf 名
```

先加 `--limit 300` 冒烟确认链路，再去掉跑全量。

## 7. 收结果

```bash
column -t -s, eval_harness/scoreboard.csv
runpodctl send eval_harness/scoreboard.csv    # 拉回本地
runpodctl send eval_harness/per_class
```

关注 **`benchmark=plantnet300k`** 的 `top1` / `macro_recall`：
- BioCLIP 2 这行 = Stage 1 要超过的**长尾地板**
- DALIP 这行 = 外部参照
- 通用基准（imagenetv2 / nabirds）BioCLIP 2 会偏低、DALIP 更低（论文里 ImageNet 49.2 vs 18.6）——记录下来，Stage 1 的目标是"植物涨、通用别更差"

## 关于文本缓存

`eval_harness/.text_cache/` 按 `(model_id, benchmark, 类名+模板)` 缓存文本原型。
同一模型重跑 / 换图像子集 → 秒出。换模型 → 重新编码。跨 pod 想复用就把这个目录也 send。
