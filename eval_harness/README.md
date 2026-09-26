# S0 — 零样本评测 harness

Stage 1 训练**之前**先建 baseline scoreboard：把 `BioCLIP 2` 和 `DALIP checkpoint` 跑进同一套基准，
之后每个训练 checkpoint 也用它评，早停 / WiSE-FT α 都看这张表。

```
eval_harness/
├── zeroshot_eval.py   # 主入口
├── benchmarks.py      # 数据集加载器（缺数据自动跳过）
├── prompts.py         # 文本模板
├── metrics.py         # top-k / macro_recall(=balanced acc) / macro_f1 / ECE
├── scoreboard.csv     # 产出（追加写）
└── per_class/         # 产出：每个 model×benchmark 的逐类 recall
```

## 基准与数据获取（都放到 `--data-root` 下）

| key | 类型 | 目录 | 获取 |
|---|---|---|---|
| `plantnet300k` | 植物（长尾主基准）| `plantnet-300k/images/test/<species_id>/*.jpg` + `plantnet300K_species_id_2_name.json` | Zenodo `records/5645731`（Pl@ntNet-300K）|
| `rare_species` | 植物（稀有种）| `rare-species/<Genus species>/*.jpg` | HF `imageomics/rare-species`（`datasets` 下载后导出成 ImageFolder，或直接放 parquet 解出的图）|
| `imagenet1k` | 通用留存 | `imagenet-1k/val/<wnid>/*.JPEG` + `imagenet_class_index.json` | 标准 ImageNet val；class_index 用 PyTorch/Keras 那份 `{"0":["n01440764","tench"],...}` |
| `nabirds` | 通用留存（非植物生物）| `nabirds/images/<class_id>/*.jpg` + `classes.txt image_class_labels.txt images.txt` | NABirds 官方（Cornell）；也接受纯 ImageFolder |
| 任意 | 植物 | `extra/<name>/<class>/*.jpg` | 通用 ImageFolder 兜底（如 iNat21-plants 子集、PlantCLEF 子集）|

数据不全没关系：`discover()` 只评实际存在的。最少放 `plantnet300k` + `imagenet1k` 就能跑。

## 运行（RunPod）

```bash
pip install open_clip_torch torch pillow numpy

# 1) BioCLIP 2 baseline
python -m eval_harness.zeroshot_eval \
    --model hf-hub:imageomics/bioclip-2 \
    --data-root /workspace/eval_data \
    --out eval_harness/scoreboard.csv

# 2) DALIP checkpoint（ViT-B/16，OpenCLIP 结构）
python -m eval_harness.zeroshot_eval \
    --arch ViT-B-16 --pretrained /workspace/ckpt/dalip.pt --model-id dalip-vitb16 \
    --data-root /workspace/eval_data

# 3) 之后：我们的 checkpoint
python -m eval_harness.zeroshot_eval \
    --arch ViT-L-14 --pretrained /workspace/out/bioclip2plant_ep4.pt \
    --model-id bioclip2-plant-r32-unfreeze4 --data-root /workspace/eval_data
```

## 本地 CPU 冒烟

```bash
python -m eval_harness.zeroshot_eval \
    --model hf-hub:imageomics/bioclip-2 \
    --data-root eval_harness/_smoke --benchmarks all \
    --limit 32 --batch-size 8 --workers 0
```

（`_smoke/extra/<toy>/<classA|classB>/*.jpg` 放几张图即可验证链路。）

## scoreboard 字段

`timestamp, model_id, benchmark, kind, n_images, n_classes, top1, top5, macro_recall, macro_f1, ece, templates, notes`

- **macro_recall** = 各类 recall 的均值 = balanced accuracy，长尾看这个（对应"macro 精度"）。
- **成功判据**：我们的 checkpoint 在 `kind=plant` 的 top1/macro_recall 均值 > BioCLIP 2；`kind=general` 不下降。DALIP 行作外部参照。
