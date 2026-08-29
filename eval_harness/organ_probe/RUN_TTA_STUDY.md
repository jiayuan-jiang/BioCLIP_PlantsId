# TTA 完整消融 —— 在 3060 laptop 上跑的操作 spec

目标：一次编码 7 个视图，产出 3 个 npz，带回 Mac 出决策表。
脚本：`tta_full_study.py`（laptop 跑）、`tta_analysis.py`（Mac 跑）。分支 `tta-study`。

产出（laptop → Mac）：
| 文件 | 大小 | 内容 |
|---|---|---|
| `tta_species_proto.npz` | ~15 MB | 5000 物种文本原型 `[5000,768]` |
| `tta_species_7v.npz` | ~54 MB | 2500 图 × 7 视图 emb + y + paths |
| `tta_organ_eval_7v.npz` | ~104 MB | 4851 图 × 7 视图 emb + organs |

7 视图 = `full · hflip · cc90 · cc80 · cc70 · cc60 · padsq`。

---

## Part 1 — Mac → U 盘（在**你自己的终端**跑；Claude Code 终端被 TCC 挡）

```bash
cd ~/Documents/GDS/NO/BioCLIP
U=/Volumes/Jiayuan_2T

mkdir -p "$U/BioCLIP/data" "$U/BioCLIP/demo" "$U/BioCLIP/scripts"
cp -R data/test_images  "$U/BioCLIP/data/"          # 3.2 GB，必传
cp -R demo/weights      "$U/BioCLIP/demo/"          # 1.7 GB，可选（省 laptop 下载）
cp eval_harness/organ_probe/tta_full_study.py "$U/BioCLIP/scripts/"
cp eval_harness/organ_probe/tta_analysis.py   "$U/BioCLIP/scripts/"
sync
```

弹权限框就点「允许」。若 `cp` 报 `Operation not permitted`：系统设置 → 隐私与安全性 → 完全磁盘访问权限 → 加终端 app、打开、重启终端。

---

## Part 2 — U 盘 → laptop，摆目录

把 U 盘里的 `BioCLIP/` 拷到 laptop 任意位置，例如 `D:\BioCLIP\`，结构：

```
D:\BioCLIP\
  data\test_images\<taxon>_<Genus species>\*.jpg
  demo\weights\open_clip_model.safetensors      (若拷了)
  scripts\tta_full_study.py
  scripts\tta_analysis.py
```

---

## Part 3 — laptop 环境 + 跑

```powershell
# CUDA 版 torch（3060）
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install open_clip_torch pillow numpy pyarrow huggingface_hub

# ROOT 指向含 data\ 和 demo\ 的目录
set BIOCLIP_ROOT=D:\BioCLIP           # PowerShell: $env:BIOCLIP_ROOT="D:\BioCLIP"

python D:\BioCLIP\scripts\tta_full_study.py
```

- 自动 `device=cuda` + fp16 + batch 192
- 自动下 Pl@ntNet-300K test 8 个 parquet（~3.3 GB，一次性；进 `~/.cache/huggingface`）
- 若没拷 `demo\weights\`：自动从 HF 下 `imageomics/bioclip-2`（~1.7 GB）
- 耗时 ~15–25 min。日志每 200 张打一次 eta。
- 产出 3 个 npz 在**脚本所在目录**（`D:\BioCLIP\scripts\`），或加 `--out D:\BioCLIP\out`
- 只想先跑 species：`--skip-organ`；只 organ：`--skip-species`

**显存**：ViT-L/14 fp16 batch 192 ≈ 4–5 GB，6 GB 够。若 OOM，改脚本里 `BATCH = 96`。

---

## Part 4 — laptop → Mac（croc，只传 3 个小 npz，共 ~170 MB）

laptop 装 croc：`winget install croc`（或 `scoop install croc` / 从 github.com/schollz/croc releases 下 exe）。

```powershell
cd D:\BioCLIP\scripts
croc send tta_species_proto.npz tta_species_7v.npz tta_organ_eval_7v.npz
# 打印一句口令，如：croc 1234-word-word-word
```

Mac（你自己的终端）：
```bash
cd ~/Documents/GDS/NO/BioCLIP/eval_harness/organ_probe
croc <口令>            # 若没装：brew install croc
```

（也可以用同一个 U 盘带回，把 3 个 npz 拷进去即可——但 croc 更快。）

---

## Part 5 — Mac 出决策表

3 个 npz 落到 `eval_harness/organ_probe/` 后：

```bash
cd ~/Documents/GDS/NO/BioCLIP
python eval_harness/organ_probe/tta_analysis.py
```

需要本地已有 `organ_train.npz` + `organ_train_cc.npz`（已在）。输出：
- **表 A** species zero-shot：单视图 / 增量组合 / leave-one-out 的 top-1/top-5 + Δ + forward 次数；emb-avg vs score-avg vs score-max
- **表 B** organ probe：视图组合 × {emb-avg / prob-avg / logit-avg} × {train=full / train=avg(full,cc60)} 的 4 头 AUC-PR + bark P≥0.8 recall + forward 次数
- 一致性校验：7v 的 full 视图 vs 原 `organ_eval.npz`，`max|Δ|` 应 ~0

我拿到输出后写进 `doc/recommend/organ-tagger-eval.md`，给出最终取舍（species 是在线每查询成本、organ 是离线索引成本）。

---

## 确定性说明

采样用 Python `random.seed(0)`，跨平台一致 → laptop 编的 organ eval 子集和 Mac 上 `organ_eval.npz` 是同一批同序（Part 5 会校验）。species 子集同理由 seed 固定。
