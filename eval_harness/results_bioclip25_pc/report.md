# BioCLIP 2.5 Huge vs BioCLIP 2：同一批基准上的零样本对比

- 机器：Windows 笔记本 WSL2，RTX 3060 Laptop 6 GB；torch 2.6.0+cu124，open_clip 3.3.0
- 模型：`hf-hub:imageomics/bioclip-2`（`bioclip2-pc`）vs `hf-hub:imageomics/bioclip-2.5-vith14`（`bioclip25-pc`）
- 两个模型都是 **fp32**，用同一套 harness、同一套模板（`prompts.py`）和各自的文本塔现算分类器；没有用官方预计算的 `txt_emb_*.npy`
- 运行时间：BioCLIP 2 是 2026-10-14 14:56–17:37，BioCLIP 2.5 是 2026-10-14 17:37 至 10-15 00:00

## 结论（先看这里）

1. **2.5 在三个基准上都比 BioCLIP 2 好**，配对自举 95% 区间都不含 0。按物种重采样的区间更保守，也都不含 0。
   - PlantNet-300K（植物，长尾）：top1 **+1.6 个点**，macro_recall +3.5。提升不大；按物种重采样的区间是 [+0.1, +2.9]，下界接近 0。
   - Rare Species（**动物**，跨生物留存）：top1 **+7.2**
   - ImageNetV2（通用留存）：top1 **+9.4**
2. **代价**：参数量是 2.3 倍（986M vs 428M），图像编码约 **2.1 倍慢**，显存峰值约 2.1 倍（4.3–4.6 GB vs 2.0–2.2 GB）。6 GB 卡上 fp32 能跑，但余量只有几百 MB。
3. **校准变差**：2.5 的 ECE 明显更高（PlantNet 0.148 vs 0.060，ImageNetV2 0.275 vs 0.149）。如果线上要用 softmax 置信度做阈值或拒识，换模型后需要重新标定。
4. **对线上植物场景最相关的是 PlantNet，那里提升最小**。提升最大的两项一个是动物、一个是通用图像，而且都有训练集泄漏风险（见"可信度"一节）。
5. 我们自己的 `plant_head5k` 不在本机（在 Mac 上），**没有跑**，也没有造替代数据。"对我们自己的植物数据好多少"这一点，本报告回答不了。

## 1. 对比表

| 基准 | 类型 | n_images | n_classes | 模型 | top1 | top5 | macro_recall | macro_f1 | ECE |
|---|---|---|---|---|---|---|---|---|---|
| plantnet300k (test) | 植物 | 31112 | 1081 | BioCLIP 2 | 0.6871 | 0.9046 | 0.5542 | 0.4370 | 0.0599 |
| | | | | BioCLIP 2.5 | 0.7035 | 0.9157 | 0.5896 | 0.4802 | 0.1481 |
| | | | | **差** | **+1.64** | **+1.11** | **+3.54** | +4.32 | +8.82 |
| rare_species | 动物 | 11983 | 400 | BioCLIP 2 | 0.7833 | 0.9361 | 0.7835 | 0.7820 | 0.0171 |
| | | | | BioCLIP 2.5 | 0.8551 | 0.9662 | 0.8552 | 0.8556 | 0.0654 |
| | | | | **差** | **+7.18** | **+3.01** | **+7.17** | +7.36 | +4.83 |
| imagenetv2 | 通用 | 10000 | 1000 | BioCLIP 2 | 0.3478 | 0.5176 | 0.3476 | 0.3337 | 0.1486 |
| | | | | BioCLIP 2.5 | 0.4420 | 0.6277 | 0.4419 | 0.4146 | 0.2750 |
| | | | | **差** | **+9.42** | **+11.01** | **+9.43** | +8.09 | +12.64 |
| plant_head5k | 植物 | — | — | — | 未跑：数据不在本机 | | | | |

差值单位是百分点。ECE 越低越好，所以 ECE 的正差值表示变差。

## 2. 复现检查：本机 BioCLIP 2 vs RunPod A40 基线

| 基准 | 指标 | 本机 | RunPod | 差（百分点） | 判定 |
|---|---|---|---|---|---|
| plantnet300k | top1 / top5 / macro_recall | 0.6871 / 0.9046 / 0.5542 | 0.6881 / 0.9046 / 0.5538 | −0.10 / 0.00 / +0.04 | ✅ 在 ±0.5 内 |
| imagenetv2 | top1 / top5 / macro_recall | 0.3478 / 0.5176 / 0.3476 | 0.3478 / 0.5176 / 0.3476 | 0 / 0 / 0 | ✅ 完全一致 |
| rare_species | top1 / top5 / macro_recall | 0.7833 / 0.9361 / 0.7835 | 0.7835 / 0.9355 / 0.7827 | −0.02 / +0.06 / +0.08 | ⚠️ 不是同一批图 |

- 两边都是 fp32，同一份 harness 和模板。
- rare_species：本机按 `_pod_prep_data.sh` 的方式导出（`all` 界）得到 **11983 张**，RunPod 那行是 **9924 张**，也就是 PROMPT.md 提到的"导出方式不同"。数字很接近，但不是严格复现。
- PlantNet 那 0.1 个点的差，可能来自 parquet 转 JPEG 时重新编码（quality=92）或 GPU 数值的细微差别，没有进一步排查。
- 结论：本机基线可信，下面的对比以本机 BioCLIP 2 为准。

## 3. 配对自举置信区间（top1 差，2.5 − 2）

对同一批图配对重采样 1000 次，seed=0。脚本会先检查两个模型的图片顺序和标签完全一致。

| 基准 | 差 | 95% CI，按图片重采样 | 95% CI，按物种整簇重采样 |
|---|---|---|---|
| plantnet300k | +1.63 | [+1.24, +2.01] | [+0.14, +2.88] |
| rare_species | +7.19 | [+6.47, +7.84] | [+5.74, +8.57] |
| imagenetv2 | +9.42 | [+8.48, +10.32] | [+8.03, +10.91] |

- 同一物种的图之间不独立，所以按物种的区间更可信。
- PlantNet 按物种的区间比按图片的宽很多（下界从 +1.2 降到 +0.1）。原因是它的测试集极度长尾，少数大类的图占了大多数，提升主要取决于这些大类。

## 4. 长尾拆分（仅 PlantNet-300K）

按测试集里每类的图片数把类别分成三等份。rare_species 每类约 30 张、ImageNetV2 每类 10 张，类别均衡，不做这个拆分：硬切三份只会按类编号切，没有意义。

| 组 | 类数 | 图片数 | 每类图片数 | BioCLIP 2 top1 | BioCLIP 2.5 top1 | 差 |
|---|---|---|---|---|---|---|
| 头部 | 361 | 29343 | 9–902 | 0.6963 | 0.7112 | +1.49 |
| 中部 | 360 | 1370 | 2–9 | 0.5314 | 0.5774 | +4.60 |
| 尾部 | 360 | 399 | 1–2 | 0.5464 | 0.5664 | +2.01 |

- 头部占了 94% 的图片，所以整体 top1 基本由头部决定。中部和尾部的提升更大，这和 macro_recall 提升（+3.5）比 top1 提升（+1.6）大是一致的。
- 尾部只有 399 张图，±2 个点的差在统计上不稳，只能看方向。
- 分组用的是测试集频率，不是训练集频率（PlantNet 的训练集没有下载）。PlantNet 的 train 和 test 分布相近，所以可以当近似。分组边界上同样图片数的类，按类编号分到两组。

## 5. 代价

| | BioCLIP 2 | BioCLIP 2.5 | 倍数 |
|---|---|---|---|
| 参数量（总 / 图像塔） | 427.6M / 304.0M | 986.1M / 632.1M | 2.3× / 2.1× |
| 图像嵌入维度 | 768 | 1024 | |
| 加载后显存（fp32 权重） | 1633 MiB | 3827 MiB | 2.3× |
| 图像编码峰值显存（batch 32） | 2045 MiB | 4335 MiB | 2.1× |
| 文本分类器峰值显存（每批 32 类） | 1989–2248 MiB | 4298–4645 MiB | 2.1× |
| 每张图编码耗时* | 207 ms | 427 ms | **2.06×** |
| 全量三个基准总耗时 | 2 小时 41 分 | 6 小时 23 分 | |

\* 耗时取 rare_species 和 ImageNetV2 两段的平均，这两段里两个模型都在同样的降频状态下。每张图耗时包含读图和解码。

**耗时的绝对值不代表这张卡的正常水平。** 整个过程中 GPU 平均 88°C，**86% 的采样点核心频率 ≤250 MHz**（最高 2100 MHz），降频原因是 `0x20`，即温度过高导致的降频（`logs/sysmon.log`）。BioCLIP 2 在 PlantNet 开头没降频时有 34 张/秒，降频后只有 5 张/秒。所以：

- **倍数（约 2.1×）可信**，和两者的理论计算量之比（约 167 vs 81 GFLOPs @224px）一致。
- 每张图多少毫秒的绝对值不要拿去估线上延迟，要在不降频的机器上重新测。

## 6. 可信度：哪些结论可信，哪些不可信

**可信**
- 本机 BioCLIP 2 基线复现正确（PlantNet 差 0.1 个点以内，ImageNetV2 完全一致）。
- 同一批图、同一套模板、同一精度下，2.5 在这三个基准上的 top1 更高，按物种重采样的区间也不含 0。
- 代价的倍数：参数量 2.3×，显存约 2.1×，计算量约 2.1×。

**不可信，或者不能外推**
- **训练集泄漏无法排除。** 官方没有说明训练集和测试集是否重叠，本评估也查不了。
  - **Rare Species** 是 Imageomics 自己发布的数据集，和训练数据（TreeOfLife 系列）是同一个团队、来源相近。2.5 在这项上 +7.2，**泄漏风险较高**；这次没有核实 2.5 的训练数据是否刻意排除了它。
  - **PlantNet-300K** 的图来自 Pl@ntNet 用户观测。这类观测会公开汇总到 GBIF 等平台，而据公开资料，TreeOfLife 训练集包含 GBIF 来源。**是否有重叠没有核实。**
  - **ImageNetV2** 是通用图像，2.5 的训练数据是否含有这类图像不清楚，+9.4 也可能部分来自训练数据的变化，而不完全是能力提升。
- **"对我们线上的植物识别好多少"回答不了**：最相关的 `plant_head5k` 没有跑，PlantNet 的提升（+1.6）最小，下界接近 0。
- **Rare Species 是动物**，不能当植物性能的证据。
- **耗时的绝对值**：全程降频，见第 5 节。
- **ECE**：harness 用各模型自带的 `logit_scale`，没有做温度标定。2.5 的 ECE 更差可能只是温度设置不同，不一定是"更不可靠"。不过线上如果直接用 softmax 分数，这个差异是实实在在的。

## 7. 和计划不同的地方，以及运行中的问题

1. **显存（已解决，没有改精度）**：harness 原来写死"每次 256 类"送进文本塔，ImageNetV2 一次要算 1792 条提示词。BioCLIP 2 在 fp32 下峰值就到了 6501 MiB，超过物理显存；WSL2 不会报 OOM，而是悄悄溢出到共享内存，速度大幅变慢。
   - 修法：加了 `--text-batch`（这次用 32 类），这只是分批算，每类仍然是自己模板嵌入的平均，**结果不变**。冒烟时验证过：分批前后 BioCLIP 2 的 PlantNet 分数完全一样。
   - 另外加了 `--max-vram-mib 5200`：超过就直接 OOM，防止静默溢出。
   - 两个模型在 fp32 下都装得下，所以**没有加 `--precision`，也没有用 fp16**。
2. **WSL 两次中断**：BioCLIP 2 前两次全量运行都在中途停了，没有任何结果。第一次停在 PlantNet 3200 张附近，第二次停在启动后扫描数据目录的阶段。之后 WSL 都重启过一次，原因没有确认（重启前的内核日志丢失）。日志在 `logs/failed_run1/`、`logs/failed_run2/`。第三次和 2.5 都完整跑完，正常退出。降频期间温度一直在 88°C 左右，过热是最可疑的原因，但没有证实。
3. **rare_species** 是 11983 张，不是 RunPod 的 9924 张，见第 2 节。

## 8. 复现方式

所有命令都在 `$TASK=/mnt/f/ssh_working_dir/bioclip25_eval` 下运行，环境是 `source ~/venvs/bioclip25/bin/activate`；长任务在 tmux 里跑。

```bash
# 0) 冒烟数据（64 张真实 PlantNet test 图 + 1081/1000 类空目录骨架；只验证加载、显存和文本分类器）
python -I scripts/make_smoke.py <smoke_dir>/data   # 另需把 imagenet_class_index.json 放进 <smoke_dir>/data/imagenetv2/
bash scripts/smoke.sh                              # 日志：logs/smoke.log、logs/smoke_cost.jsonl

# 1) 数据（约 9 GB 原始数据 + 约 11 GB 图片目录，放在 eval_data/）
bash scripts/prep_data.sh                          # 日志：logs/prep_data.log

# 2) 全量评估（先 2 后 2.5）
bash scripts/run_eval.sh hf-hub:imageomics/bioclip-2         bioclip2-pc    # logs/eval_bioclip2-pc.log
bash scripts/run_eval.sh hf-hub:imageomics/bioclip-2.5-vith14 bioclip25-pc  # logs/eval_bioclip25-pc.log
#    实际参数：--benchmarks plantnet300k,rare_species,imagenetv2 --batch-size 32 --workers 4
#             --text-batch 32 --max-vram-mib 5200（fp32）
bash scripts/sysmon.sh >> logs/sysmon.log          # 可选：GPU 温度/频率/显存监控

# 3) 分析（自举 CI、长尾拆分、代价汇总）
python scripts/analyze.py --a bioclip2-pc --b bioclip25-pc --results results   # → results/analysis.json、logs/analyze.log
```

**对 harness 的改动**（`eval_harness/zeroshot_eval.py`；原版备份在 `eval_harness_orig/`）。都是新增的可选参数，默认行为不变：
- `--text-batch N`：文本原型每次前向的类别数，默认 256，和原来一样
- `--max-vram-mib N`：PyTorch 显存上限
- `--preds-dir DIR`：保存逐图 labels 和 top5（npz），供配对分析用
- `--cost-log FILE`：每个基准的耗时、峰值显存、参数量（jsonl）

**产出文件**
- `results/scoreboard.csv`：2 个模型 × 3 个基准
- `results/per_class/`：逐类 recall
- `results/preds/`：逐图预测
- `results/cost.jsonl`：代价
- `results/analysis.json`：自举和长尾分析
- 本报告
