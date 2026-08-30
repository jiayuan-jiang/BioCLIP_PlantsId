# organ_probe — 配图推荐 organ tagger 实测

2026-08-28。现成 BioCLIP 2（`demo/weights/`，冻结，CPU）。
**完整解读 + 简历故事见 `doc/recommend/organ-tagger-eval.md`。**

## 文件

| 文件 | 作用 |
|---|---|
| `try_recommend.py` | sniff test：Test A 随机图部位分桶 / Test B Acer 属物种+部位检索（`data/test_images`）|
| `diag_organ.py` | 诊断：部位文本原型两两余弦 + image-mean centering 效果 |
| `build_organ_eval.py` | 拉 Pl@ntNet-300K **test** split（8 parquet），抽 4851 张再平衡子集，编码 → `organ_eval.npz` |
| `build_organ_train.py` | 拉 **train+val** split，**随机**抽样（leaf/flower/fruit 各 2500 + habit 3000 + bark 1851），编码 → `organ_train.npz` |
| `validate_organ.py` | zero-shot raw / centered / linear probe（5-fold）：balanced_acc + F1 + 混淆矩阵 |
| `retrain_organ.py` / `retrain_organ_v2.py` | train→test 架构对比（E1–E7：5-way/4-way/MLP/3-way/bark-OvR/负样本选择）|
| `heads_compare.py` | 头对头：A) multi-label 5-sigmoid  vs  B) 3-way+路由。**A 胜** |
| `organ_tagger.py` | **最终**：4 个 one-vs-rest sigmoid（leaf/flower/fruit/bark）+ holdout 定阈值 → `organ_tagger.npz` / `.json` |
| `tta_build.py` / `tta_compare.py` | 两视图 center-crop TTA：重编码中心裁 60% 视图 → `organ_{train,eval}_cc.npz`，比 full / cc / avg 的 AUC-PR |
| `tta_full_study.py` / `tta_analysis.py` + `RUN_TTA_STUDY.md` | **7 视图完整消融**（species zero-shot + organ probe，emb/score/prob/logit-avg）。laptop 跑 study，Mac 跑 analysis。产物 `tta_{species,organ_eval}_7v.npz` + `tta_species_proto.npz` |
| `mvp_recommend.py` | 端到端 MVP：5 个手设 query，从同属池按物种分×部位分排序，出带文字标注的 contact sheet（`runs/recommend_mvp/`）|
| `organ_eval.npz` / `organ_train.npz` (+ `_cc`) | 已编码 emb + 标签，直接跑下游脚本 |
| `organ_train_BADSAMPLING.npz.bak` | 踩坑存档：按分片顺序取前 N 的偏置版（勿用）|

## 关键结果

| | balanced_acc / AUC-PR |
|---|---|
| zero-shot raw / 减 image-mean / linear probe 5-way（5-fold，偏乐观）| 0.36 / 0.46 / 0.63 |
| train→test 5-way softmax / 3-way(仅 lff) | 0.66 / **0.85** |
| multi-label 5-sigmoid（argmax）/ 路由 | bacc 0.665 自然加权 0.750 / **路由更差** 0.637 / 0.643 |
| **最终 4 头 AUC-PR**（full 特征）| leaf **0.82** · flower **0.84** · fruit **0.88** · bark **0.60** |
| + TTA `train=avg(full,cc60)` eval `full+cc60` | leaf 0.82 · flower 0.85 · fruit 0.89 · **bark 0.62**；bark R@P≥0.8 0.44→**0.48**（all-7 同级）|
| species zero-shot TTA `full+hflip+cc90`（在线成本）| top-1 0.850→**0.860**（+1.0pt；all-7 +1.4pt）；emb-avg ≡ score-avg |
| 多标签真标签命中率（max-F1 阈值）| **0.80** |
| bark 补数据：225 / 800 / 1851 张 → AUC-PR | 0.53 / 0.60 / 0.60 |

**用法**：leaf/flower/fruit 硬阈值打标签；**bark 输出 `p_bark` 概率供检索排序**（不硬阈值）；habit 从 tagger 拿掉（取景类型，AUC-PR 0.33）。

## 两个踩坑

1. **按分片顺序取前 N** → 训练集物种偏置 → train→test bacc 假摔到 0.42（用质心 cos 诊断，改随机采样修复，回 0.66）。
2. **in-sample 挑阈值** → bark "P≥0.8" 虚高；holdout 定阈值后 bark 到不了 P 0.8（现实 P≈0.54/R≈0.56）→ 对 bark 改用 top-k 排序框架。

## 复现

```bash
cd <repo>
python eval_harness/organ_probe/build_organ_eval.py     # 若无 organ_eval.npz
python eval_harness/organ_probe/build_organ_train.py     # 若无 organ_train.npz
python eval_harness/organ_probe/validate_organ.py
python eval_harness/organ_probe/retrain_organ.py
python eval_harness/organ_probe/retrain_organ_v2.py
python eval_harness/organ_probe/heads_compare.py
python eval_harness/organ_probe/organ_tagger.py
```

依赖：`open_clip_torch torch pillow numpy pyarrow huggingface_hub scikit-learn`
