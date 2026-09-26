# Organ tagger v3 —— 干净 bark + 人工筛 stem

**日期**：2026-09-01 ~ 09-04
**代码**：`build_organ_train_v3.py`（fetch+manifest+encode）· `add_stem_v3.py`（人工标注合并）· `organ_tagger_v3.py`（5 头训练+eval）
**数据**：`organ_train_v3.npz` / `organ_eval_v3.npz` / `organ_tagger_v3.npz`+`.json`（权重）· `stem_images_used/`（1400 张人工选中的 stem 原图 + `stem_labels.json`，永久保留）

---

## 起因

v1 organ tagger 的 `bark` 类取自 Pl@ntNet-300K 的 `organ` 字段。PlantNet 没有 "stem" 类，草本茎/木本嫩枝/真树皮全塞进 `bark` → bark 头 AUC-PR 只有 0.60（leaf 0.82 / flower 0.84 / fruit 0.88）。

## 数据溯源（4 个候选源，2 个能用）

| 源 | 结果 |
|---|---|
| **BarkNet 1.0**（Zenodo 11508014，魁北克 23 种）| ✅ 干净树皮，采用 |
| **BARK-KR**（Zenodo 4749062，首尔 ~30 种）| ✅ 干净树皮，采用 |
| **NTU-Tree**（HF `liswei/NTU-Tree`，台大 15 种「stem」）| 实为树皮特写，**改归 bark**，不当 stem |
| **PlantCLEF 2015+2016 `Content=Stem`**（5786 张）| ❌ 抽样发现 ~50% 其实是树干（PlantCLEF 没有 bark 类，树皮照全塞进 Stem）。**弃用，未进最终训练集** |

结论：**没有干净的现成 herbaceous-stem 数据集**（GRASP-125、DARMA 都是同样问题：树+草混、"stem/branch" 不分家）。→ 从 PlantCLEF 的 5786 张里**人工筛**。

## 人工标注

临时网页 `tmp/stem_review/app.py`（:8200，网格点选，自动存盘）。参考本项目自己的先例（bark 正样本 225→800 张，AUC-PR 0.53→0.60，800+ 就平），定目标 ~800 张，实际标了 **1400 张**（翻了 5786 中的一部分，命中率约 40-50%）。

选中的 1400 张原图永久存在 `stem_images_used/`（256MB），`stem_labels.json` 同目录备份。**不要删**——人工标注成本高，模型/流程变了要重训时还能直接复用这批。

## 最终结果（5 头，`organ_tagger_v3.py`）

train 11122 / test 2778：`{leaf 2000/500, flower 2000/500, fruit 2000/500, bark 4002/998, stem 1120/280}`

| head | AUC-PR | 备注 |
|---|---|---|
| leaf | 0.859 | 同分布内（PlantNet），比 v1(0.82) 略升 |
| flower | 0.905 | 比 v1(0.84) 升 |
| fruit | 0.895 | 比 v1(0.88) 升 |
| bark | **1.000**⚠️ | 见下「bark 的两面性」|
| stem | **0.933** | 负样本含真树皮（BarkNet），是比其它头都硬的测试 |

leaf/flower/fruit 小幅波动（负样本里多了 stem/更干净的 bark，属正常噪声范围）。

## bark 的两面性：训练集 1.000 ≠ 能直接信

bark 正样本全来自专门拍树皮的数据集（BarkNet=GalaxyS5 魁北克、BARK-KR=iPhone 首尔、NTU=台大），负样本全是 PlantNet——两个分布在 BioCLIP 空间里**零重叠**，1.000 有「学会认拍摄风格而非纹理」的嫌疑。

**用 `demo/recommend/global_pool.npz`（25000 张真实 iNat 照片，无标签）做了定性抽查**（不是指标，是挑高分图 + 查物种名 + 肉眼看）：
- v3 判 bark 最高的图：Nyssa sylvatica / Ostrya virginiana / Pinus strobus / Quercus palustris——全是树，其中 Ostrya 那张是仰拍多棵树干的宽景，**不是 BarkNet 那种贴脸构图**，证明学到的是纹理概念，不是单纯认出「这是 BarkNet 拍法」
- v1 曾经的高分误判：`Schoenoplectus tabernaemontani`（手捏一根绿色芦苇茎，v1=0.987）、`Echium pininana`、`Darmera peltata`（巨型草本/伞叶草，v1=0.98）——v3 全部正确打低分（0.11~0.24）
- v1 在整个 25000 张池子上有 21% 判成 bark（明显过敏），v3 只有 3.5%（>0.5 的 868 张）

结论：**v3 bark 头可用**，比 v1 有实质提升，训练集 1.000 的水分主要来自「无法造出同分布负样本」，不代表部署会失败。

## 尚未做 / 已知限制

- stem 头没有做同等的 iNat 池定性抽查（不像 bark 那次有报警信号——0.933 没有饱和，优先级较低，需要时可以照搬同一方法）
- **还没接入生产**：`demo/recommend/organ_heads.npz`（部署权重）仍是 v1 4 头；`global_pool.npz` 的 `p_organ` 字段仍是 v1 算的。要用上 v3，需要：① 用 v3 5 头对 pool 的 `emb` 重算 `p_organ`（emb 已有，不用重编码）② 替换 `organ_heads.npz` ③ `demo/recommend/recommend.py` 的 `CLASSES`/`p_organ` 相关代码要适配 5 类
- root 类：讨论过，判定没有干净数据集来源，弃用（`SCORE_ABLATION.md` 待办里也提过）

## 磁盘

`_v3_cache/raw/`（BarkNet 全量 + BARK-KR 全量 + NTU 全量 + PlantCLEF 未选中部分，共 ~18.4GB）已删除——npz 里 embedding 已经保存，原图除了 `stem_images_used/` 那 1400 张（人工标注，留档）都不再需要。
