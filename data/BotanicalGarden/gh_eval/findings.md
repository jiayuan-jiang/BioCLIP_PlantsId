> 完整记录（含脚本与数据位置）见 [doc/botanical-garden-eval.md](../../../doc/botanical-garden-eval.md)；本文件是首轮简版。

# 温室测试集 · 线上版本评估 · 关键发现（2026-10-09）

数据：BotanicalGarden/gh_images（26410 张 / 602 物种；iNat 14801、Commons 8136、Pl@ntNet 3000、Wikipedia 473）。
模型：BioCLIP 2 ViT-L/14 本地权重 + bioclip_full_index.npz（35 万 Plantae 文字向量）。

## 主结果（lenient 命名）
| | top-1 | top-5 |
|---|---|---|
| 开集 350k（线上现状）| 34.8% | 54.9% |
| 闭集 温室 611 种 | 66.9% | 83.4% |
| 开集，去掉文字向量为零的 48 个物种 | 37.8% | 59.6% |
| 闭集，同上 | 72.6% | 90.5% |
| 对照：iNat 野外图 1500 张（同流程，开集）| 73.1% | 90.3% |

## 1. 线上词表有 14,514 行（4.1%）文字向量全零
- 这些物种永远预测不出来；温室里 48 个物种（2081 张图）受影响，含 Asplenium nidus / A. australasicum / A. scolopendrium /
  Cissus quadrangularis / Cibotium glaucum / Adiantum capillus-veneris / Acrostichum aureum 等常见种。
- 全零行与分类路径长度、科缺失无关（taxon_path 词数分布和正常行一致），原因未明；来源是 pybioclip 预计算向量本身。
- 清单：zero_vector_species.csv（温室内）、index_zero_vector_rows.csv（全部 14514 行）。
- 修法（未做）：用同一文字塔对这些行重新编码，补进词表。

## 2. 长尾：与 iNat 全球观测数强相关，与植物园 accession 数无关
| iNat 观测数 | 物种数 | 开集 top-1 | 闭集 top-1 |
|---|---|---|---|
| 0 | 6 | 7.8% | 57.4% |
| 1–99 | 163 | 12.5% | 46.9% |
| 100–999 | 150 | 28.7% | 63.5% |
| 1k–9.9k | 181 | 38.6% | 70.1% |
| ≥10k | 102 | 54.1% | 77.0% |
同属物种越多越难（属内 21–100 种：25.9%；1–5 种：41.7%）。错判里 39.6% 是同属内认错；属级 top-1 62.9%。

## 3. 置信度
- Top-1 相似度全部落在 0.58–0.82，线上 `confidence` 的 35%/22% 阈值下 26410 张全是 "high"，该字段无区分力。
- 开集按 top-1 相似度五分位：25.9% → 35.3% → 38.6% → 40.6% → 48.5%（弱）。
- 闭集按 top1−top2 相似度差（margin）五分位：35.0% → 54.0% → 79.0% → 95.5% → 99.5%（强）。
- 见 calibration_*.csv。

## 4. 标注质量（目视抽检）
- 有同物异名/分类意见分歧：Euphorbia splendens 的图被判成 E. milii（同一植物）；Sasa veitchii 被判成 Sasaella leucorhoda。
- 来源差异：Pl@ntNet 51.2% > Wikipedia 39.5% > iNat 33.7%（含大量栽培照）> Commons 30.4%。
- 未量化错标率，需要人工抽检 per_species.csv 里 0% 的物种。
