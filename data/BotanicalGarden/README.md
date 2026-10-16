# BotanicalGarden — 植物园测试集与评估数据

完整说明（评估口径、全部结果、踩过的坑、脚本与数据位置表）：**[doc/botanical-garden-eval.md](../../doc/botanical-garden-eval.md)**
干了什么 / 发现的问题 / 下一步（执行留痕）：**[doc/spec/botanical-garden-eval-run-log.md](../../doc/spec/botanical-garden-eval-run-log.md)**

```
MBGNA_..._8355821904764023402.csv   温室清单（1,027 条）        MBGNA_..._-3772834397469505304.csv  全园库存
gh_images/    测试图片 images/<物种>/ + manifest.csv（来源/许可/作者/sha1）+ coverage.csv
gh_eval/      评估产出：predictions.csv（逐图）· per_species.csv（逐物种）· summary.md · image_proto_* · multi_photo_results.csv · plots/
gh_notext_emb/ 51 个无文字向量物种的 embedding 与图像原型（prototypes.npz）
gh_closed/    温室闭集文字塔 closed_text.npz · 映射表 · 无文字向量物种清单 no_text_vector_species.csv
wcvp/         Kew WCVP 名录（命名对账用，约 1GB，可删后重新下载）
```
脚本在上一级 `data/bg_NN_*.py`（按 NN 顺序是流水线步骤）。
