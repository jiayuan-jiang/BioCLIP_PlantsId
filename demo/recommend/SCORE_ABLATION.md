# 配图推荐打分形式 ablation

**日期**：2026-08-30
**脚本**：`retrieval_score_ablation.py`（可复现，纯 numpy，~7s）
**数据**：`global_pool.npz`（25000 张 / 5000 种，两视图 TTA `avg(full,cc60)` emb + 4 头 organ tagger + quality）、`global_species_proto.npz`（5000 种文本原型）。3060 编码，croc/U盘 带回。
**查询**：池内 ≥3 张照片的种里随机 400 种 × 4 器官(leaf/flower/fruit/bark)。

---

## 起因

`recommend.py` / `retrieval_eval.py` 原打分：`score = cos(q_种, emb)^a · p_organ[部位]^b · quality^c`（乘积 + 指数），`A,B,C` 是占位默认。按 `retrieval_eval.py` 的 a/b/c sweep 定权重时发现联合 purity 极低（rank1=S ~0.25），遂做本 ablation 定位原因。

---

## 结果

### [1] 原 eval 的 ground-truth 覆盖率

| | 数量 | 占比 |
|---|---|---|
| 1600 个 (种,器官) query，池内 ≥1 张该器官正确照片 | 767 | 48% |
| 池内 ≥3 张 | 339 | 21% |
| 每种平均照片数 | 5.0 | |

→ 原 `retrieval_eval.py` 不加 ground-truth 掩码，一半 query **结构上就得 0**（该种没有该器官的照片），把联合 purity 机械压低。修正 = 只评「该器官池内 ≥1（或 ≥3）张」的 (种,器官) 对。

### [2] 纯 score1（cos）基线

只按 `cos` 排、不乘 p_organ/quality、不看器官（任一本种照片算对）：

```
rank1=S 0.887   purity@5 0.826   genus@5 0.939   (n=1600)
```

→ **物种检索本身没问题**。是后续「全局乘 p_organ / quality」把它砸下去。

### [3] 打分形式 ablation（只在有正确答案的 (种,器官) 上评）

**HAS≥1（n=767）**

| config | rank1=S | purity@5 | genus@5 |
|---|---|---|---|
| raw cos³ only | 0.884 | 0.821 | 0.935 |
| raw prod 3/1/.5（曾定的权重） | 0.362 | 0.187 | 0.329 |
| raw prod 1/1/1（几何平均） | 0.039 | 0.017 | 0.050 |
| z-logit(cos)·p_organ (1/1/0) | 0.051 | 0.037 | 0.128 |
| z-logit(cos)·p_organ·q (1/1/1) | 0.014 | 0.011 | 0.052 |
| softmax(cos/T=.02)·p_organ·q | 0.932 | 0.780 | 0.928 |
| **softmax(cos/T=.01)·p_organ·q** | 0.934 | 0.829 | 0.927 |
| softmax(cos/T=.005)·p_organ·q | 0.910 | 0.839 | 0.921 |
| **softmax(cos/T=.01)·p_organ（去 quality）** | **0.954** | **0.855** | **0.944** |

**HAS≥3（n=339）** 同趋势：raw prod 3/1/.5 = 0.519 / 0.298；softmax(T=.01)·p_organ = 0.956 / 0.877。

### [4] 无正确答案时 softmax 的行为（剔除本种照片模拟）

| | 赢家裸 cos | 真是 S | 同属 |
|---|---|---|---|
| 正常（本种在池） | 0.770 | 89% | 97% |
| 剔除本种（无答案） | 0.742 | 0% | 52% |

裸 cos 绝对阈值区分力弱（无答案赢家 cos 也 ~0.74；cos≥0.72 仍误报 92%）。

---

## 结论

1. **问题在打分形式，不只是权重。** raw 乘积上任何 a/b/c 都救不回来 —— 几何平均 1/1/1 = 0.039，调过的 3/1/.5 = 0.362，纯 cos³ = 0.884。原因是尺度：`cos ∈ [0.45,0.8]`（好坏比 ~1.5×）乘 `p_organ ∈ [0,1]`（比可达 100×），p_organ 主导排序，把对的种挤出候选。指数只能倾斜、不能修尺度错配。z-score→logistic 校准同样失败（0.05）。

2. **softmax(cos/T) 把物种因子从「糊在 0.6~0.75」变成近似开关**（= 识别路径的 softmax 概率，不是裸 cos）。之后**直接乘** p_organ（无指数、权重全 1）：rank1=S 0.954 / purity@5 0.855，与纯物种检索持平。器官头只在对的物种内部重排。

3. **quality 要从排序里拿掉。** 0.954 → 0.934，它把清晰的错误物种照拉进来。留作 tiebreak / 软过滤，别当乘子。

4. **eval 必须加 ground-truth 掩码**（该器官池内 ≥1/≥3 张），否则一半 query 是机械 0。

## 暂定

打分形式：**`softmax(cos / T=0.01) · p_organ[部位]`**，quality 不进排序。
（`recommend.py` 当前仍是旧的 raw `A,B,C=3,1,0.5`，改动待定。）

## 待办 / 未决（另议）

- **「查询种不在图集」的标注**：图集只有 20k 种，识图可返回更多。查询种不在池里时，ANN 自然返回最近的别种照片，靠 `provenance` 字段（exact / genus-relative / look-alike）在结果里标注。**这是注释，不是控制流** —— 不做任何分类学 fallback / 分层打分。
- **T 重标**：生产索引物种数远大于本 ablation 的 5k，打分的温度参数要在该尺度重扫。
- **pool 局限**：本 ablation 每种仅 5 张，长尾稀有种覆盖差；真 iNat 池上需复跑。
