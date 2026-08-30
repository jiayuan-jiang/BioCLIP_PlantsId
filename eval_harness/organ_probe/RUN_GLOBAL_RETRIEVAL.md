# 全局检索 eval + recommend —— 在 3060 上跑的操作 spec

验证新的检索架构（全局向量检索，无 fallback）。分工同 TTA 那次：**重的编码在 3060，轻的分析在 Mac。**

分支 `tta-study`。

---

## 只有一个重活：`global_retrieval_build.py`（3060）

对 `data/test_images`（~5000 种 × 5 张 ≈ 25000 张）编码：
- 两视图 TTA `avg(full, cc60)` → emb（fp16）
- `p_organ` = `organ_tagger_tta.npz` 的 4 头
- quality = 分辨率 + 清晰度（拉普拉斯方差）
- dHash（去重用）
- 再编码 5000 个物种文本原型

产出：
| 文件 | 大小 | 内容 |
|---|---|---|
| `global_pool.npz` | ~40 MB | `emb[N,768 fp16]` · `species_idx` · `p_organ[N,4]` · `q_res` · `q_sharp` · `dhash[N,8]` · `path` · `classes` |
| `global_species_proto.npz` | ~8 MB | `proto[5000,768 fp16]` · `species` |

耗时：25000 × 2 视图 ≈ 5 万次 forward → 3060 fp16 ~15–20 min（Mac CPU 要 ~3h）。

---

## 拷到 3060 的文件

| 文件 | 来源 | 备注 |
|---|---|---|
| `data/test_images/` | **已在**（TTA 那次拷过去了）| 3.2 GB |
| `demo/weights/open_clip_model.safetensors` | **已在** | 1.7 GB |
| `eval_harness/organ_probe/organ_tagger_tta.npz` | **要传**（~26 KB）| 4 个 sigmoid 头（在 `avg(full,cc60)` 上重训过）|
| `eval_harness/organ_probe/global_retrieval_build.py` | **要传** | 脚本 |

（`git pull` 分支 `tta-study` 能拿到脚本；`organ_tagger_tta.npz` 是 gitignored 数据，得单独拷——U 盘或 croc。）

---

## 跑

```powershell
set BIOCLIP_ROOT=D:\BioCLIP
python global_retrieval_build.py
```
- 自动 `device=cuda` + fp16
- 需同目录有 `organ_tagger_tta.npz`
- 产出 `global_pool.npz` + `global_species_proto.npz`

---

## 带回 Mac（croc，~48 MB）

```powershell
croc send global_pool.npz global_species_proto.npz
```
Mac：`croc <口令>` → 放到 `eval_harness/organ_probe/`。

---

## Mac 上分析（纯 numpy，秒级）

```bash
python eval_harness/organ_probe/global_retrieval_eval.py     # a/b/c sweep + species-purity@k + 跨物种替换分析
python eval_harness/organ_probe/recommend.py --visualize     # 示例 query + contact sheet 到 runs/recommend_v2/
```

`global_retrieval_eval.py` 输出决定 `recommend.py` 里的 `A/B/C` 权重。
