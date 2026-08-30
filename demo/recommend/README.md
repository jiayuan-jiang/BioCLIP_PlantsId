# demo/recommend/ — 植物配图推荐

全局向量检索。定物种 → 对整个照片 emb 索引按 `cos(物种文本,emb)^a · p_organ[part]^b · quality^c`
排序 → phash 去重 + MMR → provenance 后处理标注。**无 fallback、无分类学门控。**

设计文档：`doc/recommend/`（[[plant-recommend-spec]] / [[retrieval-mechanism]] / [[organ-tagger-eval]]）。

## 文件

| 文件 | 跑在哪 | 作用 |
|---|---|---|
| `inat_manifest.py` | 3060/Win | iNat Open Data 元数据 tar → DuckDB → `manifest.parquet`（top-20k 植物种 × ≤100 张）|
| `inat_download.py` | 3060/Win | 按 manifest 并行下 `medium.jpg`（S3 无限速），断点续传 |
| `inat_index_build.py` | 3060 | 两视图 TTA 编码 + `p_organ`(organ_heads) + quality + dHash + 物种原型 → `inat_pool.npz` + `inat_species_proto.npz`；`--prune` |
| `retrieval_eval.py` | Mac | 架构验证：species-purity@k + 跨物种替换分析 + `a/b/c` sweep |
| `recommend.py` | Mac / server | `Recommender` 类 + `recommend(species, parts, per_part)`；`--species "..."` 单跑 |
| `organ_heads.npz` | — | 4 个 sigmoid 头（`eval_harness/organ_probe/` 的 organ tagger 在 `avg(full,cc60)` 上重训）|
| `RUN_INAT_V1.md` | — | 完整操作 spec（元数据 → 下载 → 编码 → 拷回 → 验证 → 上服务器）|

organ tagger 的训练 / 评测 / TTA 消融代码在 `eval_harness/organ_probe/`。

## 流程

```
3060:  inat_manifest.py  →  inat_download.py  →  inat_index_build.py --prune
Mac :  retrieval_eval.py  (定 a/b/c)  →  recommend.py
server: inat_pool.npz + inat_species_proto.npz 上 /opt/bioclip/，接进 /recommend 路由
```

## 状态

设计定稿、organ tagger 训好、v1 管线脚本就绪。**待跑**：iNat Open Data 索引构建（3060 过夜）→ 权重 sweep → 接入服务。
