# iNat v1 参考索引 —— 操作 spec（全在 Windows / 3060）

目标：iNat Open Data top 20k 植物种 × flat 100 张 → 全局检索索引。
分支 `tta-study`。Claude 没权限碰 U 盘 / Windows，全程你跑。

产物链：
```
元数据 tar (~15GB)  →  manifest.parquet (~200万行)  →  下载 ~120GB 图  →
  inat_pool.npz (~3GB) + inat_species_proto.npz (~30MB)  →  拷回 Mac / 上服务器
```

---

## 0. 环境（3060）

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install open_clip_torch pillow numpy pyarrow httpx tqdm duckdb
# aws cli: https://aws.amazon.com/cli/   （或用 curl 直接下 tar）
git -C <repo> fetch && git -C <repo> checkout tta-study
```
把 `eval_harness/organ_probe/organ_tagger_tta.npz`（~26KB）拷到 3060 的同目录（gitignored，U 盘带）。
`demo/weights/open_clip_model.safetensors` 也要在 `<BIOCLIP_ROOT>/demo/weights/`。

---

## 1. 元数据 → manifest

```powershell
mkdir D:\inat && cd D:\inat
# ~15GB，无需 AWS 账号
aws s3 cp --no-sign-request s3://inaturalist-open-data/metadata/inaturalist-open-data-latest.tar.gz .
tar xzf inaturalist-open-data-latest.tar.gz          # 出 photos.csv.gz / observations.csv.gz / taxa.csv.gz / observers.csv.gz
mkdir metadata && move *.csv.gz metadata\

python <repo>\eval_harness\organ_probe\inat_manifest.py ^
    --meta D:\inat\metadata --out D:\inat\manifest.parquet ^
    --n-species 20000 --per-species 100
```
- DuckDB join，~15–30 min，需 ~12GB RAM + 临时磁盘
- ⚠ 若报列名 / license 取值不对：`gzip -dc metadata\photos.csv.gz | head` 看真实表头，改 `inat_manifest.py` 里的 SQL
- 输出末尾会打印总张数 + 预估下载 GB

---

## 2. 下载图（~120GB，可断点续传）

```powershell
python <repo>\eval_harness\organ_probe\inat_download.py ^
    --manifest D:\inat\manifest.parquet --out D:\inat\img --workers 64
```
- 存 `D:\inat\img\{taxon_id}\{photo_id}.jpg`，跳过已存在
- iNat Open Data S3 无限速，64 并发礼貌上限内
- **⚠ 目标盘**：200万个小文件在 exFAT-over-USB2 上会 I/O 颠簸。**优先下到 3060 内置 NVMe**（要 ~150GB 空闲）；U 盘只放 tar + manifest + 最终 npz
- 先 `--limit 2000` 冒烟

---

## 3. 编码索引

```powershell
set BIOCLIP_ROOT=D:\BioCLIP
python <repo>\eval_harness\organ_probe\inat_index_build.py ^
    --manifest D:\inat\manifest.parquet --img D:\inat\img ^
    --out D:\inat\index --prune
```
- 两视图 TTA `avg(full,cc60)` + `p_organ`(organ_tagger_tta) + quality + dHash + 20k 物种原型
- `--prune`：编码后每种保留 `{p_bark>0.5 ∪ p_fruit>0.5} ∪ {quality top 50}` → 索引降 ~30–40%
- 200万 × 2 视图 ≈ 400万 forward：3060 fp16 **~9–14h（过夜）**
- 产出：`inat_pool.npz`（prune 后 ~1–1.3M 张，~2–2.6GB）+ `inat_species_proto.npz`

---

## 4. 拷回 Mac（croc / U 盘，~2.6GB）

```powershell
croc send D:\inat\index\inat_pool.npz D:\inat\index\inat_species_proto.npz
```
Mac：`croc <口令>` → `eval_harness/organ_probe/`

---

## 5. Mac 上验证 + 定权重

```bash
python eval_harness/organ_probe/global_retrieval_eval.py \
    --pool eval_harness/organ_probe/inat_pool.npz --proto eval_harness/organ_probe/inat_species_proto.npz
# 看 a/b/c sweep：purity@5 / rank1=S / swap:useful / swap:genus → 定 recommend.py 的 A/B/C

python eval_harness/organ_probe/recommend.py \
    --pool eval_harness/organ_probe/inat_pool.npz --proto eval_harness/organ_probe/inat_species_proto.npz \
    --species "Acer rubrum"
```
（v1 索引里没有本地图片，`--visualize` 不可用；结果带 iNat URL，浏览器看。）

---

## 6. 上服务器（Contabo）

`inat_pool.npz` + `inat_species_proto.npz` scp 到 `/opt/bioclip/`，`recommend.py` 逻辑接进 `inference.py` 的 `/recommend` 路由（demo-engineering-execution Phase D/E）。
- 索引 ~2.6GB + 现 RSS 3.3GB ≈ 6GB / 11.7GB，加 swapfile
- 服务只返 JSON（含 `url` / `license` / `attribution`），图片走 iNat CDN，零出站
- 内存/延迟到瓶颈：PCA 768→256 或 HNSW（[[retrieval-mechanism]] §7）

---

## 磁盘占用速查（3060）

| 项 | 大小 | 用完可删 |
|---|---|---|
| 元数据 tar + csv.gz | ~30GB | ✓（manifest 出来后）|
| 下载的图 `D:\inat\img` | ~120GB | ✓（编码后）|
| `inat_pool.npz` + proto | ~2.6GB | 保留，是产物 |
