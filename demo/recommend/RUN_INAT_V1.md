# iNat v1 参考索引 —— 操作 spec（全在 Windows / 3060）

目标：iNat Open Data top 20k 植物种 × flat 100 张 → 全局检索索引。
Claude 没权限碰 U 盘 / Windows，全程你（或交付 agent）跑。

产物链：
```
元数据 tar (~15GB)  →  manifest.parquet (~200万行)  →  下载 ~120GB 图  →
  inat_pool.npz (~2.6GB) + inat_species_proto.npz (~30MB)  →  拷回 Mac / 上服务器
```

**这个 `recommend/` 文件夹是自足的**，脚本之间不依赖 repo 其它部分。`organ_heads.npz` 就在文件夹里。

---

## 0. 环境（3060）

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install open_clip_torch pillow numpy pyarrow httpx duckdb
# aws cli: https://aws.amazon.com/cli/   （或用 curl 直接下 tar）
```

**BioCLIP 权重**，二选一：
- 已有 `open_clip_model.safetensors` → 放成 `<某目录>/demo/weights/open_clip_model.safetensors`，跑第 3 步前 `set BIOCLIP_ROOT=<某目录>`。TTA 那次已拷到 3060。
- 没有 / 嫌麻烦 → **什么都不设**，`inat_index_build.py` 自动从 HuggingFace 下 `imageomics/bioclip-2`（~1.7GB，需联网 + HF 缓存盘空间）。

脚本路径下文写作 `recommend\<脚本>`（即这个文件夹）；若拷成别的名字自行替换。

---

## 1. 元数据 → manifest

```powershell
mkdir D:\inat && cd D:\inat
# ~15GB，无需 AWS 账号
aws s3 cp --no-sign-request s3://inaturalist-open-data/metadata/inaturalist-open-data-latest.tar.gz .
tar xzf inaturalist-open-data-latest.tar.gz          # 出 photos.csv.gz / observations.csv.gz / taxa.csv.gz / observers.csv.gz
mkdir metadata && move *.csv.gz metadata\

# 先自检 schema（打印各 CSV 表头 + license 取值分布，不跑 join）
python recommend\inat_manifest.py --meta D:\inat\metadata --out D:\inat\manifest.parquet --probe-only

# 确认列名 / license 取值正常后，去掉 --probe-only 正式跑
python recommend\inat_manifest.py ^
    --meta D:\inat\metadata --out D:\inat\manifest.parquet ^
    --n-species 20000 --per-species 100
```
- DuckDB join，~15–30 min，需 ~12GB RAM + 临时磁盘
- 脚本**自动**做 schema 自检 + 列名兜底（`photo_id`/`observation_uuid`/… 找不到会明确报哪个 CSV 缺哪列）；license 匹配大小写不敏感、兼容 URL 形式，规则 = CC0 + CC-BY + CC-BY-NC（排除含 nd/sa 的）
- 若 `① 植物种 0`：检查 `--plantae-id`（默认 47126）或 taxa.ancestry 格式
- 输出末尾打印总张数 + 预估下载 GB + 最终 license 分布

---

## 2. 下载图（~120GB，可断点续传）

```powershell
python recommend\inat_download.py ^
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
python recommend\inat_index_build.py ^
    --manifest D:\inat\manifest.parquet --img D:\inat\img ^
    --out D:\inat\index --prune
```
- 两视图 TTA `avg(full,cc60)` + `p_organ`(organ_heads) + quality + dHash + 20k 物种原型
- `--prune`：编码后每种保留 `{p_bark>0.5 ∪ p_fruit>0.5} ∪ {quality top 50}` → 索引降 ~30–40%
- 200万 × 2 视图 ≈ 400万 forward：3060 fp16 **~9–14h（过夜）**
- 产出：`inat_pool.npz`（prune 后 ~1–1.3M 张，~2–2.6GB）+ `inat_species_proto.npz`

---

## 4. 拷回 Mac（croc / U 盘，~2.6GB）

```powershell
croc send D:\inat\index\inat_pool.npz D:\inat\index\inat_species_proto.npz
```
Mac：`croc <口令>` → `recommend/`

---

## 5. Mac 上验证 + 定权重

```bash
python recommend/retrieval_eval.py \
    --pool recommend/inat_pool.npz --proto recommend/inat_species_proto.npz
# 看 a/b/c sweep：purity@5 / rank1=S / swap:useful / swap:genus → 定 recommend.py 的 A/B/C

python recommend/recommend.py \
    --pool recommend/inat_pool.npz --proto recommend/inat_species_proto.npz \
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
