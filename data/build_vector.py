"""
BioCLIP 2 植物向量库构建脚本（官方格式 + Prompt Ensemble）
=============================================================
BioCLIP / BioCLIP 2 训练时使用三种文本格式混合训练（论文 Table 3）：

  1. 分类学路径（最强）：
       "a photo of Plantae Tracheophyta Magnoliopsida Ericales Ericaceae Rhododendron Rhododendron simsii"
  2. 科学名：
       "a photo of Rhododendron simsii"
  3. 常用名：
       "a photo of azalea"

推理时对三种格式各编码一次，取 L2 归一化后的均值向量作为该物种的最终表示
（Prompt Ensemble），与原来单一模板相比可显著减少文本偏移误差。

此版本相比上一版新增：
  - 补全 phylum / class_ / order 三个分类学层级（iNat API ancestors）
  - 更新 parse_ancestors() 解析全部 7 个标准层级
  - 更新 taxonomy_enriched.csv 输出（含 phylum/class_/order 列）
  - 用三模板 ensemble 替换单一模板编码
  - 向量库中额外保存 texts_taxpath / texts_sci / texts_common 供调试

依赖：
  pip install torch open-clip-torch numpy pandas tqdm requests
"""

import os
import time
import math
import requests

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import open_clip


# ══════════════════════════════════════════════
#  修改这里的变量
# ══════════════════════════════════════════════

INPUT_CSV = "global_plants_top5000.csv"

WEIGHTS_DIR    = "./weights/bioclip2"
HF_REPO_ID     = "imageomics/bioclip-2"
OPEN_CLIP_ARCH = "ViT-L-14"

OUTPUT_TAXONOMY_CSV = "taxonomy_enriched.csv"
OUTPUT_INDEX_NPZ    = "plant_text_index.npz"

BATCH_SIZE     = 256
API_BATCH_SIZE = 30
API_DELAY      = 0.3
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════

# 标准 7 级分类学层级（按从高到低顺序，与 BioCLIP 训练时一致）
TAXON_RANKS = ["kingdom", "phylum", "class_", "order", "family", "genus", "species"]

_LOCAL_REQUIRED_FILES = [
    "open_clip_pytorch_model.bin",
    "open_clip_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
_LOCAL_WEIGHT_CANDIDATES = ["open_clip_pytorch_model.bin", "model.safetensors"]


# ─────────────────────────────────────────────
# Step 1  读取 CSV
# ─────────────────────────────────────────────

def load_input_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"taxon_id": str})
    required = {"taxon_id", "scientific_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}  实际列：{list(df.columns)}")
    print(f"✓ 读取 CSV：{len(df)} 行  [{path}]")
    return df


# ─────────────────────────────────────────────
# Step 2  补全完整 7 级分类学层级
# ─────────────────────────────────────────────
#
# 策略（两步批量，总请求数约 50 次，而非 5000 次）：
#   Step 2a  批量接口拿全部物种的 ancestor_ids 列表
#   Step 2b  合并去重所有 ancestor id，分批批量查 rank+name
#   Step 2c  本地字典拼装每个物种的完整 7 级分类学
#
# 关键发现：
#   /v1/taxa?id=...  批量接口返回 ancestor_ids（id 列表）但不展开 ancestors 对象
#   /v1/taxa?id=...  用相同接口查询这批 ancestor id，可以得到每个节点的 rank+name
#   因此完全不需要逐个单独请求，大幅提速。

INAT_TAXA_URL = "https://api.inaturalist.org/v1/taxa"

RANK_MAP = {
    "kingdom": "kingdom",
    "phylum":  "phylum",
    "class":   "class_",   # iNat 返回 "class"，存储为 "class_" 避免 Python 关键字冲突
    "order":   "order",
    "family":  "family",
    "genus":   "genus",
    "species": "species",
}


def _batch_request(url: str, params: dict, retry: int = 3) -> list:
    """带 429 退避重试的批量 GET 请求，返回 results 列表。"""
    for attempt in range(retry):
        try:
            r = requests.get(url, params=params, timeout=30,
                             headers={"User-Agent": "bioclip2-builder/1.0"})
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"⏳ 429 限速，等待 {wait}s…")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json().get("results", [])
        except Exception as e:
            if attempt == retry - 1:
                print(f"  ⚠ 请求失败：{e}")
            time.sleep(2)
    return []


def fetch_taxon_info_batch(taxon_ids: list) -> dict:
    """
    批量获取 taxon 信息（ancestor_ids + 自身 rank/name）。
    返回 {taxon_id_str: {"ancestor_ids": [...], "rank": ..., "name": ...}}
    """
    out = {}
    batch_size = 30
    for i in range(0, len(taxon_ids), batch_size):
        batch = taxon_ids[i: i + batch_size]
        results = _batch_request(INAT_TAXA_URL, {
            "id": ",".join(str(x) for x in batch),
            "per_page": len(batch),
        })
        for t in results:
            out[str(t["id"])] = {
                "ancestor_ids": [str(x) for x in t.get("ancestor_ids", [])],
                "rank":  t.get("rank", ""),
                "name":  t.get("name", ""),
            }
        time.sleep(API_DELAY)
    return out


def fetch_node_names(node_ids: list) -> dict:
    """
    批量获取 ancestor node 的 rank 和 name。
    返回 {node_id_str: {"rank": ..., "name": ...}}
    """
    out = {}
    batch_size = 30
    for i in tqdm(range(0, len(node_ids), batch_size),
                  desc="查询祖先节点", unit="batch"):
        batch = node_ids[i: i + batch_size]
        results = _batch_request(INAT_TAXA_URL, {
            "id": ",".join(str(x) for x in batch),
            "per_page": len(batch),
        })
        for t in results:
            out[str(t["id"])] = {
                "rank": t.get("rank", ""),
                "name": t.get("name", ""),
            }
        time.sleep(API_DELAY)
    return out


def build_taxonomy_row(taxon_id: str,
                       taxon_info: dict,
                       node_name_map: dict) -> dict:
    """
    用 ancestor_ids 列表 + node_name_map 拼装单个物种的 7 级分类学。
    """
    result = {v: "" for v in RANK_MAP.values()}
    info = taxon_info.get(str(taxon_id), {})

    # 处理祖先节点
    for anc_id in info.get("ancestor_ids", []):
        node = node_name_map.get(str(anc_id), {})
        rank = node.get("rank", "")
        name = node.get("name", "")
        if rank in RANK_MAP:
            result[RANK_MAP[rank]] = name

    # 处理物种自身（species 级别）
    self_rank = info.get("rank", "")
    self_name = info.get("name", "")
    if self_rank in RANK_MAP:
        result[RANK_MAP[self_rank]] = self_name

    return result


def enrich_taxonomy(df: pd.DataFrame, cache_path: str) -> pd.DataFrame:
    """
    两步批量策略补全全部 7 级分类学信息，总 API 请求约 50 次。
    有完整缓存则直接读取，无需重复请求。
    """
    need_cols = set(RANK_MAP.values())

    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, dtype={"taxon_id": str})
        if need_cols.issubset(set(cached.columns)):
            # 验证关键列是否真的有数据（不全是空）
            filled = cached["phylum"].notna() & (cached["phylum"] != "")
            if filled.sum() > len(cached) * 0.5:
                print(f"✓ 读取分类学缓存：{cache_path}  ({len(cached)} 行)")
                df = df.drop(columns=[c for c in need_cols if c in df.columns],
                             errors="ignore")
                df = df.merge(cached[["taxon_id"] + list(need_cols)],
                              on="taxon_id", how="left")
                return df
        print("⚠ 缓存数据不完整，重新构建…")

    all_ids = df["taxon_id"].tolist()
    print(f"Step 2a  批量获取 {len(all_ids)} 个物种的 ancestor_ids…")
    taxon_info = fetch_taxon_info_batch(all_ids)

    # 收集所有唯一 ancestor id
    all_ancestor_ids = set()
    for tid, info in taxon_info.items():
        all_ancestor_ids.update(info.get("ancestor_ids", []))
    all_ancestor_ids = list(all_ancestor_ids)
    print(f"Step 2b  查询 {len(all_ancestor_ids)} 个唯一祖先节点的 rank/name…")
    node_name_map = fetch_node_names(all_ancestor_ids)

    print("Step 2c  本地拼装分类学层级…")
    tax_rows = {}
    for tid in all_ids:
        tax_rows[tid] = build_taxonomy_row(tid, taxon_info, node_name_map)

    for col in RANK_MAP.values():
        df[col] = df["taxon_id"].map(lambda x, c=col: tax_rows.get(x, {}).get(c, ""))

    df["kingdom"] = df["kingdom"].replace("", "Plantae").fillna("Plantae")

    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"✓ 分类学信息已保存：{cache_path}")
    return df


# ─────────────────────────────────────────────
# Step 3  构建三种文本描述（Prompt Ensemble）
# ─────────────────────────────────────────────

def build_taxpath(row) -> str:
    """
    BioCLIP 官方训练格式：把 7 级分类学从根到叶拼成一条路径字符串。
    空缺层级自动跳过，不留空洞。
    示例：
      "a photo of Plantae Tracheophyta Magnoliopsida Ericales Ericaceae Rhododendron Rhododendron simsii"
    """
    parts = []
    for col in TAXON_RANKS:
        val = str(row.get(col, "") or "").strip()
        if val:
            parts.append(val)
    path = " ".join(parts)
    return f"a photo of {path}" if path else ""


def build_sciname(row) -> str:
    """
    科学名格式：
      "a photo of Rhododendron simsii"
    """
    sci = str(row.get("scientific_name", "") or "").strip()
    return f"a photo of {sci}" if sci else ""


def build_common(row) -> str:
    """
    常用名格式：
      "a photo of azalea"
    回退到科学名（BioCLIP 训练时若无常用名则用科学名）。
    """
    common = str(row.get("common_name", "") or "").strip()
    if common:
        return f"a photo of {common}"
    sci = str(row.get("scientific_name", "") or "").strip()
    return f"a photo of {sci}" if sci else ""


def build_all_texts(df: pd.DataFrame):
    """
    为每行生成三组文本列表：taxpath / sciname / common。
    同时返回有效行掩码（三种格式都非空的行）。
    """
    texts_taxpath = []
    texts_sci     = []
    texts_common  = []

    for _, row in df.iterrows():
        texts_taxpath.append(build_taxpath(row))
        texts_sci.append(build_sciname(row))
        texts_common.append(build_common(row))

    # 示例输出
    print(f"✓ 文本描述示例（第 1 行）：")
    print(f"    taxpath : {texts_taxpath[0]}")
    print(f"    sciname : {texts_sci[0]}")
    print(f"    common  : {texts_common[0]}")

    return texts_taxpath, texts_sci, texts_common


# ─────────────────────────────────────────────
# Step 4  加载模型
# ─────────────────────────────────────────────

def _local_weights_ready(weights_dir: str) -> bool:
    if not weights_dir or not os.path.isdir(weights_dir):
        return False
    has_weight = any(
        os.path.isfile(os.path.join(weights_dir, f))
        for f in _LOCAL_WEIGHT_CANDIDATES
    )
    if not has_weight:
        return False
    non_weight = [f for f in _LOCAL_REQUIRED_FILES if f not in _LOCAL_WEIGHT_CANDIDATES]
    return all(os.path.isfile(os.path.join(weights_dir, f)) for f in non_weight)


def _download_weights(repo_id: str, weights_dir: str):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("缺少 huggingface_hub，请运行：pip install huggingface-hub")
    os.makedirs(weights_dir, exist_ok=True)
    print(f"  正在从 HuggingFace 下载权重：{repo_id}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=weights_dir,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print("  ✓ 权重下载完成")


def load_bioclip2(weights_dir: str, hf_repo_id: str, arch: str):
    if weights_dir and not _local_weights_ready(weights_dir):
        print(f"⚠ 本地权重目录不完整：{os.path.abspath(weights_dir)}")
        print("  → 尝试从 HuggingFace Hub 下载…")
        try:
            _download_weights(hf_repo_id, weights_dir)
        except Exception as e:
            print(f"  ⚠ 下载失败（{e}），回退到 hf-hub 直接加载")
            weights_dir = None

    if weights_dir and _local_weights_ready(weights_dir):
        abs_dir = os.path.abspath(weights_dir)
        weight_file = None
        for c in _LOCAL_WEIGHT_CANDIDATES:
            p = os.path.join(abs_dir, c)
            if os.path.isfile(p):
                weight_file = p
                break
        if weight_file is None:
            raise FileNotFoundError(f"权重目录中未找到 .bin/.safetensors：{abs_dir}")
        print(f"加载 BioCLIP 2（本地权重）：{weight_file}  device={DEVICE}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=weight_file
        )
        try:
            tokenizer = open_clip.get_tokenizer(f"hf-hub:{hf_repo_id}")
        except Exception:
            tokenizer = open_clip.get_tokenizer(arch)
        source = f"本地  {weight_file}"
    else:
        hf_name = f"hf-hub:{hf_repo_id}"
        print(f"加载 BioCLIP 2（hf-hub 在线）：{hf_name}  device={DEVICE}")
        model, _, preprocess = open_clip.create_model_and_transforms(hf_name)
        tokenizer = open_clip.get_tokenizer(hf_name)
        source = f"hf-hub  {hf_repo_id}"

    model = model.to(DEVICE).eval()
    print(f"✓ 模型加载完成  [{source}]")
    return model, tokenizer, preprocess


# ─────────────────────────────────────────────
# Step 5  编码文本（单组）
# ─────────────────────────────────────────────

def encode_texts(model, tokenizer, texts: list, batch_size: int,
                 desc: str = "编码") -> np.ndarray:
    all_embs = []
    n_batches = math.ceil(len(texts) / batch_size)
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc=desc):
            batch = texts[i * batch_size: (i + 1) * batch_size]
            tokens = tokenizer(batch).to(DEVICE)
            embs = model.encode_text(tokens)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


# ─────────────────────────────────────────────
# Step 6  Prompt Ensemble：三组均值后再归一化
# ─────────────────────────────────────────────

def ensemble_embeddings(embs_taxpath: np.ndarray,
                        embs_sci:     np.ndarray,
                        embs_common:  np.ndarray) -> np.ndarray:
    """
    三组向量等权平均后重新 L2 归一化。
    这是 BioCLIP 论文推荐的 mixed text type inference 方式。
    """
    mean_emb = (embs_taxpath + embs_sci + embs_common) / 3.0
    norms = np.linalg.norm(mean_emb, axis=1, keepdims=True).clip(min=1e-8)
    return (mean_emb / norms).astype(np.float32)


# ─────────────────────────────────────────────
# Step 7  保存向量库
# ─────────────────────────────────────────────

def save_index(embeddings: np.ndarray, df: pd.DataFrame,
               texts_taxpath: list, texts_sci: list, texts_common: list,
               out_path: str):
    np.savez_compressed(
        out_path,
        embeddings    = embeddings.astype(np.float32),
        taxon_ids     = df["taxon_id"].fillna("").values.astype(str),
        sci_names     = df["scientific_name"].fillna("").values.astype(str),
        common_names  = df["common_name"].fillna("").values.astype(str)
                        if "common_name" in df.columns
                        else np.array([""] * len(df)),
        texts_taxpath = np.array(texts_taxpath, dtype=object),
        texts_sci     = np.array(texts_sci,     dtype=object),
        texts_common  = np.array(texts_common,  dtype=object),
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"✓ 向量库已保存：{out_path}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    # 1. 读取 CSV
    df = load_input_csv(INPUT_CSV)

    # 2. 补全完整 7 级分类学（phylum/class_/order 为新增层级）
    df = enrich_taxonomy(df, OUTPUT_TAXONOMY_CSV)

    # 3. 生成三种格式文本描述
    texts_taxpath, texts_sci, texts_common = build_all_texts(df)

    # 4. 加载模型
    model, tokenizer, _ = load_bioclip2(WEIGHTS_DIR, HF_REPO_ID, OPEN_CLIP_ARCH)

    # 5. 三组分别编码
    print("\n编码三种文本格式…")
    embs_taxpath = encode_texts(model, tokenizer, texts_taxpath,
                                BATCH_SIZE, desc="分类学路径")
    embs_sci     = encode_texts(model, tokenizer, texts_sci,
                                BATCH_SIZE, desc="科学名    ")
    embs_common  = encode_texts(model, tokenizer, texts_common,
                                BATCH_SIZE, desc="常用名    ")

    # 6. Prompt Ensemble
    print("\n合并向量（Prompt Ensemble）…")
    embeddings = ensemble_embeddings(embs_taxpath, embs_sci, embs_common)
    print(f"✓ 最终向量：shape={embeddings.shape}  dtype={embeddings.dtype}")

    # 7. 保存
    save_index(embeddings, df,
               texts_taxpath, texts_sci, texts_common,
               OUTPUT_INDEX_NPZ)

    print("\n构建完成！产出文件：")
    print(f"  {OUTPUT_TAXONOMY_CSV}  ← 含全部 7 级分类学的 CSV")
    print(f"  {OUTPUT_INDEX_NPZ}     ← Prompt Ensemble 向量库")
    print("\n向量库结构：")
    print("  embeddings    (N, D) float32  ← 三模板均值归一化向量（用于推理）")
    print("  texts_taxpath (N,)            ← 分类学路径文本（调试用）")
    print("  texts_sci     (N,)            ← 科学名文本")
    print("  texts_common  (N,)            ← 常用名文本")


if __name__ == "__main__":
    main()