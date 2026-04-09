"""
BioCLIP 2 植物向量库构建脚本
============================
步骤：
  1. 读取 iNaturalist CSV（含 taxon_id + scientific_name）
  2. 批量查询 iNat API 补全 genus / family / kingdom 分类学层级
  3. 拼接文本描述：
       "a photo of {species}, a species of {genus}, family {family}, {kingdom}"
  4. 用 BioCLIP 2 文本编码器批量编码，归一化后保存向量库
  5. 输出：taxonomy_enriched.csv  +  plant_text_index.npz

依赖安装：
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

INPUT_CSV = "global_plants_top5000.csv"   # 上一步产出的 CSV

# ── 权重目录 ──────────────────────────────────
# 优先从此目录加载本地权重文件；目录不存在或文件缺失时自动联网下载，
# 下载完成后权重会保存到该目录供下次离线使用。
# 留空字符串 "" 则直接走在线拉取（不做本地缓存覆盖）。
WEIGHTS_DIR = "./weights/bioclip2"

# HuggingFace Hub 仓库 ID（在线拉取时使用）
HF_REPO_ID  = "imageomics/bioclip-2"

# open_clip 架构名（本地加载时需要指定，与 HF config 对应）
# BioCLIP 2 使用 ViT-L/14；BioCLIP 1 使用 ViT-B/16
OPEN_CLIP_ARCH = "ViT-L-14"

# ── 其他参数 ──────────────────────────────────
TEXT_TEMPLATE = (
    "a photo of {species}, a species of {genus}, family {family}, {kingdom}"
)

OUTPUT_TAXONOMY_CSV = "taxonomy_enriched.csv"   # 补全分类学后的 CSV（可缓存复用）
OUTPUT_INDEX_NPZ    = "plant_text_index.npz"    # 向量库（embeddings + metadata）

BATCH_SIZE     = 256     # 文本编码批大小，显存不足时调小
API_BATCH_SIZE = 30      # 每次向 iNat API 查询的 taxon 数量（最大 30）
API_DELAY      = 0.3     # 请求间隔秒数
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════


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
# Step 2  补全分类学层级（带本地缓存）
# ─────────────────────────────────────────────

INAT_TAXON_BATCH_URL = "https://api.inaturalist.org/v1/taxa"
RANK_ORDER = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def parse_ancestors(taxon: dict) -> dict:
    """从 taxon JSON 中提取 genus / family / kingdom。"""
    result = {"genus": "", "family": "", "kingdom": ""}
    ancestors = taxon.get("ancestors", [])
    ancestors_plus_self = ancestors + [taxon]
    for anc in ancestors_plus_self:
        r = anc.get("rank", "")
        n = anc.get("name", "")
        if r in result:
            result[r] = n
    return result


def fetch_taxonomy_batch(taxon_ids: list[str]) -> dict[str, dict]:
    """查询一批 taxon_id，返回 {taxon_id: {genus, family, kingdom}}。"""
    ids_str = ",".join(taxon_ids)
    params = {"id": ids_str, "per_page": len(taxon_ids)}
    try:
        r = requests.get(INAT_TAXON_BATCH_URL, params=params, timeout=30,
                         headers={"User-Agent": "bioclip2-builder/1.0"})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ⚠ API 请求失败：{e}，将跳过本批次")
        return {}

    out = {}
    for taxon in data.get("results", []):
        tid = str(taxon["id"])
        out[tid] = parse_ancestors(taxon)
    return out


def enrich_taxonomy(df: pd.DataFrame, cache_path: str) -> pd.DataFrame:
    """
    为 df 中每行补全 genus / family / kingdom。
    已有缓存文件则直接读取；否则逐批请求 API 并写缓存。
    """
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, dtype={"taxon_id": str})
        need_cols = {"genus", "family", "kingdom"}
        if need_cols.issubset(set(cached.columns)):
            print(f"✓ 读取分类学缓存：{cache_path}  ({len(cached)} 行)")
            # 以 taxon_id 为键合并，左连接保留原 df 顺序
            df = df.drop(columns=[c for c in need_cols if c in df.columns], errors="ignore")
            df = df.merge(cached[["taxon_id"] + list(need_cols)],
                          on="taxon_id", how="left")
            return df

    print(f"未找到缓存，开始从 iNaturalist API 补全分类学信息…")
    all_ids = df["taxon_id"].tolist()
    total_batches = math.ceil(len(all_ids) / API_BATCH_SIZE)
    tax_map: dict[str, dict] = {}

    for i in tqdm(range(total_batches), desc="查询分类学"):
        batch = all_ids[i * API_BATCH_SIZE: (i + 1) * API_BATCH_SIZE]
        result = fetch_taxonomy_batch(batch)
        tax_map.update(result)
        time.sleep(API_DELAY)

    # 映射回 df
    df["genus"]   = df["taxon_id"].map(lambda x: tax_map.get(x, {}).get("genus",   ""))
    df["family"]  = df["taxon_id"].map(lambda x: tax_map.get(x, {}).get("family",  ""))
    df["kingdom"] = df["taxon_id"].map(lambda x: tax_map.get(x, {}).get("kingdom", ""))

    # 兜底：kingdom 为空时填 "Plantae"
    df["kingdom"] = df["kingdom"].replace("", "Plantae").fillna("Plantae")

    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"✓ 分类学信息已保存：{cache_path}")
    return df


# ─────────────────────────────────────────────
# Step 3  拼接文本描述
# ─────────────────────────────────────────────

def build_text_descriptions(df: pd.DataFrame, template: str) -> list[str]:
    texts = []
    for _, row in df.iterrows():
        species = row.get("scientific_name", "") or row.get("species", "")
        genus   = row.get("genus",   "") or species.split()[0] if species else ""
        family  = row.get("family",  "") or "unknown family"
        kingdom = row.get("kingdom", "") or "Plantae"
        txt = template.format(
            species=species,
            genus=genus,
            family=family,
            kingdom=kingdom,
        )
        texts.append(txt)
    print(f"✓ 文本描述示例：\n    {texts[0]}\n    {texts[1]}")
    return texts


# ─────────────────────────────────────────────
# Step 4  BioCLIP 2 加载（本地优先，缺失则下载）
# ─────────────────────────────────────────────

# open_clip 从本地目录加载时，目录内需要包含：
#   open_clip_pytorch_model.bin  （或 .safetensors）
#   open_clip_config.json
#   tokenizer 相关文件（vocab / merges / special_tokens_map / tokenizer_config）
# 这些文件均可从 HuggingFace Hub imageomics/bioclip2 直接下载。

_LOCAL_REQUIRED_FILES = [
    "open_clip_pytorch_model.bin",
    "open_clip_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

# safetensors 格式与 bin 二选一即可
_LOCAL_WEIGHT_CANDIDATES = [
    "open_clip_pytorch_model.bin",
    "model.safetensors",
]


def _local_weights_ready(weights_dir: str) -> bool:
    """检查本地目录是否包含全部必要文件（权重文件有任意一个即可）。"""
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
    """从 HuggingFace Hub 下载权重到本地目录。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "缺少 huggingface_hub，请运行：pip install huggingface-hub"
        )

    os.makedirs(weights_dir, exist_ok=True)
    print(f"  正在从 HuggingFace 下载权重：{repo_id}")
    print(f"  保存目录：{os.path.abspath(weights_dir)}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=weights_dir,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"  ✓ 权重下载完成")


def load_bioclip2(weights_dir: str, hf_repo_id: str, arch: str):
    """
    加载 BioCLIP 2 模型。
    优先级：本地目录 → 在线下载（下载后存入本地目录）→ hf-hub 直接加载
    """
    # ── 判断本地是否就绪 ──
    if weights_dir and not _local_weights_ready(weights_dir):
        if weights_dir:
            missing = []
            if os.path.isdir(weights_dir):
                missing = [
                    f for f in _LOCAL_REQUIRED_FILES
                    if not os.path.isfile(os.path.join(weights_dir, f))
                    and f not in _LOCAL_WEIGHT_CANDIDATES
                ]
                if not any(
                    os.path.isfile(os.path.join(weights_dir, f))
                    for f in _LOCAL_WEIGHT_CANDIDATES
                ):
                    missing.insert(0, "权重文件（.bin/.safetensors）")
            print(f"⚠ 本地权重目录不完整：{os.path.abspath(weights_dir)}")
            if missing:
                print(f"  缺少文件：{missing}")
        print("  → 尝试从 HuggingFace Hub 下载…")
        try:
            _download_weights(hf_repo_id, weights_dir)
        except Exception as e:
            print(f"  ⚠ 下载失败（{e}），回退到 hf-hub 直接加载")
            weights_dir = None   # 触发下方 hf-hub 分支

    # ── 加载模型 ──
    if weights_dir and _local_weights_ready(weights_dir):
        abs_dir = os.path.abspath(weights_dir)

        # open_clip 的 pretrained 参数必须是具体文件路径，不能是目录
        weight_file = None
        for candidate in _LOCAL_WEIGHT_CANDIDATES:
            p = os.path.join(abs_dir, candidate)
            if os.path.isfile(p):
                weight_file = p
                break
        if weight_file is None:
            raise FileNotFoundError(f"权重目录中未找到 .bin/.safetensors 文件：{abs_dir}")

        print(f"加载 BioCLIP 2（本地权重）：{weight_file}  (device={DEVICE})")
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch,
            pretrained=weight_file,   # ← 必须是文件路径
        )
        # tokenizer 用 hf-hub 路径加载以获取正确词表
        # 若离线环境无法访问 HF，则退回 arch 名称（词表可能略有差异）
        try:
            tokenizer = open_clip.get_tokenizer(f"hf-hub:{hf_repo_id}")
        except Exception:
            tokenizer = open_clip.get_tokenizer(arch)
        source = f"本地  {weight_file}"
    else:
        hf_name = f"hf-hub:{hf_repo_id}"
        print(f"加载 BioCLIP 2（hf-hub 在线）：{hf_name}  (device={DEVICE})")
        model, _, preprocess = open_clip.create_model_and_transforms(hf_name)
        tokenizer = open_clip.get_tokenizer(hf_name)
        source = f"hf-hub  {hf_repo_id}"

    model = model.to(DEVICE).eval()
    print(f"✓ 模型加载完成  [{source}]")
    return model, tokenizer, preprocess


def encode_texts(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    all_embs = []
    n_batches = math.ceil(len(texts) / batch_size)
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="文本编码"):
            batch = texts[i * batch_size: (i + 1) * batch_size]
            tokens = tokenizer(batch).to(DEVICE)
            embs = model.encode_text(tokens)
            # L2 归一化（余弦相似度搜索必要）
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().float().numpy())
    embeddings = np.concatenate(all_embs, axis=0)
    print(f"✓ 编码完成：shape={embeddings.shape}  dtype={embeddings.dtype}")
    return embeddings


# ─────────────────────────────────────────────
# Step 5  保存向量库
# ─────────────────────────────────────────────

def save_index(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    texts: list[str],
    out_path: str,
):
    """
    保存 .npz，包含：
      embeddings  : float32 (N, D)
      taxon_ids   : str array
      sci_names   : str array
      common_names: str array
      texts       : str array（拼接后的文本描述）
    """
    np.savez_compressed(
        out_path,
        embeddings   = embeddings.astype(np.float32),
        taxon_ids    = df["taxon_id"].fillna("").values.astype(str),
        sci_names    = df["scientific_name"].fillna("").values.astype(str),
        common_names = df["common_name"].fillna("").values.astype(str) if "common_name" in df.columns else np.array([""] * len(df)),
        texts        = np.array(texts, dtype=object),
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"✓ 向量库已保存：{out_path}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────
# 推理时使用示例（供参考，非构建流程）
# ─────────────────────────────────────────────

def inference_example(index_path: str, image_path: str):
    """
    示例：加载向量库 + 对单张图像做零样本分类。
    运行方式：在构建完成后单独调用此函数。
    """
    from PIL import Image

    print("\n── 推理示例 ──")
    index = np.load(index_path, allow_pickle=True)
    text_embs    = index["embeddings"]     # (N, D) float32
    sci_names    = index["sci_names"]
    common_names = index["common_names"]

    model, tokenizer, preprocess = load_bioclip2(WEIGHTS_DIR, HF_REPO_ID, OPEN_CLIP_ARCH)

    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_emb = model.encode_image(img)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        img_emb = img_emb.cpu().float().numpy()[0]   # (D,)

    sims = text_embs @ img_emb   # (N,) cosine similarity
    top5 = np.argsort(sims)[::-1][:5]

    print(f"图像：{image_path}")
    print("Top-5 预测：")
    for rank, idx in enumerate(top5, 1):
        cn = common_names[idx] if common_names[idx] else "—"
        print(f"  {rank}. {sci_names[idx]} ({cn})  sim={sims[idx]:.4f}")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    # 1. 读取 CSV
    df = load_input_csv(INPUT_CSV)

    # 2. 补全分类学（有缓存则跳过 API 请求）
    df = enrich_taxonomy(df, OUTPUT_TAXONOMY_CSV)

    # 3. 拼接文本描述
    texts = build_text_descriptions(df, TEXT_TEMPLATE)

    # 4. 加载模型并编码
    model, tokenizer, _ = load_bioclip2(WEIGHTS_DIR, HF_REPO_ID, OPEN_CLIP_ARCH)
    embeddings = encode_texts(model, tokenizer, texts, BATCH_SIZE)

    # 5. 保存向量库
    save_index(embeddings, df, texts, OUTPUT_INDEX_NPZ)

    print("\n构建完成！产出文件：")
    print(f"  {OUTPUT_TAXONOMY_CSV}  ← 含 genus/family/kingdom 的完整分类学 CSV")
    print(f"  {OUTPUT_INDEX_NPZ}     ← 文本向量库（embeddings + 元数据）")
    print("\n推理时加载方式：")
    print("  index = np.load('plant_text_index.npz', allow_pickle=True)")
    print("  embeddings = index['embeddings']   # float32 (N, D)")
    print("  sci_names  = index['sci_names']")
    print("  # img_emb @ embeddings.T → cosine similarity")


if __name__ == "__main__":
    main()