"""
BioCLIP 2 单图推理脚本
======================
输入：本地图片路径 或 图片 URL
输出：Top-K 预测结果，含完整 taxonomy_enriched 信息

用法：修改 main() 底部的 IMAGE_SOURCE 和 TOPK，直接运行
  python infer.py
"""

import os
import sys
import tempfile
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd
from PIL import Image

import torch
import open_clip


# ══════════════════════════════════════════════
#  修改这里的变量
# ══════════════════════════════════════════════

INDEX_NPZ      = "./data/bioclip_full_index.npz"     # 向量库
TAXONOMY_CSV   = "./data/taxonomy_enriched.csv"    # 完整分类学 CSV

WEIGHTS_DIR    = "./data/weights/bioclip2"
HF_REPO_ID     = "imageomics/bioclip-2"
OPEN_CLIP_ARCH = "ViT-L-14"

DEFAULT_TOPK   = 5

# ══════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LOCAL_WEIGHT_CANDIDATES = ["open_clip_pytorch_model.bin", "model.safetensors"]


# ─────────────────────────────────────────────
# 加载资源（只在首次调用时执行）
# ─────────────────────────────────────────────

_cache = {}

def get_resources():
    if _cache:
        return _cache["model"], _cache["preprocess"], \
               _cache["text_embs"], _cache["df"], _cache["taxon_ids"]

    # 向量库
    print("加载向量库…", flush=True)
    idx = np.load(INDEX_NPZ, allow_pickle=True)
    text_embs = idx["embeddings"].astype(np.float32)   # (N, D)
    taxon_ids = idx["taxon_ids"].tolist()

    # taxonomy CSV（含完整分类学字段）
    print("加载分类学数据…", flush=True)
    df = pd.read_csv(TAXONOMY_CSV, dtype={"taxon_id": str})
    # 以 taxon_id 为索引方便 O(1) 查询
    df = df.set_index("taxon_id")

    # 模型
    print("加载模型…", flush=True)
    weight_file = None
    if WEIGHTS_DIR and os.path.isdir(WEIGHTS_DIR):
        for c in _LOCAL_WEIGHT_CANDIDATES:
            p = os.path.join(WEIGHTS_DIR, c)
            if os.path.isfile(p):
                weight_file = p
                break

    if weight_file:
        print(f"  本地权重：{weight_file}  device={DEVICE}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            OPEN_CLIP_ARCH, pretrained=weight_file
        )
        try:
            tokenizer = open_clip.get_tokenizer(f"hf-hub:{HF_REPO_ID}")
        except Exception:
            tokenizer = open_clip.get_tokenizer(OPEN_CLIP_ARCH)
    else:
        hf_name = f"hf-hub:{HF_REPO_ID}"
        print(f"  hf-hub 在线：{hf_name}  device={DEVICE}")
        model, _, preprocess = open_clip.create_model_and_transforms(hf_name)
        tokenizer = open_clip.get_tokenizer(hf_name)

    model = model.to(DEVICE).eval()
    print("✓ 就绪\n")

    _cache.update(dict(model=model, preprocess=preprocess,
                       text_embs=text_embs, df=df, taxon_ids=taxon_ids))
    return model, preprocess, text_embs, df, taxon_ids


# ─────────────────────────────────────────────
# 图片加载（路径 或 URL）
# ─────────────────────────────────────────────

def load_image(source: str) -> Image.Image:
    parsed = urllib.parse.urlparse(source)
    is_url = parsed.scheme in ("http", "https")

    if is_url:
        print(f"下载图片：{source}")
        headers = {"User-Agent": "bioclip2-infer/1.0"}
        req = urllib.request.Request(source, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        # 写入临时文件再用 PIL 打开，避免某些格式解码问题
        suffix = os.path.splitext(parsed.path)[-1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        img = Image.open(tmp_path).convert("RGB")
        os.unlink(tmp_path)
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"图片文件不存在：{source}")
        img = Image.open(source).convert("RGB")

    return img


# ─────────────────────────────────────────────
# 推理
# ─────────────────────────────────────────────

def predict(image_source: str, topk: int = DEFAULT_TOPK) -> list[dict]:
    """
    输入图片路径或 URL，返回 Top-K 预测列表。
    每项是 taxonomy_enriched.csv 中对应行的完整字段 + 相似度分数。
    """
    model, preprocess, text_embs, df, taxon_ids = get_resources()

    # 编码图像
    img = load_image(image_source)
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_emb = model.encode_image(tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        img_emb = img_emb.cpu().float().numpy()[0]   # (D,)

    # 余弦相似度
    sims = text_embs @ img_emb   # (N,)
    topk_idx = np.argsort(sims)[::-1][:topk]

    results = []
    for rank, idx in enumerate(topk_idx, start=1):
        tid = taxon_ids[idx]
        sim = float(sims[idx])

        # 从 CSV 取完整分类学数据
        if tid in df.index:
            row = df.loc[tid].to_dict()
        else:
            row = {}

        results.append({
            "rank":           rank,
            "taxon_id":       tid,
            "similarity":     round(sim, 5),
            **row,            # 展开 taxonomy_enriched 中的所有字段
        })

    return results


# ─────────────────────────────────────────────
# 格式化输出
# ─────────────────────────────────────────────

def print_results(results: list[dict], image_source: str):
    width = 60
    print("─" * width)
    print(f"图片：{image_source}")
    print("─" * width)

    # 确定要展示的字段（按优先顺序，有则显示）
    display_fields = [
        ("taxon_id",        "Taxon ID"),
        ("scientific_name", "学名"),
        ("common_name",     "常用名"),
        ("genus",           "属"),
        ("family",          "科"),
        ("kingdom",         "界"),
        ("rank_level",      "分类等级"),
        ("observations_count", "iNat 观测数"),
    ]

    for res in results:
        sim_pct = res["similarity"] * 100
        # 置信度判断
        if sim_pct >= 35:
            confidence = "高"
            tag = "✓"
        elif sim_pct >= 22:
            confidence = "中"
            tag = "?"
        else:
            confidence = "低"
            tag = "✗"

        print(f"\n  #{res['rank']}  相似度 {sim_pct:.2f}%  [{confidence}置信度 {tag}]")
        for field_key, field_label in display_fields:
            val = res.get(field_key, "")
            if val != "" and not (isinstance(val, float) and np.isnan(val)):
                print(f"    {field_label:<12}：{val}")

    print("\n" + "─" * width)

    # 置信度提示
    top_sim = results[0]["similarity"] * 100 if results else 0
    if top_sim < 22:
        print("⚠ 置信度过低，建议提供更清晰的图片或尝试局部特写（叶片/花/果实）。")
    elif top_sim < 35:
        print("⚠ 置信度中等，结果仅供参考，建议结合属级信息判断。")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():"""
BioCLIP 2 单图推理脚本
======================
输入：本地图片路径 或 图片 URL
输出：Top-K 预测结果

用法：修改 main() 底部的变量，直接运行
  python infer.py
"""

import os
import tempfile
import urllib.request
import urllib.parse

import numpy as np
from PIL import Image

import torch
import open_clip


# ══════════════════════════════════════════════
#  修改这里的变量
# ══════════════════════════════════════════════

INDEX_NPZ      = "./data/bioclip_full_index.npz"

WEIGHTS_DIR    = "./data/weights/bioclip2"
HF_REPO_ID     = "imageomics/bioclip-2"
OPEN_CLIP_ARCH = "ViT-L-14"

DEFAULT_TOPK   = 5

# ══════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LOCAL_WEIGHT_CANDIDATES = ["open_clip_pytorch_model.bin", "model.safetensors"]

_cache = {}


# ─────────────────────────────────────────────
# 加载资源
# ─────────────────────────────────────────────

def get_resources():
    if _cache:
        return _cache["model"], _cache["preprocess"], _cache["index"]

    print("加载向量库…", flush=True)
    raw = np.load(INDEX_NPZ, allow_pickle=True)
    index = {}
    for k in raw.files:
        if k == "embeddings":
            index["embeddings"] = raw[k].astype(np.float32)
        else:
            index[k] = raw[k].tolist()
    n = index["embeddings"].shape[0]
    d = index["embeddings"].shape[1]
    fields = [k for k in index if k != "embeddings"]
    print(f"  物种数：{n:,}，维度：{d}，字段：{fields}")

    print("加载模型…", flush=True)
    weight_file = None
    if WEIGHTS_DIR and os.path.isdir(WEIGHTS_DIR):
        for c in _LOCAL_WEIGHT_CANDIDATES:
            p = os.path.join(WEIGHTS_DIR, c)
            if os.path.isfile(p):
                weight_file = p
                break

    if weight_file:
        print(f"  本地权重：{weight_file}  device={DEVICE}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            OPEN_CLIP_ARCH, pretrained=weight_file)
    else:
        hf_name = f"hf-hub:{HF_REPO_ID}"
        print(f"  hf-hub：{hf_name}  device={DEVICE}")
        model, _, preprocess = open_clip.create_model_and_transforms(hf_name)

    model = model.to(DEVICE).eval()
    print("✓ 就绪\n")

    _cache.update(dict(model=model, preprocess=preprocess, index=index))
    return model, preprocess, index


# ─────────────────────────────────────────────
# 图片加载
# ─────────────────────────────────────────────

def load_image(source: str) -> Image.Image:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in ("http", "https"):
        print(f"下载图片：{source}")
        req = urllib.request.Request(
            source, headers={"User-Agent": "bioclip2-infer/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        suffix = os.path.splitext(parsed.path)[-1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        img = Image.open(tmp_path).convert("RGB")
        os.unlink(tmp_path)
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"图片文件不存在：{source}")
        img = Image.open(source).convert("RGB")
    return img


# ─────────────────────────────────────────────
# 推理
# ─────────────────────────────────────────────

def predict(image_source: str, topk: int = DEFAULT_TOPK) -> list:
    model, preprocess, index = get_resources()

    img = load_image(image_source)
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_emb = model.encode_image(tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        img_emb = img_emb.cpu().float().numpy()[0]

    sims = index["embeddings"] @ img_emb
    topk_idx = np.argsort(sims)[::-1][:topk]

    results = []
    for rank, idx in enumerate(topk_idx, start=1):
        entry = {"rank": rank, "similarity": round(float(sims[idx]), 5)}
        for field in index:
            if field != "embeddings":
                entry[field] = index[field][idx]
        results.append(entry)

    return results


# ─────────────────────────────────────────────
# 格式化输出
# ─────────────────────────────────────────────

def print_results(results: list, image_source: str):
    width = 64
    print("─" * width)
    print(f"图片：{image_source}")
    print("─" * width)

    # 字段显示优先级
    display_fields = [
        ("sci_names",    "学名"),
        ("common_names", "常用名"),
        ("taxon_paths",  "分类学路径"),
        ("genera",       "属"),
        ("families",     "科"),
        ("kingdoms",     "界"),
        ("taxon_ids",    "Taxon ID"),
    ]

    for res in results:
        sim_pct = res["similarity"] * 100
        if sim_pct >= 35:
            conf, tag = "高", "✓"
        elif sim_pct >= 22:
            conf, tag = "中", "?"
        else:
            conf, tag = "低", "✗"

        print(f"\n  #{res['rank']}  相似度 {sim_pct:.2f}%  [{conf}置信度 {tag}]")
        for field_key, label in display_fields:
            val = res.get(field_key, "")
            if val and str(val).strip() and val != "nan":
                print(f"    {label:<12}：{val}")

    print("\n" + "─" * width)
    top_sim = results[0]["similarity"] * 100 if results else 0
    if top_sim < 22:
        print("⚠ 置信度过低，建议提供更清晰的图片或尝试局部特写。")
    elif top_sim < 35:
        print("⚠ 置信度中等，结果仅供参考。")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    # ── 修改这里 ──────────────────────────────
    IMAGE_SOURCE = "./demoday/red_maple.png"
    TOPK         = 10
    OUTPUT_JSON  = False
    # ──────────────────────────────────────────

    results = predict(IMAGE_SOURCE, topk=TOPK)

    if OUTPUT_JSON:
        import json
        def _clean(v):
            if isinstance(v, float) and np.isnan(v):
                return None
            return v
        print(json.dumps(
            [{k: _clean(v) for k, v in r.items()} for r in results],
            ensure_ascii=False, indent=2))
    else:
        print_results(results, IMAGE_SOURCE)



if __name__ == "__main__":
    main()