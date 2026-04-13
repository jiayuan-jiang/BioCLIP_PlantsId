"""
BioCLIP 2 — 核心推理逻辑
共享给 inference.py (FastAPI) 和 handler.py (RunPod Serverless)
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import open_clip

logger = logging.getLogger("bioclip")

# ── 路径 & 常量 ──────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
INDEX_NPZ  = BASE_DIR / "bioclip_full_index.npz"
WEIGHTS_DIR = BASE_DIR / "weights"

HF_REPO_ID      = "imageomics/bioclip-2"
OPEN_CLIP_ARCH  = "ViT-L-14"
DEFAULT_TOPK    = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_WEIGHT_CANDIDATES = ["open_clip_pytorch_model.bin", "open_clip_model.safetensors"]

# ── 全局资源（加载后填充）───────────────────────────────────────────
_res: dict = {}


def load_resources() -> None:
    """加载向量索引 + 模型，结果写入 _res。"""

    if not INDEX_NPZ.is_file():
        raise FileNotFoundError(
            f"向量索引不存在：{INDEX_NPZ}\n"
            "请先执行：cp ../data/bioclip_full_index.npz demo/"
        )
    print("加载向量库…", flush=True)
    raw = np.load(INDEX_NPZ, allow_pickle=True)
    index: dict = {}
    for k in raw.files:
        if k == "embeddings":
            index["embeddings"] = raw[k].astype(np.float32)
        else:
            index[k] = raw[k].tolist()
    n, d = index["embeddings"].shape
    meta_fields = [k for k in index if k != "embeddings"]
    print(f"  物种数：{n:,}  维度：{d}  元字段：{meta_fields}")

    print("加载模型…", flush=True)
    weight_file = None
    if WEIGHTS_DIR.is_dir():
        for name in _WEIGHT_CANDIDATES:
            p = WEIGHTS_DIR / name
            if p.is_file():
                weight_file = str(p)
                break

    if weight_file:
        print(f"  本地权重：{weight_file}  device={DEVICE}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            OPEN_CLIP_ARCH, pretrained=weight_file
        )
    else:
        hf_name = f"hf-hub:{HF_REPO_ID}"
        print(f"  HuggingFace 在线：{hf_name}  device={DEVICE}")
        model, _, preprocess = open_clip.create_model_and_transforms(hf_name)

    model = model.to(DEVICE).eval()
    print(f"✓ 就绪  device={DEVICE}\n")

    _res["model"]      = model
    _res["preprocess"] = preprocess
    _res["index"]      = index


def infer(img: Image.Image, topk: int) -> list[dict]:
    model      = _res["model"]
    preprocess = _res["preprocess"]
    index      = _res["index"]

    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_emb = model.encode_image(tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        img_emb = img_emb.cpu().float().numpy()[0]

    sims    = index["embeddings"] @ img_emb
    top_idx = np.argsort(sims)[::-1][:topk]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        sim     = float(sims[idx])
        sim_pct = round(sim * 100, 2)
        confidence = "high" if sim_pct >= 35 else ("medium" if sim_pct >= 22 else "low")

        entry: dict = {
            "rank":           rank,
            "similarity":     round(sim, 5),
            "similarity_pct": sim_pct,
            "confidence":     confidence,
        }
        for field in index:
            if field == "embeddings":
                continue
            val = index[field][idx]
            val = _to_python(val)
            entry[field] = None if val is None or str(val) in ("nan", "None", "") else val

        results.append(entry)

    return results


def _to_python(val):
    """将 numpy 标量 / 数组转为 Python 原生类型，保证 JSON 可序列化。"""
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return None if np.isnan(val) else float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, float) and (val != val):
        return None
    return val
