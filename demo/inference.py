"""
BioCLIP 2 Demo — FastAPI Service
=================================
本地植物 / 动物图像识别服务，基于 BioCLIP 2 + 全量向量索引

目录结构（启动前确保以下文件已就位）：
  demo/
  ├── inference.py                  ← 本文件
  ├── bioclip_full_index.npz
  ├── weights/
  │   ├── open_clip_pytorch_model.bin
  │   └── open_clip_model.safetensors
  ├── static/
  │   └── index.html
  └── requirements.txt

启动：
  uvicorn claude:app --host 0.0.0.0 --port 8000
"""

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import open_clip

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("bioclip")

# ── 路径配置 ─────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
INDEX_NPZ = BASE_DIR / "bioclip_full_index.npz"
WEIGHTS_DIR = BASE_DIR / "weights"

HF_REPO_ID     = "imageomics/bioclip-2"
OPEN_CLIP_ARCH = "ViT-L-14"
DEFAULT_TOPK   = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_WEIGHT_CANDIDATES = ["open_clip_pytorch_model.bin", "open_clip_model.safetensors"]

# ── 全局资源（启动后填充）────────────────────────────────────────────
_res: dict = {}


def _load_resources() -> None:
    """加载向量索引 + 模型，结果写入 _res。"""

    # 向量库
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
            index["embeddings"] = raw[k].astype(np.float32)   # (N, D)
        else:
            index[k] = raw[k].tolist()
    n, d = index["embeddings"].shape
    meta_fields = [k for k in index if k != "embeddings"]
    print(f"  物种数：{n:,}  维度：{d}  元字段：{meta_fields}")

    # 模型权重（优先本地，回退 HuggingFace）
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

    _res["model"]     = model
    _res["preprocess"] = preprocess
    _res["index"]     = index


# ── FastAPI 生命周期 ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_resources()
    yield
    _res.clear()


app = FastAPI(
    title="BioCLIP 2 Demo",
    description="植物 / 动物图像识别，基于 BioCLIP 2 模型",
    version="1.0.0",
    lifespan=lifespan,
)


# ── 推理逻辑 ─────────────────────────────────────────────────────────

def _infer(img: Image.Image, topk: int) -> list[dict]:
    model     = _res["model"]
    preprocess = _res["preprocess"]
    index     = _res["index"]

    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_emb = model.encode_image(tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        img_emb = img_emb.cpu().float().numpy()[0]   # (D,)

    sims = index["embeddings"] @ img_emb             # (N,)
    top_idx = np.argsort(sims)[::-1][:topk]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        sim = float(sims[idx])
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
            # 强制转为 Python 原生类型，避免 numpy 类型导致 JSON 序列化失败
            val = _to_python(val)
            entry[field] = None if val is None or str(val) in ("nan", "None", "") else val

        results.append(entry)

    return results


def _to_python(val):
    """将 numpy 标量/数组转为 Python 原生类型，保证 JSON 可序列化。"""
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return None if np.isnan(val) else float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, float) and (val != val):   # Python nan
        return None
    return val


# ── 路由 ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    """服务健康检查。"""
    idx = _res.get("index", {})
    n = idx["embeddings"].shape[0] if "embeddings" in idx else 0
    return {"status": "ok", "device": DEVICE, "species_count": n}


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="待识别图片（jpg/png/webp…）"),
    topk: int = Query(default=DEFAULT_TOPK, ge=1, le=20, description="返回 Top-K 结果数"),
):
    """
    上传图片，返回 Top-K 物种预测结果。

    - **file**: 图片文件
    - **topk**: 返回结果数量（1–20）
    """
    if not _res:
        raise HTTPException(status_code=503, detail="模型尚未就绪，请稍候")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解析失败：{e}")

    try:
        results = _infer(img, topk)
    except Exception as e:
        logger.error("推理异常", exc_info=True)
        raise HTTPException(status_code=500, detail=f"推理失败：{e}")

    # 用标准 json 序列化，确保 numpy 类型已被 _to_python 处理
    return Response(
        content=json.dumps({"results": results}, ensure_ascii=False),
        media_type="application/json",
    )


# ── 入口 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=False)
