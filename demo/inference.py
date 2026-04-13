"""
BioCLIP 2 Demo — FastAPI Service
=================================
本地植物 / 动物图像识别服务，基于 BioCLIP 2 + 全量向量索引

启动：
  uvicorn inference:app --host 0.0.0.0 --port 8000
"""

import io
import json
import logging
from contextlib import asynccontextmanager

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from inference_core import BASE_DIR, DEFAULT_TOPK, DEVICE, _res, load_resources, infer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("bioclip")


# ── FastAPI 生命周期 ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield
    _res.clear()


app = FastAPI(
    title="BioCLIP 2 Demo",
    description="植物 / 动物图像识别，基于 BioCLIP 2 模型",
    version="1.0.0",
    lifespan=lifespan,
)


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
    """上传图片，返回 Top-K 物种预测结果。"""
    if not _res:
        raise HTTPException(status_code=503, detail="模型尚未就绪，请稍候")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解析失败：{e}")

    try:
        results = infer(img, topk)
    except Exception as e:
        logger.error("推理异常", exc_info=True)
        raise HTTPException(status_code=500, detail=f"推理失败：{e}")

    return Response(
        content=json.dumps({"results": results}, ensure_ascii=False),
        media_type="application/json",
    )


# ── 入口 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=False)
