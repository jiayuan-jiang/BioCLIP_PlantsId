"""
BioCLIP 2 Demo — FastAPI Service
=================================
本地植物 / 动物图像识别服务，基于 BioCLIP 2 + 全量向量索引
+ 配图推荐（/recommend，两段式 ANN + 双温度打分，跑在独立进程池）

启动：
  uvicorn inference:app --host 0.0.0.0 --port 8000
可选环境变量：
  RECO_WORKERS=2            配图推荐进程池 worker 数
  RECO_ARTIFACTS=<dir>      构建产物目录（默认 <此文件目录>/deploy_artifacts）
  RECO_EF_SEARCH=384        HNSW efSearch
  BIOCLIP_LOG_DIR=<dir>     访问日志目录（默认 <此文件目录>/logs），写 access.jsonl（JSONL，按大小轮转）
"""

import asyncio
import io
import json
import logging
import multiprocessing as mp
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response

from inference_core import BASE_DIR, DEFAULT_TOPK, DEVICE, _res, load_resources, infer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("bioclip")

# ── 配图推荐：进程池配置 ──────────────────────────────────────────────
RECO_WORKERS = int(os.environ.get("RECO_WORKERS", "2"))
RECO_ARTIFACTS = os.environ.get("RECO_ARTIFACTS") or str(BASE_DIR / "deploy_artifacts")
RECO_EF_SEARCH = int(os.environ.get("RECO_EF_SEARCH", "384"))

_reco_pool: ProcessPoolExecutor | None = None
_reco_names: list[str] = []            # 图集 20k 物种名（主进程持一份，做 in-pool 判断 + 模糊建议）
_reco_sp2i: dict[str, int] = {}
_id_sci2row: dict[str, int] = {}       # 识图索引 35万种 学名 → 行号（图集外物种取文本原型用）


# ── 访问日志（JSONL，每请求一行）─────────────────────────────────────
# 记录：谁（ip/ua）· 什么时候（ts）· 查了什么（endpoint/params）· 结果是什么
# （predict 的 top 物种 / recommend 每部位返回的 photo_id）· 耗时 / 状态。
# 目的：事后能复盘、发现坏 case、给未来改进（打分、prune、organ 头）攒素材。
LOG_DIR = os.environ.get("BIOCLIP_LOG_DIR") or str(BASE_DIR / "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_alogger = logging.getLogger("bioclip.access")
_alogger.propagate = False
if not _alogger.handlers:
    _h = RotatingFileHandler(
        os.path.join(LOG_DIR, "access.jsonl"),
        maxBytes=100 * 1024 * 1024, backupCount=10, encoding="utf-8",
    )
    _h.setFormatter(logging.Formatter("%(message)s"))
    _alogger.addHandler(_h)
    _alogger.setLevel(logging.INFO)


def _client_ip(req: Request) -> str:
    xff = req.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return req.client.host if req.client else "?"


def _alog(record: dict) -> None:
    record["ts"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    try:
        _alogger.info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:
        logger.exception("access-log 写入失败")


# ── FastAPI 生命周期 ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()

    global _reco_pool, _reco_names, _reco_sp2i, _id_sci2row
    # 识图索引的学名→行号（图集外物种，用它的文本塔向量当 query）
    _idx = _res.get("index", {})
    _sci = _idx.get("sci_names", [])
    _id_sci2row = {s: i for i, s in enumerate(_sci)}

    species_json = os.path.join(RECO_ARTIFACTS, "recommend_species.json")
    if os.path.isfile(species_json):
        from recommend._worker import init_worker

        _reco_names = json.load(open(species_json))
        _reco_sp2i = {s: i for i, s in enumerate(_reco_names)}
        # 必须 spawn：主进程已加载 torch（含线程池）+ asyncio，fork 出来的 worker
        # 会继承被污染的锁/线程状态 → 死锁（Linux 默认 fork）。spawn 干净重启。
        _reco_pool = ProcessPoolExecutor(
            max_workers=RECO_WORKERS,
            mp_context=mp.get_context("spawn"),
            initializer=init_worker,
            initargs=(RECO_ARTIFACTS, RECO_EF_SEARCH),
        )
        # 预热：给每个 worker 丢一个 dummy job，让它在后台把 faiss/meta/topm 读进来。
        # 不等结果（fire-and-forget）—— 服务立刻可用；前几个真实请求可能排在预热后面几秒。
        from recommend._worker import run_job
        warm = dict(species_or_idx=0, parts=["leaf"], topk=1, t_sp=None, t_org=None, pool_k=200)
        for _ in range(RECO_WORKERS):
            _reco_pool.submit(run_job, warm)

        logger.info(
            "配图推荐就绪  artifacts=%s  workers=%d(后台预热中)  efSearch=%d  species=%d",
            RECO_ARTIFACTS, RECO_WORKERS, RECO_EF_SEARCH, len(_reco_names),
        )
    else:
        logger.warning("配图推荐产物不存在（%s），/recommend 关闭", species_json)

    logger.info("访问日志：%s", os.path.join(LOG_DIR, "access.jsonl"))
    _alog({"endpoint": "startup", "recommend_enabled": _reco_pool is not None,
           "workers": RECO_WORKERS, "ef_search": RECO_EF_SEARCH, "pool_species": len(_reco_names)})

    yield

    _alog({"endpoint": "shutdown"})
    if _reco_pool is not None:
        _reco_pool.shutdown(wait=False, cancel_futures=True)
    _res.clear()


app = FastAPI(
    title="BioCLIP 2 Demo",
    description="植物 / 动物图像识别 + 配图推荐，基于 BioCLIP 2 模型",
    version="1.1.0",
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
    return {
        "status": "ok",
        "device": DEVICE,
        "species_count": n,
        "recommend": {
            "enabled": _reco_pool is not None,
            "workers": RECO_WORKERS if _reco_pool is not None else 0,
            "pool_species": len(_reco_names),
        },
    }


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(..., description="待识别图片（jpg/png/webp…）"),
    topk: int = Query(default=DEFAULT_TOPK, ge=1, le=20, description="返回 Top-K 结果数"),
):
    """上传图片，返回 Top-K 物种预测结果。"""
    rid = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    rec = {"endpoint": "predict", "id": rid, "ip": _client_ip(request),
           "ua": request.headers.get("user-agent", "")[:200], "topk": topk,
           "filename": (file.filename or "")[:200]}

    if not _res:
        rec.update(status=503)
        _alog(rec)
        raise HTTPException(status_code=503, detail="模型尚未就绪，请稍候")

    data = await file.read()
    rec["bytes"] = len(data)
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        rec.update(status=400, error=str(e)[:200], latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        _alog(rec)
        raise HTTPException(status_code=400, detail=f"图片解析失败：{e}")

    try:
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(None, infer, img, topk)  # ViT-L 前向丢线程池，不卡事件循环
    except Exception as e:
        rec.update(status=500, error=str(e)[:300], latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        _alog(rec)
        logger.error("推理异常", exc_info=True)
        raise HTTPException(status_code=500, detail=f"推理失败：{e}")

    # infer() 可能返回 list（旧版 inference_core）或 dict（带 genus_confidence 的新版）
    if isinstance(payload, list):
        payload = {"results": payload, "genus_confidence": None}
    results = payload.get("results", [])
    rec.update(
        status=200,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        top=[{"rank": r.get("rank"), "sci": r.get("sci_names"),
              "sim": r.get("similarity"), "conf": r.get("confidence")} for r in results[:5]],
        genus_conf=payload.get("genus_confidence"),
    )
    _alog(rec)

    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        media_type="application/json",
        headers={"X-Request-Id": rid},
    )


@app.get("/recommend/species")
async def recommend_species():
    """图集里可查的 20k 物种名（前端自动补全用）。"""
    return Response(content=json.dumps(_reco_names, ensure_ascii=False),
                    media_type="application/json")


@app.post("/recommend")
async def recommend(request: Request, payload: dict):
    """配图推荐。body:
    { "species": "Acer rubrum",
      "parts":  ["leaf","flower","fruit","bark","stem"],   # 选填，默认全 5
      "topk":   12,     "t_sp": 0.01,   "t_org": 1.0,   "pool_k": 1000 }   # 均选填
    """
    rid = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    species = (payload.get("species") or "").strip()
    rec = {"endpoint": "recommend", "id": rid, "ip": _client_ip(request),
           "ua": request.headers.get("user-agent", "")[:200], "species": species[:200],
           "req": {k: payload.get(k) for k in ("parts", "topk", "t_sp", "t_org", "pool_k")}}

    if _reco_pool is None:
        rec.update(status=503)
        _alog(rec)
        raise HTTPException(status_code=503, detail="配图推荐未启用（产物缺失）")

    if not species:
        rec.update(status=400)
        _alog(rec)
        raise HTTPException(status_code=400, detail="缺少 species")

    si = _reco_sp2i.get(species)
    query_vec = None
    if si is None:
        # 图集外：用识图索引的文本塔向量当 query，结果全 look-alike / genus-relative
        row = _id_sci2row.get(species)
        if row is None:
            q = species.lower()
            hint = [s for s in _reco_names if q in s.lower()][:8]
            rec.update(status=404, in_pool=False, suggest=hint,
                       latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            _alog(rec)
            raise HTTPException(
                status_code=404,
                detail={"error": f"未知物种（不在图集 20000 种，也不在识图 {len(_id_sci2row)} 种里）：{species}",
                        "suggest": hint},
            )
        query_vec = _res["index"]["embeddings"][row].astype("float32").tolist()

    kwargs = dict(
        species_or_idx=si,
        species_name=species,
        query_vec=query_vec,
        parts=payload.get("parts"),
        topk=payload.get("topk"),
        t_sp=payload.get("t_sp"),
        t_org=payload.get("t_org"),
        pool_k=payload.get("pool_k"),
    )
    try:
        from recommend._worker import run_job

        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(_reco_pool, run_job, kwargs), timeout=90
        )
    except Exception as e:
        rec.update(status=500, in_pool=si is not None, error=str(e)[:300],
                   latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        _alog(rec)
        logger.error("配图推荐异常", exc_info=True)
        raise HTTPException(status_code=500, detail=f"推荐失败：{e}")

    # 结果摘要：每部位返回的 photo_id + provenance 分布（复盘 / 挑坏 case 用）
    summary = {}
    for part, lst in result.get("results", {}).items():
        prov = {"exact": 0, "genus": 0, "look": 0}
        for x in lst:
            p = x["provenance"].split(" ")[0]
            prov["exact" if p == "exact" else "genus" if p == "genus-relative" else "look"] += 1
        summary[part] = {"n": len(lst), **prov,
                         "photo_ids": [x["photo_id"] for x in lst]}
    rec.update(
        status=200, in_pool=si is not None,
        not_in_pool=result.get("not_in_pool"),
        params=result.get("params"),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        results=summary,
    )
    _alog(rec)

    return Response(content=json.dumps(result, ensure_ascii=False),
                    media_type="application/json", headers={"X-Request-Id": rid})


# ── 入口 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=False)
