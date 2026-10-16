"""Matthaei Botanical Gardens 温室闭集识别（特殊版本，挂在 /matthaei/ 下）。

与 /predict 的区别：候选集不是 35 万物种，而是温室清单的 611 个物种（闭集塔 closed_v1.npz）。
  611 行全部是 BioCLIP 2 文字向量（用文字塔对 "a photo of {taxon_path}." 重新编码，见 data/bg_70）。
  其中 48 行在线上词表里向量全零，这里已修复；同一模态同一尺度，不需要校准。
非植物 gate 仍用全量索引的 energy（与 /predict 同一阈值）。
"""
import asyncio
import io
import json
import time
import uuid

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from PIL import Image

import gate
from inference_core import BASE_DIR, _res
from embed_node import node as embed_node, ImageDecodeError

EDITION = "matthaei-botanical-garden-greenhouse"
VERSION = "closed-v1"
_T: dict = {}


def _load():
    if _T:
        return
    z = np.load(BASE_DIR / "matthaei" / "closed_v1.npz", allow_pickle=True)
    _T.update(E=z["embeddings"].astype(np.float32), src=[str(s) for s in z["source"]],
              names=[str(n) for n in z["names"]], family=[str(n) for n in z["family"]],
              genus=[str(n) for n in z["genus"]], common=[str(n) for n in z["common"]])


def _classify(emb, topk: int) -> dict:
    """下游：温室闭集识别。emb 来自共享 embedding 节点（与公版识别共用同一次图像编码）。"""
    _load()
    e, idx = emb.vec, _res["index"]
    if emb.energy is None:
        emb.energy = gate.energy_score(idx["embeddings"] @ e, _res["logit_scale"])
    energy = emb.energy
    if not gate.is_plant(energy):
        return {"edition": EDITION, "version": VERSION, "is_plant": False,
                "energy_score": round(energy, 3), "results": []}

    s = _T["E"] @ e
    order = np.argsort(s)[::-1][:topk]
    z = _res["logit_scale"] * (s - s[order[0]])
    p = np.exp(z) / np.exp(z).sum()
    margin = float(s[order[0]] - s[order[1]]) if len(order) > 1 else None
    results = [{
        "rank": r, "sci_names": _T["names"][i], "common_names": _T["common"][i] or None,
        "families": _T["family"][i], "genera": _T["genus"][i],
        "score": round(float(s[i]), 5), "prob": round(float(p[i]), 5),
    } for r, i in enumerate(order, 1)]
    return {"edition": EDITION, "version": VERSION, "mode": "closed_set",
            "is_plant": True, "energy_score": round(energy, 3),
            "margin_top1_top2": None if margin is None else round(margin, 5),
            "results": results}


def make_router(alog, client_ip) -> APIRouter:
    r = APIRouter(prefix="/matthaei", tags=["matthaei-greenhouse"])

    @r.get("/info")
    async def info():
        _load()
        return {"edition": EDITION, "version": VERSION, "species": len(_T["names"]),
                "by_source": {k: _T["src"].count(k) for k in set(_T["src"])}}

    @r.get("/species")
    async def species():
        _load()
        return Response(json.dumps(
            [{"name": n, "common": c or None, "family": f, "source": s}
             for n, c, f, s in zip(_T["names"], _T["common"], _T["family"], _T["src"])],
            ensure_ascii=False), media_type="application/json")

    @r.post("/predict")
    async def predict(request: Request, file: UploadFile = File(...),
                      topk: int = Query(default=5, ge=1, le=20)):
        rid = uuid.uuid4().hex[:12]; t0 = time.perf_counter()
        rec = {"endpoint": "matthaei_predict", "id": rid, "ip": client_ip(request),
               "ua": request.headers.get("user-agent", "")[:200], "topk": topk}
        if not _res:
            raise HTTPException(503, "模型尚未就绪，请稍候")
        try:
            emb, einfo = await embed_node.embed(await file.read())
            rec["embed"] = einfo
        except ImageDecodeError as e:
            alog({**rec, "status": 400}); raise HTTPException(400, str(e))
        payload = await asyncio.get_running_loop().run_in_executor(None, _classify, emb, topk)
        rs = payload["results"]
        alog({**rec, "status": 200, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
              "is_plant": payload["is_plant"], "top": [{"sci": x["sci_names"], "score": x["score"]} for x in rs[:5]]})
        return Response(json.dumps(payload, ensure_ascii=False), media_type="application/json",
                        headers={"X-Request-Id": rid, "X-Edition": EDITION})

    return r
