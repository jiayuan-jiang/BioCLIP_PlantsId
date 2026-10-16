"""植物园实拍收集 router（限时、带密钥路径）。

挂载：inference.py 里 `app.include_router(field_router)`。
所有路径都在 `/f-<TOKEN>/` 下；token 不对 = 普通 404，扫不到。
token：环境变量 FIELD_TOKEN，或 demo/field_token.txt（不入库）。没有 token → 整个 router 不挂载。
有效期：FIELD_EXPIRES_AT（ISO 时间，UTC）或首次启动后 36 小时；过期后一律 404，已存数据不删。

数据：BASE_DIR/field_data/
    images/<id>.jpg      统一转成 JPEG（HEIC 也行，装了 pillow-heif 的话）
    meta.jsonl           每张一行，追加写
"""
import hashlib
import hmac
import io
import json
import os
import re
import threading
import time
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncio

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image, ImageOps

from inference_core import BASE_DIR

try:  # iOS 原图 HEIC：可选依赖
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

UPLOADERS = ["Jiayuan Jiang", "Fangyi Li", "Jiangpan Bian", "Ping Yin", "Ningrui Zhu"]
ORGANS = ["whole", "leaf", "flower", "fruit", "bark", "stem", "other"]
MAX_BYTES = 30 * 1024 * 1024
MAX_SIDE = 4096  # 超过就缩，iOS 原图 12MP≈4032 不会被缩

API_BASE = os.environ.get("FIELD_API_BASE", "http://127.0.0.1:8000")  # 回环调自己的真实接口
PRED_TIMEOUT = 120
_inflight = 0   # 同时在途的"一对预测"数；写进 predict_log 用来观察并发

DATA = BASE_DIR / "field_data"
PRED_LOG = DATA / "predict_log.jsonl"
IMG_DIR = DATA / "images"
META = DATA / "meta.jsonl"
_lock = threading.Lock()


def _token() -> str:
    t = os.environ.get("FIELD_TOKEN", "").strip()
    if not t:
        f = BASE_DIR / "field_token.txt"
        if f.exists():
            t = f.read_text().strip()
    return t


def _expires_at() -> datetime:
    env = os.environ.get("FIELD_EXPIRES_AT", "").strip()
    if env:
        return datetime.fromisoformat(env.replace("Z", "+00:00"))
    DATA.mkdir(parents=True, exist_ok=True)
    f = DATA / "expires_at"
    if not f.exists():
        f.write_text((datetime.now(timezone.utc) + timedelta(hours=36)).isoformat())
    return datetime.fromisoformat(f.read_text().strip())


TOKEN = _token()
EXPIRES = _expires_at() if TOKEN else None


def _alive() -> bool:
    return datetime.now(timezone.utc) < EXPIRES


def _gate(token: str):
    """token 错 / 过期 → 与任何不存在的路径一样的 404。"""
    if not TOKEN or not hmac.compare_digest(token, TOKEN) or not _alive():
        raise HTTPException(status_code=404, detail="Not Found")


_ALIASES: dict = {}


def _aliases(name: str) -> list:
    if not _ALIASES:
        for sp in json.loads((BASE_DIR / "static" / "field_species.json").read_text(encoding="utf-8")):
            _ALIASES[sp["name"]] = sp.get("aliases", [])
    return _ALIASES.get(name, [])


def _norm(n) -> str:
    """学名 → 'genus species'（小写，去杂交符号/栽培品种/变型）。"""
    s = str(n or "").lower().replace("×", " ").replace("✕", " ")
    s = re.sub(r"'[^']*'", " ", s)
    s = re.sub(r"\bx\b", " ", s)
    return " ".join(re.findall(r"[a-z][a-z\-]*", s)[:2])


def _score_hits(row: dict):
    """按当前标注的物种，重算两个接口的 top1/top5 是否命中（改标注后也调用）。"""
    targets = {_norm(row["species"])} | {_norm(a) for a in _aliases(row["species"])}
    targets.discard("")
    genera = {t.split()[0] for t in targets}
    for k in ("pred_open", "pred_closed"):
        p = row.get(k)
        if not p or p.get("status") != 200:
            continue
        names = [_norm(x.get("sci")) for x in p["top5"]]
        for x, n in zip(p["top5"], names):
            x["match"] = n in targets
        p["hit_top1"] = bool(names) and names[0] in targets
        p["hit_top5"] = any(n in targets for n in names)
        p["genus_top1"] = bool(names) and names[0].split()[0] in genera if names and names[0] else False


async def _call(client, path: str, jpg: bytes) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.post(API_BASE + path, params={"topk": 5}, timeout=PRED_TIMEOUT,
                              files={"file": ("photo.jpg", jpg, "image/jpeg")})
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if r.status_code != 200:
            return {"status": r.status_code, "latency_ms": ms, "error": r.text[:200], "top5": []}
        j = r.json()
        top5 = [{"sci": x.get("sci_names"), "common": x.get("common_names"),
                 "score": x.get("similarity", x.get("score")), "prob": x.get("prob")}
                for x in j.get("results", [])[:5]]
        return {"status": 200, "latency_ms": ms, "is_plant": j.get("is_plant"),
                "energy": j.get("energy_score"), "margin": j.get("margin_top1_top2"), "top5": top5}
    except Exception as e:
        return {"status": 0, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": repr(e)[:200], "top5": []}


async def _predict_both(jpg: bytes) -> dict:
    """同时打公版 /predict 和温室 /matthaei/predict；记录各自耗时、总耗时、并发度。"""
    global _inflight
    _inflight += 1
    n_in = _inflight
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient() as c:
            o, z = await asyncio.gather(_call(c, "/predict", jpg), _call(c, "/matthaei/predict", jpg))
    finally:
        _inflight -= 1
    return {"open": o, "closed": z, "wall_ms": round((time.perf_counter() - t0) * 1000, 1), "inflight_at_start": n_in}


router = APIRouter(prefix="/f-{token}", include_in_schema=False)


def _read_meta() -> list:
    if not META.exists():
        return []
    with META.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _num(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _exif_gps(im):
    try:
        g = im.getexif().get_ifd(0x8825)
        if not g or 2 not in g or 4 not in g:
            return None
        def dms(v, ref):
            d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
            return -d if ref in ("S", "W") else d
        return round(dms(g[2], g.get(1, "N")), 6), round(dms(g[4], g.get(3, "E")), 6)
    except Exception:
        return None


@router.get("/")
async def page(token: str):
    _gate(token)
    html = (BASE_DIR / "static" / "field.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@router.get("/api/config")
async def config(token: str):
    _gate(token)
    species = json.loads((BASE_DIR / "static" / "field_species.json").read_text(encoding="utf-8"))
    return JSONResponse({
        "uploaders": UPLOADERS, "organs": ORGANS, "species": species,
        "expires_at": EXPIRES.isoformat(),
    })


@router.get("/api/stats")
async def stats(token: str):
    _gate(token)
    rows = _read_meta()
    by_user, by_sp = {}, {}
    for r in rows:
        by_user[r["uploader"]] = by_user.get(r["uploader"], 0) + 1
        by_sp[r["species"]] = by_sp.get(r["species"], 0) + 1
    return {"total": len(rows), "by_uploader": by_user, "by_species": by_sp}


@router.post("/api/upload")
async def upload(
    token: str,
    file: UploadFile = File(...),
    client_id: str = Form(...),
    uploader: str = Form(...),
    species: str = Form(...),
    species_manual: str = Form("0"),
    organ: str = Form(""),
    note: str = Form(""),
    bed: str = Form(""),
    lat: str = Form(""), lon: str = Form(""), gps_acc: str = Form(""), gps_alt: str = Form(""),
    gps_ts: str = Form(""),
    taken_ts: str = Form(""),
    client_name: str = Form(""),
):
    _gate(token)
    if uploader not in UPLOADERS:
        raise HTTPException(400, "uploader 不在名单内")
    species = species.strip()[:200]
    if not species:
        raise HTTPException(400, "缺少物种")
    if organ and organ not in ORGANS:
        raise HTTPException(400, "organ 不合法")
    if not re.fullmatch(r"[A-Za-z0-9-]{8,64}", client_id):
        raise HTTPException(400, "client_id 不合法")

    # 幂等：同一 client_id 重试 → 直接成功（弱网下重传很常见）
    img_id = hashlib.sha1(client_id.encode()).hexdigest()[:16]
    out = IMG_DIR / f"{img_id}.jpg"
    if out.exists():
        for r in _read_meta():
            if r["id"] == img_id:
                return {"ok": True, "id": img_id, "dup": True, "pred_open": r.get("pred_open"), "pred_closed": r.get("pred_closed")}

    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "图片过大")
    try:
        im = Image.open(io.BytesIO(raw))
        exif_gps = _exif_gps(im)
        im = ImageOps.exif_transpose(im)  # 先按 EXIF 转正，再丢 EXIF
        im = im.convert("RGB")
    except Exception:
        raise HTTPException(415, "无法解码图片（HEIC 需服务端 pillow-heif，或让前端转 JPEG）")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    # 浏览器没拿到 GPS（iOS Safari 在 http 下禁用定位）→ 退回照片 EXIF（相册选图且开了位置才有）
    gps_source = "browser" if _num(lat) is not None and _num(lon) is not None else ""
    if not gps_source and exif_gps:
        lat, lon = str(exif_gps[0]), str(exif_gps[1])
        gps_source = "exif"

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    im.save(tmp, "JPEG", quality=92)
    os.replace(tmp, out)

    row = {
        "id": img_id, "client_id": client_id,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "taken_ts": taken_ts, "uploader": uploader,
        "species": species, "species_manual": species_manual == "1",
        "organ": organ, "note": note.strip()[:1000], "bed": bed.strip()[:20],
        "lat": _num(lat), "lon": _num(lon), "gps_acc_m": _num(gps_acc),
        "gps_alt_m": _num(gps_alt), "gps_ts": gps_ts, "gps_source": gps_source,
        "width": im.size[0], "height": im.size[1],
        "orig_bytes": len(raw), "orig_name": client_name[:200],
        "orig_type": file.content_type or "",
    }
    with _lock, META.open("a", encoding="utf-8") as f:   # 先落盘元数据，预测失败/断线也不丢照片
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    both = await _predict_both(out.read_bytes())
    row["pred_open"], row["pred_closed"] = both["open"], both["closed"]
    _score_hits(row)
    with _lock:
        rows = _read_meta()
        for r in rows:
            if r["id"] == img_id:
                r["pred_open"], r["pred_closed"] = row["pred_open"], row["pred_closed"]
        _write_meta(rows)
        with PRED_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(), "id": img_id, "uploader": uploader,
                "species": species, "wall_ms": both["wall_ms"], "inflight_at_start": both["inflight_at_start"],
                "open": row["pred_open"], "closed": row["pred_closed"]}, ensure_ascii=False) + "\n")
    return {"ok": True, "id": img_id, "pred_open": row["pred_open"], "pred_closed": row["pred_closed"],
            "wall_ms": both["wall_ms"]}


def _write_meta(rows: list):
    tmp = META.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, META)


@router.get("/api/mine")
async def mine(token: str, uploader: str):
    _gate(token)
    rows = [r for r in _read_meta() if r["uploader"] == uploader]
    rows.sort(key=lambda r: r["uploaded_at"], reverse=True)
    return [{k: r.get(k) for k in ("id", "uploaded_at", "species", "species_manual", "organ", "bed", "note", "edited_at", "pred_open", "pred_closed")}
            for r in rows]


@router.get("/api/thumb/{img_id}")
async def thumb(token: str, img_id: str):
    _gate(token)
    if not re.fullmatch(r"[0-9a-f]{16}", img_id):
        raise HTTPException(404)
    src = IMG_DIR / f"{img_id}.jpg"
    if not src.exists():
        raise HTTPException(404)
    dst = DATA / "thumbs" / f"{img_id}.jpg"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(src)
        im.thumbnail((240, 240))
        im.save(dst, "JPEG", quality=80)
    return FileResponse(dst, headers={"Cache-Control": "private, max-age=3600"})


@router.post("/api/edit/{img_id}")
async def edit(token: str, img_id: str, uploader: str = Form(...), species: str = Form(...),
               species_manual: str = Form("0"), organ: str = Form(""), bed: str = Form(""),
               note: str = Form("")):
    _gate(token)
    if organ and organ not in ORGANS:
        raise HTTPException(400, "organ 不合法")
    if not species.strip():
        raise HTTPException(400, "缺少物种")
    with _lock:
        rows = _read_meta()
        for r in rows:
            if r["id"] == img_id and r["uploader"] == uploader:
                r.update(species=species.strip()[:200], species_manual=species_manual == "1",
                         organ=organ, bed=bed.strip()[:20], note=note.strip()[:1000],
                         edited_at=datetime.now(timezone.utc).isoformat())
                _score_hits(r)
                _write_meta(rows)
                return {"ok": True}
    raise HTTPException(404, "没找到这张照片")


@router.post("/api/delete/{img_id}")
async def delete(token: str, img_id: str, uploader: str = Form(...)):
    """删除 = 移到 trash/（不真删）。"""
    _gate(token)
    with _lock:
        rows = _read_meta()
        keep = [r for r in rows if not (r["id"] == img_id and r["uploader"] == uploader)]
        if len(keep) == len(rows):
            raise HTTPException(404, "没找到这张照片")
        _write_meta(keep)
        (DATA / "trash").mkdir(exist_ok=True)
        src = IMG_DIR / f"{img_id}.jpg"
        if src.exists():
            os.replace(src, DATA / "trash" / src.name)
    return {"ok": True}


@router.get("/api/export.zip")
async def export(token: str):
    """打包下载：images/ + meta.jsonl。"""
    _gate(token)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        if META.exists():
            z.write(META, "meta.jsonl")
        if IMG_DIR.exists():
            for p in sorted(IMG_DIR.glob("*.jpg")):
                z.write(p, f"images/{p.name}")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": "attachment; filename=field_data.zip"})
