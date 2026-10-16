"""
温室 accepted species → 多源图片抓取 (Botanical Garden pipeline · step 10)
=========================================================================
输入：bg_00_resolve_names.py 产出的 bg_greenhouse_species_accepted.csv
来源（各来源并行取候选，轮询交错，直到凑够 --per-species 张）：
  inat      iNaturalist 观测照片（先 captive=true 栽培照，再野外；无 taxon_id 时按学名匹配）
  plantnet  Pl@ntNet 用户观测图（经 GBIF 数据集过滤；Pl@ntNet 开放 API 只有识别、无下载图接口）
  commons   Wikimedia Commons：Category:<物种名> + 文件名搜索（标题/描述须含物种名）
  wikipedia 英文维基物种页首图
  gbif      GBIF occurrence（默认关闭：排除 iNat/Pl@ntNet 后几乎全是标本馆压制标本，且每物种多耗约 24 秒）

默认不限许可（--cc-only 可只留 CC 系列）；许可、作者、原页面都记在 manifest。
注意：不限许可的图仅限内部评估使用，不得再分发。

可断点续跑 / 增量补图：已有图片按 manifest 计数，只补到 --per-species 的差额。
输出：
  OUT_DIR/images/{accepted_name}/{source}_{id}.jpg
  OUT_DIR/manifest.csv   一行一张图
  OUT_DIR/coverage.csv   每物种各来源张数
  OUT_DIR/source_errors.csv  失败的 API 调用

依赖：pip install requests tqdm pandas pillow
"""

import os
import re
import csv
import time
import hashlib
import argparse
from io import BytesIO
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm

HEADERS = {"User-Agent": "bg-image-fetcher/1.0 (jiayuanj@umich.edu)"}
INAT_OBS = "https://api.inaturalist.org/v1/observations"
GBIF_OCC = "https://api.gbif.org/v1/occurrence/search"
COMMONS = "https://commons.wikimedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"      # GBIF 里的 iNat research-grade 数据集
PLANTNET_DATASET_KEY = "7a3679ef-5582-4aaa-81f0-8c2545cafc81"  # GBIF 里的 Pl@ntNet 用户观测（CC-BY-SA）

INAT_LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"
OK_LICENSE = re.compile(r"(publicdomain/zero|creativecommons\.org/licenses/(by|by-nc|by-sa|by-nc-sa)/|"
                        r"\bcc0\b|\bcc[- ]by|public domain|^pd)", re.I)
ANY_LICENSE = True         # main() 按 --cc-only 改写
MAX_SIDE = 640
GBIF_MAX_PAGES = 2
QUALITY_SPLIT = False      # True：iNat 先取 research 级，再取 needs_id 级（来源记为 inat_nr，便于之后过滤）
PLANTNET_PAGES = 1         # Pl@ntNet(GBIF 数据集) 翻页数，每页 300 条
SOURCE_DEADLINE = 60       # 单来源单物种（或单块下载）的整体时限（秒），防止连接挂住
CHUNK = 12                 # 每块下载张数；不够再下一块

MANIFEST_COLS = ["accepted_name", "source", "source_id", "file", "url", "page",
                 "license", "author", "sha1"]

API_TIMEOUT = (5, 15)      # (connect, read) 秒；iNat 偶尔会让连接一直挂着，必须短超时
DL_TIMEOUT = (5, 20)
ERRORS = []                # (api, key, 原因)，结束时写 source_errors.csv


def lic_ok(lic):
    return ANY_LICENSE or bool(OK_LICENSE.search(lic or ""))


def get_json(url, params, delay, retries=2):
    key = params.get("taxon_id") or params.get("taxon_name") or params.get("taxonKey") \
        or params.get("gcmtitle") or params.get("gsrsearch")
    for i in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=API_TIMEOUT)
            if r.status_code == 429:
                time.sleep(min(10 * i, 20))
                continue
            r.raise_for_status()
            time.sleep(delay)
            return r.json()
        except Exception as e:
            if i == retries:
                ERRORS.append((url, key, type(e).__name__))
                return None
            time.sleep(1)
    ERRORS.append((url, key, "429"))
    return None


def name_variants(row):
    """accepted_name + 原始记录名的二名法（去杂交号/栽培品种），用于没有 taxon_id 时按名字查。"""
    names = [row["accepted_name"]]
    for raw in str(row.get("raw_names", "")).split(" | "):
        raw = re.sub(r"'[^']*'", "", raw)
        t = re.sub(r"[^A-Za-z\- ]", " ", raw).split()
        if len(t) >= 2:
            names.append(f"{t[0]} {t[1]}")
    return list(dict.fromkeys(names))


# ─────────────────────────────────────────────
# 各来源：返回候选 dict 列表 {source, source_id, url, page, license, author}
# ─────────────────────────────────────────────

def cand_inat(row, n):
    per = min(max(n, 20), 200)
    if "inat_ids" in row:                      # 来自 bg_05：已排除模糊匹配、已跟随同物异名
        ids = str(row["inat_ids"]) if pd.notna(row["inat_ids"]) else ""
        keys = [{"taxon_id": ids}] if ids else []
    else:                                      # 旧路径（没跑 bg_05 时）
        tid = row.get("inat_taxon_id")
        keys = [{"taxon_id": int(tid)}] if not pd.isna(tid) else \
               [{"taxon_name": nm} for nm in name_variants(row)[:3]]
    out, seen = [], set()
    max_pages = min(10, -(-n // 200))                      # iNat 翻页上限：page*per_page ≤ 10000
    quals = [("research", "inat"), ("needs_id", "inat_nr")] if QUALITY_SPLIT else [(None, "inat")]
    for k in keys:
        for qual, src in quals:
            for captive in ("true", "false"):              # 先栽培，后野外
                for page in range(1, max_pages + 1):
                    params = {**k, "photos": "true", "captive": captive, "per_page": per, "page": page,
                              "order_by": "votes", "order": "desc"}
                    if qual:
                        params["quality_grade"] = qual
                    if not ANY_LICENSE:
                        params["photo_license"] = INAT_LICENSES
                    d = get_json(INAT_OBS, params, delay=1.0)
                    res = (d or {}).get("results", [])
                    for o in res:
                        for p in o.get("photos", [])[:1]:   # 每条观测只取首图，避免同一株重复
                            if p["id"] in seen or not p.get("url"):
                                continue
                            seen.add(p["id"])
                            out.append({
                                "source": src, "source_id": str(p["id"]),
                                "url": p["url"].replace("square", "medium"),
                                "page": o.get("uri"), "license": p.get("license_code") or "arr",
                                "author": (o.get("user") or {}).get("login", ""),
                            })
                    if len(res) < per or len(out) >= n:
                        break
                if len(out) >= n:
                    break
            if len(out) >= n:
                break
        if out:
            break          # 已用某个名字拿到图，不再试别的名字
    return out


def _gbif(row, n, source, dataset_key=None):
    key = row.get("accepted_key")
    if pd.isna(key):
        return []
    params = {"taxonKey": int(key), "mediaType": "StillImage", "limit": 300}
    if dataset_key:
        params["datasetKey"] = dataset_key
    out = []
    # 通用 gbif：前几页多半是 iNat 数据集，需翻页才能凑到非 iNat 的图
    for page in range(PLANTNET_PAGES if dataset_key else GBIF_MAX_PAGES):
        params["offset"] = page * 300
        d = get_json(GBIF_OCC, params, delay=0.3)
        for o in (d or {}).get("results", []):
            if not dataset_key and o.get("datasetKey") in (INAT_DATASET_KEY, PLANTNET_DATASET_KEY):
                continue                               # 这两个有各自的来源，避免重复
            if o.get("basisOfRecord") in ("PRESERVED_SPECIMEN", "FOSSIL_SPECIMEN"):
                continue                               # 压制标本照片不是活体照，不适合测试
            for m in o.get("media", [])[:1]:
                url, lic = m.get("identifier"), m.get("license") or ""
                if not url or m.get("type") not in (None, "StillImage") or not lic_ok(lic):
                    continue
                out.append({
                    "source": source, "source_id": f"{o['key']}",
                    "url": url, "page": m.get("references") or f"https://www.gbif.org/occurrence/{o['key']}",
                    "license": lic or "arr", "author": m.get("creator") or m.get("rightsHolder") or "",
                })
        if len(out) >= n or (d or {}).get("endOfRecords", True):
            break
    return out


def cand_gbif(row, n):
    return _gbif(row, n, "gbif")


def cand_plantnet(row, n):
    return _gbif(row, n, "plantnet", PLANTNET_DATASET_KEY)


def _commons_pages(d, source_check=None):
    out = []
    for p in ((d or {}).get("query", {}).get("pages", {})).values():
        ii = (p.get("imageinfo") or [{}])[0]
        if not ii.get("mime", "").startswith(("image/jpeg", "image/png")):
            continue
        md = ii.get("extmetadata", {})
        lic = (md.get("LicenseShortName") or {}).get("value", "")
        if not lic_ok(lic):
            continue
        if source_check:    # 搜索结果须在标题/描述里带物种名，否则噪声太大
            blob = (p.get("title", "") + " " + re.sub(r"<[^>]+>", " ",
                    (md.get("ImageDescription") or {}).get("value", ""))).lower()
            if not source_check(blob):
                continue
        author = re.sub(r"<[^>]+>", "", (md.get("Artist") or {}).get("value", ""))[:80]
        out.append({
            "source": "commons", "source_id": str(p["pageid"]),
            "url": ii.get("thumburl") or ii["url"], "page": ii.get("descriptionurl", ""),
            "license": lic or "unknown", "author": author,
        })
    return out


def cand_commons(row, n):
    names = [row["accepted_name"]]
    if pd.notna(row.get("synonyms")):
        names += [x for x in str(row["synonyms"]).split(" | ") if x][:4]
    base = {"action": "query", "format": "json", "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime", "iiurlwidth": MAX_SIDE}
    out, seen = [], set()
    for name in names:
        d = get_json(COMMONS, {**base, "generator": "categorymembers", "gcmtitle": f"Category:{name}",
                               "gcmtype": "file", "gcmlimit": min(max(n * 2, 50), 500)}, delay=0.3)
        out += [c for c in _commons_pages(d) if c["source_id"] not in seen and not seen.add(c["source_id"])]
        if len(out) < n:    # 分类里不够，再按文件名/描述搜索
            d = get_json(COMMONS, {**base, "generator": "search", "gsrsearch": f'"{name}"',
                                   "gsrnamespace": 6, "gsrlimit": 50}, delay=0.3)
            low = name.lower()
            out += [c for c in _commons_pages(d, lambda b, low=low: low in b)
                    if c["source_id"] not in seen and not seen.add(c["source_id"])]
        if len(out) >= n:
            break
    return out


def cand_wikipedia(row, n):
    name = row["accepted_name"]
    try:
        r = requests.get(WIKI_SUMMARY + name.replace(" ", "_"), headers=HEADERS, timeout=API_TIMEOUT)
        if r.status_code != 200:
            return []
        j = r.json()
        img = (j.get("originalimage") or j.get("thumbnail") or {}).get("source")
        if not img or j.get("type") == "disambiguation":
            return []
        return [{"source": "wikipedia", "source_id": re.sub(r"\W+", "_", img.rsplit("/", 1)[-1])[:60],
                 "url": img, "page": (j.get("content_urls") or {}).get("desktop", {}).get("page", ""),
                 "license": "unknown", "author": ""}]
    except Exception as e:
        ERRORS.append(("wikipedia", name, type(e).__name__))
        return []


SOURCES = {"inat": cand_inat, "plantnet": cand_plantnet, "commons": cand_commons,
           "wikipedia": cand_wikipedia, "gbif": cand_gbif}


def interleave(lists):
    """各来源轮询交错（不截断）。"""
    out = []
    for group in zip_longest(*lists):
        out.extend(c for c in group if c)
    return out


# ─────────────────────────────────────────────
# 下载
# ─────────────────────────────────────────────

def download(c, dest_dir):
    try:
        r = requests.get(c["url"], headers=HEADERS, timeout=DL_TIMEOUT)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if min(img.size) < 200:                 # 太小的图没用
            return None
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        fn = f"{c['source']}_{c['source_id']}.jpg"
        buf = BytesIO()
        img.save(buf, "JPEG", quality=92)
        data = buf.getvalue()
        with open(os.path.join(dest_dir, fn), "wb") as f:
            f.write(data)
        return {**c, "file": fn, "sha1": hashlib.sha1(data).hexdigest()}
    except Exception:
        return None


def run_with_deadline(fn_args, workers):
    """并行跑一组 (fn, *args)，整体 SOURCE_DEADLINE；超时的线程直接放弃（不能用 with，会等它们）。"""
    ex = ThreadPoolExecutor(workers)
    futs = [ex.submit(*fa) for fa in fn_args]
    out, t0 = [], time.time()
    for f in futs:
        try:
            out.append(f.result(timeout=max(1, SOURCE_DEADLINE - (time.time() - t0))))
        except Exception:
            out.append(None)
    ex.shutdown(wait=False, cancel_futures=True)
    return out


def process(row, need, srcs, img_root, have_ids, seen_sha, workers, have_files=frozenset()):
    """为一个物种取候选并下载，直到凑够 need 张；返回 manifest 行列表。"""
    name = row["accepted_name"]
    results = run_with_deadline([(SOURCES[s], row, need * 2) for s in srcs], len(srcs))
    for s, r in zip(srcs, results):
        if r is None:
            ERRORS.append((s, name, "deadline_or_error"))
    cands = interleave([r or [] for r in results])
    cands = [c for c in cands if (c["source"], c["source_id"]) not in have_ids]

    dest = os.path.join(img_root, name.replace(" ", "_"))
    os.makedirs(dest, exist_ok=True)
    rows = []
    for i in range(0, len(cands), CHUNK):
        if len(rows) >= need:
            break
        chunk = cands[i:i + CHUNK] if len(cands) - i >= CHUNK else cands[i:]
        # 只下载还缺的数量 + 少量余量
        chunk = chunk[:max(need - len(rows), 1) + 3]
        for res in run_with_deadline([(download, c, dest) for c in chunk], workers):
            if not res:
                continue
            if res["sha1"] in seen_sha or len(rows) >= need:
                fp = os.path.join(dest, res["file"])
                if (name, res["file"]) not in have_files:      # 已登记的文件绝不能删
                    os.remove(fp)
                continue
            seen_sha.add(res["sha1"])
            rows.append({"accepted_name": name, **{k: res.get(k, "") for k in MANIFEST_COLS[1:]}})
    return rows


def main():
    global ANY_LICENSE, QUALITY_SPLIT, PLANTNET_PAGES, SOURCE_DEADLINE, CHUNK
    ap = argparse.ArgumentParser()
    ap.add_argument("--species-csv", default="./bg_greenhouse_species_accepted.csv")
    ap.add_argument("--out-dir", default="./BotanicalGarden/gh_images")
    ap.add_argument("--per-species", type=int, default=50, help="每物种上限（含已有）")
    ap.add_argument("--sources", default="inat,plantnet,commons,wikipedia")
    ap.add_argument("--cc-only", action="store_true", help="只保留 CC 系列许可（默认不限）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 K 个物种（冒烟测试）")
    ap.add_argument("--workers", type=int, default=6, help="单物种内的下载并发")
    ap.add_argument("--species-workers", type=int, default=2, help="同时处理的物种数（iNat API 限速，勿大）")
    ap.add_argument("--species-file", default="", help="只处理该文件里列出的物种（一行一个 accepted_name）")
    ap.add_argument("--quality-split", action="store_true", help="iNat 先 research 级，再 needs_id 级（来源记为 inat_nr）")
    ap.add_argument("--plantnet-pages", type=int, default=1)
    ap.add_argument("--deadline", type=int, default=60, help="单来源取候选的整体时限（秒）；翻页多时要调大")
    ap.add_argument("--chunk", type=int, default=12, help="每块下载张数")
    args = ap.parse_args()
    ANY_LICENSE = not args.cc_only
    QUALITY_SPLIT, PLANTNET_PAGES, SOURCE_DEADLINE, CHUNK = args.quality_split, args.plantnet_pages, args.deadline, args.chunk

    srcs = args.sources.split(",")
    sp = pd.read_csv(args.species_csv)
    ids_path = "./bg_greenhouse_inat_ids.csv"
    if os.path.exists(ids_path):               # bg_05 的 iNat 解析结果（含 GBIF 同物异名）
        ids = pd.read_csv(ids_path, dtype=str)[["accepted_name", "inat_ids", "synonyms"]]
        sp = sp.merge(ids, on="accepted_name", how="left")
        print(f"使用 {ids_path}：有 iNat taxon 的物种 {(sp['inat_ids'].fillna('') != '').sum()} / {len(sp)}")
    if args.species_file:
        want = [x.strip() for x in open(args.species_file, encoding="utf-8") if x.strip()]
        sp = sp[sp.accepted_name.isin(want)].reset_index(drop=True)
        print(f"--species-file：{len(want)} 个名字，匹配到 {len(sp)} 个物种")
    if args.limit:
        sp = sp.head(args.limit)

    img_root = os.path.join(args.out_dir, "images")
    os.makedirs(img_root, exist_ok=True)
    man_path = os.path.join(args.out_dir, "manifest.csv")
    if os.path.exists(man_path):
        m = pd.read_csv(man_path, dtype={"source_id": str})
        m["source_id"] = m["source_id"].str.replace(r"\.0$", "", regex=True)   # 防止被读成浮点后写成 123.0
        m = m[m["source"] != "none"]          # 清掉旧版的"无图占位"，让这些物种重试
        m.to_csv(man_path, index=False)
    else:
        m = pd.DataFrame(columns=MANIFEST_COLS)
        m.to_csv(man_path, index=False)
    existing = m.groupby("accepted_name").size().to_dict()
    have_ids = set(zip(m["source"], m["source_id"].astype(str)))
    seen_sha = set(m["sha1"].dropna())
    # 注意：source_id 读回来可能是数字，统一成 str 后比对
    have_ids = {(a, str(b)) for a, b in have_ids}

    todo = [r for r in sp.to_dict("records")
            if existing.get(r["accepted_name"], 0) < args.per_species]
    print(f"物种 {len(sp)}；需补图 {len(todo)}；已有图 {len(m)}；目标每种 {args.per_species}；"
          f"来源 {srcs}；{'不限许可' if ANY_LICENSE else '仅 CC'}")

    mf = open(man_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(mf, fieldnames=MANIFEST_COLS)
    pool = ThreadPoolExecutor(args.species_workers)
    have_files = frozenset(zip(m["accepted_name"], m["file"]))
    futs = {pool.submit(process, r, args.per_species - existing.get(r["accepted_name"], 0),
                        srcs, img_root, have_ids, seen_sha, args.workers, have_files): r["accepted_name"]
            for r in todo}
    try:
        for f in tqdm(as_completed(futs), total=len(futs), desc="物种", unit="spp"):
            try:
                for row in f.result():
                    w.writerow(row)
                mf.flush()
            except Exception as e:
                ERRORS.append(("process", futs[f], type(e).__name__))
    finally:
        mf.close()

    if ERRORS:
        pd.DataFrame(ERRORS, columns=["api", "key", "error"]).to_csv(
            os.path.join(args.out_dir, "source_errors.csv"), index=False)
        print(f"API 失败 {len(ERRORS)} 次，见 source_errors.csv")
    m = pd.read_csv(man_path, dtype={"source_id": str})
    cov = (m.groupby(["accepted_name", "source"]).size().unstack(fill_value=0))
    cov["total"] = cov.sum(axis=1)
    cov.to_csv(os.path.join(args.out_dir, "coverage.csv"))
    print(f"\n物种有图：{len(cov)} / {len(sp)}；总图数 {len(m)}")
    print(m["source"].value_counts().to_string())
    print(f"<5 张：{(cov['total'] < 5).sum()}；<{args.per_species} 张：{(cov['total'] < args.per_species).sum()}；"
          f"无图：{len(sp) - len(cov)}")


if __name__ == "__main__":
    main()
