"""Phase 1 —— 数据预计算（方案 C：预计算 + 降采样）。pod 上跑。

**并行版**：每个分片一个独立 worker 进程（stream + caption 清洗 + pHash + 跨 eval 泄漏
过滤 + 分片内近重去重 + resize/存盘），最后一个 merge 进程做跨分片精确去重 + 冻结文本塔
GPU 编码 + 汇总。

产出:
  {out}/img/{shard}/{key}.jpg
  {out}/text_emb.f16.npy        [M, 768] fp16, L2 归一化
  {out}/index.parquet           row, shard, key, caption, binomial, family, phash, is_general
  {out}/leakage_report.txt
  {out}/parts/part_{k}.parquet  中间产物（可删）

依据: doc/spec/stage1-execution.md Phase 1 · doc/stage1-spec.md §6（防泄漏）§9（工程）。

用法:
  # 一把梭（orchestrator 起 N 个 worker + merge）:
  python -m stage1.prepare --shards 0-23 --out /workspace/data/stage1 \
     --eval-roots /workspace/eval_data --jobs 24 --device cuda

  # 或分步:
  python -m stage1.prepare --worker-shard 5 --out ... --eval-roots ...
  python -m stage1.prepare --merge --out ... --device cuda

  # 冒烟: --shards 0-0 --limit 4000 --jobs 1
"""

from __future__ import annotations

import os

# 必须在 import numpy/scipy 之前：单线程 BLAS。
# imagehash.phash 用 scipy.fftpack -> OpenBLAS 默认按核数(96)开线程池；
# N 个 worker 各开 ~200 线程 -> 线程颠簸，CPU 空转 88%。锁死单线程后 N-way 才真并行。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import html
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────
# caption 清洗
# ─────────────────────────────────────────────────────────────

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_SEO = re.compile(
    r"(stock\s*image|stock\s*photo|shutterstock|istockphoto|getty\s*images|dreamstime|"
    r"alamy|123rf|depositphotos|royalty[- ]free|\|\s*gallery|\bhttps?://|www\.|"
    r"\.com\b|\.net\b|photograph by|image credit|all rights reserved)",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")
_ALLCAPS_RUN = re.compile(r"\b[A-Z0-9]{6,}\b")
_BINOMIAL = re.compile(r"^([A-Z][a-z]+)\s([a-z][a-z\-]+)\b")


def clean_caption(raw: str) -> str | None:
    if not raw:
        return None
    s = html.unescape(raw)
    if _CJK.search(s):
        return None
    if _SEO.search(s):
        return None
    if len(_ALLCAPS_RUN.findall(s)) >= 2:
        return None
    s = _WS.sub(" ", s).strip()
    if len(s) < 5 or len(s) > 2000:
        return None
    if len(s.split()) < 2:
        return None
    return s


def parse_binomial(caption: str) -> str:
    m = _BINOMIAL.match(caption)
    if not m:
        return ""
    genus, sp = m.group(1), m.group(2)
    if sp in ("a", "the", "is", "of", "in", "and", "with", "sp", "spp"):
        return ""
    return f"{genus} {sp}"


# ─────────────────────────────────────────────────────────────
# pHash（64-bit -> uint64）
# ─────────────────────────────────────────────────────────────

def phash_u64(img) -> int:
    import imagehash
    h = imagehash.phash(img, hash_size=8)
    v = 0
    for b in h.hash.flatten():
        v = (v << 1) | int(b)
    return v


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def build_eval_phash_set(eval_roots: list[Path], cache: Path,
                         subdirs=("plantnet-300k", "rare-species", "imagenetv2")) -> np.ndarray:
    if cache.is_file():
        print(f"  eval pHash 命中缓存 {cache}", flush=True)
        return np.load(cache)
    from PIL import Image
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPEG", ".JPG", ".PNG"}
    hs = []
    for root in eval_roots:
        for sub in subdirs:
            d = root / sub
            if not d.is_dir():
                continue
            files = [p for p in d.rglob("*") if p.suffix in exts]
            print(f"  eval pHash: {d} -> {len(files)} 图", flush=True)
            for i, p in enumerate(files):
                try:
                    hs.append(phash_u64(Image.open(p).convert("RGB")))
                except Exception:
                    pass
                if i % 5000 == 0 and i:
                    print(f"    {i}/{len(files)}", flush=True)
    arr = np.array(sorted(set(hs)), dtype=np.uint64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    print(f"  eval pHash 集: {len(arr)} 唯一 -> {cache}", flush=True)
    return arr


_POPC = [
    (np.uint64(1), np.uint64(0x5555555555555555)),
    (np.uint64(2), np.uint64(0x3333333333333333)),
    (np.uint64(4), np.uint64(0x0F0F0F0F0F0F0F0F)),
]


def near_eval(h: int, eval_sorted: np.ndarray, thr: int) -> bool:
    if eval_sorted.size == 0:
        return False
    x = np.bitwise_xor(eval_sorted, np.uint64(h))
    x = x - ((x >> _POPC[0][0]) & _POPC[0][1])
    x = (x & _POPC[1][1]) + ((x >> _POPC[1][0]) & _POPC[1][1])
    x = (x + (x >> _POPC[2][0])) & _POPC[2][1]
    pc = (x * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return bool((pc <= thr).any())


# ─────────────────────────────────────────────────────────────
# family 查询（可选，失败留空）
# ─────────────────────────────────────────────────────────────

class FamilyResolver:
    def __init__(self, local_csv: Path | None):
        self.map: dict[str, str] = {}
        if local_csv and local_csv.is_file():
            import csv
            with open(local_csv, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    sci = (r.get("scientific_name") or r.get("species") or "").strip()
                    fam = (r.get("family") or "").strip()
                    if sci and fam:
                        self.map[sci] = fam
                        self.map.setdefault(sci.split(" ")[0], fam)

    def get(self, binomial: str) -> str:
        if not binomial:
            return ""
        return self.map.get(binomial) or self.map.get(binomial.split(" ")[0], "")


def parse_shard_arg(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# ─────────────────────────────────────────────────────────────
# worker：处理单个分片（无 GPU）
# ─────────────────────────────────────────────────────────────

def run_worker(shard: int, args):
    import webdataset as wds
    from PIL import Image
    import pandas as pd

    out = Path(args.out)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    img_dir = out / "img" / str(shard)
    img_dir.mkdir(parents=True, exist_ok=True)

    eval_ph = np.load(args.eval_phash_cache) if Path(args.eval_phash_cache).is_file() \
        else np.array([], dtype=np.uint64)
    fam = FamilyResolver(Path(args.family_csv))

    hdr = ""
    tok = os.environ.get("HF_TOKEN", "")
    if tok:
        hdr = f"-H 'Authorization: Bearer {tok}' "
    url = args.url_tmpl.format(shard=shard)
    pipe = f"pipe:curl -sL --retry 5 --retry-delay 3 {hdr}'{url}'"
    ds = wds.WebDataset(pipe, handler=wds.warn_and_continue, shardshuffle=False)

    C = dict(seen=0, cjk=0, seo=0, empty=0, exact_cap=0, exact_img=0,
             leak=0, internal=0, kept=0, img_err=0)
    seen_cap, seen_img = set(), set()
    buckets: dict[int, list[int]] = {}
    rows = []
    t0 = time.time()

    for sample in ds:
        C["seen"] += 1
        if args.limit and C["seen"] > args.limit:
            break
        jpg = sample.get("jpg") or sample.get("jpeg") or sample.get("png")
        txt = sample.get("txt") or sample.get("caption") or b""
        if jpg is None:
            C["img_err"] += 1
            continue
        raw = txt.decode("utf-8", "ignore") if isinstance(txt, bytes) else str(txt)
        cap = clean_caption(raw)
        if cap is None:
            C["cjk" if _CJK.search(raw) else "seo" if _SEO.search(raw) else "empty"] += 1
            continue
        cs = hashlib.sha1(cap.encode()).hexdigest()
        if cs in seen_cap:
            C["exact_cap"] += 1
            continue
        isha = hashlib.sha1(jpg).hexdigest()
        if isha in seen_img:
            C["exact_img"] += 1
            continue
        try:
            img = Image.open(io.BytesIO(jpg)).convert("RGB")
            ph = phash_u64(img)
        except Exception:
            C["img_err"] += 1
            continue
        if near_eval(ph, eval_ph, args.leak_thr):
            C["leak"] += 1
            continue
        pref = ph >> 48
        bk = buckets.setdefault(pref, [])
        if len(bk) >= args.bucket_keep and any(hamming(ph, x) <= args.leak_thr for x in bk):
            C["internal"] += 1
            continue
        bk.append(ph)

        seen_cap.add(cs)
        seen_img.add(isha)
        binom = parse_binomial(cap)
        key = isha[:16]
        w, h = img.size
        sc = args.resize_short / min(w, h)
        if sc < 1.0:
            img = img.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.BICUBIC)
        img.save(img_dir / f"{key}.jpg", format="JPEG", quality=args.jpeg_q)
        rows.append((str(shard), key, isha, cs, cap, binom, fam.get(binom), int(ph), 0))
        C["kept"] += 1
        if C["kept"] % 10000 == 0:
            print(f"  [shard {shard}] kept={C['kept']} seen={C['seen']} "
                  f"({C['kept']/(time.time()-t0):.0f}/s) leak={C['leak']} int={C['internal']}",
                  flush=True)

    df = pd.DataFrame(rows, columns=["shard", "key", "img_sha", "cap_sha", "caption",
                                     "binomial", "family", "phash", "is_general"])
    df.to_parquet(parts / f"part_{shard}.parquet", index=False)
    (parts / f"stats_{shard}.json").write_text(json.dumps(C))
    print(f"[shard {shard}] DONE seen={C['seen']} kept={C['kept']} "
          f"wall={(time.time()-t0)/60:.1f}min", flush=True)


# ─────────────────────────────────────────────────────────────
# merge：跨分片精确去重 + GPU 文本编码 + 汇总
# ─────────────────────────────────────────────────────────────

def run_merge(args):
    import pandas as pd
    import torch
    import open_clip

    out = Path(args.out)
    parts = out / "parts"
    pfiles = sorted(parts.glob("part_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
    if not pfiles:
        sys.exit("没有 part_*.parquet，先跑 worker")
    df = pd.concat([pd.read_parquet(p) for p in pfiles], ignore_index=True)
    n_raw = len(df)
    # 跨分片精确去重（保留首次）
    df = df.drop_duplicates(subset="cap_sha", keep="first")
    df = df.drop_duplicates(subset="img_sha", keep="first").reset_index(drop=True)
    n_dedup = n_raw - len(df)
    M = len(df)
    print(f"[merge] parts {n_raw} 行 -> 跨分片去重 -{n_dedup} -> M={M}", flush=True)

    df.insert(0, "row", np.arange(M, dtype=np.int64))
    caps = df["caption"].tolist()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(args.model)
    model = model.eval().to(device)
    tok = open_clip.get_tokenizer(args.model)
    for p in model.parameters():
        p.requires_grad_(False)

    emb = np.zeros((M, 768), dtype=np.float16)
    with torch.no_grad():
        for i in range(0, M, args.text_batch):
            chunk = caps[i:i + args.text_batch]
            t = tok(chunk).to(device)
            f = model.encode_text(t).float()
            f = torch.nn.functional.normalize(f, dim=-1)
            emb[i:i + len(chunk)] = f.cpu().numpy().astype(np.float16)
            if i % (args.text_batch * 20) == 0:
                print(f"  text-encode {i}/{M}", flush=True)
    np.save(out / "text_emb.f16.npy", emb)

    df.drop(columns=["img_sha", "cap_sha"]).to_parquet(out / "index.parquet", index=False)

    # 汇总 stats
    tot = dict(seen=0, cjk=0, seo=0, empty=0, exact_cap=0, exact_img=0,
               leak=0, internal=0, kept=0, img_err=0)
    for sf in sorted(parts.glob("stats_*.json")):
        s = json.loads(sf.read_text())
        for k in tot:
            tot[k] += s.get(k, 0)
    n_bin = int((df["binomial"] != "").sum())
    fam_n = df["family"].replace("", np.nan).nunique()
    rep = out / "leakage_report.txt"
    rep.write_text(
        "Stage 1 数据预计算 —— 泄漏/清洗审计（并行版）\n"
        f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"parts: {[p.name for p in pfiles]}\n\n"
        f"流式样本 seen           : {tot['seen']}\n"
        f"  丢 CJK                : {tot['cjk']}\n"
        f"  丢 SEO 噪声           : {tot['seo']}\n"
        f"  丢 空/过短/过长        : {tot['empty']}\n"
        f"  丢 caption 精确重(分片内): {tot['exact_cap']}\n"
        f"  丢 image 精确重(分片内)  : {tot['exact_img']}\n"
        f"  丢 跨 eval 集泄漏      : {tot['leak']}   (pHash Hamming <= {args.leak_thr})\n"
        f"  丢 训练集内部近重(分片内): {tot['internal']}\n"
        f"  丢 图像损坏           : {tot['img_err']}\n"
        f"  丢 跨分片精确重        : {n_dedup}\n"
        f"最终保留 M              : {M}\n"
        f"  其中含 binomial       : {n_bin} ({100*n_bin/max(M,1):.1f}%)\n"
        f"  唯一 family           : {fam_n}\n\n"
        f"text_emb.f16.npy shape : {emb.shape}\n"
    )
    print(rep.read_text(), flush=True)
    print(f"✓ 产出 -> {out}/  (index.parquet {M} 行, text_emb {emb.shape})", flush=True)


# ─────────────────────────────────────────────────────────────
# orchestrator
# ─────────────────────────────────────────────────────────────

def run_orchestrate(args):
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    shards = parse_shard_arg(args.shards)
    print(f"orchestrate shards={shards} jobs={args.jobs} out={out}", flush=True)

    # 先一次性建 eval pHash 缓存（避免 worker 竞争）
    build_eval_phash_set([Path(r) for r in args.eval_roots], Path(args.eval_phash_cache))

    base = [sys.executable, "-m", "stage1.prepare", "--out", str(out),
            "--eval-roots", *args.eval_roots,
            "--eval-phash-cache", args.eval_phash_cache,
            "--family-csv", args.family_csv, "--model", args.model,
            "--leak-thr", str(args.leak_thr), "--bucket-keep", str(args.bucket_keep),
            "--resize-short", str(args.resize_short), "--jpeg-q", str(args.jpeg_q),
            "--url-tmpl", args.url_tmpl]
    if args.limit:
        base += ["--limit", str(args.limit)]

    running: dict[int, subprocess.Popen] = {}
    todo = list(shards)
    failed = []
    while todo or running:
        (out / "parts").mkdir(parents=True, exist_ok=True)
        while todo and len(running) < args.jobs:
            k = todo.pop(0)
            lf = open(out / "parts" / f"worker_{k}.log", "w")
            p = subprocess.Popen(base + ["--worker-shard", str(k)], stdout=lf, stderr=lf)
            running[k] = p
            print(f"  → worker shard {k} pid {p.pid}", flush=True)
        done = [k for k, p in running.items() if p.poll() is not None]
        for k in done:
            rc = running.pop(k).returncode
            ok = (Path(args.out) / "parts" / f"part_{k}.parquet").is_file()
            print(f"  ✓ worker {k} rc={rc} part={'OK' if ok else 'MISSING'}", flush=True)
            if rc != 0 or not ok:
                failed.append(k)
        time.sleep(5)

    if failed:
        print(f"[orchestrate] 失败分片 {failed} —— 可单独重跑 --worker-shard K；继续 merge 用现有 parts", flush=True)
    run_merge(args)


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="0-23")
    ap.add_argument("--out", default="data/stage1")
    ap.add_argument("--eval-roots", nargs="+", default=["/workspace/eval_data"])
    ap.add_argument("--eval-phash-cache", default="/workspace/data/_eval_phash.npy")
    ap.add_argument("--family-csv", default="data/taxonomy_enriched.csv")
    ap.add_argument("--model", default="hf-hub:imageomics/bioclip-2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text-batch", type=int, default=1024)
    ap.add_argument("--leak-thr", type=int, default=8)
    ap.add_argument("--bucket-keep", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help=">0 每分片最多 seen N（冒烟）")
    ap.add_argument("--resize-short", type=int, default=256)
    ap.add_argument("--jpeg-q", type=int, default=85)
    ap.add_argument("--jobs", type=int, default=24, help="并行 worker 数")
    ap.add_argument("--url-tmpl",
                    default="https://huggingface.co/datasets/HEART77/plantmix/resolve/main/{shard}.tar")
    ap.add_argument("--worker-shard", type=int, default=None, help="内部：处理单分片")
    ap.add_argument("--merge", action="store_true", help="内部/手动：合并 parts")
    args = ap.parse_args()

    if args.worker_shard is not None:
        run_worker(args.worker_shard, args)
    elif args.merge:
        run_merge(args)
    else:
        run_orchestrate(args)


if __name__ == "__main__":
    main()
