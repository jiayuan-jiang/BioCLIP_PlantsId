r"""按 manifest 并行下载 iNat medium 图。跳过已存在，断点续传安全。

用法：
  pip install httpx pyarrow tqdm
  python inat_download.py --manifest D:\inat\manifest.parquet --out D:\inat\img --workers 64

落盘：{out}/{taxon_id}/{photo_id}.{ext}
"""
import argparse, asyncio, os, sys
import pyarrow.parquet as pq
import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=64)
ap.add_argument("--limit", type=int, default=0, help="调试用，只下前 N 张")
A = ap.parse_args()

t = pq.read_table(A.manifest, columns=["photo_id", "taxon_id", "extension", "url"]).to_pydict()
rows = list(zip(t["photo_id"], t["taxon_id"], t["extension"], t["url"]))
if A.limit:
    rows = rows[:A.limit]
print(f"{len(rows):,} 张待下", flush=True)

done = ok = fail = skip = 0
lock = asyncio.Lock()

async def fetch(client, sem, pid, tid, ext, url):
    global done, ok, fail, skip
    d = os.path.join(A.out, str(tid))
    fp = os.path.join(d, f"{pid}.{ext}")
    if os.path.exists(fp) and os.path.getsize(fp) > 1000:
        async with lock:
            skip += 1; done += 1
        return
    os.makedirs(d, exist_ok=True)
    async with sem:
        for attempt in range(3):
            try:
                r = await client.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 1000:
                    with open(fp, "wb") as f:
                        f.write(r.content)
                    async with lock:
                        ok += 1
                    break
                if r.status_code in (403, 404):
                    async with lock:
                        fail += 1
                    break
            except Exception:
                await asyncio.sleep(1 + attempt)
        else:
            async with lock:
                fail += 1
    async with lock:
        done += 1
        if done % 2000 == 0:
            print(f"  {done:,}/{len(rows):,}  ok={ok:,} skip={skip:,} fail={fail:,}", flush=True)

async def main():
    sem = asyncio.Semaphore(A.workers)
    limits = httpx.Limits(max_connections=A.workers + 16, max_keepalive_connections=A.workers)
    async with httpx.AsyncClient(limits=limits, headers={"User-Agent": "bioclip-refpool/1.0"}) as client:
        await asyncio.gather(*(fetch(client, sem, *r) for r in rows))

asyncio.run(main())
print(f"\n✓ ok={ok:,}  skip={skip:,}  fail={fail:,}  →  {A.out}", flush=True)
