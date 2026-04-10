"""
补充下载缺失物种图片
====================
扫描 taxonomy_enriched.csv 中全部 5000 个物种，
找出在 OUTPUT_DIR 中没有目录或图片不足 IMAGES_PER_SPP 张的物种，
只下载这些缺失的，已有图片的物种完全跳过。

依赖：pip install requests tqdm pandas
"""

import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ══════════════════════════════════════════════
#  修改这里的变量
# ══════════════════════════════════════════════

TAXONOMY_CSV    = "taxonomy_enriched.csv"
OUTPUT_DIR      = "./test_images"

IMAGES_PER_SPP  = 5        # 每个物种目标图片数量
QUALITY_GRADE   = "research"
IMAGE_SIZE      = "medium"

MAX_WORKERS     = 2
REQUEST_DELAY   = 1.2
THROTTLE_WAIT   = 60.0
RETRY_TIMES     = 4
IMG_RETRY       = 3
IMG_RETRY_DELAY = 2.0

# ══════════════════════════════════════════════

API_BASE = "https://api.inaturalist.org/v1"
HEADERS  = {"User-Agent": "inat-missing-downloader/1.0"}
SIZE_SUFFIX = {"square": "square", "small": "small",
               "medium": "medium", "large": "large", "original": "original"}


def safe_dirname(taxon_id: str, sci_name: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|]', "_", sci_name).strip()
    return f"{taxon_id}_{safe}"


def count_existing(taxon_id: str, sci_name: str) -> int:
    d = os.path.join(OUTPUT_DIR, safe_dirname(taxon_id, sci_name))
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d)
               if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))


def fetch_observations(taxon_id: str, n: int) -> list:
    params = {
        "taxon_id":      taxon_id,
        "quality_grade": QUALITY_GRADE,
        "photos":        "true",
        "per_page":      min(n * 3, 50),
        "order":         "random",
        "order_by":      "random",
    }
    results = []
    for attempt in range(1, RETRY_TIMES + 1):
        try:
            resp = requests.get(f"{API_BASE}/observations",
                                params=params, headers=HEADERS, timeout=20)
        except Exception as e:
            if attempt < RETRY_TIMES:
                time.sleep(REQUEST_DELAY * 2)
                continue
            print(f"\n    ⚠ 网络错误 taxon_id={taxon_id}：{e}")
            return []

        if resp.status_code == 429:
            wait = THROTTLE_WAIT * attempt
            print(f"\n    ⏳ 429 限速 taxon_id={taxon_id}，等待 {wait:.0f}s…")
            time.sleep(wait)
            continue

        if not resp.ok:
            print(f"\n    ⚠ HTTP {resp.status_code} taxon_id={taxon_id}")
            return []

        results = resp.json().get("results", [])
        break
    else:
        print(f"\n    ⚠ 持续 429，放弃 taxon_id={taxon_id}")
        return []

    target = SIZE_SUFFIX.get(IMAGE_SIZE, "medium")
    photos = []
    for obs in results:
        obs_id = obs.get("id")
        for photo in obs.get("photos", []):
            url = photo.get("url", "")
            if not url:
                continue
            url = re.sub(r"/(square|small|medium|large|original)\.",
                         f"/{target}.", url)
            photos.append({"obs_id": obs_id,
                           "photo_id": photo.get("id"), "url": url})
            if len(photos) >= n:
                return photos
    return photos


def download_image(url: str, save_path: str) -> bool:
    for attempt in range(IMG_RETRY):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
            r.raise_for_status()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt < IMG_RETRY - 1:
                time.sleep(IMG_RETRY_DELAY)
            else:
                print(f"\n    ✗ 图片下载失败（{url}）：{e}")
    return False


def process_species(row: dict) -> dict:
    taxon_id = str(row["taxon_id"])
    sci_name = str(row.get("scientific_name", taxon_id))
    species_dir = os.path.join(OUTPUT_DIR, safe_dirname(taxon_id, sci_name))
    os.makedirs(species_dir, exist_ok=True)

    existing = {f for f in os.listdir(species_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))}
    need = IMAGES_PER_SPP - len(existing)
    if need <= 0:
        return {"taxon_id": taxon_id, "sci_name": sci_name,
                "downloaded": 0, "skipped": len(existing)}

    photos = fetch_observations(taxon_id, need)
    time.sleep(REQUEST_DELAY)

    downloaded = 0
    for p in photos:
        fname = f"{p['obs_id']}_{p['photo_id']}.jpg"
        if fname in existing:
            continue
        if download_image(p["url"], os.path.join(species_dir, fname)):
            downloaded += 1

    return {"taxon_id": taxon_id, "sci_name": sci_name,
            "downloaded": downloaded, "skipped": len(existing)}


def main():
    df = pd.read_csv(TAXONOMY_CSV, dtype={"taxon_id": str})
    print(f"✓ 读取 {len(df)} 个物种  [{TAXONOMY_CSV}]")

    # 扫描哪些物种图片不足
    print("扫描已有图片…")
    missing_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="扫描", unit="spp"):
        existing = count_existing(str(row["taxon_id"]),
                                  str(row.get("scientific_name", "")))
        if existing < IMAGES_PER_SPP:
            missing_rows.append(row.to_dict())

    if not missing_rows:
        print("✓ 全部物种图片已齐全，无需下载。")
        return

    print(f"\n需要补充的物种：{len(missing_rows)} 个"
          f"（已完整：{len(df) - len(missing_rows)} 个）")
    print(f"并发线程：{MAX_WORKERS}，API 间隔：{REQUEST_DELAY}s\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_dl = 0
    failed   = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_species, row): row
                   for row in missing_rows}
        with tqdm(total=len(missing_rows), desc="补充下载", unit="spp") as pbar:
            for future in as_completed(futures):
                res = future.result()
                total_dl += res["downloaded"]
                if res["downloaded"] == 0 and res["skipped"] == 0:
                    failed.append(res["sci_name"])
                pbar.set_postfix(dl=total_dl, fail=len(failed))
                pbar.update(1)

    print(f"\n完成！")
    print(f"  补充下载：{total_dl} 张")
    print(f"  无图物种：{len(failed)} 个")
    if failed:
        for name in failed[:20]:
            print(f"    - {name}")
        if len(failed) > 20:
            print(f"    … 共 {len(failed)} 个")


if __name__ == "__main__":
    main()