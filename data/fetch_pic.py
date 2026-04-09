"""
iNaturalist 测试图片下载脚本
============================
从 taxonomy_enriched.csv 读取 taxon_id，
随机覆盖 80% 物种，每个物种下载 N 张 research-grade 观测图片，
保存到指定目录，目录结构：
  OUTPUT_DIR/
    {taxon_id}_{scientific_name}/
      {observation_id}_{photo_id}.jpg
      ...

依赖：
  pip install requests tqdm pandas
"""

import os
import re
import time
import random
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ══════════════════════════════════════════════
#  修改这里的变量
# ══════════════════════════════════════════════

TAXONOMY_CSV   = "taxonomy_enriched.csv"  # build_vector.py 产出的分类学 CSV
OUTPUT_DIR     = "./test_images"          # 图片保存根目录

SPECIES_RATIO  = 0.80    # 随机覆盖物种比例
IMAGES_PER_SPP = 5       # 每个物种下载图片数量
QUALITY_GRADE  = "research"  # research / needs_id / any
IMAGE_SIZE     = "medium"    # square / small / medium / large / original

RANDOM_SEED    = 42      # 随机种子，固定后结果可复现；改为 None 则每次不同

MAX_WORKERS    = 2       # 并发线程数，iNat 限速严格，建议 1-2
REQUEST_DELAY  = 1.2     # 每次 API 请求后的固定间隔（秒）
THROTTLE_WAIT  = 60.0    # 遇到 429 时的基础等待时间（秒），实际 = THROTTLE_WAIT * 重试次数
RETRY_TIMES    = 4       # API 请求最大重试次数（含 429 退避）
IMG_RETRY      = 3       # 单张图片下载失败后的重试次数
IMG_RETRY_DELAY = 2.0    # 图片下载重试间隔（秒）

# ══════════════════════════════════════════════

API_BASE = "https://api.inaturalist.org/v1"
HEADERS  = {"User-Agent": "inat-test-downloader/1.0"}

SIZE_SUFFIX = {"square": "square", "small": "small",
               "medium": "medium", "large": "large", "original": "original"}


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def safe_dirname(taxon_id: str, sci_name: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|]', "_", sci_name).strip()
    return f"{taxon_id}_{safe}"


def fetch_observations(taxon_id: str, n: int) -> list:
    """
    获取指定物种的观测图片列表。
    遇到 429 时按 THROTTLE_WAIT * attempt 秒退避重试。
    """
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
            resp = requests.get(
                f"{API_BASE}/observations",
                params=params,
                headers=HEADERS,
                timeout=20,
            )
        except Exception as e:
            if attempt < RETRY_TIMES:
                time.sleep(REQUEST_DELAY * 2)
                continue
            print(f"\n    ⚠ 网络错误 taxon_id={taxon_id}：{e}")
            return []

        if resp.status_code == 429:
            wait = THROTTLE_WAIT * attempt
            print(f"\n    ⏳ 429 限速 taxon_id={taxon_id}，"
                  f"等待 {wait:.0f}s（第 {attempt}/{RETRY_TIMES} 次）…")
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
            url = re.sub(r"/(square|small|medium|large|original)\.", f"/{target}.", url)
            photos.append({"obs_id": obs_id, "photo_id": photo.get("id"), "url": url})
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


# ─────────────────────────────────────────────
# 处理单个物种（供线程池调用）
# ─────────────────────────────────────────────

def process_species(row: dict) -> dict:
    taxon_id = str(row["taxon_id"])
    sci_name = str(row.get("scientific_name", taxon_id))
    species_dir = os.path.join(OUTPUT_DIR, safe_dirname(taxon_id, sci_name))
    os.makedirs(species_dir, exist_ok=True)

    existing = {
        f for f in os.listdir(species_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }
    if len(existing) >= IMAGES_PER_SPP:
        return {"taxon_id": taxon_id, "sci_name": sci_name,
                "downloaded": 0, "skipped": len(existing)}

    need   = IMAGES_PER_SPP - len(existing)
    photos = fetch_observations(taxon_id, need)
    time.sleep(REQUEST_DELAY)   # 固定间隔，避免并发叠加请求过密

    downloaded = 0
    for p in photos:
        fname = f"{p['obs_id']}_{p['photo_id']}.jpg"
        if fname in existing:
            continue
        if download_image(p["url"], os.path.join(species_dir, fname)):
            downloaded += 1

    return {"taxon_id": taxon_id, "sci_name": sci_name,
            "downloaded": downloaded, "skipped": len(existing)}


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    df = pd.read_csv(TAXONOMY_CSV, dtype={"taxon_id": str})
    if "taxon_id" not in df.columns:
        raise ValueError(f"CSV 中未找到 taxon_id 列，实际列：{list(df.columns)}")
    print(f"✓ 读取 {len(df)} 个物种  [{TAXONOMY_CSV}]")

    n_sample = int(len(df) * SPECIES_RATIO)
    sampled  = df.sample(n=n_sample, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"✓ 随机抽取 {n_sample} 个物种（{SPECIES_RATIO*100:.0f}%），"
          f"每种 {IMAGES_PER_SPP} 张，预计最多 {n_sample * IMAGES_PER_SPP} 张")
    print(f"✓ 并发线程：{MAX_WORKERS}，API 间隔：{REQUEST_DELAY}s，"
          f"429 退避基数：{THROTTLE_WAIT}s")
    print(f"✓ 图片保存目录：{os.path.abspath(OUTPUT_DIR)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = sampled.to_dict("records")

    total_dl   = 0
    total_skip = 0
    failed     = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_species, row): row for row in rows}
        with tqdm(total=n_sample, desc="下载进度", unit="spp") as pbar:
            for future in as_completed(futures):
                res = future.result()
                total_dl   += res["downloaded"]
                total_skip += res["skipped"]
                if res["downloaded"] == 0 and res["skipped"] == 0:
                    failed.append(res["sci_name"])
                pbar.set_postfix(dl=total_dl, skip=total_skip, fail=len(failed))
                pbar.update(1)

    print(f"\n完成！")
    print(f"  物种数：{n_sample}")
    print(f"  新下载：{total_dl} 张")
    print(f"  已跳过：{total_skip} 张（断点续传）")
    print(f"  无图物种：{len(failed)} 个")
    if failed:
        for name in failed[:20]:
            print(f"    - {name}")
        if len(failed) > 20:
            print(f"    … 共 {len(failed)} 个")


if __name__ == "__main__":
    main()