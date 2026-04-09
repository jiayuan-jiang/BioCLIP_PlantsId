"""
iNaturalist 北美植物物种排行榜抓取脚本
输出：CSV 文件，按观测数量降序排列
"""

import csv
import time
import urllib.request
import urllib.parse
import json
import os


def fetch_species_counts(place_id, iconic_taxa, quality_grade, rank, per_page, page):
    base_url = "https://api.inaturalist.org/v1/observations/species_counts"
    params = {
        "place_id": place_id,
        "iconic_taxa[]": iconic_taxa,
        "quality_grade": quality_grade,
        "rank": rank,
        "per_page": per_page,
        "page": page,
        "order_by": "observations_count",
        "order": "desc",
    }
    query_string = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_string}"

    req = urllib.request.Request(url, headers={"User-Agent": "inat-plant-fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data


def main():
    # ──────────────────────────────────────────
    # 修改这里的变量来调整抓取行为
    # ──────────────────────────────────────────

    PLACE_ID = 1         # iNaturalist place_id：97394 = 北美洲
                              # 其他常用：1=全球, 6712=美国, 6712=加拿大(6712换成682)
    ICONIC_TAXA = "Plantae"   # 分类群：Plantae / Animalia / Fungi / Insecta 等
    QUALITY_GRADE = "research"  # 观测质量：research（已验证）/ needs_id / any
    RANK = "species"          # 分类等级：species / genus / any
    TOTAL_SPECIES = 5000      # 最多抓取物种数量（iNat 上限约 500*页数）
    PER_PAGE = 500            # 每页条数，最大 500
    REQUEST_DELAY = 0.4       # 请求间隔秒数，避免触发限速

    OUTPUT_FILE = "global_plants_top5000.csv"  # 输出 CSV 文件名

    # ──────────────────────────────────────────

    total_pages = -(-TOTAL_SPECIES // PER_PAGE)  # 向上取整
    all_species = []

    print(f"开始抓取：北美植物 Top {TOTAL_SPECIES}")
    print(f"place_id={PLACE_ID}, taxa={ICONIC_TAXA}, quality={QUALITY_GRADE}")
    print(f"共需请求 {total_pages} 页（每页 {PER_PAGE} 条）\n")

    for page in range(1, total_pages + 1):
        print(f"  第 {page}/{total_pages} 页…", end=" ", flush=True)
        try:
            data = fetch_species_counts(
                place_id=PLACE_ID,
                iconic_taxa=ICONIC_TAXA,
                quality_grade=QUALITY_GRADE,
                rank=RANK,
                per_page=PER_PAGE,
                page=page,
            )
        except Exception as e:
            print(f"请求失败：{e}")
            break

        results = data.get("results", [])
        if not results:
            print("无数据，提前结束。")
            break

        for r in results:
            taxon = r.get("taxon", {})
            all_species.append({
                "rank": len(all_species) + 1,
                "common_name": taxon.get("preferred_common_name", ""),
                "scientific_name": taxon.get("name", ""),
                "taxon_id": taxon.get("id", ""),
                "observations_count": r.get("count", 0),
                "iconic_taxon": taxon.get("iconic_taxon_name", ""),
                "rank_level": taxon.get("rank", ""),
            })

        print(f"累计 {len(all_species)} 个物种，本页 {len(results)} 条")

        if len(all_species) >= TOTAL_SPECIES:
            all_species = all_species[:TOTAL_SPECIES]
            break
        if len(results) < PER_PAGE:
            print("  数据已全部返回，提前结束。")
            break

        time.sleep(REQUEST_DELAY)

    if not all_species:
        print("未获取到任何数据，请检查参数或网络。")
        return

    # 写入 CSV
    fieldnames = [
        "rank",
        "common_name",
        "scientific_name",
        "taxon_id",
        "observations_count",
        "iconic_taxon",
        "rank_level",
    ]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_species)

    abs_path = os.path.abspath(OUTPUT_FILE)
    print(f"\n完成！共写入 {len(all_species)} 个物种")
    print(f"输出文件：{abs_path}")


if __name__ == "__main__":
    main()