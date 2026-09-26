"""
MBGNA 藏品清单 → 学名归一 (Botanical Garden pipeline · step 00)
=============================================================
读取 data/BotanicalGarden/ 下的两个 MBGNA Living Collections CSV，
把 "Scientific Name"（含栽培品种名、杂交号、命名人、sp. 等）清洗后：

  1. 用 GBIF species/match 归一到 accepted species，拿 family / rank / usageKey
  2. 用 iNaturalist /v1/taxa 查对应 taxon_id（后续 bg_10 下图需要）

输出 CSV，一行一个去重后的原始学名，带解析结果与状态标记。
结果缓存到 JSON，可断点续跑。

依赖：pip install requests tqdm pandas
"""

import os
import re
import csv
import json
import time
import argparse

import requests
import pandas as pd
from tqdm import tqdm


# ══════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════

BG_DIR = "./BotanicalGarden"

CSV_GREENHOUSE = os.path.join(
    BG_DIR, "MBGNA_LivingCollectionsDatabase(PublicView)_8355821904764023402.csv"
)
CSV_FULL = os.path.join(
    BG_DIR, "MBGNA_LivingCollectionsDatabase(PublicView)_-3772834397469505304.csv"
)

CACHE_FILE = "./bg_name_resolution_cache.json"

GBIF_MATCH = "https://api.gbif.org/v1/species/match"
INAT_TAXA = "https://api.inaturalist.org/v1/taxa"

HEADERS = {"User-Agent": "bg-name-resolver/1.0 (jiayuanj@umich.edu)"}

REQUEST_DELAY = 1.0     # 每次 iNat 请求后的间隔（GBIF 无需限速，iNat 严格）
RETRY_TIMES = 4
THROTTLE_WAIT = 30.0    # iNat 429 退避基数

# ══════════════════════════════════════════════


# ─────────────────────────────────────────────
# 学名清洗
# ─────────────────────────────────────────────

_CULTIVAR_QUOTE = re.compile(r"'[^']*'")           # 'Bronze Venus'
_CULTIVAR_CV = re.compile(r"\bcv\.?\s+\S+", re.I)
_PAREN = re.compile(r"\([^)]*\)")                  # (Author) 等
_INFRA = re.compile(r"\b(subsp|ssp|var|f|forma)\.?\s+", re.I)
_SPP = re.compile(r"\b(sp{1,2}|spec|indet|aff|cf)\.?(\s|$)", re.I)
_MULTISPACE = re.compile(r"\s+")

# 杂交标记统一成 " x "
_HYBRID = re.compile(r"[×✕✖]|(?<=\s)x(?=\s)")


def clean_name(raw: str):
    """返回 (clean_query, note)。note 记录清洗中触发的特殊情况。"""
    s = raw.strip()
    notes = []

    if _HYBRID.search(s):
        notes.append("hybrid")
    s = _HYBRID.sub(" x ", s)

    if _CULTIVAR_QUOTE.search(s) or _CULTIVAR_CV.search(s):
        notes.append("cultivar")
    s = _CULTIVAR_QUOTE.sub("", s)
    s = _CULTIVAR_CV.sub("", s)
    s = _PAREN.sub("", s)

    genus_only = False
    if _SPP.search(s):
        genus_only = True
        notes.append("genus_only")
        s = _SPP.sub(" ", s)

    # 去掉 infraspecific 前缀词，但保留其后的加词（GBIF 能处理三名法，这里做保守裁剪）
    s = _INFRA.sub("", s)

    # 只保留字母、空格、点、连字符、乘号 x
    s = re.sub(r"[^A-Za-z\- ]", " ", s)
    s = _MULTISPACE.sub(" ", s).strip()

    tokens = s.split()
    if genus_only and tokens:
        s = tokens[0]
    elif len(tokens) >= 2:
        # genus + 第一个加词（含 "x hybridus" 形式时取 genus + x + 加词）
        if tokens[1] == "x" and len(tokens) >= 3:
            s = " ".join(tokens[:3])
        else:
            s = " ".join(tokens[:2])
    elif tokens:
        s = tokens[0]
    else:
        s = ""

    return s, ";".join(notes)


# ─────────────────────────────────────────────
# 读取藏品 CSV
# ─────────────────────────────────────────────

def load_species(csv_path: str) -> pd.DataFrame:
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            name = (r.get("Scientific Name") or "").strip()
            if not name:
                continue
            rows.append({
                "raw_name": name,
                "subzone": (r.get("Management Subzone") or "").strip(),
            })
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("raw_name")
        .agg(accession_count=("raw_name", "size"),
             subzones=("subzone", lambda s: "|".join(sorted(set(x for x in s if x)))))
        .reset_index()
        .sort_values("raw_name")
    )
    return agg


# ─────────────────────────────────────────────
# 解析：GBIF + iNat
# ─────────────────────────────────────────────

def gbif_match(name: str) -> dict:
    try:
        resp = requests.get(
            GBIF_MATCH,
            params={"name": name, "kingdom": "Plantae", "strict": "false"},
            headers=HEADERS, timeout=20,
        )
        resp.raise_for_status()
        d = resp.json()
    except Exception as e:
        return {"gbif_error": str(e)}
    return {
        "gbif_usage_key": d.get("usageKey"),
        "gbif_name": d.get("canonicalName") or d.get("scientificName"),
        "gbif_rank": (d.get("rank") or "").lower(),
        "gbif_status": d.get("status"),
        "gbif_match_type": d.get("matchType"),
        "gbif_confidence": d.get("confidence"),
        "family": d.get("family"),
        "genus": d.get("genus"),
    }


def inat_taxon(name: str) -> dict:
    for attempt in range(1, RETRY_TIMES + 1):
        try:
            resp = requests.get(
                INAT_TAXA,
                params={"q": name, "is_active": "true", "per_page": 5},
                headers=HEADERS, timeout=20,
            )
        except Exception as e:
            if attempt < RETRY_TIMES:
                time.sleep(REQUEST_DELAY * 2)
                continue
            return {"inat_error": str(e)}

        if resp.status_code == 429:
            time.sleep(THROTTLE_WAIT * attempt)
            continue
        if not resp.ok:
            return {"inat_error": f"HTTP {resp.status_code}"}

        results = resp.json().get("results", [])
        # 优先：完全同名且为植物
        best = None
        for r in results:
            if r.get("iconic_taxon_name") != "Plantae":
                continue
            if r.get("name", "").lower() == name.lower():
                best = r
                break
            if best is None:
                best = r
        if best is None:
            return {"inat_taxon_id": None}
        return {
            "inat_taxon_id": best.get("id"),
            "inat_name": best.get("name"),
            "inat_rank": best.get("rank"),
            "inat_common": best.get("preferred_common_name"),
            "inat_obs_count": best.get("observations_count"),
        }
    return {"inat_error": "persistent 429"}


def resolve_one(clean_query: str, cache: dict) -> dict:
    if not clean_query:
        return {"status": "empty_after_clean"}
    if clean_query in cache:
        return cache[clean_query]

    out = {}
    out.update(gbif_match(clean_query))
    out.update(inat_taxon(clean_query))
    time.sleep(REQUEST_DELAY)

    # 状态判定
    mt = out.get("gbif_match_type")
    rank = out.get("gbif_rank")
    if mt in (None, "NONE"):
        status = "no_match"
    elif rank == "species":
        status = "ok" if mt == "EXACT" else "ok_fuzzy"
    elif rank in ("genus", "family"):
        status = "genus_only"
    else:
        status = f"rank_{rank}"
    if out.get("inat_taxon_id") is None and status.startswith("ok"):
        status += "_no_inat"
    out["status"] = status

    cache[clean_query] = out
    return out


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["greenhouse", "full", "both"],
                    default="greenhouse")
    ap.add_argument("--out", default=None,
                    help="输出 CSV 路径，默认按 scope 命名")
    args = ap.parse_args()

    scopes = {
        "greenhouse": [("greenhouse", CSV_GREENHOUSE)],
        "full": [("full", CSV_FULL)],
        "both": [("greenhouse", CSV_GREENHOUSE), ("full", CSV_FULL)],
    }[args.scope]
    out_path = args.out or f"./bg_{args.scope}_species.csv"

    # 缓存
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"✓ 载入缓存 {len(cache)} 条  [{CACHE_FILE}]")

    # 汇总所有 scope 的物种
    frames = []
    for scope_label, path in scopes:
        d = load_species(path)
        d.insert(0, "scope", scope_label)
        frames.append(d)
        print(f"✓ {scope_label}: {len(d)} 个去重学名  [{os.path.basename(path)}]")
    df = pd.concat(frames, ignore_index=True)

    # 若 both：同名跨 scope 合并，scope 记为 greenhouse+full
    if args.scope == "both":
        df = (
            df.groupby("raw_name")
            .agg(scope=("scope", lambda s: "+".join(sorted(set(s)))),
                 accession_count=("accession_count", "sum"),
                 subzones=("subzones", lambda s: "|".join(sorted(set(
                     x for v in s for x in v.split("|") if x)))))
            .reset_index()
        )

    # 清洗
    cleaned = df["raw_name"].map(clean_name)
    df["clean_query"] = [c[0] for c in cleaned]
    df["clean_note"] = [c[1] for c in cleaned]

    # 解析
    records = []
    try:
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="解析", unit="spp"):
            res = resolve_one(row.clean_query, cache)
            rec = row._asdict()
            rec.update(res)
            records.append(rec)
    finally:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
        print(f"\n✓ 缓存已保存（{len(cache)} 条）")

    out_df = pd.DataFrame(records)

    col_order = [
        "scope", "raw_name", "accession_count", "clean_query", "clean_note",
        "status", "gbif_name", "gbif_rank", "gbif_match_type", "gbif_confidence",
        "gbif_status", "family", "genus", "gbif_usage_key",
        "inat_taxon_id", "inat_name", "inat_rank", "inat_common", "inat_obs_count",
        "subzones",
    ]
    out_df = out_df[[c for c in col_order if c in out_df.columns]]
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 摘要
    print(f"\n===== 解析摘要（{args.scope}）=====")
    print(f"输入去重学名：{len(out_df)}")
    print(out_df["status"].value_counts().to_string())
    n_species = out_df.loc[out_df["status"].str.startswith("ok"), "gbif_name"].nunique()
    n_inat = out_df["inat_taxon_id"].notna().sum()
    print(f"\n解析到 accepted species（去重）：{n_species}")
    print(f"拿到 iNat taxon_id：{n_inat} / {len(out_df)}")
    print(f"\n输出：{os.path.abspath(out_path)}")
    print("下一步：人工抽查 status=ok_fuzzy / genus_only / no_match 的行，再跑 bg_10_fetch_images.py")


if __name__ == "__main__":
    main()
