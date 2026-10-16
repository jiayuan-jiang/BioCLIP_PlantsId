"""
accepted species → iNaturalist taxon_id 重新解析 (Botanical Garden pipeline · step 05)
======================================================================================
bg_00 的 iNat 查询有三个问题，本步骤修正：
  1. 只查 is_active=true，已停用的同名 taxon 被忽略，随后"取第一条结果"会拿到别的物种
     （例：Tillandsia cyanea → 误取 T. guatemalensis，正解是已并入的 Wallisia cyanea）。
  2. 同物异名（GBIF 与 iNat 命名不同）没有用 GBIF synonyms 补查。
  3. 杂交种去掉了 "×"，iNat 上的杂交种名带 "×"。

规则：只接受**名字完全一致**的 taxon（忽略大小写与 ×）；已停用的沿 current_synonymous_taxon_ids
跳到现行 taxon；绝不取"第一条模糊结果"。先查 accepted 名（及带 × 变体），查不到有观测的再查
GBIF synonyms 与原始记录名。

输入：bg_greenhouse_species_accepted.csv
输出：bg_greenhouse_inat_ids.csv（accepted_name, inat_ids, inat_names, how, synonyms, old_id, changed）
缓存：bg_inat_ids_cache.json（可断点续跑）

依赖：pip install requests tqdm pandas
"""

import os
import re
import json
import time

import requests
import pandas as pd
from tqdm import tqdm

HEADERS = {"User-Agent": "bg-inat-resolver/1.0 (jiayuanj@umich.edu)"}
INAT_TAXA = "https://api.inaturalist.org/v1/taxa"
GBIF_SYN = "https://api.gbif.org/v1/species/{}/synonyms"
SPECIES_CSV = "./bg_greenhouse_species_accepted.csv"
OUT_CSV = "./bg_greenhouse_inat_ids.csv"
CACHE = "./bg_inat_ids_cache.json"
DELAY = 1.0


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[×✕✖]|\bx\b", " ", (s or "").lower())).strip()


def get(url, params=None, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=(5, 20))
            if r.status_code == 429:
                time.sleep(15 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(2)
    return None            # 请求失败；调用方不得把失败当成"查无此物"缓存


def inat_exact(name, cache):
    """返回该名字对应的现行 taxon 列表 [(id, name, obs_count)]；已停用的沿同物异名跳转。"""
    key = norm(name)
    if key in cache:
        return cache[key]
    data = get(INAT_TAXA, {"q": name, "per_page": 8, "is_active": "any"})
    time.sleep(DELAY)
    if data is None:
        return []          # 失败：不写缓存，下次重跑会重试
    res = data.get("results", [])
    found = []
    for t in res:
        if norm(t["name"]) != key:
            continue
        if t.get("is_active"):
            found.append((t["id"], t["name"], t.get("observations_count") or 0))
        else:
            for sid in t.get("current_synonymous_taxon_ids") or []:
                d = (get(f"{INAT_TAXA}/{sid}") or {}).get("results", [])
                time.sleep(DELAY)
                if d:
                    found.append((d[0]["id"], d[0]["name"], d[0].get("observations_count") or 0))
    found = list(dict.fromkeys(found))
    cache[key] = found
    return found


def gbif_synonyms(key):
    res = (get(GBIF_SYN.format(int(key)), {"limit": 30}) or {}).get("results", [])
    names = [s.get("canonicalName") for s in res if (s.get("rank") or "").upper() == "SPECIES"]
    return list(dict.fromkeys(n for n in names if n))


def main():
    sp = pd.read_csv(SPECIES_CSV)
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    rows = []
    try:
        for r in tqdm(sp.to_dict("records"), desc="解析 iNat", unit="spp"):
            name = r["accepted_name"]
            raws = str(r.get("raw_names", "")).split(" | ")
            hybrid = any("×" in x or re.search(r"\sx\s", x) for x in raws)
            g, e = name.split()[0], name.split()[1] if len(name.split()) > 1 else ""
            tries = [("accepted", name)]
            if hybrid:
                tries.insert(0, ("hybrid", f"{g} × {e}"))
            found, how = [], ""
            for h, n in tries:
                f = inat_exact(n, cache)
                if f:
                    found, how = f, h
                    break
            syn = gbif_synonyms(r["accepted_key"]) if pd.notna(r.get("accepted_key")) else []
            if not any(o > 0 for _, _, o in found):       # 没有有观测的 taxon → 才去试同物异名/原始名
                alt = []
                for x in raws:
                    t = re.sub(r"[^A-Za-z\- ]", " ", re.sub(r"'[^']*'", "", x)).split()
                    if len(t) >= 2:
                        alt.append(("raw", f"{t[0]} {t[1]}"))
                alt += [("synonym", s) for s in syn]
                for h, n in alt:
                    f = [x for x in inat_exact(n, cache) if x[2] > 0]
                    if f:
                        found, how = f, f"{h}:{n}"
                        break
            old = r.get("inat_taxon_id")
            ids = [str(i) for i, _, _ in found]
            rows.append({
                "accepted_name": name, "inat_ids": ",".join(ids),
                "inat_names": " | ".join(n for _, n, _ in found),
                "inat_obs": ",".join(str(o) for _, _, o in found),
                "how": how, "synonyms": " | ".join(syn),
                "old_id": "" if pd.isna(old) else int(old),
                "changed": (str(int(old)) if pd.notna(old) else "") not in ids,
            })
    finally:
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n解析完成 {len(out)}；有 iNat taxon：{(out.inat_ids != '').sum()}；"
          f"与旧 id 不同：{out.changed.sum()}；→ {OUT_CSV}")


if __name__ == "__main__":
    main()
