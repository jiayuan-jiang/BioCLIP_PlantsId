"""
温室闭集文字塔 (Botanical Garden pipeline · step 30)
====================================================
把温室 accepted species 映射到 BioCLIP 2 官方文本向量库（bioclip_full_index.npz，
即 pybioclip TreeOfLifeClassifier 预计算向量筛出 Plantae 后的导出），抽出子集作为闭集文字塔。

映射优先级（逐物种，取第一个命中）：
  exact    accepted_name 与词表二名法完全一致
  synonym  iNat 现行名 / GBIF 同物异名 / 原始记录名二名法 命中
  (未命中的写入 unmatched，不做属级兜底——要不要兜底由评估时决定)
多行命中（词表里有重名条目）时，向量取均值后重新 L2 归一；全部行号保留在 row_idx 里。
科校验：词表里的科 vs GBIF 的科，不一致的标 family_mismatch（同名异属/同音异物的预警）。

输出（BotanicalGarden/gh_closed/）：
  closed_text.npz       embeddings(n,768) L2 归一, names, match_type, family, genus, index_names ...
  closed_mapping.csv    逐物种映射表，供人工核对
  closed_unmatched.csv  没有映射到的物种
"""
import os
import re
import json
import numpy as np
import pandas as pd

OUT = "./BotanicalGarden/gh_closed"
INDEX = "./bioclip_full_index.npz"


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[×✕✖]|\bx\b", " ", str(s).lower())).strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = np.load(INDEX, allow_pickle=True)
    emb = raw["embeddings"].astype(np.float32)
    sci = [norm(s) for s in raw["sci_names"]]
    name2rows = {}
    for i, s in enumerate(sci):
        name2rows.setdefault(s, []).append(i)

    acc = pd.read_csv("bg_greenhouse_species_accepted.csv")
    ids = pd.read_csv("bg_greenhouse_inat_ids.csv", dtype=str)[["accepted_name", "inat_names", "synonyms"]]
    sp = acc.merge(ids, on="accepted_name", how="left")

    recs, vecs, kept = [], [], []
    for r in sp.itertuples():
        cands = []                                    # (match_type, 名字)
        cands.append(("exact", norm(r.accepted_name)))
        for col in ("inat_names", "synonyms"):
            if pd.notna(getattr(r, col)):
                cands += [("synonym", norm(x)) for x in str(getattr(r, col)).split(" | ") if x]
        for rawn in str(r.raw_names).split(" | "):
            t = re.sub(r"[^A-Za-z\- ]", " ", re.sub(r"'[^']*'", "", rawn)).split()
            if len(t) >= 2:
                cands.append(("synonym", norm(f"{t[0]} {t[1]}")))
        mt, hit, rows = None, None, []
        for typ, n in cands:
            if n in name2rows:
                mt, hit, rows = typ, n, name2rows[n]
                break
        base = {"accepted_name": r.accepted_name, "family_gbif": r.family, "accession_count": r.accession_count,
                "raw_names": r.raw_names}
        if not rows:
            recs.append({**base, "match_type": "unmatched"})
            continue
        v = emb[rows].mean(0)
        v /= np.linalg.norm(v)
        fam_idx = str(raw["families"][rows[0]])
        recs.append({**base, "match_type": mt, "index_name": raw["sci_names"][rows[0]], "n_rows": len(rows),
                     "row_idx": json.dumps(rows), "family_index": fam_idx,
                     "genus_index": str(raw["genera"][rows[0]]),
                     "common_index": str(raw["common_names"][rows[0]]),
                     "taxon_path_index": str(raw["taxon_paths"][rows[0]]),
                     "family_mismatch": bool(fam_idx and str(r.family) != "nan" and fam_idx.lower() != str(r.family).lower())})
        vecs.append(v)
        kept.append(len(recs) - 1)

    df = pd.DataFrame(recs)
    df.to_csv(f"{OUT}/closed_mapping.csv", index=False, encoding="utf-8-sig")
    df[df.match_type == "unmatched"][["accepted_name", "family_gbif", "accession_count", "raw_names"]].to_csv(
        f"{OUT}/closed_unmatched.csv", index=False, encoding="utf-8-sig")
    m = df[df.match_type != "unmatched"].reset_index(drop=True)
    np.savez_compressed(f"{OUT}/closed_text.npz", embeddings=np.stack(vecs).astype(np.float32),
                        names=m.accepted_name.values, match_type=m.match_type.values,
                        index_names=m.index_name.values, family=m.family_index.values, genus=m.genus_index.values,
                        row_idx=m.row_idx.values)
    print(f"温室物种 {len(sp)}")
    print(df.match_type.value_counts().to_string())
    print(f"闭集文字塔：{len(m)} × {np.stack(vecs).shape[1]}  → {OUT}/closed_text.npz")
    mm = m[m.family_mismatch]
    print(f"科不一致（需人工看）：{len(mm)}")
    if len(mm):
        print(mm[["accepted_name", "index_name", "family_gbif", "family_index"]].head(20).to_string(index=False))
    print("多行命中（词表重名）：", int((m.n_rows > 1).sum()))
    print("未映射：", list(df[df.match_type == "unmatched"].accepted_name))


if __name__ == "__main__":
    main()
