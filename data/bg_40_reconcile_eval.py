"""
命名对账 + 重新评分 (Botanical Garden pipeline · step 40)
========================================================
问题：评估里一部分"识别错误"其实是命名问题——标准答案（GBIF accepted）与模型词表（BioCLIP/iNat 命名）
用了同一植物的不同学名（同物异名、属名更迭、拼写变体、杂交种写法）。
做法：用 Kew WCVP 名录（BotanicalGarden/wcvp/wcvp_names.csv）把【标准答案的各种名字】和【模型预测的名字】
都映射到同一个"现行接受种名"，再比较。映射规则：
  物种名 → 若是 Synonym 沿 accepted_plant_name_id 跳到现行 taxon；若现行 taxon 是种下等级则取其所属种；
  同一名字对应多个现行 taxon（同名异物）→ 优先 Accepted，仍歧义则标记 ambiguous 并保持原名；
  WCVP 里没有（苔藓/化石等）→ 保持原名。
标准答案集合 GT = canon({accepted_name, 原始记录名二名法, iNat 现行名})。
  （iNat 图的标签来自 iNat taxon，所以 iNat 名也属于这批图的合法标注；GBIF 同物异名表不纳入，噪声大。）
Top-1 / Top-5：预测名映射后命中 GT；开集 Top-5 取前 20 个候选里按"现行名去重"后的前 5 个。

输出（BotanicalGarden/gh_eval/）：
  canon_predictions.csv        逐图：开集/闭集在对账口径下的命中
  canon_per_species.csv        逐物种：对账前后成功率、变化、新增被判对的预测名
  name_reconcile_gt.csv        标准答案侧的命名冲突清单（GBIF accepted ≠ WCVP 现行名等）
  canon_summary.json / canon_summary.md
"""
import re, json, os
import numpy as np, pandas as pd

E = "BotanicalGarden/gh_eval/"
WCVP = "BotanicalGarden/wcvp/wcvp_names.csv"
CACHE = "BotanicalGarden/wcvp/canon_map.json"


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[×✕✖]|\bx\b", " ", str(s).lower())).strip()


def build_canon_map():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    cols = ["plant_name_id", "taxon_rank", "taxon_status", "taxon_name", "accepted_plant_name_id", "parent_plant_name_id"]
    w = pd.read_csv(WCVP, sep="|", usecols=cols, dtype=str, low_memory=False)
    info = {r.plant_name_id: (r.taxon_rank, r.taxon_name, r.parent_plant_name_id) for r in w.itertuples()}

    def to_species(pid):
        rank, name, parent = info.get(pid, (None, None, None))
        if rank == "Species":
            return name
        if rank in ("Subspecies", "Variety", "Form", "Subvariety", "Subform") and parent in info and info[parent][0] == "Species":
            return info[parent][1]
        return None
    cand = {}
    sp = w[w.taxon_rank == "Species"]
    for r in sp.itertuples():
        tgt = r.accepted_plant_name_id if isinstance(r.accepted_plant_name_id, str) else (r.plant_name_id if r.taxon_status == "Accepted" else None)
        if tgt is None:
            continue
        t = to_species(tgt)
        if t:
            cand.setdefault(norm(r.taxon_name), {}).setdefault(norm(t), []).append(r.taxon_status)
    out = {}
    for k, d in cand.items():
        if len(d) == 1:
            out[k] = next(iter(d))
        else:                                   # 同名异物：优先自己就是 Accepted 的那个
            acc = [t for t, st in d.items() if "Accepted" in st]
            out[k] = acc[0] if len(acc) == 1 else k
    json.dump(out, open(CACHE, "w"))
    return out


def main():
    cm = build_canon_map()
    print("WCVP 种级名字映射条目", len(cm), flush=True)
    canon = lambda n: cm.get(norm(n), norm(n))

    # ── 标准答案侧 ──
    acc = pd.read_csv("bg_greenhouse_species_accepted.csv")
    ids = pd.read_csv("bg_greenhouse_inat_ids.csv", dtype=str)[["accepted_name", "inat_names"]]
    sp = acc.merge(ids, on="accepted_name", how="left")
    gt_set, gt_strict, gl = {}, {}, []
    for r in sp.itertuples():
        raws = []
        for rawn in str(r.raw_names).split(" | "):
            t = re.sub(r"[^A-Za-z\- ]", " ", re.sub(r"'[^']*'", "", rawn)).split()
            if len(t) >= 2:
                raws.append(f"{t[0]} {t[1]}")
        inat = [x for x in str(r.inat_names).split(" | ")] if isinstance(r.inat_names, str) else []
        ca, craw, cin = canon(r.accepted_name), {canon(x) for x in raws}, {canon(x) for x in inat}
        gt_strict[r.accepted_name] = ca
        gt_set[r.accepted_name] = {ca} | craw | cin
        in_w = norm(r.accepted_name) in cm
        gl.append({"accepted_name(GBIF)": r.accepted_name, "WCVP收录": in_w, "canon(accepted)": ca,
                   "与WCVP一致": ca == norm(r.accepted_name), "canon(原始记录名)": " | ".join(sorted(craw)),
                   "canon(iNat名)": " | ".join(sorted(cin)), "各名字canon不一致": len(gt_set[r.accepted_name]) > 1})
    gl = pd.DataFrame(gl)
    gl.to_csv(E + "name_reconcile_gt.csv", index=False, encoding="utf-8-sig")

    # ── 预测侧 ──
    df = pd.read_csv(E + "predictions.csv")
    raw = np.load("bioclip_full_index.npz", allow_pickle=True)
    sci_raw = raw["sci_names"]
    canon_row = {}                                           # 词表行 → 现行名（只对出现的行算）
    z = np.load(E + "sim_cache.npz")
    top20 = z["top20_idx"]
    for i in np.unique(top20):
        canon_row[i] = canon(sci_raw[i])
    N = len(df)
    sids_names = df.accepted_name.values
    o1, o5, o20 = np.zeros(N, bool), np.zeros(N, bool), np.zeros(N, bool)
    o1s = np.zeros(N, bool)
    newly = {}
    for g in range(N):
        gs = gt_set[sids_names[g]]
        cs = [canon_row[i] for i in top20[g]]
        o1[g] = cs[0] in gs
        o1s[g] = cs[0] == gt_strict[sids_names[g]]
        seen, uniq = set(), []
        for c in cs:
            if c not in seen:
                seen.add(c); uniq.append(c)
        o5[g] = any(c in gs for c in uniq[:5])
        o20[g] = any(c in gs for c in uniq)
        if o1[g] and not df.open_hit1_len.iloc[g] == True:      # 对账前判错、对账后判对
            newly.setdefault(sids_names[g], []).append(str(sci_raw[top20[g, 0]]))
    df["c_open_hit1"], df["c_open_hit1_strict"], df["c_open_hit5"], df["c_open_hit20"] = o1, o1s, o5, o20
    ctop = [str(x) for x in df.closed_top1]
    df["c_closed_hit1"] = [canon(p) in gt_set[a] for a, p in zip(df.accepted_name, ctop)]
    df["c_closed_hit5"] = [any(canon(p) in gt_set[a] for p in str(t).split(" | ")) for a, t in zip(df.accepted_name, df.closed_top5)]
    df.to_csv(E + "canon_predictions.csv", index=False, encoding="utf-8-sig")

    # ── 逐物种 ──
    zero = set(pd.read_csv(E + "zero_vector_species.csv").accepted_name)
    ps0 = pd.read_csv(E + "per_species.csv")
    g = df.groupby("accepted_name")
    ps = pd.DataFrame({"n_img": g.size(),
                       "open_top1_before": g.open_hit1_len.mean(), "open_top1_canon": g.c_open_hit1.mean(),
                       "open_top5_canon": g.c_open_hit5.mean(),
                       "closed_top1_before": g.closed_hit1.mean(), "closed_top1_canon": g.c_closed_hit1.mean(),
                       "closed_top5_canon": g.c_closed_hit5.mean()})
    ps["open_gain"] = ps.open_top1_canon - ps.open_top1_before
    ps["closed_gain"] = ps.closed_top1_canon - ps.closed_top1_before
    ps["zero_vec"] = ps.index.isin(zero)
    ps["newly_correct_preds(open)"] = pd.Series({k: "; ".join(f"{n}×{c}" for n, c in pd.Series(v).value_counts().head(3).items()) for k, v in newly.items()})
    ps = ps.join(ps0.set_index("accepted_name")[["family", "accession_count", "inat_obs_total", "top_wrong_open", "top_wrong_closed"]])
    ps = ps.reset_index().sort_values("open_gain", ascending=False)
    ps.to_csv(E + "canon_per_species.csv", index=False, encoding="utf-8-sig")

    # ── 汇总 ──
    S = {"n_img": int(N), "n_species": int(len(ps)),
         "micro": {"open_top1_before": float(df.open_hit1_len.mean()), "open_top1_canon": float(df.c_open_hit1.mean()),
                   "open_top5_before": float(df.open_hit5_len.mean()), "open_top5_canon": float(df.c_open_hit5.mean()),
                   "closed_top1_before": float(df.closed_hit1.mean()), "closed_top1_canon": float(df.c_closed_hit1.mean()),
                   "closed_top5_before": float(df.closed_hit5.mean()), "closed_top5_canon": float(df.c_closed_hit5.mean())},
         "gt_name_audit": {"GBIF_accepted_not_in_WCVP": int((~gl["WCVP收录"]).sum()),
                           "GBIF_accepted_is_WCVP_synonym": int((~gl["与WCVP一致"] & gl["WCVP收录"]).sum()),
                           "gt_names_disagree": int(gl["各名字canon不一致"].sum())},
         "species_changed_open": {"up>=10pt": int((ps.open_gain >= .1).sum()), "up>=30pt": int((ps.open_gain >= .3).sum()),
                                  "down": int((ps.open_gain < -1e-9).sum())}}
    json.dump(S, open(E + "canon_summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(S, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
