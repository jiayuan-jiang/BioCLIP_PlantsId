"""
温室测试集 · 当前线上版本识别精度评估 (Botanical Garden pipeline · step 21)
==========================================================================
"当前版本" = demo/inference_core.py：BioCLIP 2 图像 embedding × 35 万物种文字原型，余弦相似度取 Top-K，
外加 energy gate（THRESHOLD=68.537）。本脚本在 bg_20_embed.py 的 embedding 上离线复现，并记录完整数据：

  开放集 (open) ：候选 = 全量 350,398 个物种（线上现状）
  闭集   (closed)：候选 = 温室 614 个物种里在词表中有对应行的那些（"植物园场景闭集"设想的上界）

标注匹配两档（报告里都给，差值即"命名问题"的大小）：
  strict  只认 accepted_name
  lenient 认 accepted_name + iNat 名 + GBIF 同物异名 + 原始记录名二名法

输出（BotanicalGarden/gh_eval/）：
  predictions.csv   逐图：Top-5、相似度、rank、energy、gate、开放集/闭集命中
  per_species.csv   逐物种：张数、各口径 acc、主要误判去向、长尾特征
  confusions.csv    最常见的误判对
  summary.json / summary.md   汇总表（总体、按来源/许可、按长尾分桶、置信度校准、gate 影响）
  plots/*.png

用法：/opt/miniconda3/bin/python bg_21_eval.py
"""

import os
import re
import sys
import json
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

OUT = "./BotanicalGarden/gh_eval"
INDEX = "./bioclip_full_index.npz"
WEIGHTS = "./weights/bioclip2/open_clip_model.safetensors"
SPECIES_CSV = "./bg_greenhouse_species_accepted.csv"
IDS_CSV = "./bg_greenhouse_inat_ids.csv"
sys.path.insert(0, "../demo")
import gate  # noqa: E402  线上同一份 gate

BATCH = 256
TOPK = 20


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[×✕✖]|\bx\b", " ", str(s).lower())).strip()


def names_for(r):
    strict = {norm(r.accepted_name)}
    lenient = set(strict)
    for col in ("inat_names", "synonyms"):
        if pd.notna(getattr(r, col)):
            lenient |= {norm(x) for x in str(getattr(r, col)).split(" | ") if x}
    for raw in str(r.raw_names).split(" | "):
        t = re.sub(r"[^A-Za-z\- ]", " ", re.sub(r"'[^']*'", "", raw)).split()
        if len(t) >= 2:
            lenient.add(norm(f"{t[0]} {t[1]}"))
    return strict, lenient


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    os.makedirs(f"{OUT}/plots", exist_ok=True)
    E = np.load(f"{OUT}/emb.f16.npy").astype(np.float32)
    files = pd.read_csv(f"{OUT}/emb_files.csv")
    assert len(E) == len(files)
    valid = np.linalg.norm(E, axis=1) > 0.5
    print(f"图片 {len(E)}；坏图 {(~valid).sum()}", flush=True)

    # ── 词表 ──
    raw = np.load(INDEX, allow_pickle=True)
    X = torch.from_numpy(raw["embeddings"].astype(np.float32))
    sci_raw = raw["sci_names"]                      # NpzFile 每次取值都会重新解压，必须只读一次
    sci = [norm(s) for s in sci_raw]
    genera = np.array([str(g) for g in raw["genera"]])
    families = np.array([str(f) for f in raw["families"]])
    name2rows = {}
    for i, s in enumerate(sci):
        name2rows.setdefault(s, []).append(i)
    with safe_open(WEIGHTS, "pt") as f:
        scale = float(torch.exp(f.get_tensor("logit_scale")))
    print(f"词表 {len(sci):,}；logit_scale={scale:.2f}；gate 阈值 {gate.THRESHOLD}", flush=True)

    # ── 物种表 / 标注 ──
    acc = pd.read_csv(SPECIES_CSV)
    ids = pd.read_csv(IDS_CSV, dtype=str)[["accepted_name", "inat_names", "synonyms", "inat_obs"]]
    sp = acc.merge(ids, on="accepted_name", how="left").reset_index(drop=True)
    sp["genus"] = sp.accepted_name.str.split().str[0]
    sid_of = {n: i for i, n in enumerate(sp.accepted_name)}
    rows_s, rows_l, genus_set = [], [], []
    for r in sp.itertuples():
        st, le = names_for(r)
        rs = sorted({i for n in st for i in name2rows.get(n, [])})
        rl = sorted({i for n in le for i in name2rows.get(n, [])})
        rows_s.append(np.array(rs, dtype=np.int64))
        rows_l.append(np.array(rl, dtype=np.int64))
        genus_set.append({n.split()[0] for n in le})
    sp["vocab_strict"] = [len(x) > 0 for x in rows_s]
    sp["vocab_lenient"] = [len(x) > 0 for x in rows_l]
    print(f"物种 {len(sp)}：词表里有对应行 strict={sp.vocab_strict.sum()} lenient={sp.vocab_lenient.sum()}", flush=True)

    # 闭集候选：lenient 有词表行的物种；每个物种的行并成一组，物种分数 = 组内最大
    cand = [i for i in range(len(sp)) if len(rows_l[i])]
    c_rows = np.concatenate([rows_l[i] for i in cand])
    c_starts = np.cumsum([0] + [len(rows_l[i]) for i in cand[:-1]])
    cand_arr = np.array(cand)
    sid_to_cpos = {s: k for k, s in enumerate(cand)}

    # ── 逐图 ──
    sids = np.array([sid_of[n] for n in files.accepted_name])
    N = len(E)
    rec = {k: np.full(N, np.nan) for k in
           ["top1_sim", "energy", "rank_strict", "rank_len", "closed_rank", "closed_top1_sim", "closed_margin"]}
    top20_idx = np.zeros((N, TOPK), np.int64)
    top20_sim = np.zeros((N, TOPK), np.float32)
    closed_top5 = np.full((N, 5), -1, np.int64)
    cache = f"{OUT}/sim_cache.npz"
    Et = torch.from_numpy(E)
    if os.path.exists(cache):
        z = np.load(cache)
        top20_idx, top20_sim, closed_top5 = z["top20_idx"], z["top20_sim"], z["closed_top5"]
        rec = {k: z[k] for k in rec}
        print("读取相似度缓存", cache, flush=True)
    for b0 in (range(0, N, BATCH) if not os.path.exists(cache) else []):
        S = Et[b0:b0 + BATCH] @ X.T
        en = torch.logsumexp(S * scale, dim=1).numpy()
        v, ix = S.topk(TOPK, dim=1)
        top20_idx[b0:b0 + len(S)] = ix.numpy()
        top20_sim[b0:b0 + len(S)] = v.numpy()
        rec["energy"][b0:b0 + len(S)] = en
        Sn = S.numpy()
        Sc = np.maximum.reduceat(Sn[:, c_rows], c_starts, axis=1)          # (B, n_cand)
        order = np.argsort(-Sc, axis=1)[:, :5]
        for j in range(len(S)):
            g = b0 + j
            s = sids[g]
            for key, rows in (("rank_strict", rows_s[s]), ("rank_len", rows_l[s])):
                if len(rows):
                    rec[key][g] = int((Sn[j] > Sn[j, rows].max()).sum()) + 1
            closed_top5[g] = cand_arr[order[j]]
            rec["closed_top1_sim"][g] = Sc[j, order[j, 0]]
            rec["closed_margin"][g] = Sc[j, order[j, 0]] - Sc[j, order[j, 1]]
            if s in sid_to_cpos:
                rec["closed_rank"][g] = int((Sc[j] > Sc[j, sid_to_cpos[s]]).sum()) + 1
        if (b0 // BATCH) % 10 == 0:
            print(f"  {b0 + len(S)}/{N}", flush=True)

    if not os.path.exists(cache):
        np.savez(cache, top20_idx=top20_idx, top20_sim=top20_sim, closed_top5=closed_top5, **rec)
    # ── 逐图表 ──
    df = files.copy()
    df["gt_vocab_strict"] = sp.vocab_strict.values[sids]
    df["gt_vocab_lenient"] = sp.vocab_lenient.values[sids]
    df["valid"] = valid
    df["top1_name"] = [sci_raw[i] for i in top20_idx[:, 0]]
    df["top1_sim"] = top20_sim[:, 0]
    df["top5_names"] = [" | ".join(sci_raw[i] for i in r[:5]) for r in top20_idx]
    df["top5_sims"] = [" | ".join(f"{x:.3f}" for x in r[:5]) for r in top20_sim]
    df["rank_strict"], df["rank_len"] = rec["rank_strict"], rec["rank_len"]
    for k in (1, 5, 20):
        df[f"open_hit{k}_strict"] = df.rank_strict <= k
        df[f"open_hit{k}_len"] = df.rank_len <= k
    pred_genus = genera[top20_idx[:, 0]]
    df["open_genus_hit1"] = [pred_genus[g].lower() in genus_set[sids[g]] for g in range(N)]
    pred_fam = families[top20_idx[:, 0]]
    gt_fam = sp.family.values[sids]
    df["pred_family"] = pred_fam
    df["family_known"] = (pred_fam != "") & (pred_fam != "nan")
    df["open_family_hit1"] = [(pred_fam[g].lower() == str(gt_fam[g]).lower()) for g in range(N)]
    df["energy"] = rec["energy"]
    df["gate_pass"] = rec["energy"] >= gate.THRESHOLD
    df["closed_top1"] = sp.accepted_name.values[closed_top5[:, 0]]
    df["closed_top5"] = [" | ".join(sp.accepted_name.values[r]) for r in closed_top5]
    df["closed_top1_sim"], df["closed_margin"], df["closed_rank"] = (
        rec["closed_top1_sim"], rec["closed_margin"], rec["closed_rank"])
    df["closed_hit1"] = df.closed_rank <= 1
    df["closed_hit5"] = df.closed_rank <= 5
    if (~valid).any():
        hit_cols = [c for c in df.columns if c.startswith(("open_hit", "closed_hit"))]
        df[hit_cols] = df[hit_cols].astype(object)
        df.loc[~valid, hit_cols] = np.nan
    df.to_csv(f"{OUT}/predictions.csv", index=False, encoding="utf-8-sig")

    # ── 逐物种 ──
    ok = df[df.valid]
    g = ok.groupby("accepted_name")
    ps = pd.DataFrame({
        "n_img": g.size(),
        "open_top1_strict": g.open_hit1_strict.mean(), "open_top1_len": g.open_hit1_len.mean(),
        "open_top5_len": g.open_hit5_len.mean(), "open_genus_top1": g.open_genus_hit1.mean(),
        "closed_top1": g.closed_hit1.mean(), "closed_top5": g.closed_hit5.mean(),
        "median_rank_len": g.rank_len.median(), "mean_energy": g.energy.mean(),
    })
    wrong = ok[~ok.open_hit1_len.astype(bool)].groupby("accepted_name").top1_name.agg(
        lambda s: "; ".join(f"{k}×{v}" for k, v in s.value_counts().head(3).items()))
    ps["top_wrong_open"] = wrong
    wc = ok[~ok.closed_hit1.astype(bool)].groupby("accepted_name").closed_top1.agg(
        lambda s: "; ".join(f"{k}×{v}" for k, v in s.value_counts().head(3).items()))
    ps["top_wrong_closed"] = wc
    ps = sp.set_index("accepted_name")[["family", "genus", "accession_count", "inat_obs",
                                        "vocab_strict", "vocab_lenient"]].join(ps, how="inner")
    ps["inat_obs_total"] = [sum(int(float(x)) for x in str(v).split(",") if x not in ("", "nan"))
                            if pd.notna(v) else 0 for v in ps.inat_obs]
    ps["genus_vocab_size"] = [int((genera == str(gn)).sum()) for gn in ps.genus]
    ps = ps.drop(columns=["inat_obs"]).reset_index()
    ps.to_csv(f"{OUT}/per_species.csv", index=False, encoding="utf-8-sig")

    # ── 误判对 ──
    bad = ok[~ok.open_hit1_len.astype(bool)]
    cf = bad.groupby(["accepted_name", "top1_name"]).size().reset_index(name="n").sort_values("n", ascending=False)
    cf["same_genus"] = [a.split()[0].lower() == str(b).split()[0].lower() for a, b in zip(cf.accepted_name, cf.top1_name)]
    cf.head(300).to_csv(f"{OUT}/confusions.csv", index=False, encoding="utf-8-sig")

    # ── 汇总 ──
    S = {}
    n = len(ok)

    def block(d, tag):
        return {"n_img": int(len(d)),
                "open_top1_strict": float(d.open_hit1_strict.mean()), "open_top1_len": float(d.open_hit1_len.mean()),
                "open_top5_len": float(d.open_hit5_len.mean()), "open_top20_len": float(d.open_hit20_len.mean()),
                "open_genus_top1": float(d.open_genus_hit1.mean()),
                "closed_top1": float(d.closed_hit1.mean()), "closed_top5": float(d.closed_hit5.mean())}
    S["overall_micro"] = block(ok, "all")
    macro = ps[["open_top1_strict", "open_top1_len", "open_top5_len", "open_genus_top1", "closed_top1", "closed_top5"]].mean()
    S["overall_macro_over_species"] = {k: float(v) for k, v in macro.items()}
    S["n_species_evaluated"] = int(len(ps))
    S["vocab_coverage"] = {"species_total": int(len(sp)), "in_vocab_strict": int(sp.vocab_strict.sum()),
                           "in_vocab_lenient": int(sp.vocab_lenient.sum())}
    inv = ok[ok.gt_vocab_lenient]
    S["in_vocab_only_micro"] = block(inv, "inv")
    S["family_acc_known_only"] = {"n_known": int(ok.family_known.sum()),
                                  "acc": float(ok[ok.family_known].open_family_hit1.mean())}
    S["gate"] = {"threshold": gate.THRESHOLD, "pass_rate": float(ok.gate_pass.mean()),
                 "rejected_true_plants": int((~ok.gate_pass).sum()),
                 "acc_top1_len_if_gate_pass": float(ok[ok.gate_pass].open_hit1_len.mean()),
                 "acc_top1_len_if_gate_reject": float(ok[~ok.gate_pass].open_hit1_len.mean()) if (~ok.gate_pass).any() else None}
    S["by_source"] = {k: block(d, k) for k, d in ok.groupby("source")}
    lic = ok.license.fillna("").str.lower()
    tier = np.where(lic == "arr", "all-rights-reserved", np.where(lic.isin(["unknown", ""]), "unknown", "CC/PD"))
    S["by_license_tier"] = {k: block(d, k) for k, d in ok.assign(tier=tier).groupby("tier")}

    def bucket(col, bins, labels):
        b = pd.cut(ps[col], bins=bins, labels=labels, right=True)
        out = {}
        for lb, grp in ps.groupby(b, observed=True):
            out[str(lb)] = {"n_species": int(len(grp)),
                            "open_top1_len": float(grp.open_top1_len.mean()),
                            "open_top5_len": float(grp.open_top5_len.mean()),
                            "closed_top1": float(grp.closed_top1.mean())}
        return out
    S["longtail_by_accession_count"] = bucket("accession_count", [0, 1, 2, 5, 10, 1e9], ["1", "2", "3-5", "6-10", ">10"])
    S["longtail_by_inat_obs"] = bucket("inat_obs_total", [-1, 0, 99, 999, 9999, 1e12],
                                       ["0", "1-99", "100-999", "1k-9.9k", ">=10k"])
    S["longtail_by_n_img"] = bucket("n_img", [0, 4, 9, 49, 50], ["<5", "5-9", "10-49", "50"])
    S["longtail_by_genus_vocab_size"] = bucket("genus_vocab_size", [-1, 0, 5, 20, 100, 1e9],
                                              ["0 (genus absent)", "1-5", "6-20", "21-100", ">100"])
    fam = ps.groupby("family").agg(n_species=("accepted_name", "size"), open_top1_len=("open_top1_len", "mean"),
                                   closed_top1=("closed_top1", "mean")).sort_values("n_species", ascending=False)
    S["top_families"] = fam.head(15).round(4).reset_index().to_dict("records")
    sim = ok.top1_sim
    cal = {}
    for lb, mk in (("<0.22 (low)", sim < 0.22), ("0.22-0.35 (medium)", (sim >= 0.22) & (sim < 0.35)), (">=0.35 (high)", sim >= 0.35)):
        d = ok[mk]
        cal[lb] = {"n_img": int(len(d)), "open_top1_len": float(d.open_hit1_len.mean()) if len(d) else None}
    S["calibration_by_top1_sim"] = cal
    rk = ok.rank_len.dropna()
    S["rank_len_quantiles"] = {str(q): float(rk.quantile(q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)}
    S["species_acc_distribution_open_top1_len"] = {
        "=0": int((ps.open_top1_len == 0).sum()), "(0,0.5)": int(((ps.open_top1_len > 0) & (ps.open_top1_len < 0.5)).sum()),
        "[0.5,0.9)": int(((ps.open_top1_len >= 0.5) & (ps.open_top1_len < 0.9)).sum()), ">=0.9": int((ps.open_top1_len >= 0.9).sum())}
    S["wrong_top1_same_genus_share"] = float(cf[cf.same_genus].n.sum() / max(cf.n.sum(), 1))
    json.dump(S, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1)

    # ── markdown ──
    def tb(d, keys=("n_img", "open_top1_strict", "open_top1_len", "open_top5_len", "open_genus_top1", "closed_top1", "closed_top5")):
        hdr = "| 分组 | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1) + "\n"
        return hdr + "".join("| " + str(k) + " | " + " | ".join(
            (str(v[c]) if c == "n_img" else pct(v[c])) if v.get(c) is not None else "-" for c in keys) + " |\n" for k, v in d.items())
    md = ["# 温室测试集 · 当前线上版本识别精度\n",
          f"图片 {n}（坏图 {(~valid).sum()}）；物种 {len(ps)}；词表 350,398。\n",
          "## 总体\n", tb({"micro(按图)": S["overall_micro"], "macro(按物种)": {**S["overall_macro_over_species"], "n_img": n},
                          "仅词表内物种": S["in_vocab_only_micro"]}),
          f"\n词表覆盖：strict {S['vocab_coverage']['in_vocab_strict']}/{len(sp)}，lenient {S['vocab_coverage']['in_vocab_lenient']}/{len(sp)}\n",
          "\n## 按来源\n", tb(S["by_source"]), "\n## 按许可档\n", tb(S["by_license_tier"]),
          "\n## 长尾（物种级均值）\n"]
    for key, title in (("longtail_by_accession_count", "植物园 accession 数"), ("longtail_by_inat_obs", "iNat 全球观测数"),
                       ("longtail_by_n_img", "测试图张数"), ("longtail_by_genus_vocab_size", "词表里同属物种数")):
        md.append(f"\n### {title}\n")
        md.append(tb({k: {**v, "n_img": v["n_species"]} for k, v in S[key].items()},
                     keys=("n_img", "open_top1_len", "open_top5_len", "closed_top1")).replace("n_img", "物种数", 1))
    md += ["\n## 置信度校准（Top-1 相似度分档，开放集 lenient）\n"] + [f"- {k}: n={v['n_img']}, acc={pct(v['open_top1_len']) if v['open_top1_len'] is not None else '-'}\n" for k, v in cal.items()]
    md += [f"\n## Gate\n通过率 {pct(S['gate']['pass_rate'])}，误拦真植物 {S['gate']['rejected_true_plants']} 张\n",
           f"\n## 物种级 top-1 分布\n{S['species_acc_distribution_open_top1_len']}\n",
           f"\n误判中同属占比：{pct(S['wrong_top1_same_genus_share'])}；真值 rank 分位数：{S['rank_len_quantiles']}\n"]
    open(f"{OUT}/summary.md", "w").write("".join(md))

    # ── 图 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].hist(ps.open_top1_len, bins=20, color="#4c78a8"); ax[0].set_title("per-species top-1 (open, lenient)")
        ax[0].set_xlabel("accuracy"); ax[0].set_ylabel("species")
        ax[1].hist(ps.closed_top1, bins=20, color="#59a14f"); ax[1].set_title("per-species top-1 (closed set)")
        ax[1].set_xlabel("accuracy"); plt.tight_layout(); plt.savefig(f"{OUT}/plots/per_species_hist.png", dpi=130); plt.close()
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for a, key, t in zip(ax, ("longtail_by_accession_count", "longtail_by_inat_obs", "longtail_by_genus_vocab_size"),
                             ("garden accession count", "iNat global observations", "genus size in vocabulary")):
            d = S[key]; x = np.arange(len(d))
            a.bar(x - 0.2, [v["open_top1_len"] for v in d.values()], 0.4, label="open top-1", color="#4c78a8")
            a.bar(x + 0.2, [v["closed_top1"] for v in d.values()], 0.4, label="closed top-1", color="#59a14f")
            a.set_xticks(x); a.set_xticklabels(list(d.keys()), fontsize=8); a.set_title(t); a.set_ylim(0, 1)
        ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/plots/longtail.png", dpi=130); plt.close()
    except Exception as e:
        print("画图跳过:", e)
    print(open(f"{OUT}/summary.md").read())


if __name__ == "__main__":
    main()
