"""
无文字向量物种的图像原型 (Botanical Garden pipeline · step 31)
==============================================================
输入：gh_notext_emb/emb.f16.npy + emb_files.csv（bg_20 对 51 个无文字向量物种的图算的向量）
筛选：iNat 只留研究级（来源 inat 且 inat_quality.csv 为 research；丢弃 inat_nr、casual、needs_id）；
      其他来源（commons/plantnet/wikipedia/gbif）照常保留。
去噪：物种内图数 ≥8 时，用与 bg_41 相同的留一法一致性 z < -2.5 去掉离群图。
输出：gh_notext_emb/prototypes.npz（names, embeddings(L2 归一), n_used, n_research_inat, n_other）
      gh_notext_emb/prototypes_summary.csv
自检：留一法，在这些原型彼此之间分类（只对 ≥5 张的物种），看它们之间是否可分。
"""
import numpy as np, pandas as pd

D = "BotanicalGarden/gh_notext_emb/"
E = np.load(D + "emb.f16.npy").astype(np.float32)
f = pd.read_csv(D + "emb_files.csv", dtype={"source_id": str})
q = pd.read_csv("BotanicalGarden/gh_images/inat_quality.csv", dtype=str)
qg = dict(zip(q.obs, q.quality_grade))
f["obs"] = f.page.str.extract(r"observations/(\d+)")[0]
keep = ((f.source == "inat") & (f.obs.map(qg) == "research")) | f.source.isin(["commons", "plantnet", "wikipedia", "gbif"])
print(f"图 {len(f)}；研究级 iNat + 其他来源 保留 {keep.sum()}；丢弃 {(~keep).sum()}（inat_nr {(f.source=='inat_nr').sum()}，iNat 非研究级 {((f.source=='inat') & (f.obs.map(qg) != 'research')).sum()}）")
f["keep"] = keep.values

names, P, rows = [], [], []
for n, g in f[f.keep].groupby("accepted_name"):
    ix = g.index.values; X = E[ix]; used = np.ones(len(ix), bool)
    if len(ix) >= 8:
        s = X.sum(0); loo = (s - X) / (len(ix) - 1); loo /= np.linalg.norm(loo, axis=1, keepdims=True)
        c = (X * loo).sum(1); med = np.median(c); mad = np.median(np.abs(c - med)) * 1.4826 + 1e-6
        used = (c - med) / mad >= -2.5
    p = X[used].mean(0); p /= np.linalg.norm(p)
    names.append(n); P.append(p)
    rows.append({"accepted_name": n, "n_total_downloaded": int((f.accepted_name == n).sum()), "n_after_quality_filter": len(ix),
                 "n_used_for_proto": int(used.sum()), "n_inat_research": int((g.source == "inat").sum()),
                 "n_other_sources": int((g.source != "inat").sum())})
P = np.stack(P)
np.savez(D + "prototypes.npz", names=np.array(names), embeddings=P.astype(np.float32), n_used=np.array([r["n_used_for_proto"] for r in rows]))
S = pd.DataFrame(rows).sort_values("n_used_for_proto"); S.to_csv(D + "prototypes_summary.csv", index=False)
print("原型数", len(names), "| n_used 中位数", int(S.n_used_for_proto.median()), "| <10 张:", int((S.n_used_for_proto < 10).sum()))

# 自检：留一法，原型两两之间分类
idx = {n: i for i, n in enumerate(names)}; ok = []
top1 = top5 = tot = 0
for n, g in f[f.keep].groupby("accepted_name"):
    ix = g.index.values
    if len(ix) < 5: continue
    s = E[ix].sum(0)
    for j in ix:
        loo = (s - E[j]) / (len(ix) - 1); loo /= np.linalg.norm(loo)
        PP = P.copy(); PP[idx[n]] = loo
        sc = PP @ E[j]; r = (sc > sc[idx[n]]).sum()
        top1 += r == 0; top5 += r < 5; tot += 1
print(f"留一法自检（{tot} 张图，物种间 {len(names)} 类）：top1 {top1/tot:.3f}  top5 {top5/tot:.3f}")
