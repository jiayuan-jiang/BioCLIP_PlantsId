"""
图像原型 vs 文字向量（闭集，温室物种）(Botanical Garden pipeline · step 50)
============================================================================
问题：闭集准确率不高，是不是"文字向量"本身不好？用每个物种的 K 张图的均值向量当原型（few-shot prototype），
在同一批测试图上与文字向量、以及两者融合做对比。
防泄漏：
  - 支撑集（原型）与测试图不重叠；
  - 同一物种内，凡是与支撑图同来源且同作者的图一律不作测试图（避免同一人同一株的近重复）；
  - 支撑图只从"未被噪声过滤标记"的图里选（保证原型干净）；测试图报告 全部 / 未标记 两档。
候选集：参与评估的温室物种（支撑图足够的）。文字向量为零的物种保留在候选集里（图像原型对它们照样有效）。
融合：每个测试图对候选集先各自做 z 分数标准化再加权：(1-w)·z(text) + w·z(img_proto)。
重复 R 次随机抽支撑集取平均。
"""
import numpy as np, pandas as pd, json

E = "BotanicalGarden/gh_eval/"
df = pd.read_csv(E + "canon_predictions_clean.csv")
emb = np.load(E + "emb.f16.npy").astype(np.float32)
files = pd.read_csv(E + "emb_files.csv")
assert (files.file.values == df.file.values).all()
df["grp"] = [f"{s}|{a}" if isinstance(a, str) and a else f"{s}|#{i}" for i, (s, a) in enumerate(zip(files.source, files.author))]

ct = np.load("BotanicalGarden/gh_closed/closed_text.npz", allow_pickle=True)
T_all = {n: np.nan_to_num(v) for n, v in zip(ct["names"], ct["embeddings"])}
zero = set(pd.read_csv(E + "zero_vector_species.csv").accepted_name)
ps = pd.read_csv(E + "species_difficulty_clean.csv").set_index("accepted_name")
names_all = sorted(df.accepted_name.unique())
idx_of = df.groupby("accepted_name").indices
clean = ~df.flag_outlier.values


def zs(S, valid=None):
    """逐测试图在候选集上做 z 分数；valid 给出哪些候选的分数可用（文字向量为零的候选不参与统计，z 记 0=中性）。"""
    if valid is None:
        return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
    V = S[:, valid]
    out = np.zeros_like(S)
    out[:, valid] = (V - V.mean(1, keepdims=True)) / (V.std(1, keepdims=True) + 1e-9)
    return out


def run(K, R=5, weights=(0.0, 0.25, 0.5, 0.75, 1.0)):
    out = {w: {"hit1": [], "hit5": [], "hit1_clean": [], "hit5_clean": []} for w in weights}
    per_sp = {w: {} for w in weights}
    for r in range(R):
        rng = np.random.default_rng(r)
        # 候选物种：干净图 ≥ K+3 且 有文字向量记录（零向量物种也在 T_all 里，为 0）
        cand = [n for n in names_all if n in T_all and clean[idx_of[n]].sum() >= K + 3]
        P, T, qs, ql, qc = [], [], [], [], []
        for si, n in enumerate(cand):
            ix = idx_of[n]
            pool = [i for i in ix if clean[i]]
            rng.shuffle(pool)
            sup, used = [], set()
            for i in pool:                                   # 尽量选不同作者/来源
                if df.grp.values[i] not in used:
                    sup.append(i); used.add(df.grp.values[i])
                if len(sup) == K:
                    break
            for i in pool:
                if len(sup) == K:
                    break
                if i not in sup:
                    sup.append(i); used.add(df.grp.values[i])
            sset = set(sup)
            q = [i for i in ix if i not in sset and df.grp.values[i] not in used]
            if len(q) < 3:
                q = [i for i in ix if i not in sset]          # 退而求其次：只保证不重叠
            p = emb[sup].mean(0); P.append(p / np.linalg.norm(p)); T.append(T_all[n])
            qs += q; ql += [si] * len(q); qc += [clean[i] for i in q]
        P, T = np.stack(P), np.stack(T)
        Q = emb[qs]; ql = np.array(ql); qc = np.array(qc)
        tv = np.linalg.norm(T, axis=1) > 0.5
        St, Si = zs(Q @ T.T, tv), zs(Q @ P.T)
        for w in weights:
            S = (1 - w) * St + w * Si
            rank = (S > S[np.arange(len(ql)), ql][:, None]).sum(1)
            h1, h5 = rank == 0, rank < 5
            o = out[w]
            o["hit1"].append(h1.mean()); o["hit5"].append(h5.mean())
            o["hit1_clean"].append(h1[qc].mean()); o["hit5_clean"].append(h5[qc].mean())
            for si, n in enumerate(cand):
                per_sp[w].setdefault(n, []).append(h1[ql == si].mean())
    return {w: {k: float(np.mean(v)) for k, v in d.items()} for w, d in out.items()}, {w: {n: float(np.mean(v)) for n, v in d.items()} for w, d in per_sp.items()}, len(cand)


res, per = {}, {}
import sys
KS = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [1, 3, 5, 10]
for K in KS:
    r, p, nc = run(K)
    res[K], per[K] = r, p
    print(f"\nK={K}（候选 {nc} 个物种）  [测试图=未标记图]   w=0 纯文字 ... w=1 纯图像原型")
    for w, v in r.items():
        print(f"  w={w:<4}  top1 {v['hit1_clean']:.3f}  top5 {v['hit5_clean']:.3f}   (含噪声图: top1 {v['hit1']:.3f})")
json.dump({str(K): {str(w): v for w, v in r.items()} for K, r in res.items()}, open(E + f"image_proto_eval_K{'_'.join(map(str, KS))}.json", "w"), indent=1)
# 物种级：按文字向量是否为零、按 iNat 观测数分桶，比较纯文字 vs 纯图像原型 vs 融合 w=0.5（K=5）
K = KS[-1]
rows = []
for n in per[K][0.0]:
    rows.append({"accepted_name": n, "text": per[K][0.0][n], "proto": per[K][1.0][n], "fuse0.5": per[K][0.5][n],
                 "zero_vec": n in zero, "inat_obs": ps.loc[n, "inat_obs_total"]})
d = pd.DataFrame(rows); d.to_csv(E + f"image_proto_per_species_K{K}.csv", index=False, encoding="utf-8-sig")
print(f"\nK={K} 物种级均值：")
print(" 文字向量为零的物种（%d个）: text %.3f | proto %.3f | fuse %.3f" % ((d.zero_vec).sum(), *d[d.zero_vec][["text", "proto", "fuse0.5"]].mean()))
print(" 其余物种（%d个）           : text %.3f | proto %.3f | fuse %.3f" % ((~d.zero_vec).sum(), *d[~d.zero_vec][["text", "proto", "fuse0.5"]].mean()))
b = pd.cut(d[~d.zero_vec].inat_obs, [-1, 99, 999, 9999, 1e12], labels=["<100", "100-999", "1k-9.9k", ">=10k"])
print(d[~d.zero_vec].groupby(b, observed=True)[["text", "proto", "fuse0.5"]].mean().round(3).assign(n=d[~d.zero_vec].groupby(b, observed=True).size()).to_string())
