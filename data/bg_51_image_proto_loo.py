"""
图像原型 vs 文字向量：留一法、不丢物种 (Botanical Garden pipeline · step 51)
=========================================================================
bg_50 要求物种有 K+3 张干净图才参与，会把最稀有的物种整体筛掉 → 偏乐观。
本脚本：每张测试图 i（物种 s）
  - 自己物种的原型 = s 的"其余干净图"均值，排除与 i 同来源且同作者的图（防近重复泄漏）；上限 CAP 张（随机）
  - 其他物种的原型 = 该物种全部干净图均值（上限 CAP 张）
  - 候选集 = 全部有干净图的物种（602 个，含零向量物种）；自己物种排除后没图可用的测试图跳过（统计在 skipped）
  - 原型来自"其余图"，所以图越少的物种原型越粗（真实反映稀有种处境）
文字分：零向量物种的文字分记 0（中性），z 分数只在非零候选上统计。融合 = (1-w)·z(text)+w·z(img)。
按物种稀有程度 / 图数分层报告；按物种平均（macro）和按图平均（micro）两种。
"""
import sys, json, numpy as np, pandas as pd

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
E = "BotanicalGarden/gh_eval/"
df = pd.read_csv(E + "canon_predictions_clean.csv")
emb = np.load(E + "emb.f16.npy").astype(np.float32)
files = pd.read_csv(E + "emb_files.csv")
df["grp"] = [f"{s}|{a}" if isinstance(a, str) and a else f"{s}|#{i}" for i, (s, a) in enumerate(zip(files.source, files.author))]
ct = np.load("BotanicalGarden/gh_closed/closed_text.npz", allow_pickle=True)
Tm = {n: np.nan_to_num(v) for n, v in zip(ct["names"], ct["embeddings"])}
zero = set(pd.read_csv(E + "zero_vector_species.csv").accepted_name)
ps = pd.read_csv(E + "species_difficulty_clean.csv").set_index("accepted_name")
clean = ~df.flag_outlier.values
rng = np.random.default_rng(0)

names = [n for n, ix in df.groupby("accepted_name").indices.items() if clean[ix].sum() >= 1 and n in Tm]
ix_all = df.groupby("accepted_name").indices
pool = {n: np.array([i for i in ix_all[n] if clean[i]]) for n in names}
T = np.stack([Tm[n] for n in names]); tv = np.linalg.norm(T, axis=1) > 0.5
sp_id = {n: k for k, n in enumerate(names)}
# 其他物种的原型：全部（≤CAP）干净图的均值
def mean_unit(ix):
    p = emb[ix].mean(0); return p / np.linalg.norm(p)
P0 = np.stack([mean_unit(rng.permutation(pool[n])[:CAP]) for n in names])

def zs(S, valid):
    out = np.zeros_like(S); V = S[:, valid]
    out[:, valid] = (V - V.mean(1, keepdims=True)) / (V.std(1, keepdims=True) + 1e-9); return out
Wts = (0.0, 0.25, 0.5, 0.75, 1.0)
rec = []
skipped = 0
for n in names:
    s = sp_id[n]
    for i in ix_all[n]:
        sup = [j for j in pool[n] if j != i and df.grp.values[j] != df.grp.values[i]]
        if not sup:
            skipped += 1; continue
        if len(sup) > CAP: sup = list(rng.permutation(sup)[:CAP])
        P = P0.copy(); P[s] = mean_unit(sup)
        q = emb[i][None]
        St = zs(q @ T.T, tv); Si = zs(q @ P.T, np.ones(len(names), bool))
        r = {"accepted_name": n, "clean": bool(clean[i]), "k_used": len(sup)}
        for w in Wts:
            sc = (1 - w) * St + w * Si
            rank = int((sc[0] > sc[0, s]).sum())
            r[f"h1_{w}"], r[f"h5_{w}"] = rank == 0, rank < 5
        rec.append(r)
R = pd.DataFrame(rec); R.to_csv(E + f"image_proto_loo_cap{CAP}.csv", index=False)
print(f"CAP={CAP}  候选物种 {len(names)}；测试图 {len(R)}（因无可用支撑图跳过 {skipped}）")
Rc = R[R.clean]
print("\n【按图平均 micro，未标记测试图】  w=0 纯文字 … w=1 纯图像原型")
for w in Wts: print(f"  w={w:<4} top1 {Rc[f'h1_{w}'].mean():.3f}  top5 {Rc[f'h5_{w}'].mean():.3f}")
sp = Rc.groupby("accepted_name")[[f"h1_{w}" for w in Wts] + [f"h5_{w}" for w in Wts]].mean()
print("\n【按物种平均 macro，%d 个物种】" % len(sp))
for w in Wts: print(f"  w={w:<4} top1 {sp[f'h1_{w}'].mean():.3f}  top5 {sp[f'h5_{w}'].mean():.3f}")
sp = sp.join(ps[["n_clean", "inat_obs_total"]]); sp["zero_vec"] = sp.index.isin(zero)
nb = pd.cut(sp.n_clean, [0, 4, 9, 22, 1000], labels=["<5张", "5-9张", "10-22张", "≥23张"])
ob = pd.cut(sp.inat_obs_total, [-1, 99, 999, 9999, 1e12], labels=["<100", "100-999", "1k-9.9k", "≥1万"])
for lab, b in (("按物种图数", nb), ("按 iNat 观测数", ob)):
    t = sp[~sp.zero_vec].groupby(b, observed=True).agg(n=("h1_0.0", "size"), 文字=("h1_0.0", "mean"), 图像原型=("h1_1.0", "mean"),
                                                     融合0_5=("h1_0.5", "mean"), 融合0_75=("h1_0.75", "mean")).round(3)
    print(f"\n【{lab}（不含零向量物种）top-1 物种平均】"); print(t.to_string())
z = sp[sp.zero_vec]
print(f"\n【零向量物种 {len(z)} 个】 文字 {z['h1_0.0'].mean():.3f} | 图像原型 {z['h1_1.0'].mean():.3f} | 融合0.75 {z['h1_0.75'].mean():.3f}")
sp.to_csv(E + f"image_proto_loo_per_species_cap{CAP}.csv", encoding="utf-8-sig")
