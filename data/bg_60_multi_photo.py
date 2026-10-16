"""
多张照片合成一个 embedding 能否提升识别 (Botanical Garden pipeline · step 60)
===========================================================================
模拟用户对同一物种拍 m 张照片（m=1,2,3,5），把 m 个图像向量合成一个再检索。
合成方式：mean（L2 归一化后取均值再归一化，等价于各图相似度平均）；闭集另测 prob（各图 softmax 概率平均）。
取图方式：random=同物种随机 m 张；same_author=同来源同作者的 m 张（更像同一次拍摄）。
为公平：所有 m 在同一批物种上比较（干净图 ≥5 张的物种）；每物种每档 B 个 bag。
评分与 bg_40 一致：名字经 WCVP 对账，开集 top-5 取现行名去重后的前 5。
注意：我们的同物种图来自不同植株/拍摄者，不是"同一棵植物的多角度"，same_author 只是近似。
"""
import re, json, numpy as np, pandas as pd, torch

E = "BotanicalGarden/gh_eval/"; B = 8; MS = (1, 2, 3, 5); SCALE = 100.0
cm = json.load(open("BotanicalGarden/wcvp/canon_map.json"))
def norm(s): return re.sub(r"\s+", " ", re.sub(r"[×✕✖]|\bx\b", " ", str(s).lower())).strip()
canon = lambda n: cm.get(norm(n), norm(n))

df = pd.read_csv(E + "canon_predictions_clean.csv")
emb = np.load(E + "emb.f16.npy").astype(np.float32)
files = pd.read_csv(E + "emb_files.csv")
df["grp"] = [f"{s}|{a}" if isinstance(a, str) and a else f"{s}|#{i}" for i, (s, a) in enumerate(zip(files.source, files.author))]
clean = ~df.flag_outlier.values
acc = pd.read_csv("bg_greenhouse_species_accepted.csv")
ids = pd.read_csv("bg_greenhouse_inat_ids.csv", dtype=str)[["accepted_name", "inat_names"]]
sp = acc.merge(ids, on="accepted_name", how="left")
gt = {}
for r in sp.itertuples():
    s = {canon(r.accepted_name)}
    for rawn in str(r.raw_names).split(" | "):
        t = re.sub(r"[^A-Za-z\- ]", " ", re.sub(r"'[^']*'", "", rawn)).split()
        if len(t) >= 2: s.add(canon(f"{t[0]} {t[1]}"))
    if isinstance(r.inat_names, str): s |= {canon(x) for x in r.inat_names.split(" | ")}
    gt[r.accepted_name] = s

raw = np.load("bioclip_full_index.npz", allow_pickle=True)
X = torch.from_numpy(raw["embeddings"].astype(np.float32)); sci = raw["sci_names"]
ct = np.load("BotanicalGarden/gh_closed/closed_text.npz", allow_pickle=True)
cnames = list(ct["names"]); CT = torch.from_numpy(np.nan_to_num(ct["embeddings"]).astype(np.float32))
ccanon = [canon(n) for n in cnames]
rowcanon = {}
def rc(i):
    if i not in rowcanon: rowcanon[i] = canon(sci[i])
    return rowcanon[i]

ix = df.groupby("accepted_name").indices
pool = {n: np.array([i for i in v if clean[i]]) for n, v in ix.items()}
species = sorted(n for n, v in pool.items() if len(v) >= 5)
print("参与物种（干净图≥5）:", len(species), "/", len(pool), flush=True)
rng = np.random.default_rng(0)

def bags(n, m, mode):
    p = pool[n]
    out = []
    if mode == "random":
        for _ in range(B): out.append(rng.choice(p, m, replace=False))
    else:
        g = df.grp.values; byg = {}
        for i in p: byg.setdefault(g[i], []).append(i)
        ok = [v for v in byg.values() if len(v) >= m]
        if not ok: return []
        for _ in range(B):
            v = ok[rng.integers(len(ok))]; out.append(rng.choice(v, m, replace=False))
    return out

def evaluate(items, m, mode):
    """items: list of (species, idx array)；返回 dict 指标（按物种平均）"""
    res = {"o1": [], "o5": [], "c1m": [], "c5m": [], "c1p": [], "c5p": [], "sp": []}
    for b0 in range(0, len(items), 256):
        ch = items[b0:b0 + 256]
        E_ = torch.from_numpy(np.stack([emb[a] for _, a in ch]))            # (b, m, d)
        q = E_.mean(1); q = q / q.norm(dim=1, keepdim=True)
        S = q @ X.T; top = S.topk(20, dim=1).indices.numpy()
        Sc = q @ CT.T
        Sp = torch.softmax(SCALE * (E_ @ CT.T), dim=2).mean(1)                # prob 平均 (b, C)
        for j, (n, _) in enumerate(ch):
            g = gt[n]
            cs = [rc(i) for i in top[j]]; seen, uq = set(), []
            for c in cs:
                if c not in seen: seen.add(c); uq.append(c)
            res["o1"].append(cs[0] in g); res["o5"].append(any(c in g for c in uq[:5]))
            for key, sc in (("m", Sc[j]), ("p", Sp[j])):
                o = torch.argsort(sc, descending=True)[:5].numpy()
                res[f"c1{key}"].append(ccanon[o[0]] in g); res[f"c5{key}"].append(any(ccanon[k] in g for k in o))
            res["sp"].append(n)
    d = pd.DataFrame(res); g = d.groupby("sp").mean(numeric_only=True)
    return {k: float(g[k].mean()) for k in ("o1", "o5", "c1m", "c5m", "c1p", "c5p")} | {"n_bags": len(d), "n_species": len(g)}

rows = []
for mode in ("random", "same_author"):
    for m in MS:
        items = [(n, a) for n in species for a in bags(n, m, mode)]
        if not items: continue
        r = evaluate(items, m, mode); r.update(mode=mode, m=m); rows.append(r)
        print(f"{mode:12s} m={m}  bags={r['n_bags']:5d} 物种={r['n_species']:3d} | 开集 top1 {r['o1']:.3f} top5 {r['o5']:.3f} | 闭集(均值向量) top1 {r['c1m']:.3f} top5 {r['c5m']:.3f} | 闭集(概率平均) top1 {r['c1p']:.3f} top5 {r['c5p']:.3f}", flush=True)
pd.DataFrame(rows).to_csv(E + "multi_photo_results.csv", index=False)
