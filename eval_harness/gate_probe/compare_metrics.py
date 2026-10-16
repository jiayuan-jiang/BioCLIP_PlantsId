"""线上 energy vs v1 头 vs v2 头：同一批整图样本、各自线上阈值，输出 recall / precision / 拦截率。分数均为 5 折分组 OOF（新来源对 v1 为外部测试）。"""
import json
import numpy as np
import train as T1, train2 as T2

d = T2.build()
full = d["view"] == "full"
src, y, X = d["src"], d["y"], d["X"]
# energy（线上同款，logit_scale=100，35万物种索引）
idx = np.load(T1.ROOT / "data/bioclip_full_index.npz", allow_pickle=True)["embeddings"].astype(np.float32)
en = np.full(len(y), np.nan)
ii = np.where(full)[0]
for i in range(0, len(ii), 512):
    z = 100.0 * (X[ii[i:i+512]] @ idx.T); m = z.max(1, keepdims=True)
    en[ii[i:i+512]] = m[:, 0] + np.log(np.exp(z - m).sum(1))
mA = full & ~np.isin(src, ["plantnet", "stem"])
sA = T2.oof(d, mA, 1.0)                       # v1 配置：只用 iNat 正样本+原负样本，full 视图
sC = T2.oof(d, np.ones(len(y), bool), 0.1)    # v2 配置：全部数据+裁切视图
models = {"energy(线上)": (en, 68.537), "v1头": (sA, 3.584), "v2头": (sC, 2.548)}
groups = np.array([T1.GROUP.get(k, "pos") for k in src])
res = {}
for name, (s, thr) in models.items():
    pred = s >= thr
    r = {"recall_by_src": {k: float(pred[full & (src == k)].mean()) for k in ("plant", "plantnet", "stem")}}
    pos = full & (y == 1); neg = full & (y == 0)
    r["recall_pooled"] = float(pred[pos].mean())
    # 各来源等权的宏平均召回（避免 iNat 25k 张主导）
    r["recall_macro"] = float(np.mean(list(r["recall_by_src"].values())))
    r["fpr"] = float(pred[neg].mean())
    r["block_rate_by_group"] = {g: float(1 - pred[full & (groups == g)].mean()) for g in sorted(set(groups[neg]))}
    # precision 随先验变化：π=植物占比
    tpr, fpr = r["recall_macro"], r["fpr"]
    r["precision_at_prior"] = {str(p): float(tpr * p / (tpr * p + fpr * (1 - p))) for p in (0.5, 0.8, 0.95)}
    res[name] = r
json.dump(res, open("compare_metrics.json", "w"), indent=1, ensure_ascii=False)
print(f"测试集(整图)：植物 iNat {int((full&(src=='plant')).sum())} / plantnet {int((full&(src=='plantnet')).sum())} / stem {int((full&(src=='stem')).sum())}；非植物 {int((full&(y==0)).sum())}")
for n, r in res.items():
    print(f"\n== {n}")
    print(" recall:", {k: round(v, 3) for k, v in r["recall_by_src"].items()}, "macro", round(r["recall_macro"], 3))
    print(" 非植物误放 FPR:", round(r["fpr"], 4), " 拦截率:", {k: round(v, 3) for k, v in r["block_rate_by_group"].items()})
    print(" precision(植物占比 50/80/95%):", {k: round(v, 4) for k, v in r["precision_at_prior"].items()})
