"""
gate v2：补植物多样性(plantnet + stem) + 训练时加裁切视图。三组对照：
  A  v1 数据（只有 iNat 风格正样本、无裁切）→ 在 plantnet/stem 上当"外部测试"，量化 v1 的域外召回
  B  + plantnet/stem 正样本（仅 full 视图）
  C  + 所有源的裁切视图 cc60/rrc 作为额外训练样本（同一张图的所有视图同组，CV 不泄漏）
评测只在 full 视图（=线上真实输入形态）上做；阈值 = 池化正样本 OOF 分数 2% 分位（98% TPR）。
"""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import train as T1

HERE = Path(__file__).parent
F1, F2 = HERE / "feats", HERE / "feats2"
GROUP = T1.GROUP
rng = np.random.RandomState(0)


def build():
    rows = []  # (emb[N,768], y, src, gid, view)
    def add(emb, y, src, gid, view):
        rows.append(dict(X=emb, y=np.full(len(emb), y), src=np.full(len(emb), src), gid=np.asarray(gid), view=np.full(len(emb), view)))
    p = np.load(F1 / "plant.npy"); add(p, 1, "plant", np.arange(len(p)) // 5, "full")
    for s in GROUP:
        if not (F1 / f"{s}.npy").exists(): continue
        a = np.load(F1 / f"{s}.npy"); add(a, 0, s, 10**6 + (hash(s) % 997) * 10**4 + np.arange(len(a)) // 5, "full")
    for name, base in (("plantnet", 2 * 10**7), ("stem", 3 * 10**7)):
        z = np.load(F2 / f"{name}.npz")
        for v in ("full", "cc60", "rrc"):
            add(z[v], 1, name, base + np.arange(len(z[v])) // 20, v)
    z = np.load(F2 / "plant_crops.npz")
    for v in ("cc60", "rrc"):
        add(z[v], 1, "plant", z["idx"] // 5, v)
    for s in GROUP:
        f = F2 / f"{s}_crops.npz"
        if not f.exists(): continue
        z = np.load(f)
        for v in ("cc60", "rrc"):
            add(z[v], 0, s, 10**6 + (hash(s) % 997) * 10**4 + z["idx"] // 5, v)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def thr98(s): return float(np.quantile(s, 0.02))


def summarize(s, d, thr):
    full = d["view"] == "full"
    y, src = d["y"][full], d["src"][full]; sc = s[full]
    out = {"thr": thr, "auroc": float(roc_auc_score(y, sc))}
    out["tpr_by_src"] = {k: float((sc[src == k] >= thr).mean()) for k in ("plant", "plantnet", "stem")}
    neg = y == 0
    out["fpr"] = float((sc[neg] >= thr).mean())
    gs = np.array([GROUP.get(k, "plant") for k in src])
    out["fpr_by_group"] = {g: float((sc[gs == g] >= thr).mean()) for g in sorted(set(gs[neg]))}
    return out


def oof(d, mask_train, C):
    """5 折分组 OOF：返回每个样本(含未参与训练的)的分数；训练只用 mask_train 的行。"""
    s = np.zeros(len(d["y"]))
    for tr, te in GroupKFold(5).split(d["X"], d["y"], d["gid"]):
        trm = tr[mask_train[tr]]
        clf = T1.fit(d["X"][trm], d["y"][trm], C)
        s[te] = T1.score(clf, d["X"][te])
    return s


def main():
    d = build()
    src, view, y = d["src"], d["view"], d["y"]
    print({k: int((src == k).sum()) for k in sorted(set(src))}, flush=True)
    res = {}

    # A: 只用 v1 数据，在新正样本上外部测试
    mA = (view == "full") & ~np.isin(src, ["plantnet", "stem"])
    clfA = T1.fit(d["X"][mA], y[mA], 1.0)
    sA = T1.score(clfA, d["X"])
    thrA = thr98(oof(d, mA, 1.0)[mA & (y == 1)])        # v1 的阈值（仅 iNat 正样本 OOF）
    res["A_v1"] = summarize(sA, d, thrA)
    print("A", json.dumps(res["A_v1"]["tpr_by_src"]), res["A_v1"]["fpr"], flush=True)

    # B: + plantnet/stem (full)   C: + 全部裁切视图
    for name, m in (("B_newpos", view == "full"), ("C_newpos_crops", np.ones(len(y), bool))):
        best = None
        for C in (0.1, 1.0, 10.0):
            s = oof(d, m, C)
            full_pos = (view == "full") & (y == 1)
            thr = thr98(s[full_pos])
            r = summarize(s, d, thr); r["C"] = C
            print(name, "C", C, "fpr", round(r["fpr"], 4), "tpr", {k: round(v, 3) for k, v in r["tpr_by_src"].items()}, flush=True)
            if best is None or r["fpr"] < best[0]["fpr"]: best = (r, s, C)
        res[name] = best[0]
        if name.startswith("C"):
            bestC, sC = best[2], best[1]
    # 终版 C
    full_pos = (view == "full") & (y == 1)
    thr = thr98(sC[full_pos])
    clf = T1.fit(d["X"], y, bestC)
    np.savez(HERE / "gate_head_v2.npz", w=(clf.coef_[0] * 10).astype(np.float32), b=np.float32(clf.intercept_[0]), threshold=np.float32(thr))
    res["final_thr"] = thr
    sw = {}
    for t in (0.95, 0.98, 0.99, 0.995):
        tt = float(np.quantile(sC[full_pos], 1 - t)); full = view == "full"
        sw[str(t)] = {"thr": tt, "fpr": float((sC[full & (y == 0)] >= tt).mean()),
                      "fpr_screenshot": float((sC[full & (np.array([GROUP.get(k, '') for k in src]) == "screenshot")] >= tt).mean())}
    res["tpr_sweep_C"] = sw

    # leave-group-out（C 配置，整组负样本含其裁切视图都不训练）
    gs = np.array([GROUP.get(k, "pos") for k in src])
    res["lgo_C"] = {}
    for g in sorted(set(GROUP.values())):
        te_neg = gs == g
        tg = np.unique(d["gid"][y == 1]); held = set(rng.choice(tg, int(.2 * len(tg)), replace=False))
        pos_te = (y == 1) & np.isin(d["gid"], list(held))
        tr = ~te_neg & ~pos_te
        clf_g = T1.fit(d["X"][tr], y[tr], bestC)
        te = (view == "full") & (te_neg | pos_te)
        s = T1.score(clf_g, d["X"][te])
        thr_g = thr98(s[y[te] == 1])
        res["lgo_C"][g] = float((s[y[te] == 0] >= thr_g).mean())
        print("LGO", g, round(res["lgo_C"][g], 4), flush=True)
    (HERE / "results_v2.json").write_text(json.dumps(res, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
