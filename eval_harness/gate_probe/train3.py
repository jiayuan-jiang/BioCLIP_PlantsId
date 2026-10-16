"""gate v3：在 v2 基础上补通用网图风格植物正样本(oxford_flowers / plant_village / sun397_plant)。
关键改动：①阈值按「各正样本来源等权」的 98% 分位定（不让 25k iNat 主导）；
②对三个新来源做 leave-source-out：不训练该来源，看召回——真实反映域偏移。"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import train as T1, train2 as T2

HERE = Path(__file__).parent
NEW = ["oxford_flowers", "plant_village", "sun397_plant"]
POS = ["plant", "plantnet", "stem"] + NEW
GROUP = T1.GROUP


def build3():
    d = T2.build()
    parts = [d]
    for i, name in enumerate(NEW):
        z = np.load(T2.F2 / f"{name}.npz")
        for v in ("full", "cc60", "rrc"):
            n = len(z[v])
            parts.append(dict(X=z[v], y=np.ones(n, int), src=np.full(n, name), gid=(4 + i) * 10**7 + np.arange(n) // 20, view=np.full(n, v)))
    return {k: np.concatenate([p[k] for p in parts]) for k in d}


def wq(s, w, q):
    o = np.argsort(s); c = np.cumsum(w[o]) / w.sum()
    return float(s[o][np.searchsorted(c, q)])


def src_thr(s, d, srcs, tpr=0.98):
    """各来源等权的 (1-tpr) 分位阈值，只用整图视图的正样本。"""
    m = (d["view"] == "full") & np.isin(d["src"], srcs)
    w = np.zeros(len(s))
    for k in srcs:
        mk = m & (d["src"] == k); w[mk] = 1.0 / max(mk.sum(), 1)
    return wq(s[m], w[m], 1 - tpr)


def report(s, d, thr, srcs=POS):
    full = d["view"] == "full"; y, src = d["y"], d["src"]
    r = {"thr": thr, "recall": {k: float((s[full & (src == k)] >= thr).mean()) for k in srcs}}
    r["recall_macro"] = float(np.mean(list(r["recall"].values())))
    gs = np.array([GROUP.get(k, "pos") for k in src]); neg = full & (y == 0)
    r["fpr"] = float((s[neg] >= thr).mean())
    r["fpr_by_group"] = {g: float((s[full & (gs == g)] >= thr).mean()) for g in sorted(set(gs[neg]))}
    return r


def main():
    d = build3(); y, src, view = d["y"], d["src"], d["view"]
    print({k: int((src == k).sum()) for k in sorted(set(src))}, flush=True)
    res = {}
    allm = np.ones(len(y), bool)
    best = None
    for C in (0.1, 1.0):
        s = T2.oof(d, allm, C); thr = src_thr(s, d, POS); r = report(s, d, thr); r["C"] = C
        print("C", C, "fpr", round(r["fpr"], 4), "recall", {k: round(v, 3) for k, v in r["recall"].items()}, flush=True)
        if best is None or r["fpr"] < best[0]["fpr"]: best = (r, s, C)
    res["v3"] = best[0]; sC, bestC = best[1], best[2]
    # leave-source-out：新来源完全不参与训练 → 域外召回
    res["lso"] = {}
    for k in NEW:
        m = src != k
        s = np.zeros(len(y))
        # 用其余数据 5 折 OOF 定阈值；k 用全量其余数据训练的模型打分
        s_oof = T2.oof(d, m, bestC)
        clf = T1.fit(d["X"][m], y[m], bestC); s[src == k] = T1.score(clf, d["X"][src == k])
        s[m] = s_oof[m]
        thr = src_thr(s, d, [p for p in POS if p != k])
        rec = float((s[(view == "full") & (src == k)] >= thr).mean())
        res["lso"][k] = {"recall_unseen": rec, "thr": thr, "fpr": float((s[(view == "full") & (y == 0)] >= thr).mean())}
        print("LSO", k, res["lso"][k], flush=True)
    thr = src_thr(sC, d, POS)
    clf = T1.fit(d["X"], y, bestC)
    np.savez(HERE / "gate_head_v3.npz", w=(clf.coef_[0] * 10).astype(np.float32), b=np.float32(clf.intercept_[0]), threshold=np.float32(thr))
    res["final_thr"] = thr
    np.savez(HERE / "pr_scores_v3.npz", y=y[view == "full"], src=src[view == "full"].astype(str), v3=sC[view == "full"])
    (HERE / "results_v3.json").write_text(json.dumps(res, indent=1)); print("DONE", flush=True)


if __name__ == "__main__":
    main()
