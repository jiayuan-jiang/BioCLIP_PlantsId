"""
植物/非植物二元 gate probe：冻结 BioCLIP 2 embedding 上训练 logistic 头(768→1)。

评测三件事：
  1. 随机 80/20 留出（分布内）  2. leave-group-out（整类负样本组完全没见过，测泛化，截图最关键）
  3. 对比线上 energy-only 基线（同一批测试样本，同样按 98% TPR 定阈值）
输出 results.json / gate_head.npz；阈值用 5 折 out-of-fold 正样本分数的 2% 分位数定（98% TPR，召回优先）。
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).parent
FEATS = HERE / "feats"
ROOT = HERE.parents[1]
TPR = 0.98

GROUP = {  # 源 → 组
    "cifar10": "object", "imagenet": "object", "coco": "object", "caltech101": "object", "cars": "object",
    "sun397": "scene", "resisc45": "scene", "dtd": "texture",
    "screenspot": "screenshot", "wave_ui": "screenshot", "rico": "screenshot", "websight": "screenshot",
    "funsd": "document",
}


def load():
    X, y, src, grp = [], [], [], []
    p = np.load(FEATS / "plant.npy")
    X.append(p); y += [1] * len(p); src += ["plant"] * len(p)
    pg = np.arange(len(p)) // 5           # 近似按物种分组（每种 5 张）
    grp_id = list(pg)
    for s, g in GROUP.items():
        f = FEATS / f"{s}.npy"
        if not f.exists():
            print("missing", s); continue
        a = np.load(f)
        X.append(a); y += [0] * len(a); src += [s] * len(a)
        base = 10**6 + len(grp_id)
        grp_id += list(base + np.arange(len(a)) // 5)
    return np.concatenate(X), np.array(y), np.array(src), np.array(grp_id)


def thr_at_tpr(pos_scores, tpr=TPR):
    return float(np.quantile(pos_scores, 1 - tpr))


def report(name, s, y, src, thr):
    pos = s[y == 1]
    out = {"tpr": float((pos >= thr).mean()), "auroc": float(roc_auc_score(y, s)),
           "ap": float(average_precision_score(y, s)), "fpr_by_src": {}, "fpr_by_group": {}}
    neg = y == 0
    out["fpr"] = float((s[neg] >= thr).mean())
    for k in sorted(set(src[neg])):
        m = src == k
        out["fpr_by_src"][k] = float((s[m] >= thr).mean())
    gs = np.array([GROUP.get(k, "plant") for k in src])
    for g in sorted(set(gs[neg])):
        m = gs == g
        out["fpr_by_group"][g] = float((s[m] >= thr).mean())
    return out


def fit(X, y, C=1.0):
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(X * 10.0, y)   # 归一化向量尺度太小，放大 10 倍方便正则
    return clf


def score(clf, X):
    return clf.decision_function(X * 10.0)


def energy_baseline(X):
    """线上同款 energy：logit_scale · cos 对 35 万物种 logsumexp。"""
    idx = np.load(ROOT / "data/bioclip_full_index.npz", allow_pickle=True)["embeddings"].astype(np.float32)
    ls = 100.0  # open_clip bioclip logit_scale.exp()，线上 gate 用的值；见 inference_core
    out = []
    for i in range(0, len(X), 512):
        z = ls * (X[i:i + 512] @ idx.T)
        m = z.max(1, keepdims=True)
        out.append((m[:, 0] + np.log(np.exp(z - m).sum(1))))
    return np.concatenate(out)


def main():
    X, y, src, gid = load()
    print("data", X.shape, "pos", y.sum(), "neg", (y == 0).sum(), flush=True)
    res = {"n": {"pos": int(y.sum()), "neg": int((y == 0).sum()), "by_src": {k: int((src == k).sum()) for k in sorted(set(src))}}}

    # ── 能量基线（用于对比）
    en = energy_baseline(X)
    np.save(HERE / "energy_scores.npy", en)

    # ── 1. 分组 5 折 out-of-fold（分布内）：每个样本的分数都来自没训练过它的模型
    oof = np.zeros(len(y))
    for C in [0.1, 1.0, 10.0]:
        oof_c = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, gid):
            oof_c[te] = score(fit(X[tr], y[tr], C), X[te])
        thr = thr_at_tpr(oof_c[y == 1])
        r = report(f"C={C}", oof_c, y, src, thr)
        print(f"C={C}: AUROC {r['auroc']:.4f} FPR@98TPR {r['fpr']:.4f}", flush=True)
        res.setdefault("cv", {})[str(C)] = r
        if C == 1.0:
            oof = oof_c
    bestC = min(res["cv"], key=lambda c: res["cv"][c]["fpr"])
    print("best C", bestC, flush=True)
    bestC = float(bestC)
    thr_en = thr_at_tpr(en[y == 1])
    res["energy_baseline"] = report("energy", en, y, src, thr_en)
    res["energy_baseline"]["thr"] = thr_en
    print("energy:", res["energy_baseline"]["auroc"], res["energy_baseline"]["fpr"], flush=True)

    # ── 2. leave-group-out：整组负样本不参与训练
    res["lgo"] = {}
    gs = np.array([GROUP.get(k, "plant") for k in src])
    rng = np.random.RandomState(0)
    for g in sorted(set(GROUP.values())):
        te_neg = gs == g
        tr = ~te_neg
        # 正样本留 20% 当测试 / 定阈值
        pos_idx = np.where(y == 1)[0]
        pos_te = np.zeros(len(y), bool)
        te_species = set(rng.choice(np.unique(gid[pos_idx]), int(0.2 * len(np.unique(gid[pos_idx]))), replace=False))
        pos_te[pos_idx] = [gg in te_species for gg in gid[pos_idx]]
        tr = tr & ~pos_te
        te = te_neg | pos_te
        clf = fit(X[tr], y[tr], bestC)
        s = score(clf, X[te])
        thr = thr_at_tpr(s[y[te] == 1])
        r = report(g, s, y[te], src[te], thr)
        thr_e = thr_at_tpr(en[te][y[te] == 1])
        r["energy_fpr"] = float((en[te][y[te] == 0] >= thr_e).mean())
        res["lgo"][g] = r
        print(f"LGO {g}: probe FPR {r['fpr']:.4f} | energy FPR {r['energy_fpr']:.4f} | AUROC {r['auroc']:.4f}", flush=True)

    # ── 3. 终版：全数据拟合，阈值取 OOF 正样本 98% 分位
    clf = fit(X, y, bestC)
    w = clf.coef_[0] * 10.0
    b = float(clf.intercept_[0])
    oof_best = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, gid):
        oof_best[te] = score(fit(X[tr], y[tr], bestC), X[te])
    thr = thr_at_tpr(oof_best[y == 1])
    res["final"] = report("final", oof_best, y, src, thr)
    res["final"].update(C=bestC, thr=thr)
    for tpr in (0.95, 0.98, 0.99, 0.995):
        t = thr_at_tpr(oof_best[y == 1], tpr)
        res.setdefault("tpr_sweep", {})[str(tpr)] = {"thr": t, "fpr": float((oof_best[y == 0] >= t).mean()),
                                                      "fpr_screenshot": float((oof_best[(gs == "screenshot")] >= t).mean())}
    np.savez(HERE / "gate_head.npz", w=w.astype(np.float32), b=np.float32(b), threshold=np.float32(thr))
    (HERE / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("FINAL", json.dumps(res["final"], indent=1), flush=True)


if __name__ == "__main__":
    main()
