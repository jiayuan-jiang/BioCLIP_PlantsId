"""v2：按数据现实的架构。
 E5  3-way softmax leaf/flower/fruit（无 class_weight）
 E6  bark one-vs-rest —— 大 bark 集(1716) vs 小 bark 集(225) 头对头，看补数据是否有用
 E7  bark OvR 负样本只用混淆类(leaf+habit) vs 全部
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, balanced_accuracy_score, confusion_matrix,
                             precision_recall_curve, average_precision_score)

HERE = os.path.dirname(os.path.abspath(__file__))
tr = np.load(os.path.join(HERE, "organ_train.npz"), allow_pickle=True)
te = np.load(os.path.join(HERE, "organ_eval.npz"), allow_pickle=True)
Xtr, otr = tr["emb"], tr["organs"].astype(str)
Xte, ote = te["emb"], te["organs"].astype(str)
rng = np.random.default_rng(0)


def pr_table(tag, bte, sc):
    ap = average_precision_score(bte, sc)
    prec, rec, thr = precision_recall_curve(bte, sc)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    ib = f1.argmax()
    print(f"  {tag:34}  AUC-PR={ap:.3f}  maxF1: P={prec[ib]:.2f} R={rec[ib]:.2f} F1={f1[ib]:.2f}", end="")
    for pt in (0.7, 0.8, 0.9):
        ok = np.where(prec[:-1] >= pt)[0]
        r = rec[:-1][ok].max() if len(ok) else 0.0
        print(f"  |P>={pt}:R={r:.2f}", end="")
    print()


# ---------- E5: 3-way leaf/flower/fruit ----------
O3 = ["leaf", "flower", "fruit"]
mtr = np.isin(otr, O3); mte = np.isin(ote, O3)
for cw in (None, "balanced"):
    clf = LogisticRegression(max_iter=4000, class_weight=cw)
    clf.fit(Xtr[mtr], np.array([O3.index(o) for o in otr[mtr]]))
    yte = np.array([O3.index(o) for o in ote[mte]]); pred = clf.predict(Xte[mte])
    print(f"\n===== E5 3-way (class_weight={cw})  bacc={balanced_accuracy_score(yte,pred):.3f} =====")
    print(classification_report(yte, pred, target_names=O3, digits=3, zero_division=0))
    cm = confusion_matrix(yte, pred)
    for i, n in enumerate(O3):
        print(f"  {n:6}", " ".join(f"{v:5d}" for v in cm[i]))

# ---------- E6: bark OvR, 大 vs 小 bark 集 ----------
print("\n===== E6 bark one-vs-rest：补数据有没有用 =====")
bte = (ote == "bark").astype(int)
bark_idx = np.where(otr == "bark")[0]
neg_idx = np.where(otr != "bark")[0]

# 大集：全部 1716 bark + 全部负样本
btr_full = (otr == "bark").astype(int)
c = LogisticRegression(max_iter=4000, class_weight="balanced").fit(Xtr, btr_full)
pr_table(f"大 bark 集 (n_bark={len(bark_idx)})", bte, c.predict_proba(Xte)[:, 1])

# 小集：随机 225 bark + 同样负样本
for seed in (0, 1, 2):
    rng2 = np.random.default_rng(seed)
    sub = rng2.choice(bark_idx, 225, replace=False)
    keep = np.concatenate([sub, neg_idx])
    y = np.zeros(len(keep), int); y[:225] = 1
    c = LogisticRegression(max_iter=4000, class_weight="balanced").fit(Xtr[keep], y)
    pr_table(f"小 bark 集 225 (seed={seed})", bte, c.predict_proba(Xte)[:, 1])

# 中集：800
rng2 = np.random.default_rng(0)
sub = rng2.choice(bark_idx, 800, replace=False)
keep = np.concatenate([sub, neg_idx]); y = np.zeros(len(keep), int); y[:800] = 1
c = LogisticRegression(max_iter=4000, class_weight="balanced").fit(Xtr[keep], y)
pr_table("中 bark 集 800", bte, c.predict_proba(Xte)[:, 1])

# ---------- E7: 负样本只用混淆类 ----------
print("\n===== E7 bark OvR：负样本选择 =====")
for negset in (["leaf", "flower", "fruit", "habit"], ["leaf", "habit"], ["habit"], ["leaf", "flower", "fruit"]):
    m = np.isin(otr, ["bark"] + negset)
    y = (otr[m] == "bark").astype(int)
    c = LogisticRegression(max_iter=4000, class_weight="balanced").fit(Xtr[m], y)
    pr_table(f"neg={negset}", bte, c.predict_proba(Xte)[:, 1])
