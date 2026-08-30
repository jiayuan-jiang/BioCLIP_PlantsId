"""比 full / cc60 / avg(full,cc60) 三种特征下 4 头 organ tagger 的 AUC-PR + bark P/R。
train = organ_train[_cc].npz, test = organ_eval[_cc].npz。按 index 配对（确定性采样保证同序）。"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve

H = os.path.dirname(os.path.abspath(__file__))
def L(n): d = np.load(os.path.join(H, n), allow_pickle=True); return d["emb"], d["organs"].astype(str)
Xtr_f, otr = L("organ_train.npz"); Xte_f, ote = L("organ_eval.npz")
Xtr_c, otr_c = L("organ_train_cc.npz"); Xte_c, ote_c = L("organ_eval_cc.npz")

assert (otr == otr_c).all() and (ote == ote_c).all(), "顺序没对齐，配对无效"
def avg(a, b):
    m = a + b; return m / np.linalg.norm(m, axis=1, keepdims=True)
FEATS = {"full": (Xtr_f, Xte_f), "cc60": (Xtr_c, Xte_c), "avg": (avg(Xtr_f, Xtr_c), avg(Xte_f, Xte_c))}
CLS = ["leaf", "flower", "fruit", "bark"]

print(f"{'feat':6} | " + " | ".join(f"{c:>6} AUCPR" for c in CLS) + " || bark P>=.7:R  P>=.8:R  maxF1")
for name, (Xtr, Xte) in FEATS.items():
    row = [name]
    aucs = []
    for c in CLS:
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(Xtr, (otr == c).astype(int))
        s = clf.predict_proba(Xte)[:, 1]
        aucs.append(average_precision_score((ote == c).astype(int), s))
    # bark holdout 阈值 + PR
    Xf, Xh, of, oh = train_test_split(Xtr, otr, test_size=0.25, stratify=otr, random_state=0)
    bh = LogisticRegression(max_iter=5000, class_weight="balanced").fit(Xf, (of == "bark").astype(int))
    ph = bh.predict_proba(Xh)[:, 1]
    p, r, t = precision_recall_curve((oh == "bark").astype(int), ph)
    f1 = 2 * p * r / (p + r + 1e-9)
    bark_full = LogisticRegression(max_iter=5000, class_weight="balanced").fit(Xtr, (otr == "bark").astype(int))
    sc = bark_full.predict_proba(Xte)[:, 1]
    bte = (ote == "bark").astype(int)
    pt, rt, tt = precision_recall_curve(bte, sc)
    def rat(pr):
        ok = np.where(pt[:-1] >= pr)[0]
        return rt[:-1][ok].max() if len(ok) else 0.0
    f1t = 2 * pt * rt / (pt + rt + 1e-9)
    print(f"{name:6} | " + " | ".join(f"{a:11.3f}" for a in aucs) +
          f" || {rat(0.7):.2f}      {rat(0.8):.2f}      {f1t[:-1].max():.2f}")
