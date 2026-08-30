"""头对头：A) multi-label 5-sigmoid(+逐类阈值)  vs  B) 3-way softmax + bark/habit OvR。
train = organ_train.npz, test = organ_eval.npz。阈值在 train 的 holdout 上调。"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, balanced_accuracy_score, confusion_matrix,
                             average_precision_score, precision_recall_curve, f1_score)

HERE = os.path.dirname(os.path.abspath(__file__))
tr = np.load(os.path.join(HERE, "organ_train.npz"), allow_pickle=True)
te = np.load(os.path.join(HERE, "organ_eval.npz"), allow_pickle=True)
Xtr, otr = tr["emb"], tr["organs"].astype(str)
Xte, ote = te["emb"], te["organs"].astype(str)
ORG = ["leaf", "flower", "fruit", "bark", "habit"]
ytr = np.array([ORG.index(o) for o in otr]);  yte = np.array([ORG.index(o) for o in ote])
# 自然先验（Pl@ntNet 全体分布）用于加权准确率
PRIOR = np.array([0.362, 0.583, 0.037, 0.0074, 0.0147]); PRIOR /= PRIOR.sum()

Xf, Xh, yf, yh = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=0)


def summarize(tag, pred):
    bacc = balanced_accuracy_score(yte, pred)
    rec = confusion_matrix(yte, pred, labels=range(5)).astype(float)
    rec = rec.diagonal() / rec.sum(1)
    wacc = float((PRIOR * rec).sum())  # 自然先验加权准确率
    f1s = f1_score(yte, pred, labels=range(5), average=None, zero_division=0)
    print(f"\n=== {tag} ===  bacc={bacc:.3f}  自然加权acc={wacc:.3f}")
    print("  F1  " + "  ".join(f"{o}:{v:.2f}" for o, v in zip(ORG, f1s)))
    print("  recall " + "  ".join(f"{o}:{v:.2f}" for o, v in zip(ORG, rec)))
    cm = confusion_matrix(yte, pred, labels=range(5))
    print("  cm(rows=true) " + " ".join(f"{o[:4]:>5}" for o in ORG))
    for i, o in enumerate(ORG):
        print(f"    {o:6}", " ".join(f"{v:5d}" for v in cm[i]))


def best_thr(y_true_bin, score):
    p, r, t = precision_recall_curve(y_true_bin, score)
    f1 = 2 * p * r / (p + r + 1e-9)
    return t[max(f1[:-1].argmax(), 0)]


# ---------------- A: multi-label 5 sigmoid ----------------
print("#" * 60, "\nA) multi-label：5 个独立 one-vs-rest sigmoid")
Pte = np.zeros((len(Xte), 5)); Ph = np.zeros((len(Xh), 5))
aucpr = []
for c in range(5):
    clf = LogisticRegression(max_iter=4000, class_weight="balanced").fit(Xtr, (ytr == c).astype(int))
    Pte[:, c] = clf.predict_proba(Xte)[:, 1]
    clf_f = LogisticRegression(max_iter=4000, class_weight="balanced").fit(Xf, (yf == c).astype(int))
    Ph[:, c] = clf_f.predict_proba(Xh)[:, 1]
    aucpr.append(average_precision_score((yte == c).astype(int), Pte[:, c]))
print("  逐头 AUC-PR: " + "  ".join(f"{o}:{v:.3f}" for o, v in zip(ORG, aucpr)))

summarize("A1 纯 argmax(5 sigmoid 概率)", Pte.argmax(1))

# A2: 逐类阈值（holdout 上 max-F1），命中多个取概率最高，命中 0 个取 argmax
thr = np.array([best_thr((yh == c).astype(int), Ph[:, c]) for c in range(5)])
print("  逐类阈值(holdout max-F1):", {o: round(float(x), 3) for o, x in zip(ORG, thr)})
hit = Pte >= thr
pred_a2 = np.where(hit.any(1),
                   np.where(hit, Pte, -1).argmax(1),
                   Pte.argmax(1))
summarize("A2 逐类阈值 + 概率消歧", pred_a2)

# ---------------- B: 3-way softmax + bark/habit OvR ----------------
print("\n" + "#" * 60, "\nB) 3-way softmax(leaf/flower/fruit) + bark OvR + habit OvR + 路由")
m3 = np.isin(otr, ["leaf", "flower", "fruit"])
sm = LogisticRegression(max_iter=4000).fit(Xtr[m3], np.array([["leaf", "flower", "fruit"].index(o) for o in otr[m3]]))
P3 = sm.predict_proba(Xte)  # -> idx 0,1,2 = leaf,flower,fruit

bark = LogisticRegression(max_iter=4000, class_weight="balanced").fit(Xtr, (otr == "bark").astype(int))
habit = LogisticRegression(max_iter=4000, class_weight="balanced").fit(Xtr, (otr == "habit").astype(int))
pb, phb = bark.predict_proba(Xte)[:, 1], habit.predict_proba(Xte)[:, 1]

# holdout 上给 bark/habit 定阈值
bf = LogisticRegression(max_iter=4000, class_weight="balanced").fit(Xf, (yf == 3).astype(int))
hf = LogisticRegression(max_iter=4000, class_weight="balanced").fit(Xf, (yf == 4).astype(int))
tb = best_thr((yh == 3).astype(int), bf.predict_proba(Xh)[:, 1])
th = best_thr((yh == 4).astype(int), hf.predict_proba(Xh)[:, 1])
print(f"  bark thr={tb:.3f}  habit thr={th:.3f}")

# 路由：bark 优先(它 precision 高时可信) -> habit -> 3way argmax
pred_b = np.full(len(Xte), -1)
route_bark = pb >= tb
route_habit = (~route_bark) & (phb >= th)
route_3 = ~(route_bark | route_habit)
pred_b[route_bark] = 3
pred_b[route_habit] = 4
pred_b[route_3] = P3[route_3].argmax(1)
summarize("B 路由(bark>habit>3way)", pred_b)

# B 变体：bark 用高精度阈值 P>=0.8
p, r, t = precision_recall_curve((yh == 3).astype(int), bf.predict_proba(Xh)[:, 1])
ok = np.where(p[:-1] >= 0.8)[0]
tb80 = t[ok[np.argmax(r[:-1][ok])]] if len(ok) else tb
pred_b2 = np.full(len(Xte), -1)
rb = pb >= tb80
rh = (~rb) & (phb >= th)
pred_b2[rb] = 3; pred_b2[rh] = 4; pred_b2[~(rb | rh)] = P3[~(rb | rh)].argmax(1)
summarize(f"B2 bark 高精度阈值({tb80:.3f})", pred_b2)

print("\n[参考] E1 5-way softmax: bacc 0.664 | E5 3-way(仅3类): bacc 0.851")
