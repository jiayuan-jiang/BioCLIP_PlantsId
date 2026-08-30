"""organ tagger 重训对比。train = organ_train.npz (Pl@ntNet train+val)，test = organ_eval.npz (Pl@ntNet test 4851)。
E1 5-way LR · E2 4-way LR (drop habit) · E3 bark one-vs-rest (PR/阈值) · E4 5-way MLP。"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, balanced_accuracy_score,
                             confusion_matrix, precision_recall_curve, average_precision_score)

HERE = os.path.dirname(os.path.abspath(__file__))
tr = np.load(os.path.join(HERE, "organ_train.npz"), allow_pickle=True)
te = np.load(os.path.join(HERE, "organ_eval.npz"), allow_pickle=True)
Xtr, otr = tr["emb"], tr["organs"].astype(str)
Xte, ote = te["emb"], te["organs"].astype(str)
print("train:", {o: int((otr == o).sum()) for o in set(otr)})
print("test :", {o: int((ote == o).sum()) for o in set(ote)})


def ev(tag, ytr, yte, clf, names):
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(f"\n===== {tag}  acc={ (pred==yte).mean():.3f}  bacc={balanced_accuracy_score(yte,pred):.3f} =====")
    print(classification_report(yte, pred, target_names=names, digits=3, zero_division=0))
    cm = confusion_matrix(yte, pred, labels=range(len(names)))
    print("cm rows=true:", "  ".join(f"{n[:5]:>6}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:6}", " ".join(f"{v:6d}" for v in cm[i]))


# ---------- E1: 5-way ----------
O5 = ["leaf", "flower", "fruit", "bark", "habit"]
y5tr = np.array([O5.index(o) for o in otr]);  y5te = np.array([O5.index(o) for o in ote])
ev("E1 5-way LogisticRegression(balanced)", y5tr, y5te,
   LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced"), O5)

# ---------- E2: 4-way, drop habit ----------
O4 = ["leaf", "flower", "fruit", "bark"]
mtr = np.isin(otr, O4);  mte = np.isin(ote, O4)
g_Xtr, g_Xte = Xtr, Xte  # closure hack
def ev4(tag, clf):
    clf.fit(Xtr[mtr], np.array([O4.index(o) for o in otr[mtr]]))
    yte = np.array([O4.index(o) for o in ote[mte]])
    pred = clf.predict(Xte[mte])
    print(f"\n===== {tag}  bacc={balanced_accuracy_score(yte,pred):.3f} =====")
    print(classification_report(yte, pred, target_names=O4, digits=3, zero_division=0))
ev4("E2 4-way LogisticRegression(balanced)", LogisticRegression(max_iter=4000, class_weight="balanced"))

# ---------- E3: bark one-vs-rest ----------
btr = (otr == "bark").astype(int);  bte = (ote == "bark").astype(int)
clf = LogisticRegression(max_iter=4000, class_weight="balanced")
clf.fit(Xtr, btr)
sc = clf.predict_proba(Xte)[:, 1]
ap = average_precision_score(bte, sc)
prec, rec, thr = precision_recall_curve(bte, sc)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
i_best = f1.argmax()
print(f"\n===== E3 bark one-vs-rest =====  AUC-PR={ap:.3f}  (bark {bte.sum()}/{len(bte)})")
print(f"  max-F1: thr={thr[max(i_best-1,0)]:.3f}  P={prec[i_best]:.3f} R={rec[i_best]:.3f} F1={f1[i_best]:.3f}")
for p_target in (0.6, 0.7, 0.8):
    ok = np.where(prec[:-1] >= p_target)[0]
    if len(ok):
        j = ok[np.argmax(rec[:-1][ok])]
        print(f"  P>={p_target}: thr={thr[j]:.3f}  P={prec[j]:.3f} R={rec[j]:.3f}")
    else:
        print(f"  P>={p_target}: 达不到")

# ---------- E4: 5-way MLP ----------
ev("E4 5-way MLP(256,)", y5tr, y5te,
   MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=200, early_stopping=True), O5)

print("\n[参考] 旧 5-fold CV(仅 test 4851): bacc 0.632, bark F1 0.47")
