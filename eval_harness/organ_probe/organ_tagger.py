"""多标签 organ tagger：4 个独立 one-vs-rest sigmoid (leaf/flower/fruit/bark)。
阈值在 train 的 25% holdout 上定（max-F1 点 + precision>=0.80 点两套）。
train = organ_train.npz, test = organ_eval.npz（test 里的 habit 图当干扰项）。
产出：organ_tagger.npz (coef/intercept/thresholds) + 控制台报告。
"""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve

HERE = os.path.dirname(os.path.abspath(__file__))
tr = np.load(os.path.join(HERE, "organ_train.npz"), allow_pickle=True)
te = np.load(os.path.join(HERE, "organ_eval.npz"), allow_pickle=True)
Xtr, otr = tr["emb"], tr["organs"].astype(str)
Xte, ote = te["emb"], te["organs"].astype(str)
CLS = ["leaf", "flower", "fruit", "bark"]
PREC_TARGET = 0.80

Xf, Xh, of, oh = train_test_split(Xtr, otr, test_size=0.25, stratify=otr, random_state=0)

heads = {}
for c in CLS:
    full = LogisticRegression(max_iter=5000, class_weight="balanced").fit(Xtr, (otr == c).astype(int))
    hold = LogisticRegression(max_iter=5000, class_weight="balanced").fit(Xf, (of == c).astype(int))
    ph = hold.predict_proba(Xh)[:, 1]
    p, r, t = precision_recall_curve((oh == c).astype(int), ph)
    f1 = 2 * p * r / (p + r + 1e-9)
    t_f1 = float(t[max(f1[:-1].argmax(), 0)])
    ok = np.where(p[:-1] >= PREC_TARGET)[0]
    t_prec = float(t[ok[np.argmax(r[:-1][ok])]]) if len(ok) else t_f1
    heads[c] = dict(clf=full, t_f1=t_f1, t_prec=t_prec)

P = np.column_stack([heads[c]["clf"].predict_proba(Xte)[:, 1] for c in CLS])
aucpr = {c: average_precision_score((ote == c).astype(int), P[:, i]) for i, c in enumerate(CLS)}

def head_pr(i, thr):
    pred = P[:, i] >= thr
    tp = (pred & (ote == CLS[i])).sum()
    prec = tp / max(pred.sum(), 1)
    rec = tp / max((ote == CLS[i]).sum(), 1)
    return prec, rec, int(pred.sum())

print(f"test 组成: {dict(zip(*np.unique(ote, return_counts=True)))}\n")
print(f"{'head':7} {'AUC-PR':>7} | {'thr(F1)':>8} {'P':>5} {'R':>5} | {'thr(P>=.8)':>10} {'P':>5} {'R':>5}")
for i, c in enumerate(CLS):
    pf, rf, _ = head_pr(i, heads[c]["t_f1"])
    pp, rp, _ = head_pr(i, heads[c]["t_prec"])
    print(f"{c:7} {aucpr[c]:7.3f} | {heads[c]['t_f1']:8.3f} {pf:5.2f} {rf:5.2f} | "
          f"{heads[c]['t_prec']:10.3f} {pp:5.2f} {rp:5.2f}")

for tag, key in [("max-F1 阈值", "t_f1"), ("precision>=0.80 阈值", "t_prec")]:
    thr = np.array([heads[c][key] for c in CLS])
    tags = P >= thr
    ntag = tags.sum(1)
    m = np.isin(ote, CLS)
    tc = np.array([CLS.index(o) for o in ote[m]])
    covered = tags[m][np.arange(m.sum()), tc]
    extra = (ntag[m] - covered.astype(int)).clip(min=0)
    hab = ote == "habit"
    print(f"\n===== {tag} =====")
    print(f"  标签数分布(全 test): " + "  ".join(f"{k}:{int((ntag==k).sum())}" for k in range(5)))
    print(f"  真标签命中率 P(真organ ∈ tags) = {covered.mean():.3f}   平均额外标签 = {extra.mean():.2f}")
    print(f"  habit 干扰图({hab.sum()}张): 被打≥1个标签的比例 = {(ntag[hab]>=1).mean():.2f}  "
          f"其中最常见 = {CLS[np.bincount(tags[hab].argmax(1)[tags[hab].any(1)], minlength=4).argmax()] if tags[hab].any() else '-'}")
    # 组合分布
    from collections import Counter
    combo = Counter(tuple(np.array(CLS)[row].tolist()) for row in tags[m])
    print("  最常见 tag 组合:", [(("+".join(k) or "∅"), v) for k, v in combo.most_common(6)])

np.savez(os.path.join(HERE, "organ_tagger.npz"),
         classes=np.array(CLS),
         coef=np.stack([heads[c]["clf"].coef_[0] for c in CLS]),
         intercept=np.array([heads[c]["clf"].intercept_[0] for c in CLS]),
         thr_f1=np.array([heads[c]["t_f1"] for c in CLS]),
         thr_prec=np.array([heads[c]["t_prec"] for c in CLS]))
json.dump({c: {"aucpr": round(aucpr[c], 3), "thr_f1": round(heads[c]["t_f1"], 3),
               "thr_prec80": round(heads[c]["t_prec"], 3)} for c in CLS},
          open(os.path.join(HERE, "organ_tagger.json"), "w"), indent=2)
print("\nsaved organ_tagger.npz / .json")
