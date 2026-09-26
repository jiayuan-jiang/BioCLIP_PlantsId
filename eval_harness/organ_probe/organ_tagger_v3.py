"""多标签 organ tagger v3 —— 5 个独立 one-vs-rest sigmoid：leaf/flower/fruit/bark/stem。

相对 v1：bark 换成 BarkNet+BARK-KR（干净树皮），新增 stem（PlantCLEF Stem + NTU-Tree）。
数据由 build_organ_train_v3.py 产出。

train = organ_train_v3.npz   test = organ_eval_v3.npz   （keys: emb / organs / source / species）
FP 探针：复用旧 organ_eval.npz 里的 habit 图（整株照，非任一器官）当硬负样本，测误报。

产出：organ_tagger_v3.npz (classes/coef/intercept/thr_f1/thr_prec) + organ_tagger_v3.json + 控制台报告。

用法：python organ_tagger_v3.py
"""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve

HERE = os.path.dirname(os.path.abspath(__file__))
tr = np.load(os.path.join(HERE, "organ_train_v3.npz"), allow_pickle=True)
te = np.load(os.path.join(HERE, "organ_eval_v3.npz"), allow_pickle=True)
Xtr, otr = tr["emb"].astype(np.float32), tr["organs"].astype(str)
Xte, ote = te["emb"].astype(np.float32), te["organs"].astype(str)
src_te = te["source"].astype(str) if "source" in te.files else np.array(["?"] * len(ote))
CLS = ["leaf", "flower", "fruit", "bark", "stem"]   # v3.2: stem 用人工标的 1400 张（tmp/stem_review）
PREC_TARGET = 0.80

# ── 硬负样本：旧 eval 的 habit 图（label 干净，只是整株取景）──
HAB = None
_old = os.path.join(HERE, "organ_eval.npz")
if os.path.exists(_old):
    z = np.load(_old, allow_pickle=True)
    m = z["organs"].astype(str) == "habit"
    if m.any():
        HAB = z["emb"][m].astype(np.float32)

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
    return tp / max(pred.sum(), 1), tp / max((ote == CLS[i]).sum(), 1), int(pred.sum())


print(f"train 组成: {dict(zip(*np.unique(otr, return_counts=True)))}")
print(f"test  组成: {dict(zip(*np.unique(ote, return_counts=True)))}\n")
print(f"{'head':7} {'AUC-PR':>7} | {'thr(F1)':>8} {'P':>5} {'R':>5} | {'thr(P>=.8)':>10} {'P':>5} {'R':>5}")
for i, c in enumerate(CLS):
    pf, rf, _ = head_pr(i, heads[c]["t_f1"])
    pp, rp, _ = head_pr(i, heads[c]["t_prec"])
    print(f"{c:7} {aucpr[c]:7.3f} | {heads[c]['t_f1']:8.3f} {pf:5.2f} {rf:5.2f} | "
          f"{heads[c]['t_prec']:10.3f} {pp:5.2f} {rp:5.2f}")

# ── bark/stem 分来源看泛化 ──
print("\n分来源 AUC-PR（该来源正样本 vs 全部其它）:")
for c in ("bark", "stem"):
    ci = CLS.index(c)
    for s in sorted(set(src_te[ote == c])):
        pos = (ote == c) & (src_te == s)
        neg = ote != c
        mm = pos | neg
        if pos.sum() >= 5:
            ap = average_precision_score(pos[mm].astype(int), P[mm, ci])
            print(f"  {c:5} {s:16} n_pos={int(pos.sum()):4d}  AUC-PR={ap:.3f}")

for tag, key in [("max-F1 阈值", "t_f1"), ("precision>=0.80 阈值", "t_prec")]:
    thr = np.array([heads[c][key] for c in CLS])
    tags = P >= thr
    ntag = tags.sum(1)
    m = np.isin(ote, CLS)
    tc = np.array([CLS.index(o) for o in ote[m]])
    covered = tags[m][np.arange(m.sum()), tc]
    extra = (ntag[m] - covered.astype(int)).clip(min=0)
    print(f"\n===== {tag} =====")
    print(f"  标签数分布(全 test): " + "  ".join(f"{k}:{int((ntag==k).sum())}" for k in range(6)))
    print(f"  真标签命中率 P(真organ ∈ tags) = {covered.mean():.3f}   平均额外标签 = {extra.mean():.2f}")
    if HAB is not None:
        ph = np.column_stack([heads[c]["clf"].predict_proba(HAB)[:, 1] for c in CLS]) >= thr
        top = CLS[np.bincount(ph.argmax(1)[ph.any(1)], minlength=len(CLS)).argmax()] if ph.any() else "-"
        print(f"  habit 硬负样本({len(HAB)}张): 被打≥1标签 = {(ph.sum(1)>=1).mean():.2f}  最常见 = {top}")
    from collections import Counter
    combo = Counter(tuple(np.array(CLS)[row].tolist()) for row in tags[m])
    print("  最常见 tag 组合:", [(("+".join(k) or "∅"), v) for k, v in combo.most_common(6)])

np.savez(os.path.join(HERE, "organ_tagger_v3.npz"),
         classes=np.array(CLS),
         coef=np.stack([heads[c]["clf"].coef_[0] for c in CLS]),
         intercept=np.array([heads[c]["clf"].intercept_[0] for c in CLS]),
         thr_f1=np.array([heads[c]["t_f1"] for c in CLS]),
         thr_prec=np.array([heads[c]["t_prec"] for c in CLS]))
json.dump({c: {"aucpr": round(aucpr[c], 3), "thr_f1": round(heads[c]["t_f1"], 3),
               "thr_prec80": round(heads[c]["t_prec"], 3)} for c in CLS},
          open(os.path.join(HERE, "organ_tagger_v3.json"), "w"), indent=2)
print("\nsaved organ_tagger_v3.npz / .json")
