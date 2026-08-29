"""TTA 完整消融分析（在 Mac 跑，需要 laptop 带回的 3 个 npz + 本地 organ_train[_cc].npz）。
表 A：species zero-shot —— 视图消融 + emb-avg / score-avg / score-max
表 B：organ linear probe —— 视图消融 + emb-avg / prob-avg / logit-avg + train-view 消融
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

H = os.path.dirname(os.path.abspath(__file__))
VIEWS = ["full", "hflip", "cc90", "cc80", "cc70", "cc60", "padsq"]
VI = {v: i for i, v in enumerate(VIEWS)}
def npz(n): return np.load(os.path.join(H, n), allow_pickle=True)

def eavg(E, idx):
    m = E[:, idx, :].sum(1)
    return m / np.linalg.norm(m, axis=1, keepdims=True)

# ============================================================ 表 A
S = npz("tta_species_7v.npz"); P = npz("tta_species_proto.npz")
ES, Y, PROTO = S["emb"], S["y"], P["proto"]
print(f"# 表 A  species zero-shot  |  {len(Y)} 图 / {PROTO.shape[0]} 类\n")

def sp_metrics(sims):
    order = np.argsort(-sims, axis=1)
    t1 = (order[:, 0] == Y).mean()
    t5 = np.mean([Y[i] in order[i, :5] for i in range(len(Y))])
    return t1, t5
def sp_embavg(idx):  return sp_metrics(eavg(ES, idx) @ PROTO.T)
def sp_scoreavg(idx): return sp_metrics(np.mean([ES[:, j, :] @ PROTO.T for j in idx], 0))
def sp_scoremax(idx): return sp_metrics(np.max([ES[:, j, :] @ PROTO.T for j in idx], 0))

b1, b5 = sp_embavg([0])
print(f"{'config':26} {'top1':>7} {'top5':>7} {'Δtop1':>8}  fwd")
CFG_A = {
    "full (baseline)": [0], "hflip": [1], "cc90": [2], "cc80": [3], "cc70": [4], "cc60": [5], "padsq": [6],
    "full+hflip": [0, 1], "full+cc90": [0, 2], "full+cc80": [0, 3], "full+padsq": [0, 6],
    "full+hflip+cc90": [0, 1, 2], "full+hflip+cc80": [0, 1, 3],
    "full+hflip+cc90+padsq": [0, 1, 2, 6], "all 7": list(range(7)),
}
for name, idx in CFG_A.items():
    t1, t5 = sp_embavg(idx)
    print(f"{name:26} {t1:7.4f} {t5:7.4f} {t1-b1:+8.4f}  {len(idx)}x")

print("\n-- 合并方式对比 (full+hflip+cc90) --")
for tag, fn in [("emb-avg", sp_embavg), ("score-avg", sp_scoreavg), ("score-max", sp_scoremax)]:
    t1, t5 = fn([0, 1, 2])
    print(f"  {tag:10} top1={t1:.4f} top5={t5:.4f}")

print("\n-- leave-one-out from all 7 --")
a1, _ = sp_embavg(list(range(7)))
for d in range(7):
    t1, _ = sp_embavg([j for j in range(7) if j != d])
    print(f"  -{VIEWS[d]:6} top1={t1:.4f}  Δvs_all7={t1-a1:+.4f}")

# ============================================================ 表 B
print("\n\n# 表 B  organ linear probe  |  train=organ_train.npz\n")
tr = npz("organ_train.npz"); otr = tr["organs"].astype(str); Xtr = tr["emb"]
try:
    trc = npz("organ_train_cc.npz"); Xtrc = trc["emb"]
    Xtr_tta = (Xtr + Xtrc); Xtr_tta = Xtr_tta / np.linalg.norm(Xtr_tta, axis=1, keepdims=True)
except FileNotFoundError:
    Xtr_tta = None
TE = npz("tta_organ_eval_7v.npz"); ETE = TE["emb"]; ote = TE["organs"].astype(str)
CLS = ["leaf", "flower", "fruit", "bark"]
sig = lambda z: 1 / (1 + np.exp(-z))

# 一致性校验：7v 的 full 视图应≈原 organ_eval.npz
try:
    oe = npz("organ_eval.npz")
    if (oe["organs"].astype(str) == ote).all():
        d = np.abs(oe["emb"] - ETE[:, 0, :]).max()
        print(f"[一致性] 7v full 视图 vs organ_eval.npz  max|Δ|={d:.2e}  (应 ~0)\n")
except Exception:
    pass

def organ_row(trainX, view_idx, method):
    out = {}
    for c in CLS:
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(trainX, (otr == c).astype(int))
        W, b = clf.coef_[0], clf.intercept_[0]
        if method == "emb-avg":
            m = ETE[:, view_idx, :].sum(1); m = m / np.linalg.norm(m, axis=1, keepdims=True)
            s = sig(m @ W + b)
        elif method == "prob-avg":
            s = np.mean([sig(ETE[:, j, :] @ W + b) for j in view_idx], 0)
        else:  # logit-avg
            s = sig(np.mean([ETE[:, j, :] @ W + b for j in view_idx], 0))
        out[c] = average_precision_score((ote == c).astype(int), s)
        if c == "bark":
            pr, rc, _ = precision_recall_curve((ote == "bark").astype(int), s)
            ok = np.where(pr[:-1] >= 0.8)[0]
            out["bark_R@P.8"] = rc[:-1][ok].max() if len(ok) else 0.0
    return out

print(f"{'trainX':18} {'views':22} {'method':9} " + " ".join(f"{c:>6}" for c in CLS) + "  bark_R@P.8  fwd")
COMBOS = [("full", [0]), ("full+cc60", [0, 5]), ("full+hflip", [0, 1]),
          ("full+hflip+cc70", [0, 1, 4]), ("full+hflip+cc60+padsq", [0, 1, 5, 6]), ("all 7", list(range(7)))]
trains = [("train=full", Xtr)] + ([("train=avg(full,cc60)", Xtr_tta)] if Xtr_tta is not None else [])
for ttag, TX in trains:
    for vname, vidx in COMBOS:
        methods = ["emb-avg"] if len(vidx) == 1 else ["emb-avg", "prob-avg", "logit-avg"]
        for m in methods:
            r = organ_row(TX, vidx, m)
            print(f"{ttag:18} {vname:22} {m:9} " +
                  " ".join(f"{r[c]:6.3f}" for c in CLS) + f"   {r['bark_R@P.8']:.3f}     {len(vidx)}x")
