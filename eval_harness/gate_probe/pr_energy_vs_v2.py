import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
z = np.load("pr_scores.npz"); y, src, en = z["y"], z["src"], z["energy"]
plt.rcParams.update({"font.family": ["PingFang HK", "Arial Unicode MS"], "axes.spines.top": False, "axes.spines.right": False})
def weights(prior):
    w = np.zeros(len(y))
    for s in ("plant", "plantnet", "stem"):
        m = src == s; w[m] = prior / 3 / m.sum()
    m = y == 0; w[m] = (1 - prior) / m.sum(); return w
THR = [(68.537, "68.537（线上）", "#2a78d6", "o"), (68.0, "68.0", "#eb6834", "s"), (67.0, "#7f7f7f", "#7f7f7f", "^")][:2]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
for ax, prior in zip(axes, (.5, .9)):
    w = weights(prior)
    p, r, _ = precision_recall_curve(y, en, sample_weight=w)
    ax.plot(r, p, color="#2a78d6", lw=2, alpha=.45, label="energy 全曲线")
    p2, r2, _ = precision_recall_curve(y, z["v2"], sample_weight=w)
    ax.plot(r2, p2, color="#1baf7a", lw=2.2, label="v2 二元头 全曲线")
    pred = z["v2"] >= 0.67; tp = w[pred & (y == 1)].sum(); fp = w[pred & (y == 0)].sum(); rec = tp / w[y == 1].sum(); pr = tp / (tp + fp)
    ax.plot([rec], [pr], "D", ms=11, color="#1baf7a", mec="white", mew=2, zorder=6, label=f"v2 阈值 0.67（recall 99.5%）: R={rec:.3f}  P={pr:.3f}")
    for t, lab, c, mk in THR:
        pred = en >= t; tp = w[pred & (y == 1)].sum(); fp = w[pred & (y == 0)].sum(); rec = tp / w[y == 1].sum(); pr = tp / (tp + fp)
        ax.plot([rec], [pr], mk, ms=11, color=c, mec="white", mew=2, label=f"阈值 {lab}: R={rec:.3f}  P={pr:.3f}", zorder=5)
        ax.annotate(f"{t:g}", (rec, pr), textcoords="offset points", xytext=(-6, -16) if t > 68.2 else (-6, -18), ha="right", fontsize=9, color=c)
    ax.set_title(f"energy vs v2 的 Precision–Recall（植物为正类，植物占比 {int(prior*100)}%）", loc="left", fontsize=11)
    ax.set_xlabel("Recall（植物被放行的比例）"); ax.set_ylabel("Precision（放行的里真是植物的比例）")
    ax.set_xlim(0.80, 1.002); ax.grid(alpha=.2); ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_ylim(0.6 if prior == .5 else 0.9, 1.0)
fig.savefig("pr_energy_vs_v2.png", dpi=150)
