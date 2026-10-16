import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

z = np.load("pr_scores.npz"); y, src = z["y"], z["src"]
M = {"线上 energy": (z["energy"], 68.537, "#2a78d6"), "v1 二元头": (z["v1"], 3.584, "#eb6834"), "v2 二元头": (z["v2"], 2.548, "#1baf7a")}
plt.rcParams.update({"font.family": ["PingFang HK", "Arial Unicode MS", "sans-serif"], "axes.spines.top": False, "axes.spines.right": False})

def weights(mask, prior):
    """正样本各来源等权，正负总权重按 prior 配比。"""
    w = np.zeros(len(y))
    pos_srcs = [s for s in ("plant", "plantnet", "stem") if (mask & (src == s)).any()]
    for s in pos_srcs:
        m = mask & (src == s) & (y == 1); w[m] = prior / len(pos_srcs) / m.sum()
    m = mask & (y == 0); w[m] = (1 - prior) / m.sum()
    return w

panels = [("全部正样本（iNat+Pl@ntNet+stem 等权），植物占比 50%", np.ones(len(y), bool), .5),
          ("同上，植物占比 90%（真实用户多数传植物）", np.ones(len(y), bool), .9),
          ("仅域外正样本（Pl@ntNet+stem），占比 50%\n对 v1 是没训练过的数据", np.isin(src, ["plantnet", "stem"]) | (y == 0), .5)]
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), constrained_layout=True)
for ax, (title, mask, prior) in zip(axes, panels):
    w = weights(mask, prior)
    for name, (s, thr, c) in M.items():
        p, r, _ = precision_recall_curve(y[mask], s[mask], sample_weight=w[mask])
        ax.plot(r, p, color=c, lw=2, label=name)
        pred = (s >= thr) & mask
        tp = w[pred & (y == 1)].sum(); fp = w[pred & (y == 0)].sum(); rec = tp / w[mask & (y == 1)].sum()
        ax.plot([rec], [tp / (tp + fp)], "o", ms=9, color=c, mec="white", mew=2)
    ax.set_title(title, fontsize=11, loc="left"); ax.set_xlabel("Recall（植物）"); ax.set_ylabel("Precision（判为植物的里真是植物的比例）")
    lo = 0.5 if prior == .5 else 0.85
    ax.set_xlim(0.6, 1.005); ax.set_ylim(lo if prior != .5 else 0.6, 1.005); ax.grid(alpha=.2)
axes[0].legend(loc="lower left", frameon=False, title="● = 线上实际阈值", title_fontsize=9)
fig.savefig("pr_curves.png", dpi=150)
# 圆点处的数值
for title, mask, prior in panels:
    w = weights(mask, prior); print(title.split("\n")[0])
    for name, (s, thr, c) in M.items():
        pred = (s >= thr) & mask; tp = w[pred & (y == 1)].sum(); fp = w[pred & (y == 0)].sum()
        print(f"  {name}: recall {tp/w[mask&(y==1)].sum():.3f} precision {tp/(tp+fp):.4f}")
