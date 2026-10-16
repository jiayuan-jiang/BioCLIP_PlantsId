import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import train as T1, train2 as T2
NEW = ["oxford_flowers", "plant_village", "sun397_plant"]
a = np.load("pr_scores.npz"); b = np.load("pr_scores_v3.npz")
n0 = len(a["y"]); assert (b["src"][:n0] == a["src"]).all()
# 新来源：energy 与 v2(最终头，未训练过这些来源=外部测试) 直接由 embedding 算
idx = np.load(T1.ROOT / "data/bioclip_full_index.npz", allow_pickle=True)["embeddings"].astype(np.float32)
h2 = np.load("gate_head_v2.npz"); en_new, v2_new = [], []
for k in NEW:
    X = np.load(T2.F2 / f"{k}.npz")["full"]
    e = []
    for i in range(0, len(X), 512):
        zz = 100.0 * (X[i:i+512] @ idx.T); m = zz.max(1, keepdims=True); e.append(m[:, 0] + np.log(np.exp(zz - m).sum(1)))
    en_new.append(np.concatenate(e)); v2_new.append(X @ h2["w"] + h2["b"])
y, src = b["y"], b["src"]
S = {"线上 energy": (np.concatenate([a["energy"]] + en_new), 68.537, "#2a78d6", "o"),
     "v2 二元头": (np.concatenate([a["v2"]] + v2_new), 0.67, "#eb6834", "s"),
     "v3 二元头": (b["v3"], float(np.load("gate_head_v3.npz")["threshold"]), "#1baf7a", "D")}
assert all(len(v[0]) == len(y) for v in S.values())
np.savez("pr_three_scores.npz", y=y, src=src, energy=S["线上 energy"][0], v2=S["v2 二元头"][0], v3=S["v3 二元头"][0])
plt.rcParams.update({"font.family": ["PingFang HK", "Arial Unicode MS"], "axes.spines.top": False, "axes.spines.right": False})
def weights(mask, srcs, prior=.5):
    w = np.zeros(len(y))
    for k in srcs: m = mask & (src == k); w[m] = prior / len(srcs) / m.sum()
    m = mask & (y == 0); w[m] = (1 - prior) / m.sum(); return w
panels = [("A  同源三类正样本（iNat / Pl@ntNet / stem）", ["plant", "plantnet", "stem"]),
          ("B  再加网图类正样本（Oxford 花 / PlantVillage 叶 / SUN397 植物场景）", ["plant", "plantnet", "stem"] + NEW)]
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)
for ax, (title, srcs) in zip(axes, panels):
    mask = np.isin(src, srcs) | (y == 0); w = weights(mask, srcs)
    for name, (s, thr, c, mk) in S.items():
        p, r, _ = precision_recall_curve(y[mask], s[mask], sample_weight=w[mask]); ax.plot(r, p, color=c, lw=2.2, label=name)
        pred = (s >= thr) & mask; tp = w[pred & (y == 1)].sum(); fp = w[pred & (y == 0)].sum(); rec = tp / w[mask & (y == 1)].sum(); pr = tp / (tp + fp)
        ax.plot([rec], [pr], mk, ms=11, color=c, mec="white", mew=2, zorder=5, label=f"  ● {name.split()[0] if 'energy' in name else name[:2]} 阈值 {thr:g}：R={rec:.3f}  P={pr:.3f}")
        print(title[:1], name, f"thr {thr:g} R {rec:.3f} P {pr:.3f}")
    ax.set_title(title + "；植物占 50%", loc="left", fontsize=10.5); ax.set_xlabel("Recall（植物被放行的比例）"); ax.set_ylabel("Precision（放行的里真是植物的比例）")
    ax.set_xlim(0.6, 1.002); ax.set_ylim(0.6, 1.003); ax.grid(alpha=.2); ax.legend(loc="lower left", frameon=False, fontsize=8.5)
fig.savefig("pr_three.png", dpi=150)
