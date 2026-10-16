"""逐物种柱状图：每个物种两根柱（开集 top-1 / 闭集 top-1），按 iNat 全球观测数降序分面板。"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]   # 中文字形
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
from matplotlib.patches import Patch

E = "BotanicalGarden/gh_eval/"
ps = pd.read_csv(E + "per_species.csv")
zero = set(pd.read_csv(E + "zero_vector_species.csv").accepted_name)
ps["zero"] = ps.accepted_name.isin(zero)
ps = ps.sort_values(["inat_obs_total", "accepted_name"], ascending=[False, True]).reset_index(drop=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"                 # 调色板 slot 1/2，已过 validate_palette
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
PER = 86
n_pan = int(np.ceil(len(ps) / PER))
fig, axes = plt.subplots(n_pan, 1, figsize=(24, 3.9 * n_pan), facecolor=SURF)
for k, ax in enumerate(axes):
    d = ps.iloc[k * PER:(k + 1) * PER]
    x = np.arange(len(d))
    ax.set_facecolor(SURF)
    for i, z in enumerate(d.zero):                  # 文字向量为零的物种：灰底
        if z:
            ax.axvspan(i - 0.5, i + 0.5, color="#d9d8d3", zorder=0, lw=0)
    ax.bar(x - 0.2, d.open_top1_len, 0.38, color=BLUE, zorder=2, lw=0)
    ax.bar(x + 0.2, d.closed_top1, 0.38, color=ORANGE, zorder=2, lw=0)
    ax.set_xlim(-0.7, PER - 0.3)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, .25, .5, .75, 1.0]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}  ({int(o):,})" for n, o in zip(d.accepted_name, d.inat_obs_total)],
                       rotation=90, fontsize=5.2, color=INK, fontstyle="italic", fontfamily="DejaVu Sans")
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0)
    lo, hi = int(d.inat_obs_total.max()), int(d.inat_obs_total.min())
    ax.set_title(f"物种 {k*PER+1}–{k*PER+len(d)}   ·   iNat 全球观测数 {lo:,} → {hi:,}   ·   本面板均值 开集 {d.open_top1_len.mean():.0%} / 闭集 {d.closed_top1.mean():.0%}",
                 loc="left", fontsize=10, color=INK, pad=6)
fig.legend(handles=[Patch(color=BLUE, label="开集 top-1（35 万物种）"), Patch(color=ORANGE, label="闭集 top-1（温室 611 种）"),
                    Patch(color="#d9d8d3", label="该物种文字向量为全零（线上词表缺陷，必为 0）")],
           loc="upper center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.995))
fig.suptitle("")
fig.text(0.01, 0.995 - 0.0, "", fontsize=1)
plt.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.2)
fig.savefig(E + "plots/per_species_open_vs_closed.png", dpi=170, facecolor=SURF)
fig.savefig(E + "plots/per_species_open_vs_closed.pdf", facecolor=SURF)
print("saved", len(ps), "species,", n_pan, "panels")
# 面板单独导出一个裁剪预览
from PIL import Image
im = Image.open(E + "plots/per_species_open_vs_closed.png"); w, h = im.size
im.crop((0, 0, w, int(h / n_pan * 1.05))).save("/private/tmp/claude-501/-Users-avalanchy-Documents-GDS-NO-BioCLIP/627a400b-7e0a-41bb-a340-8723cf68f24a/scratchpad/panel1.png")
im.crop((0, int(h / n_pan * 4), w, h)).save("/private/tmp/claude-501/-Users-avalanchy-Documents-GDS-NO-BioCLIP/627a400b-7e0a-41bb-a340-8723cf68f24a/scratchpad/panel56.png")
