"""
标注噪声过滤 + 清洗后重评 (Botanical Garden pipeline · step 41)
==============================================================
只用【图像向量】（不看文字、不看模型预测，避免循环论证）：
  同一物种的图应当彼此相似。对每张图算"与本物种其余图中心的余弦相似度"（留一法），
  在物种内做稳健 z 分数（中位数/MAD）；z < -2.5 记为"与本物种其余图不一致"（疑似错标 / 非植物主体 / 极端远景特写）。
  图数 < 8 的物种不做过滤（统计不可靠）。
输出：canon_predictions_clean.csv（在 canon_predictions 上加 flag_outlier / loo_sim / loo_z）、clean_summary.json
"""
import json, numpy as np, pandas as pd

E = "BotanicalGarden/gh_eval/"
df = pd.read_csv(E + "canon_predictions.csv")
emb = np.load(E + "emb.f16.npy").astype(np.float32)
files = pd.read_csv(E + "emb_files.csv")
assert len(emb) == len(df) and (files.file.values == df.file.values).all()

sim = np.full(len(df), np.nan); z = np.full(len(df), np.nan)
for name, idx in df.groupby("accepted_name").indices.items():
    if len(idx) < 8:
        continue
    X = emb[idx]; s = X.sum(0)
    loo = (s - X) / (len(idx) - 1)
    loo /= np.linalg.norm(loo, axis=1, keepdims=True)
    c = (X * loo).sum(1)
    med = np.median(c); mad = np.median(np.abs(c - med)) * 1.4826 + 1e-6
    sim[idx], z[idx] = c, (c - med) / mad
df["loo_sim"], df["loo_z"] = sim, z
df["flag_outlier"] = df.loo_z < -2.5
df["judged"] = df.loo_z.notna()
df.to_csv(E + "canon_predictions_clean.csv", index=False, encoding="utf-8-sig")

def blk(d):
    return {"n_img": int(len(d)), "open_top1": float(d.c_open_hit1.mean()), "open_top5": float(d.c_open_hit5.mean()),
            "closed_top1": float(d.c_closed_hit1.mean()), "closed_top5": float(d.c_closed_hit5.mean())}
j = df[df.judged]
S = {"n_img": int(len(df)), "judged_imgs(species n>=8)": int(len(j)), "flagged": int(df.flag_outlier.sum()),
     "flag_rate_among_judged": float(j.flag_outlier.mean()),
     "all": blk(df), "flagged_imgs": blk(df[df.flag_outlier]), "unflagged_judged": blk(j[~j.flag_outlier]),
     "clean_set(unflagged, all)": blk(df[~df.flag_outlier]),
     "by_source_flag_rate": {k: float(v) for k, v in j.groupby("source").flag_outlier.mean().items()},
     "by_source_flagged_open_top1": {k: float(v) for k, v in df[df.flag_outlier].groupby("source").c_open_hit1.mean().items()}}
json.dump(S, open(E + "clean_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False, indent=1))
