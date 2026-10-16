"""
实拍优先级清单 (Botanical Garden pipeline · step 74)
===================================================
目的：明天/之后的实拍优先拍"网上图少"和"现在识别不准"的物种。
  - 网络图数量：A 组（563 个原有文字向量物种）用去噪后的 n_clean；B 组（原零向量物种）用 bg_31 同口径筛选后的张数
  - 闭集精度：用线上正在用的闭集塔 demo/matthaei/closed_v1.npz（重编码向量）对所有网络图重新算 top1/top5
  - 开集精度：A 组用已有逐物种结果 open1_clean；B 组（词表向量全零/缺失）开集必为 0
  - 位置：新清单（含 Garden Bed）
优先级分数 = 0.5·稀缺度 + 0.5·(1-闭集top1)；稀缺度 = 1 - min(n,50)/50（50 张以上视为够）。
输出 BotanicalGarden/shoot_priority.csv，同时供实拍页使用的 demo/static/field_species.json。
"""
import glob, json
import numpy as np, pandas as pd

B = "BotanicalGarden/"
T = np.load("../demo/matthaei/closed_v1.npz", allow_pickle=True)
tn = np.array([str(n) for n in T["names"]]); TE = T["embeddings"].astype(np.float64)

# A 组
df = pd.read_csv(B + "gh_eval/canon_predictions_clean.csv"); E = np.load(B + "gh_eval/emb.f16.npy").astype(np.float64)
m = (~df.flag_outlier.values)
XA, yA = E[m], df.accepted_name.values[m]
# B 组
f = pd.read_csv(B + "gh_notext_emb/emb_files.csv", dtype={"source_id": str}); EN = np.load(B + "gh_notext_emb/emb.f16.npy").astype(np.float64)
q = pd.read_csv(B + "gh_images/inat_quality.csv", dtype=str); qg = dict(zip(q.obs, q.quality_grade)); f["obs"] = f.page.str.extract(r"observations/(\d+)")[0]
keep = ((f.source == "inat") & (f.obs.map(qg) == "research")) | f.source.isin(["commons", "plantnet", "wikipedia", "gbif"])
f = f[keep.values]; XB, yB = EN[f.index.values], f.accepted_name.values
X = np.concatenate([XA, XB]); y = np.concatenate([yA, yB])
S = X @ TE.T; order = np.argsort(-S, 1)[:, :5]
top1 = tn[order[:, 0]] == y; top5 = (tn[order] == y[:, None]).any(1)
acc = pd.DataFrame({"sp": y, "t1": top1, "t5": top5}).groupby("sp").agg(n_web=("t1", "size"), closed_top1=("t1", "mean"), closed_top5=("t5", "mean"))
# 最常把它认成谁
wrong = pd.DataFrame({"sp": y, "pred": tn[order[:, 0]]})[~top1].groupby("sp").pred.agg(lambda s: "; ".join(f"{k}×{v}" for k, v in s.value_counts().head(2).items()))
acc["confused_with"] = wrong
op = pd.read_csv(B + "gh_eval/species_difficulty_clean.csv").set_index("accepted_name")
acc["open_top1"] = op.open1_clean.reindex(acc.index)
acc["family"] = op.family.reindex(acc.index)
acc["zero_vec"] = op.zero_vec.reindex(acc.index)

# 新清单：位置 + 现存株数
new = pd.read_csv(glob.glob(B + "MBGNA*2241858699287564569.csv")[0], encoding="utf-8-sig")
accd = pd.read_csv("bg_greenhouse_species_accepted.csv")
raw2acc = {}
for _, r in accd.iterrows():
    for rn in str(r.raw_names).split("|"):
        raw2acc[rn.strip()] = r.accepted_name
new["acc"] = new["Scientific Name"].map(raw2acc)
H = lambda s: "Arid" if "Arid" in s else "Temperate" if "Temperate" in s else "Tropical"
new["house"] = new["Management Subzone"].map(H)
loc = new.dropna(subset=["acc"]).groupby("acc").agg(
    n_plants=("acc", "size"), houses=("house", lambda s: "/".join(sorted(set(s)))),
    beds=("Garden Bed", lambda s: ",".join(str(int(b)) for b in sorted(set(s.dropna())))),
    garden_name=("Scientific Name", lambda s: s.iloc[0]),
    location_note=("Location Details", lambda s: " / ".join(sorted(set(s.dropna().astype(str))))))
d = loc.join(acc, how="left")
d["n_web"] = d.n_web.fillna(0).astype(int)
d["closed_top1"] = d.closed_top1.fillna(0.0)          # 完全没图的按 0 处理
d["closed_top5"] = d.closed_top5.fillna(0.0)
d["no_open_vec"] = d.open_top1.isna() | d.zero_vec.fillna(False).astype(bool)
d["open_top1"] = d.open_top1.fillna(0.0)
d["scarcity"] = 1 - np.minimum(d.n_web, 50) / 50
d["score"] = 0.5 * d.scarcity + 0.5 * (1 - d.closed_top1)
def reason(r):
    t = []
    if r.n_web == 0: t.append("网上0张（精度未评估）")
    else:
        if r.n_web < 20: t.append(f"网上仅{r.n_web}张")
        if r.closed_top1 < 0.5: t.append(f"温室识别top1仅{r.closed_top1:.0%}")
    if r.no_open_vec: t.append("公版词表无向量")
    elif r.n_web and r.open_top1 < 0.2: t.append(f"公版top1仅{r.open_top1:.0%}")
    return "；".join(t)
d["reason"] = d.apply(reason, axis=1)
d = d.sort_values("score", ascending=False).reset_index().rename(columns={"index": "accepted_name", "acc": "accepted_name"})
d.insert(0, "rank", range(1, len(d) + 1))
d.to_csv(B + "shoot_priority.csv", index=False, encoding="utf-8-sig")

# 供实拍页：全部物种都进列表（拍到什么都能选），前 TOP_N 个标 ★ 优先
TOP_N = 60
com = accd.set_index("accepted_name").inat_common.fillna("")
raws = accd.set_index("accepted_name").raw_names
out = []
for _, r in d.iterrows():
    al = {r.accepted_name} | {x.strip() for x in str(raws.get(r.accepted_name, "")).split("|")}
    al.discard(r.garden_name)
    out.append(dict(name=r.garden_name, aliases=sorted(a for a in al if a), common=str(com.get(r.accepted_name, "")),
                    houses=r.houses.split("/"), beds=[int(b) for b in str(r.beds).split(",") if b and b != "nan"],
                    n_plants=int(r.n_plants), web_photos=int(r.n_web), rank=int(r["rank"]), priority=bool(r["rank"] <= TOP_N),
                    reason=r.reason, closed_top1=round(float(r.closed_top1), 2), loc=r.location_note if isinstance(r.location_note, str) else ""))
json.dump(out, open("../demo/static/field_species.json", "w"), ensure_ascii=False)
print("field_species.json:", len(out), "个物种，★", TOP_N)
print(len(d), "species")

