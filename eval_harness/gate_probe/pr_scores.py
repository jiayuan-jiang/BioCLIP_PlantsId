"""算 energy / v1头 / v2头 在同一批整图样本上的分数 → pr_scores.npz（分数均为 5 折分组 OOF；v1 对 plantnet/stem 是外部测试）"""
import numpy as np
import train as T1, train2 as T2
d = T2.build()
full = d["view"] == "full"; src, y, X = d["src"], d["y"], d["X"]
idx = np.load(T1.ROOT / "data/bioclip_full_index.npz", allow_pickle=True)["embeddings"].astype(np.float32)
ii = np.where(full)[0]; en = np.zeros(len(ii))
for i in range(0, len(ii), 512):
    z = 100.0 * (X[ii[i:i+512]] @ idx.T); m = z.max(1, keepdims=True)
    en[i:i+512] = m[:, 0] + np.log(np.exp(z - m).sum(1))
print("energy done", flush=True)
mA = full & ~np.isin(src, ["plantnet", "stem"])
sA = T2.oof(d, mA, 1.0)[ii]; print("v1 done", flush=True)
sC = T2.oof(d, np.ones(len(y), bool), 0.1)[ii]; print("v2 done", flush=True)
np.savez("pr_scores.npz", y=y[ii], src=src[ii].astype(str), energy=en, v1=sA, v2=sC)
