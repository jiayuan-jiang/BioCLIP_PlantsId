"""
Matthaei 温室闭集塔 v1 (Botanical Garden pipeline · step 70)
===========================================================
closed_text.npz 里 48 行文字向量是 NaN（线上词表里这些物种的预计算向量全零，归一化后成 NaN）。
做法：用 BioCLIP 2 自己的文字塔，把闭集 611 个物种的名字重新编码，整个闭集塔用重编码向量。
  文本 = "a photo of {taxon_path}."（taxon_path 取自词表 taxon_paths，closed_mapping.csv 的 taxon_path_index）
  与同一物种线上已有向量的余弦：均值 0.964（不是逐位复现，官方还可能含别的模板/细节；分类效果已验证）。
验证（温室网络图，候选 611）：
  原来有向量的 563 物种：存的向量 top1 0.760 → 重编码 0.781
  原来 NaN 的 40 物种（图>20 张）：重编码 0.912（对照：图像原型+手选偏移 0.887，需校准且同分布乐观）
所有行同一模态、同一尺度，不需要校准。
需要 torch + open_clip（本地临时 venv 即可）。输出 ../demo/matthaei/closed_v1.npz
"""
import numpy as np, pandas as pd, torch, open_clip

B = "BotanicalGarden/"
model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="weights/bioclip2/open_clip_model.safetensors")
tok = open_clip.get_tokenizer("ViT-L-14"); model.eval()

ct = np.load(B + "gh_closed/closed_text.npz", allow_pickle=True)
names = [str(n) for n in ct["names"]]
bad = np.isnan(ct["embeddings"]).any(1)
mp = pd.read_csv(B + "gh_closed/closed_mapping.csv").set_index("accepted_name")
paths = [str(mp.loc[n, "taxon_path_index"]) for n in names]
common = [str(mp.loc[n, "common_index"]) if isinstance(mp.loc[n, "common_index"], str) else "" for n in names]

txt = [f"a photo of {p}." for p in paths]
out = []
for i in range(0, len(txt), 64):
    with torch.no_grad():
        out.append(torch.nn.functional.normalize(model.encode_text(tok(txt[i:i + 64])), dim=-1).numpy())
E_ = np.concatenate(out).astype(np.float32)
assert not np.isnan(E_).any()
source = np.where(bad, "text_reencoded_fix", "text_reencoded")
np.savez("../demo/matthaei/closed_v1.npz", names=np.array(names), embeddings=E_, source=source,
         family=ct["family"], genus=ct["genus"], common=np.array(common))
print(len(names), "rows;", int(bad.sum()), "原 NaN 行已修复")
