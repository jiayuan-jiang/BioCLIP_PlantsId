"""
对照实验：同一套 embedding + 开集检索流程，跑在 data/test_images（iNat 研究级野外图）上，
用来区分"评估流程有问题"还是"温室场景本身难"。每个物种随机取 1 张，排除词表里向量为零的物种。
"""
import os, re, glob, random
import numpy as np, pandas as pd, torch, open_clip
from PIL import Image
from tqdm import tqdm

N = 1500
random.seed(0)
raw = np.load("bioclip_full_index.npz", allow_pickle=True)
X = torch.from_numpy(raw["embeddings"].astype(np.float32))
sci = [re.sub(r"\s+", " ", str(s).lower()) for s in raw["sci_names"]]
nrm = np.linalg.norm(raw["embeddings"], axis=1)
name2rows = {}
for i, s in enumerate(sci):
    name2rows.setdefault(s, []).append(i)

items = []
for d in sorted(glob.glob("test_images/*")):
    nm = os.path.basename(d).split("_", 1)[1].lower()
    rows = name2rows.get(nm, [])
    if not rows or (nrm[rows] < 0.5).all():          # 词表里没有 / 向量为零 → 不作为对照
        continue
    fs = glob.glob(d + "/*.jpg")
    if fs:
        items.append((random.choice(fs), nm, rows))
random.shuffle(items)
items = items[:N]
print("对照图", len(items), "物种", len({i[1] for i in items}), flush=True)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
model, _, pp = open_clip.create_model_and_transforms("ViT-L-14", pretrained="./weights/bioclip2/open_clip_model.safetensors")
model = model.to(dev).eval()
hit1 = hit5 = 0
with torch.no_grad():
    for k in tqdm(range(0, len(items), 32)):
        b = items[k:k + 32]
        x = torch.stack([pp(Image.open(p).convert("RGB")) for p, _, _ in b]).to(dev)
        e = model.encode_image(x); e = (e / e.norm(dim=-1, keepdim=True)).float().cpu()
        S = e @ X.T
        top5 = S.topk(5, dim=1).indices.numpy()
        for j, (_, nm, rows) in enumerate(b):
            rs = set(rows)
            hit1 += top5[j, 0] in rs
            hit5 += any(t in rs for t in top5[j])
n = len(items)
print(f"\n对照（iNat 野外图，开集 350k，strict 名字）：n={n}  top1={hit1/n:.3f}  top5={hit5/n:.3f}")
