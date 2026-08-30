"""诊断：为什么 Test A 部位分桶塌成 habit。
1) 5 个部位文本原型两两余弦
2) 减 image-mean 校准后再分桶
3) image-query 部位检索（用几张肉眼确认的种子图当 query，不用文本）
"""
import os, glob, random, collections
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from PIL import Image
import torch, open_clip

ROOT = "/Users/avalanchy/Documents/GDS/NO/BioCLIP"
W = f"{ROOT}/demo/weights/open_clip_model.safetensors"
TI = f"{ROOT}/data/test_images"
random.seed(0)

model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=W)
model.eval(); tok = open_clip.get_tokenizer("ViT-L-14")

@torch.no_grad()
def et(ps):
    e = model.encode_text(tok(ps)).float(); return (e/e.norm(dim=-1,keepdim=True)).numpy()
@torch.no_grad()
def ei(paths, bs=16):
    o=[]
    for i in range(0,len(paths),bs):
        b=[preprocess(Image.open(p).convert("RGB")) for p in paths[i:i+bs]]
        e=model.encode_image(torch.stack(b)).float(); o.append((e/e.norm(dim=-1,keepdim=True)).numpy())
        print(f"  {min(i+bs,len(paths))}/{len(paths)}",end="\r")
    print(); return np.concatenate(o,0)
def proto(ps): e=et(ps).mean(0); return e/np.linalg.norm(e)

OP = {
 "leaf":["a photo of a leaf","a close-up photo of the leaves of a plant","plant foliage, leaves"],
 "flower":["a photo of a flower","a close-up photo of a flower","a plant in bloom with flowers"],
 "fruit":["a photo of a fruit","a close-up photo of berries and fruit","the fruit and seeds of a plant"],
 "bark":["a photo of tree bark","a close-up photo of the bark of a tree trunk","the trunk of a tree"],
 "habit":["a photo of a whole plant","a photo of a whole tree outdoors","a shrub growing in a landscape"],
}
ORG=list(OP); P=np.stack([proto(OP[o]) for o in ORG])

print("=== 1) 部位文本原型两两余弦 ===")
print("        "+"  ".join(f"{o:6}" for o in ORG))
for i,o in enumerate(ORG):
    print(f"{o:6}  "+"  ".join(f"{P[i]@P[j]:.3f} " for j in range(len(ORG))))

# 样本
sp=sorted(glob.glob(f"{TI}/*/")); random.shuffle(sp)
paths=[]
for s in sp:
    im=glob.glob(s+"*.jpg")
    if im: paths.append(random.choice(im))
    if len(paths)>=200: break
E=ei(paths)

print("\n=== 2) 减 image-mean 校准 ===")
mu=E.mean(0); Ec=E-mu; Ec=Ec/np.linalg.norm(Ec,axis=1,keepdims=True)
for tag,X in [("raw",E),("centered",Ec)]:
    pred=(X@P.T).argmax(1)
    print(f"  {tag:9}", dict(collections.Counter(ORG[i] for i in pred)))

print("\n=== 3) image-query 部位检索（种子图当 query）===")
# 从样本里挑：肉眼无法确认，改用文本弱标 + 人工 seed 思路的近似——
# 这里用每个部位文本原型在样本中取 top-8 当"伪种子"，再用伪种子均值做 image-query 检索
for oi,o in enumerate(ORG):
    s=E@P[oi]; seed=E[np.argsort(s)[::-1][:8]].mean(0); seed/=np.linalg.norm(seed)
    q=E@seed
    top=np.argsort(q)[::-1][:5]
    print(f"  [{o:6}] 伪种子 image-query top5: {[os.path.basename(paths[t]) for t in top]}")
    print(f"           这些图的 raw 文本分桶 = {[ORG[(E[t]@P.T).argmax()] for t in top]}")
