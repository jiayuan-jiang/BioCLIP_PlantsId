"""
文字向量复现实验 (Botanical Garden pipeline · step 73)
=====================================================
按官方 make_txt_embedding.py（TreeOfLife-toolbox）的写法重编码 25 个已知物种，和线上存的向量比余弦。
官方写法：名字=7 级空格拼接 [+ " with common name X"]；80 个 OpenAI 模板各编码、逐条归一化、取均值再归一化。
结果（2026-10-10）：余弦均值 0.972，没有复现到 1.0。已排除：tokenizer、模型配置、权重版本(sha256 一致)。未排除：open_clip/torch 版本、GPU 精度。
80 个模板文件：HF Space imageomics/bioclip-2-demo 的 components/templates.py，放到 off/ 目录。
"""
import sys, numpy as np, torch, random, open_clip
sys.path.insert(0,"off")
from components_templates import openai_imagenet_template as TP
import torch.nn.functional as F
W="weights/bioclip2"
model,_,_=open_clip.create_model_and_transforms("ViT-L-14",pretrained=W+"/open_clip_model.safetensors"); model.eval()
tk_old=open_clip.get_tokenizer("ViT-L-14")
try: tk_new=open_clip.get_tokenizer("local-dir:"+W); print("tokenizer local-dir ok",type(tk_new).__name__)
except Exception as e: print("local-dir tokenizer fail:",repr(e)[:200]); tk_new=None
z=np.load("bioclip_full_index.npz",allow_pickle=True); E=z["embeddings"]; path=z["taxon_paths"]; com=z["common_names"]
ok=np.where(np.linalg.norm(E,axis=1)>0.5)[0]; random.seed(1); ix=random.sample(list(ok),25); ref=E[ix]
s=lambda a:"" if a is None or str(a)=="nan" else str(a)
names=[s(path[i])+(" with common name "+s(com[i]) if s(com[i]) else "") for i in ix]   # 官方 make_txt_embedding.py 的拼法
print(names[0], "|", sum("common name" in n for n in names),"/25 带俗名",flush=True)
def enc(tk,names):
    out=[]
    for n in names:
        with torch.no_grad(): e=F.normalize(model.encode_text(tk([t(n) for t in TP])),dim=-1).mean(0)
        out.append((e/e.norm()).numpy())
    return np.stack(out)
for label,tk in [("旧 tokenizer(ViT-L-14 默认)",tk_old),("官方 tokenizer(bioclip-2 本地)",tk_new)]:
    if tk is None: continue
    c=(enc(tk,names)*ref).sum(1); print(f"{label:34s} mean {c.mean():.5f} min {c.min():.5f}",flush=True)
