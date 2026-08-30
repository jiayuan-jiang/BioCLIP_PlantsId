"""全局检索 eval / recommend 的离线索引构建。
对 data/test_images（~5000 种 × 5 张）编码 —— 两视图 TTA avg(full,cc60)，
算 p_organ（organ_tagger_tta 的 4 头）+ quality（分辨率 + 清晰度）+ dHash，
再编码 5000 个物种文本原型。

产出（拷回 Mac 给 global_retrieval_eval.py / recommend.py）：
  global_pool.npz          emb[N,768 fp16] · species_idx[N] · p_organ[N,4] · q_res[N] · q_sharp[N] · dhash[N,8] · path[N] · classes
  global_species_proto.npz proto[S,768 fp16] · species[S]

用法（3060）：
  export BIOCLIP_ROOT=/path/to/BioCLIP      # 含 data/test_images, demo/weights
  python global_retrieval_build.py
需要同目录有 organ_tagger_tta.npz（从 Mac 带过来，~26KB）。
"""
import os, glob, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from PIL import Image
import torch, open_clip

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("BIOCLIP_ROOT") or os.path.normpath(os.path.join(HERE, "..", ".."))
TI = os.path.join(ROOT, "data", "test_images")
LOCAL_W = os.path.join(ROOT, "demo", "weights", "open_clip_model.safetensors")
TAG = np.load(os.path.join(HERE, "organ_tagger_tta.npz"), allow_pickle=True)
CLS = list(TAG["classes"]); COEF = TAG["coef"].astype(np.float32); ICPT = TAG["intercept"].astype(np.float32)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda"
BATCH = 256 if DEVICE == "cuda" else 32
print(f"device={DEVICE} amp={AMP} batch={BATCH}", flush=True)

pre = LOCAL_W if os.path.exists(LOCAL_W) else "hf-hub:imageomics/bioclip-2"
if pre.startswith("hf-hub"):
    model, _, preprocess = open_clip.create_model_and_transforms(pre); tok = open_clip.get_tokenizer(pre)
else:
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=pre)
    tok = open_clip.get_tokenizer("ViT-L-14")
model = model.to(DEVICE).eval()

def cc60(im):
    w, h = im.size; cw, ch = int(w * .6), int(h * .6)
    l, t = (w - cw) // 2, (h - ch) // 2
    return im.crop((l, t, l + cw, t + ch))

def sharpness(im):
    g = np.asarray(im.convert("L").resize((256, 256)), np.float32)
    lap = -4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)
    return float(lap[1:-1, 1:-1].var())

def dhash(im, s=8):
    g = np.asarray(im.convert("L").resize((s + 1, s)), np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())

@torch.no_grad()
def enc(tensors):
    out = []
    for i in range(0, len(tensors), BATCH):
        x = torch.stack(tensors[i:i + BATCH]).to(DEVICE)
        if AMP:
            with torch.autocast("cuda", dtype=torch.float16):
                e = model.encode_image(x)
        else:
            e = model.encode_image(x)
        e = e.float()
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out, 0)

# ---- 物种文本原型 ----
species = sorted({os.path.basename(d).split("_", 1)[1] for d in glob.glob(f"{TI}/*") if "_" in os.path.basename(d)})
print(f"{len(species)} 物种", flush=True)
TEMPL = ["a photo of {n}.", "a photo of {n}, a species of plant.", "a close-up photo of {n}.", "a photo of the plant {n}."]

@torch.no_grad()
def proto(names, bs=512):
    rows = []
    for i in range(0, len(names), bs):
        ck = names[i:i + bs]; acc = None
        for tm in TEMPL:
            t = tok([tm.format(n=n) for n in ck]).to(DEVICE)
            if AMP:
                with torch.autocast("cuda", dtype=torch.float16):
                    e = model.encode_text(t)
            else:
                e = model.encode_text(t)
            e = e.float(); e = e / e.norm(dim=-1, keepdim=True)
            acc = e if acc is None else acc + e
        acc = acc / acc.norm(dim=-1, keepdim=True)
        rows.append(acc.cpu().numpy())
        print(f"  proto {i+len(ck)}/{len(names)}", end="\r", flush=True)
    print(); return np.concatenate(rows, 0)

PROTO = proto(species)
np.savez(os.path.join(HERE, "global_species_proto.npz"), proto=PROTO.astype(np.float16), species=np.array(species))
print("saved global_species_proto.npz", PROTO.shape, flush=True)
sp2i = {s: i for i, s in enumerate(species)}

# ---- 照片池：两视图 TTA emb + p_organ + quality + dhash ----
items = []
for d in glob.glob(f"{TI}/*"):
    b = os.path.basename(d)
    if "_" not in b:
        continue
    sp = b.split("_", 1)[1]
    for p in glob.glob(d + "/*.jpg"):
        items.append((p, sp))
print(f"照片 {len(items)}", flush=True)

N = len(items)
EMB = np.zeros((N, 768), np.float16)
QRES = np.zeros(N, np.float32); QSHARP = np.zeros(N, np.float32); DH = np.zeros((N, 8), np.uint8)
buf, idx = [], []; t0 = time.time()

def flush():
    if not buf:
        return
    arr = enc(buf)  # [2m, 768], 每图 [full, cc60] 连续两行
    for k, n in enumerate(idx):
        m = arr[2 * k] + arr[2 * k + 1]
        EMB[n] = (m / np.linalg.norm(m)).astype(np.float16)
    buf.clear(); idx.clear()

for n, (p, _) in enumerate(items):
    try:
        im = Image.open(p).convert("RGB")
        QRES[n] = max(im.size); QSHARP[n] = sharpness(im); DH[n] = dhash(im)
        buf.append(preprocess(im)); buf.append(preprocess(cc60(im))); idx.append(n)
    except Exception as ex:
        print("skip", p, ex, flush=True)
    if len(buf) >= BATCH * 3:
        flush()
    if n % 500 == 0:
        el = time.time() - t0
        print(f"  {n}/{N} {el:.0f}s eta {el/max(n,1)*(N-n):.0f}s", flush=True)
flush()

PORG = 1 / (1 + np.exp(-(EMB.astype(np.float32) @ COEF.T + ICPT)))  # [N,4]
np.savez(os.path.join(HERE, "global_pool.npz"),
         emb=EMB, species_idx=np.array([sp2i[s] for _, s in items], np.int32),
         p_organ=PORG.astype(np.float32), q_res=QRES, q_sharp=QSHARP, dhash=DH,
         path=np.array([os.path.basename(p) for p, _ in items]), classes=np.array(CLS))
print("saved global_pool.npz", EMB.shape, flush=True)
