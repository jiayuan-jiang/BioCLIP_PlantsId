r"""iNat v1 参考索引构建（3060）。读 manifest + 已下载的图 →
两视图 TTA avg(full,cc60) emb + p_organ + quality + dhash + 物种文本原型。

用法（3060）：
  set BIOCLIP_ROOT=D:\BioCLIP
  python inat_index_build.py --manifest D:\inat\manifest.parquet --img D:\inat\img \
      --out D:\inat\index --prune

产出 --out 目录：
  inat_pool.npz          photo_id[N] · taxon_id[N] · species_idx[N] · emb[N,768 fp16] ·
                         p_organ[N,4] · q_res[N] · q_sharp[N] · dhash[N,8] · license[N] ·
                         observer_login[N] · url[N] · classes[4]
  inat_species_proto.npz proto[S,768 fp16] · species[S]
--prune：编码后每种保留 {p_bark>0.5 ∪ p_fruit>0.5} ∪ {quality top 50}
"""
import argparse, os, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from PIL import Image
import torch, open_clip
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--img", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prune", action="store_true")
ap.add_argument("--prune-keep", type=int, default=50)
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("BIOCLIP_ROOT") or os.path.normpath(os.path.join(HERE, "..", ".."))
LOCAL_W = os.path.join(ROOT, "demo", "weights", "open_clip_model.safetensors")
TAG = np.load(os.path.join(HERE, "organ_heads.npz"), allow_pickle=True)
CLS = list(TAG["classes"]); COEF = TAG["coef"].astype(np.float32); ICPT = TAG["intercept"].astype(np.float32)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda"
BATCH = 256 if DEVICE == "cuda" else 32
pre = LOCAL_W if os.path.exists(LOCAL_W) else "hf-hub:imageomics/bioclip-2"
if pre.startswith("hf-hub"):
    model, _, preprocess = open_clip.create_model_and_transforms(pre); tok = open_clip.get_tokenizer(pre)
else:
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=pre)
    tok = open_clip.get_tokenizer("ViT-L-14")
model = model.to(DEVICE).eval()
print(f"device={DEVICE} amp={AMP}", flush=True)

def cc60(im):
    w, h = im.size; cw, ch = int(w * .6), int(h * .6); l, t = (w - cw) // 2, (h - ch) // 2
    return im.crop((l, t, l + cw, t + ch))
def sharp(im):
    g = np.asarray(im.convert("L").resize((256, 256)), np.float32)
    lap = -4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)
    return float(lap[1:-1, 1:-1].var())
def dhash(im, s=8):
    g = np.asarray(im.convert("L").resize((s + 1, s)), np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())

@torch.no_grad()
def enc(tensors):
    o = []
    for i in range(0, len(tensors), BATCH):
        x = torch.stack(tensors[i:i + BATCH]).to(DEVICE)
        if AMP:
            with torch.autocast("cuda", dtype=torch.float16):
                e = model.encode_image(x)
        else:
            e = model.encode_image(x)
        e = e.float(); o.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(o, 0)

m = pq.read_table(A.manifest).to_pydict()
rows = list(zip(m["photo_id"], m["taxon_id"], m["scientific_name"], m["extension"],
                m["license"], m["observer_login"], m["url"]))
species = sorted(set(m["scientific_name"])); sp2i = {s: i for i, s in enumerate(species)}
print(f"{len(rows):,} 行 / {len(species):,} 种", flush=True)

# ---- 物种文本原型 ----
TEMPL = ["a photo of {n}.", "a photo of {n}, a species of plant.", "a close-up photo of {n}.", "a photo of the plant {n}."]
@torch.no_grad()
def proto(names, bs=512):
    R = []
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
        acc = acc / acc.norm(dim=-1, keepdim=True); R.append(acc.cpu().numpy())
        print(f"  proto {i+len(ck)}/{len(names)}", end="\r", flush=True)
    print(); return np.concatenate(R, 0)
np.savez(os.path.join(A.out, "inat_species_proto.npz"),
         proto=proto(species).astype(np.float16), species=np.array(species))
print("saved inat_species_proto.npz", flush=True)

# ---- 照片编码 ----
N = len(rows)
EMB = np.zeros((N, 768), np.float16)
QRES = np.zeros(N, np.float32); QSHARP = np.zeros(N, np.float32); DH = np.zeros((N, 8), np.uint8)
valid = np.zeros(N, bool)
buf, idx = [], []; t0 = time.time()
def flush():
    if not buf:
        return
    arr = enc(buf)
    for k, n in enumerate(idx):
        v = arr[2 * k] + arr[2 * k + 1]
        EMB[n] = (v / np.linalg.norm(v)).astype(np.float16)
    buf.clear(); idx.clear()
for n, (pid, tid, sci, ext, lic, obs, url) in enumerate(rows):
    fp = os.path.join(A.img, str(tid), f"{pid}.{ext}")
    try:
        im = Image.open(fp).convert("RGB")
        QRES[n] = max(im.size); QSHARP[n] = sharp(im); DH[n] = dhash(im); valid[n] = True
        buf.append(preprocess(im)); buf.append(preprocess(cc60(im))); idx.append(n)
    except Exception:
        pass
    if len(buf) >= BATCH * 3:
        flush()
    if n % 5000 == 0:
        el = time.time() - t0
        print(f"  {n:,}/{N:,}  {el:.0f}s eta {el/max(n,1)*(N-n):.0f}s  valid={valid[:n+1].sum():,}", flush=True)
flush()

PORG = 1 / (1 + np.exp(-(EMB.astype(np.float32) @ COEF.T + ICPT)))
keep = valid.copy()
if A.prune:
    def z(x): x = np.log1p(np.maximum(x, 0)); return (x - x[valid].mean()) / (x[valid].std() + 1e-6)
    qual = 0.5 + 0.5 * np.tanh((z(QRES) + z(QSHARP)) / 2)
    tid_arr = np.array([r[1] for r in rows])
    for tid in np.unique(tid_arr[valid]):
        mk = valid & (tid_arr == tid)
        ii = np.where(mk)[0]
        rare = (PORG[ii, CLS.index("bark")] > 0.5) | (PORG[ii, CLS.index("fruit")] > 0.5)
        topq = ii[np.argsort(-qual[ii])[:A.prune_keep]]
        k = set(ii[rare]) | set(topq)
        drop = set(ii) - k
        keep[list(drop)] = False
    print(f"prune: {valid.sum():,} → {keep.sum():,}", flush=True)

sel = np.where(keep)[0]
np.savez(os.path.join(A.out, "inat_pool.npz"),
         photo_id=np.array([rows[i][0] for i in sel]),
         taxon_id=np.array([rows[i][1] for i in sel]),
         species_idx=np.array([sp2i[rows[i][2]] for i in sel], np.int32),
         emb=EMB[sel], p_organ=PORG[sel].astype(np.float32),
         q_res=QRES[sel], q_sharp=QSHARP[sel], dhash=DH[sel],
         license=np.array([rows[i][4] for i in sel]),
         observer_login=np.array([str(rows[i][5]) for i in sel]),
         url=np.array([rows[i][6] for i in sel]), classes=np.array(CLS))
print(f"saved inat_pool.npz  {len(sel):,} 张", flush=True)
