"""TTA 完整消融 —— 一次编码 7 个视图，两个下游（species zero-shot / organ probe）。
可在 CPU 或 CUDA(3060) 跑；device 自动选，cuda 用 fp16。确定性采样跨平台一致。

产出（拷回 Mac 给 tta_analysis.py）：
  tta_species_proto.npz   [n_species,768] 文本原型
  tta_species_7v.npz      [N,7,768] emb + y + paths
  tta_organ_eval_7v.npz   [M,7,768] emb + organs

用法（laptop）：
  export BIOCLIP_ROOT=/path/to/BioCLIP        # 含 data/test_images
  python tta_full_study.py                    # 全跑
  python tta_full_study.py --skip-organ       # 只 species
"""
import os, sys, glob, random, time, argparse, io, csv, collections
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from PIL import Image
import torch, open_clip

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("BIOCLIP_ROOT"))
ap.add_argument("--out", default=None)
ap.add_argument("--n-species", type=int, default=2500)
ap.add_argument("--skip-species", action="store_true")
ap.add_argument("--skip-organ", action="store_true")
A = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = A.root or os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = A.out or HERE
TI = os.path.join(ROOT, "data", "test_images")
LOCAL_W = os.path.join(ROOT, "demo", "weights", "open_clip_model.safetensors")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda"
BATCH = 192 if DEVICE == "cuda" else 32
print(f"device={DEVICE}  amp={AMP}  batch={BATCH}", flush=True)

pre = LOCAL_W if os.path.exists(LOCAL_W) else "hf-hub:imageomics/bioclip-2"
if pre.startswith("hf-hub"):
    model, _, preprocess = open_clip.create_model_and_transforms(pre)
    tok = open_clip.get_tokenizer(pre)
else:
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=pre)
    tok = open_clip.get_tokenizer("ViT-L-14")
model = model.to(DEVICE).eval()
print(f"weights: {pre}", flush=True)

VIEWS = ["full", "hflip", "cc90", "cc80", "cc70", "cc60", "padsq"]

def _cc(im, f):
    w, h = im.size
    cw, ch = max(1, int(w * f)), max(1, int(h * f))
    l, t = (w - cw) // 2, (h - ch) // 2
    return im.crop((l, t, l + cw, t + ch))

def _padsq(im):
    w, h = im.size
    s = max(w, h)
    c = Image.new("RGB", (s, s), (124, 116, 104))
    c.paste(im, ((s - w) // 2, (s - h) // 2))
    return c

def make_views(im):
    return [im, im.transpose(Image.FLIP_LEFT_RIGHT),
            _cc(im, .9), _cc(im, .8), _cc(im, .7), _cc(im, .6), _padsq(im)]

@torch.no_grad()
def _enc(tensors):
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

@torch.no_grad()
def encode_7v(open_fns):
    """open_fns: list of callables returning a PIL.Image. -> [N,7,768]"""
    N = len(open_fns)
    EMB = np.zeros((N, 7, 768), np.float32)
    buf, idx = [], []
    t0 = time.time()
    def flush():
        if not buf:
            return
        arr = _enc(buf)
        for (n, v), row in zip(idx, arr):
            EMB[n, v] = row
        buf.clear(); idx.clear()
    for n, fn in enumerate(open_fns):
        try:
            for v, vi in enumerate(make_views(fn().convert("RGB"))):
                buf.append(preprocess(vi)); idx.append((n, v))
        except Exception as ex:
            print("skip", n, ex, flush=True)
        if len(buf) >= BATCH * 3:
            flush()
        if n % 200 == 0:
            el = time.time() - t0
            print(f"  {n}/{N}  {el:.0f}s eta {el/max(n,1)*(N-n):.0f}s", flush=True)
    flush()
    return EMB

PLANT_TEMPLATES = ["a photo of {name}.", "a photo of {name}, a species of plant.",
                   "a close-up photo of {name}.", "a photo of the plant {name}."]

@torch.no_grad()
def proto_matrix(names, bs=512):
    rows = []
    for i in range(0, len(names), bs):
        ck = names[i:i + bs]
        acc = None
        for tm in PLANT_TEMPLATES:
            t = tok([tm.format(name=n) for n in ck]).to(DEVICE)
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
    print()
    return np.concatenate(rows, 0)

# ---------------- SPECIES ----------------
if not A.skip_species:
    print("=== SPECIES ===", flush=True)
    random.seed(0)
    species = sorted({os.path.basename(d).split("_", 1)[1]
                      for d in glob.glob(f"{TI}/*") if "_" in os.path.basename(d)})
    print(f"{len(species)} 物种类", flush=True)
    PROTO = proto_matrix(species)
    np.savez(os.path.join(OUT, "tta_species_proto.npz"), proto=PROTO, species=np.array(species))
    sp2i = {s: i for i, s in enumerate(species)}
    pool = []
    for d in glob.glob(f"{TI}/*"):
        b = os.path.basename(d)
        if "_" not in b:
            continue
        sp = b.split("_", 1)[1]
        for p in glob.glob(d + "/*.jpg"):
            pool.append((p, sp))
    random.shuffle(pool)
    pool = [x for x in pool if x[1] in sp2i][:A.n_species]
    print(f"采样 {len(pool)}", flush=True)
    EMB = encode_7v([(lambda p=p: Image.open(p)) for p, _ in pool])
    np.savez(os.path.join(OUT, "tta_species_7v.npz"), emb=EMB,
             y=np.array([sp2i[s] for _, s in pool]),
             paths=np.array([os.path.basename(p) for p, _ in pool]))
    print("saved tta_species_7v.npz", EMB.shape, flush=True)

# ---------------- ORGAN EVAL ----------------
if not A.skip_organ:
    print("=== ORGAN EVAL ===", flush=True)
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    random.seed(0)
    REPO = "mikehemberger/plantnet300K"
    fn2organ = {r["file_name"]: r["organ"] for r in
                csv.DictReader(open(hf_hub_download(REPO, "data/test/metadata.csv", repo_type="dataset")))}
    KEEP = {"leaf", "flower", "fruit", "bark", "habit"}
    CAP = {"leaf": 1500, "flower": 1500, "fruit": 1176, "bark": 225, "habit": 450}
    TEST = ["data/test-00000-of-00008-ccdf9a8395f11a02.parquet",
            "data/test-00001-of-00008-4cc7b1523923ba5e.parquet",
            "data/test-00002-of-00008-952e223123a49903.parquet",
            "data/test-00003-of-00008-4e3b69c54dfa5948.parquet",
            "data/test-00004-of-00008-6882acfdcab0bbdd.parquet",
            "data/test-00005-of-00008-f6f2fd539fae591e.parquet",
            "data/test-00006-of-00008-5932eb3f032283e1.parquet",
            "data/test-00007-of-00008-8caa262cecd65437.parquet"]
    buckets = collections.defaultdict(list)
    for tf in TEST:
        p = hf_hub_download(REPO, tf, repo_type="dataset")
        for batch in pq.ParquetFile(p).iter_batches(batch_size=256):
            for im in batch.to_pydict()["image"]:
                org = fn2organ.get(im["path"])
                if org in KEEP and im["bytes"]:
                    buckets[org].append(im["bytes"])
    samples = []
    for org, lst in buckets.items():
        random.shuffle(lst)
        for b in lst[:CAP[org]]:
            samples.append((b, org))
    random.shuffle(samples)
    print(f"organ eval 采样 {len(samples)}", flush=True)
    EMB = encode_7v([(lambda b=b: Image.open(io.BytesIO(b))) for b, _ in samples])
    np.savez(os.path.join(OUT, "tta_organ_eval_7v.npz"), emb=EMB,
             organs=np.array([o for _, o in samples]))
    print("saved tta_organ_eval_7v.npz", EMB.shape, flush=True)

print("DONE", flush=True)
