"""species 识别的 TTA 测试。data/test_images 采样 2500 张，每张编 5 视图，
zero-shot cos vs 5000 物种文本原型，比各 TTA 组合的 top-1 / top-5。
一次性编码全部视图 → 存 npz，之后分析免费。"""
import os, glob, random, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from PIL import Image
import torch, open_clip

ROOT = "/Users/avalanchy/Documents/GDS/NO/BioCLIP"
TI = f"{ROOT}/data/test_images"
W = f"{ROOT}/demo/weights/open_clip_model.safetensors"
HERE = os.path.dirname(os.path.abspath(__file__))
EMB_NPZ = os.path.join(HERE, "tta_species_emb.npz")
PROTO_NPZ = os.path.join(HERE, "tta_species_proto.npz")
N_SAMPLE = 2500
random.seed(0)

PLANT_TEMPLATES = [
    "a photo of {name}.",
    "a photo of {name}, a species of plant.",
    "a close-up photo of {name}.",
    "a photo of the plant {name}.",
]
VIEWS = ["full", "hflip", "cc80", "cc60", "padsq"]

print("加载模型…", flush=True)
model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=W)
model.eval();  tok = open_clip.get_tokenizer("ViT-L-14")

def cc(im, f):
    w, h = im.size; cw, ch = int(w * f), int(h * f)
    l, t = (w - cw) // 2, (h - ch) // 2
    return im.crop((l, t, l + cw, t + ch))

def padsq(im):
    w, h = im.size; s = max(w, h)
    c = Image.new("RGB", (s, s), (124, 116, 104))
    c.paste(im, ((s - w) // 2, (s - h) // 2)); return c

def views(im):
    return [im, im.transpose(Image.FLIP_LEFT_RIGHT), cc(im, 0.8), cc(im, 0.6), padsq(im)]

@torch.no_grad()
def enc_text_proto(names, bs=256):
    out = []
    for i in range(0, len(names), bs):
        chunk = names[i:i + bs]
        protos = []
        for tmpl in PLANT_TEMPLATES:
            e = model.encode_text(tok([tmpl.format(name=n) for n in chunk])).float()
            protos.append((e / e.norm(dim=-1, keepdim=True)).numpy())
        p = np.mean(protos, axis=0)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        out.append(p)
        print(f"  proto {i+len(chunk)}/{len(names)}", end="\r", flush=True)
    print();  return np.concatenate(out, 0)

# ---------- 物种原型 ----------
species = sorted(os.path.basename(d).split("_", 1)[1] for d in glob.glob(f"{TI}/*"))
if os.path.exists(PROTO_NPZ):
    z = np.load(PROTO_NPZ, allow_pickle=True); PROTO = z["proto"]; species = list(z["species"])
    print(f"载入原型 {PROTO.shape}", flush=True)
else:
    print(f"编码 {len(species)} 物种原型…", flush=True)
    PROTO = enc_text_proto(species)
    np.savez(PROTO_NPZ, proto=PROTO, species=np.array(species))
sp2i = {s: i for i, s in enumerate(species)}

# ---------- 采样 + 编码 5 视图 ----------
if os.path.exists(EMB_NPZ):
    z = np.load(EMB_NPZ, allow_pickle=True); EMB = z["emb"]; Y = z["y"]
    print(f"载入 emb {EMB.shape}", flush=True)
else:
    allimg = []
    for d in glob.glob(f"{TI}/*"):
        sp = os.path.basename(d).split("_", 1)[1]
        for p in glob.glob(d + "/*.jpg"):
            allimg.append((p, sp))
    random.shuffle(allimg)
    allimg = [x for x in allimg if x[1] in sp2i][:N_SAMPLE]
    print(f"采样 {len(allimg)} 张，编码 {len(VIEWS)} 视图…", flush=True)
    EMB = np.zeros((len(allimg), len(VIEWS), 768), np.float32)
    Y = np.array([sp2i[sp] for _, sp in allimg])
    t0 = time.time()
    with torch.no_grad():
        for k, (p, _) in enumerate(allimg):
            try:
                vs = views(Image.open(p).convert("RGB"))
                x = torch.stack([preprocess(v) for v in vs])
                e = model.encode_image(x).float()
                EMB[k] = (e / e.norm(dim=-1, keepdim=True)).numpy()
            except Exception as ex:
                print("skip", p, ex)
            if k % 100 == 0:
                el = time.time() - t0
                print(f"  {k}/{len(allimg)}  {el:.0f}s eta {el/max(k,1)*(len(allimg)-k):.0f}s", flush=True)
    np.savez(EMB_NPZ, emb=EMB, y=Y)
    print("saved", EMB_NPZ, EMB.shape, flush=True)

# ---------- 评估各 TTA 组合 ----------
def topk_acc(feat):                       # feat [N,768] 已归一
    sims = feat @ PROTO.T                 # [N,5000]
    r = np.argsort(-sims, axis=1)
    t1 = (r[:, 0] == Y).mean()
    t5 = np.array([Y[i] in r[i, :5] for i in range(len(Y))]).mean()
    return t1, t5

def combo(idxs):
    m = EMB[:, idxs, :].sum(1)
    return m / np.linalg.norm(m, axis=1, keepdims=True)

vi = {v: i for i, v in enumerate(VIEWS)}
CONFIGS = {
    "full (baseline)":        [vi["full"]],
    "hflip only":             [vi["hflip"]],
    "cc80 only":              [vi["cc80"]],
    "cc60 only":              [vi["cc60"]],
    "padsq only":             [vi["padsq"]],
    "full+hflip":             [vi["full"], vi["hflip"]],
    "full+cc80":              [vi["full"], vi["cc80"]],
    "full+cc60":              [vi["full"], vi["cc60"]],
    "full+padsq":             [vi["full"], vi["padsq"]],
    "full+hflip+cc80":        [vi["full"], vi["hflip"], vi["cc80"]],
    "full+hflip+cc80+padsq":  [vi["full"], vi["hflip"], vi["cc80"], vi["padsq"]],
    "all 5":                  list(range(5)),
}
print(f"\n{'config':26} {'top1':>7} {'top5':>7}   Δtop1")
b1, b5 = topk_acc(combo(CONFIGS["full (baseline)"]))
for name, idxs in CONFIGS.items():
    t1, t5 = topk_acc(combo(idxs))
    print(f"{name:26} {t1:7.4f} {t5:7.4f}   {t1-b1:+.4f}")
