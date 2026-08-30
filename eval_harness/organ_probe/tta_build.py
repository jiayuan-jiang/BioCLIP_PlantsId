"""center-crop TTA：用和 organ_train/eval.npz 完全相同的确定性采样，
重新编码一个【中心裁 60%】视图 → organ_{train,eval}_cc.npz（emb, organs[, species]）。
之后 tta_compare.py 按 index 与原 npz 配对，比 full / cc / avg 的 AUC-PR。"""
import os, io, csv, random, collections, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from PIL import Image
import torch, open_clip
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfFileSystem

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
W = f"{ROOT}/demo/weights/open_clip_model.safetensors"
REPO = "mikehemberger/plantnet300K"
FRAC = 0.60

model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=W)
model.eval()

def cc(im):
    w, h = im.size
    cw, ch = int(w * FRAC), int(h * FRAC)
    l, t = (w - cw) // 2, (h - ch) // 2
    return im.crop((l, t, l + cw, t + ch))

@torch.no_grad()
def encode(byte_list, bs=32):
    out, t0 = [], time.time()
    for i in range(0, len(byte_list), bs):
        b = []
        for by in byte_list[i:i + bs]:
            try: b.append(preprocess(cc(Image.open(io.BytesIO(by)).convert("RGB"))))
            except Exception: b.append(torch.zeros(3, 224, 224))
        e = model.encode_image(torch.stack(b)).float()
        out.append((e / e.norm(dim=-1, keepdim=True)).numpy())
        if i % (bs * 20) == 0:
            el = time.time() - t0; done = i + len(b)
            print(f"    {done}/{len(byte_list)}  {el:.0f}s eta {el/max(done,1)*(len(byte_list)-done):.0f}s", flush=True)
    return np.concatenate(out, 0)

# ---------- EVAL：同 build_organ_eval.py ----------
print("=== eval cc ===", flush=True)
random.seed(0)
meta = hf_hub_download(REPO, "data/test/metadata.csv", repo_type="dataset")
fn2organ = {r["file_name"]: r["organ"] for r in csv.DictReader(open(meta))}
KEEP = {"leaf", "flower", "fruit", "bark", "habit"}
CAP = {"leaf": 1500, "flower": 1500, "fruit": 1176, "bark": 225, "habit": 450}
TEST = [f"data/test-0000{i}-of-00008-{h}.parquet" for i, h in enumerate(
    ["ccdf9a8395f11a02", "4cc7b1523923ba5e", "952e223123a49903", "4e3b69c54dfa5948",
     "6882acfdcab0bbdd", "f6f2fd539fae591e", "5932eb3f032283e1", "8caa262cecd65437"])]
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
emb = encode([b for b, _ in samples])
np.savez(os.path.join(HERE, "organ_eval_cc.npz"), emb=emb, organs=np.array([o for _, o in samples]))
print("saved organ_eval_cc.npz", emb.shape, flush=True)

# ---------- TRAIN：同 build_organ_train.py（v2 随机采样）----------
print("=== train cc ===", flush=True)
random.seed(0)
fs = HfFileSystem()
recs = []
for split in ["train", "val"]:
    m = hf_hub_download(REPO, f"data/{split}/metadata.csv", repo_type="dataset")
    for r in csv.DictReader(open(m)):
        recs.append((r["file_name"], r["organ"], r["species"]))
by_organ = collections.defaultdict(list)
for fn, org, sp in recs:
    by_organ[org].append((fn, sp))
CAP2 = {"leaf": 2500, "flower": 2500, "fruit": 2500, "habit": 3000, "bark": None}
want = {}
for org, cap in CAP2.items():
    lst = by_organ[org][:]; random.shuffle(lst)
    for fn, sp in (lst if cap is None else lst[:cap]):
        want[fn] = (org, sp)
shards = sorted(fs.glob(f"datasets/{REPO}/data/train-*.parquet")) + \
         sorted(fs.glob(f"datasets/{REPO}/data/validation-*.parquet"))
shards = [s.split(f"datasets/{REPO}/")[1] for s in shards]
got = {}
for sh in shards:
    p = hf_hub_download(REPO, sh, repo_type="dataset")
    for batch in pq.ParquetFile(p).iter_batches(batch_size=256):
        for im in batch.to_pydict()["image"]:
            if im["path"] in want and im["bytes"] and im["path"] not in got:
                got[im["path"]] = im["bytes"]
    if len(got) >= len(want):
        break
samples = [(b, *want[fn]) for fn, b in got.items()]
random.shuffle(samples)
emb = encode([b for b, _, _ in samples])
np.savez(os.path.join(HERE, "organ_train_cc.npz"), emb=emb,
         organs=np.array([o for _, o, _ in samples]),
         species=np.array([s for _, _, s in samples]))
print("saved organ_train_cc.npz", emb.shape, flush=True)
