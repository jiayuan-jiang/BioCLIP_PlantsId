"""Pl@ntNet-300K train(+val) split → 抽 organ 训练集，BioCLIP 编码 → organ_train.npz。
v2: 先从 metadata 【随机】选目标 file_name（避免按分片顺序取前 N 造成物种偏置），再扫分片只留这些。
bark 全要，其余类随机 capped。"""
import os, io, csv, random, collections, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from PIL import Image
import torch, open_clip
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfFileSystem

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "organ_train.npz")
W = f"{ROOT}/demo/weights/open_clip_model.safetensors"
random.seed(0)

CAP = {"leaf": 2500, "flower": 2500, "fruit": 2500, "habit": 3000, "bark": None}  # None = 全要
REPO = "mikehemberger/plantnet300K"
fs = HfFileSystem()

# 全量 (file_name -> organ, species)，train + val
recs = []
for split in ["train", "val"]:
    m = hf_hub_download(REPO, f"data/{split}/metadata.csv", repo_type="dataset")
    for r in csv.DictReader(open(m)):
        recs.append((r["file_name"], r["organ"], r["species"]))
by_organ = collections.defaultdict(list)
for fn, org, sp in recs:
    by_organ[org].append((fn, sp))
print("全量 organ 计数:", {k: len(v) for k, v in sorted(by_organ.items())}, flush=True)

# 随机选目标
want = {}   # file_name -> (organ, species)
for org, cap in CAP.items():
    lst = by_organ[org][:]
    random.shuffle(lst)
    pick = lst if cap is None else lst[:cap]
    for fn, sp in pick:
        want[fn] = (org, sp)
print("目标子集:", collections.Counter(o for o, _ in want.values()), "总", len(want), flush=True)

shards = sorted(fs.glob(f"datasets/{REPO}/data/train-*.parquet")) + \
         sorted(fs.glob(f"datasets/{REPO}/data/validation-*.parquet"))
shards = [s.split(f"datasets/{REPO}/")[1] for s in shards]

got = {}  # file_name -> bytes
for si, sh in enumerate(shards):
    p = hf_hub_download(REPO, sh, repo_type="dataset")
    pf = pq.ParquetFile(p)
    for batch in pf.iter_batches(batch_size=256):
        for im in batch.to_pydict()["image"]:
            if im["path"] in want and im["bytes"] and im["path"] not in got:
                got[im["path"]] = im["bytes"]
    print(f"  [{si+1}/{len(shards)}] got {len(got)}/{len(want)}", flush=True)
    if len(got) >= len(want):
        break

samples = [(b, *want[fn]) for fn, b in got.items()]   # (bytes, organ, species)
random.shuffle(samples)
print("命中:", collections.Counter(o for _, o, _ in samples), "总", len(samples), flush=True)

print("加载模型…", flush=True)
model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=W)
model.eval()

@torch.no_grad()
def encode(byte_list, bs=32):
    out, t0 = [], time.time()
    for i in range(0, len(byte_list), bs):
        batch = []
        for b in byte_list[i:i + bs]:
            try:
                batch.append(preprocess(Image.open(io.BytesIO(b)).convert("RGB")))
            except Exception:
                batch.append(torch.zeros(3, 224, 224))
        e = model.encode_image(torch.stack(batch)).float()
        out.append((e / e.norm(dim=-1, keepdim=True)).numpy())
        if i % (bs * 20) == 0:
            el = time.time() - t0; done = i + len(batch)
            print(f"  {done}/{len(byte_list)}  {el:.0f}s  eta {el/max(done,1)*(len(byte_list)-done):.0f}s", flush=True)
    return np.concatenate(out, 0)

emb = encode([b for b, _, _ in samples])
np.savez(OUT, emb=emb,
         organs=np.array([o for _, o, _ in samples]),
         species=np.array([s for _, _, s in samples]))
print("saved", OUT, emb.shape, flush=True)
