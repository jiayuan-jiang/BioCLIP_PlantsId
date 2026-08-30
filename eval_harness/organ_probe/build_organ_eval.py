"""下载 Pl@ntNet-300K test split（8 parquet），按 organ 抽一个较均衡的子集，
用现成 BioCLIP 2 编码图像，存 embeddings + organ 标签到 npz。"""
import os, io, csv, random, collections, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from PIL import Image
import torch, open_clip
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "organ_eval.npz")
W = f"{ROOT}/demo/weights/open_clip_model.safetensors"
random.seed(0)

KEEP = {"leaf", "flower", "fruit", "bark", "habit"}
CAP = {"leaf": 1500, "flower": 1500, "fruit": 1176, "bark": 225, "habit": 450}

print("meta…", flush=True)
meta_path = hf_hub_download("mikehemberger/plantnet300K", "data/test/metadata.csv", repo_type="dataset")
fn2organ = {}
for r in csv.DictReader(open(meta_path)):
    fn2organ[r["file_name"]] = r["organ"]

TEST_FILES = [
    "data/test-00000-of-00008-ccdf9a8395f11a02.parquet",
    "data/test-00001-of-00008-4cc7b1523923ba5e.parquet",
    "data/test-00002-of-00008-952e223123a49903.parquet",
    "data/test-00003-of-00008-4e3b69c54dfa5948.parquet",
    "data/test-00004-of-00008-6882acfdcab0bbdd.parquet",
    "data/test-00005-of-00008-f6f2fd539fae591e.parquet",
    "data/test-00006-of-00008-5932eb3f032283e1.parquet",
    "data/test-00007-of-00008-8caa262cecd65437.parquet",
]

# 收集 (bytes, organ)
buckets = collections.defaultdict(list)
for tf in TEST_FILES:
    p = hf_hub_download("mikehemberger/plantnet300K", tf, repo_type="dataset")
    pf = pq.ParquetFile(p)
    for batch in pf.iter_batches(batch_size=256):
        d = batch.to_pydict()
        for im in d["image"]:
            org = fn2organ.get(im["path"])
            if org in KEEP and im["bytes"]:
                buckets[org].append(im["bytes"])
    print(f"  {tf.split('/')[-1]}  ", {k: len(v) for k, v in buckets.items()}, flush=True)

samples = []  # (bytes, organ)
for org, lst in buckets.items():
    random.shuffle(lst)
    for b in lst[:CAP[org]]:
        samples.append((b, org))
random.shuffle(samples)
print("最终子集:", dict(collections.Counter(o for _, o in samples)), "总", len(samples), flush=True)

print("加载模型…", flush=True)
model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=W)
model.eval()

@torch.no_grad()
def encode(byte_list, bs=32):
    out = []
    t0 = time.time()
    for i in range(0, len(byte_list), bs):
        batch = []
        for b in byte_list[i:i + bs]:
            try:
                batch.append(preprocess(Image.open(io.BytesIO(b)).convert("RGB")))
            except Exception:
                batch.append(torch.zeros(3, 224, 224))
        e = model.encode_image(torch.stack(batch)).float()
        e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.numpy())
        if i % (bs * 10) == 0:
            el = time.time() - t0
            print(f"  {i+len(batch)}/{len(byte_list)}  {el:.0f}s  eta {el/max(i+len(batch),1)*(len(byte_list)-i):.0f}s", flush=True)
    return np.concatenate(out, 0)

emb = encode([b for b, _ in samples])
organs = np.array([o for _, o in samples])
np.savez(OUT, emb=emb, organs=organs)
print("saved", OUT, emb.shape, flush=True)
