"""把人工标注的 stem（tmp/stem_review/stem_labels.json）编码后并入 organ_train_v3 / organ_eval_v3。

只编码这批新增的，不碰已经编好的 leaf/flower/fruit/bark —— 避免重跑 BarkNet 那 15min。
80/20 分层切，追加到现有 organ_train_v3.npz / organ_eval_v3.npz。

用法：python add_stem_v3.py
"""
import os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
W = os.path.join(ROOT, "demo", "weights", "open_clip_model.safetensors")
IMG_DIR = os.path.join(HERE, "_v3_cache", "raw", "plantclef_stem")
LABELS = os.path.join(ROOT, "tmp", "stem_review", "stem_labels.json")
TRAIN_NPZ = os.path.join(HERE, "organ_train_v3.npz")
EVAL_NPZ = os.path.join(HERE, "organ_eval_v3.npz")
EVAL_FRAC = 0.20
random_seed = 0

if "stem" in set(np.load(TRAIN_NPZ, allow_pickle=True)["organs"].astype(str)):
    raise SystemExit("organ_train_v3.npz 里已经有 stem 了，别重复跑（会重复 append）。"
                      "要重来先从 git / 备份恢复 organ_*_v3.npz。")

files = json.load(open(LABELS))["selected"]
print(f"人工标的 stem：{len(files)} 张", flush=True)
paths = [os.path.join(IMG_DIR, f) for f in files]

import torch, open_clip
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_num_threads(4)
print("加载 open_clip ViT-L-14 …", flush=True)
model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=W)
model.eval()


def _load(p):
    im = Image.open(p)
    im.draft("RGB", (640, 640))
    im = im.convert("RGB")
    im.thumbnail((640, 640), Image.BILINEAR)
    return preprocess(im)


@torch.no_grad()
def encode(paths, bs=16):
    embs, t0 = [], time.time()
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            try:
                batch.append(_load(p))
            except Exception as e:
                print(f"  !! 跳过 {p}: {e}", flush=True)
                batch.append(torch.zeros(3, 224, 224))
        e = model.encode_image(torch.stack(batch)).float()
        embs.append((e / e.norm(dim=-1, keepdim=True)).numpy())
        done = i + len(batch)
        if (i // bs) % 10 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(paths)}  {el:.0f}s  eta {el/max(done,1)*(len(paths)-done):.0f}s", flush=True)
    return np.concatenate(embs, 0)


emb = encode(paths)
org = np.array(["stem"] * len(emb))
src = np.array(["plantclef_manual"] * len(emb))
sp = np.array([""] * len(emb))

rng = np.random.default_rng(random_seed)
idx = np.arange(len(emb)); rng.shuffle(idx)
n_ev = max(1, int(len(idx) * EVAL_FRAC))
ev_idx, tr_idx = idx[:n_ev], idx[n_ev:]

for tag, sel, path in [("train", tr_idx, TRAIN_NPZ), ("eval", ev_idx, EVAL_NPZ)]:
    old = np.load(path, allow_pickle=True)
    merged = dict(
        emb=np.concatenate([old["emb"], emb[sel]]),
        organs=np.concatenate([old["organs"], org[sel]]),
        source=np.concatenate([old["source"], src[sel]]),
        species=np.concatenate([old["species"], sp[sel]]),
    )
    np.savez(path, **merged)
    import collections
    print(f"{tag:5} → {os.path.basename(path)}  now {len(merged['emb'])} 行  "
          f"{dict(sorted(collections.Counter(merged['organs'].tolist()).items()))}", flush=True)

print("\n下一步：python organ_tagger_v3.py 里把 CLS 加回 stem（5类），重训 5 头。")
