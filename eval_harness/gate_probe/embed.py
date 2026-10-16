"""
植物/非植物二元 gate 的特征提取：用线上同款 BioCLIP 2 + preprocess 把各数据源编码成 L2 归一 768 维向量。
每个源存一个 feats/<name>.npy，已存在则跳过（可断点续跑）。

用法: python embed.py [源名 ...]   (缺省=全部)
"""
import sys, time, io, itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import open_clip
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "feats"
WEIGHTS = ROOT / "data/weights/bioclip2/open_clip_model.safetensors"
PLANT_DIR = ROOT / "data/test_images"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BS = 64

# ImageNet-1k 里的植物/蔬果/真菌/自然景观类（936-980 & 984-998）整段剔掉，避免把植物当负样本
IMNET_EXCLUDE = set(range(936, 981)) | set(range(984, 999))
KW_EXCLUDE = ("forest", "garden", "field", "jungle", "orchard", "greenhouse", "vineyard", "wood", "tree",
              "plant", "flower", "farm", "meadow", "lawn", "park", "grass", "crop", "rice_field", "swamp",
              "bamboo", "yard", "hedge", "lotus", "lily", "sunflower", "leaf", "terrace", "wetland", "rainforest",
              "bog", "marsh", "cemetery", "campsite", "pasture", "arbor", "canopy", "botanic", "nursery",
              "vegetable", "fruit", "mountain", "island", "tundra", "stream", "river", "lake", "beach")


def _names(ds, col="label"):
    f = ds.features.get(col)
    return getattr(f, "names", None)


def hf(repo, n, group, split="train", col="image", labelcol=None, exclude=None, config=None, seed=0, buf=2000, skip_fn=None):
    def gen():
        from datasets import load_dataset
        ds = load_dataset(repo, config, split=split, streaming=True)
        names = _names(ds, labelcol) if labelcol else None
        ds = ds.shuffle(seed=seed, buffer_size=buf)
        k = 0
        for ex in ds:
            if labelcol is not None:
                y = ex[labelcol]
                if exclude is not None and exclude(y, names):
                    continue
            if skip_fn and skip_fn(ex):
                continue
            im = ex[col]
            if not isinstance(im, Image.Image):
                continue
            yield im
            k += 1
            if k >= n:
                break
    return dict(gen=gen, n=n, group=group)


def kw_exclude(y, names):
    if names is None: return False
    nm = names[y].lower()
    return any(k in nm for k in KW_EXCLUDE)


def plants():
    def gen():
        rng = np.random.RandomState(0)
        dirs = sorted(p for p in PLANT_DIR.iterdir() if p.is_dir())
        for d in dirs:
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    try:
                        yield Image.open(f)
                    except Exception:
                        pass
    return dict(gen=gen, n=10**9, group="plant")


SOURCES = {
    "plant":        plants(),
    # ── 通用物体 / 动物 / 车辆（非植物里最大的一块）
    "cifar10":      hf("uoft-cs/cifar10", 2000, "object", col="img"),
    "imagenet":     hf("visual-layer/imagenet-1k-vl-enriched", 4000, "object", labelcol="label",
                       exclude=lambda y, n: y in IMNET_EXCLUDE),
    "coco":         hf("detection-datasets/coco", 3000, "object"),
    "caltech101":   hf("Multimodal-Fatima/Caltech101_not_background_train", 1500, "object", labelcol="label",
                       exclude=lambda y, n: n is not None and any(k in n[y].lower() for k in ("lotus", "lily", "sunflower", "ferry", "wild_cat")) ),
    "cars":         hf("tanganke/stanford_cars", 1500, "object", labelcol="label"),
    # ── 场景 / 纹理 / 航拍（剔掉带植被的类）
    "sun397":       hf("tanganke/sun397", 2500, "scene", labelcol="label", exclude=kw_exclude),
    "resisc45":     hf("tanganke/resisc45", 1000, "scene", labelcol="label", exclude=kw_exclude),
    "dtd":          hf("tanganke/dtd", 1500, "texture"),
    # ── 截图 / UI / 文档（用户最可能误传的一类）
    "screenspot":   hf("rootsautomation/ScreenSpot", 1300, "screenshot", split="test"),
    "wave_ui":      hf("agentsea/wave-ui", 2000, "screenshot"),
    "rico":         hf("Voxel51/rico", 1500, "screenshot"),
    "websight":     hf("HuggingFaceM4/WebSight", 1500, "screenshot"),
    "funsd":        hf("nielsr/funsd", 300, "document"),
}


def prep(preprocess, im):
    try:
        im.load()
        return preprocess(im.convert("RGB"))
    except Exception:
        return None


@torch.no_grad()
def run(name, model, preprocess, pool):
    path = OUT / f"{name}.npy"
    if path.exists():
        print(f"[skip] {name}", flush=True); return
    src = SOURCES[name]
    t0 = time.time()
    feats = []
    it = itertools.islice(src["gen"](), src["n"])
    total = 0
    while True:
        batch = list(itertools.islice(it, BS))
        if not batch: break
        xs = [x for x in pool.map(lambda im: prep(preprocess, im), batch) if x is not None]
        if not xs: continue
        x = torch.stack(xs).to(DEVICE)
        e = model.encode_image(x).float()
        e = e / e.norm(dim=-1, keepdim=True)
        feats.append(e.cpu().numpy())
        total += len(xs)
        if (total // BS) % 10 == 0:
            print(f"  {name}: {total} ({total/(time.time()-t0):.1f} img/s)", flush=True)
            np.save(OUT / f"{name}.partial.npy", np.concatenate(feats))
    arr = np.concatenate(feats).astype(np.float32)
    np.save(path, arr)
    (OUT / f"{name}.partial.npy").unlink(missing_ok=True)
    print(f"[done] {name}: {arr.shape} in {time.time()-t0:.0f}s", flush=True)


def main():
    names = sys.argv[1:] or [k for k in SOURCES if k != "plant"] + ["plant"]
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=str(WEIGHTS))
    model = model.to(DEVICE).eval()
    pool = ThreadPoolExecutor(6)
    for n in names:
        try:
            run(n, model, preprocess, pool)
        except Exception as e:
            print(f"[FAIL] {n}: {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
