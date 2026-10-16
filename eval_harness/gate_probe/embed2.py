"""
v2 数据补充：(1) 新植物源 plantnet300K(Pl@ntNet 用户拍摄，含花/叶/果/树皮/整株) + 人工筛过的 stem 图；
(2) 多视图增强：对已有源的子集编码 cc60(中心裁 60%) 与 rrc(随机裁 10–50% 面积) 两个裁切视图，训练时当额外样本。
输出 feats2/<源>.npz：每个视图一个数组 + idx(对应 v1 feats 里的行号，同一张图的各视图共用 idx → CV 分组不泄漏)。
"""
import sys, io, time, glob, itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, open_clip
from PIL import Image
import torchvision.transforms as T

import embed as E1   # 复用 v1 的 SOURCES / 模型常量

HERE = Path(__file__).parent
OUT = HERE / "feats2"; OUT.mkdir(exist_ok=True)
ROOT = E1.ROOT
DEV = E1.DEVICE
STEM_DIR = ROOT / "eval_harness/organ_probe/stem_images_used"
PLANTNET = sorted(glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--mikehemberger--plantnet300K/snapshots/*/data/train-*.parquet")))
NEG_FRAC = 0.5      # 每个负样本源前 50% 做裁切视图
PLANT_STEP = 3      # 已有植物每 3 张取 1 张做裁切视图


def cc60(im):
    w, h = im.size
    cw, ch = int(w * .6), int(h * .6)
    l, t = (w - cw) // 2, (h - ch) // 2
    return im.crop((l, t, l + cw, t + ch))


def make_rrc(seed):
    rng = np.random.RandomState(seed)
    def f(im):
        w, h = im.size
        for _ in range(10):
            a = rng.uniform(.10, .50) * w * h
            r = np.exp(rng.uniform(np.log(.6), np.log(1.6)))
            cw, ch = int(round((a * r) ** .5)), int(round((a / r) ** .5))
            if 8 <= cw <= w and 8 <= ch <= h:
                l, t = rng.randint(0, w - cw + 1), rng.randint(0, h - ch + 1)
                return im.crop((l, t, l + cw, t + ch))
        return cc60(im)
    return f


def plantnet_gen(n_per_shard=200):
    rng = np.random.RandomState(1)
    import pyarrow.parquet as pq
    for f in PLANTNET:
        t = pq.read_table(f, columns=["image"])
        pick = rng.choice(t.num_rows, min(n_per_shard, t.num_rows), replace=False)
        col = t.column("image")
        for i in sorted(pick):
            d = col[int(i)].as_py()
            try:
                yield Image.open(io.BytesIO(d["bytes"]))
            except Exception:
                pass


def stem_gen():
    for f in sorted(STEM_DIR.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            try: yield Image.open(f)
            except Exception: pass


def plant_sub_gen():
    for i, im in enumerate(E1.SOURCES["plant"]["gen"]()):
        if i % PLANT_STEP == 0:
            yield im


@torch.no_grad()
def encode(model, xs):
    e = model.encode_image(torch.stack(xs).to(DEV)).float()
    return (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()


def run(name, gen, views, model, pre, pool, n=None, idx=None):
    path = OUT / f"{name}.npz"
    if path.exists():
        print("[skip]", name, flush=True); return
    t0 = time.time()
    store = {v: [] for v in views}
    rrc = make_rrc(abs(hash(name)) % 1000)
    fns = {"full": lambda im: im, "cc60": cc60, "rrc": rrc}
    it = gen if n is None else itertools.islice(gen, n)
    BATCH = 24
    tot = 0
    while True:
        batch = list(itertools.islice(it, BATCH))
        if not batch: break
        ims = []
        for im in batch:
            try: im.load(); ims.append(im.convert("RGB"))
            except Exception: ims.append(None)
        # rrc 的随机状态必须在主线程顺序消费，保证可复现
        crops = {v: [fns[v](im) if im is not None else None for im in ims] for v in views}
        for v in views:
            xs = list(pool.map(lambda im: pre(im) if im is not None else None, crops[v]))
            ok = [x for x in xs if x is not None]
            store[v].append(encode(model, ok) if len(ok) == len(xs) else None)
        tot += len(batch)
        if (tot // BATCH) % 10 == 0:
            print(f"  {name}: {tot} ({tot/(time.time()-t0):.1f} img/s)", flush=True)
    out = {}
    for v in views:
        if any(s is None for s in store[v]):
            raise RuntimeError(f"{name}: 有坏图，idx 对齐会错位")  # 极少见；v1 也跳了坏图，此处保守报错
        out[v] = np.concatenate(store[v]).astype(np.float32)
    out["idx"] = np.arange(len(out[views[0]])) if idx is None else np.asarray(idx)
    np.savez(path, **out)
    print(f"[done] {name}: {len(out['idx'])} in {time.time()-t0:.0f}s", flush=True)


def main():
    model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained=str(E1.WEIGHTS))
    model = model.to(DEV).eval()
    pool = ThreadPoolExecutor(6)
    # 1) 新植物源（full + 两个裁切）
    run("plantnet", plantnet_gen(), ["full", "cc60", "rrc"], model, pre, pool, n=10200)
    run("stem", stem_gen(), ["full", "cc60", "rrc"], model, pre, pool)
    # 2) 已有源子集的裁切视图
    n_plant = 25000 // PLANT_STEP + 1
    run("plant_crops", plant_sub_gen(), ["cc60", "rrc"], model, pre, pool, idx=np.arange(0, 25000, PLANT_STEP)[:n_plant])
    for s, spec in E1.SOURCES.items():
        if s == "plant" or not (E1.OUT / f"{s}.npy").exists(): continue
        k = int(len(np.load(E1.OUT / f"{s}.npy", mmap_mode="r")) * NEG_FRAC)
        run(s + "_crops", spec["gen"](), ["cc60", "rrc"], model, pre, pool, n=k)


if __name__ == "__main__":
    main()
