"""
配图推荐 workflow sniff test —— 现成 BioCLIP 2，CPU。
Test A: zero-shot 部位分类（在 data/test_images 随机抽样上，无标注，肉眼看分布）
Test B: 物种+部位 检索（pool 一个属的图，按部位 query 排序，同时看物种原型分）
"""
import os, sys, random, glob, collections
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from PIL import Image
import torch
import open_clip

ROOT = "/Users/avalanchy/Documents/GDS/NO/BioCLIP"
WEIGHTS = f"{ROOT}/demo/weights/open_clip_model.safetensors"
TEST_IMAGES = f"{ROOT}/data/test_images"
DEVICE = "cpu"
random.seed(0)

print("加载模型…", flush=True)
model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=WEIGHTS)
model = model.to(DEVICE).eval()
tok = open_clip.get_tokenizer("ViT-L-14")
LOGIT_SCALE = model.logit_scale.exp().item()
print(f"✓ logit_scale={LOGIT_SCALE:.2f}", flush=True)


@torch.no_grad()
def enc_text(prompts):
    t = tok(prompts).to(DEVICE)
    e = model.encode_text(t).float()
    e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().numpy()


@torch.no_grad()
def enc_images(paths, bs=16):
    out = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            try:
                batch.append(preprocess(Image.open(p).convert("RGB")))
            except Exception:
                batch.append(torch.zeros(3, 224, 224))
        x = torch.stack(batch).to(DEVICE)
        e = model.encode_image(x).float()
        e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.cpu().numpy())
        print(f"  encoded {min(i+bs,len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return np.concatenate(out, 0)


def proto(prompts):
    """一组 prompt 取均值再归一化 = 一个原型向量。"""
    e = enc_text(prompts).mean(0)
    return e / np.linalg.norm(e)


ORGAN_PROMPTS = {
    "leaf":   ["a photo of a leaf", "a close-up photo of the leaves of a plant", "plant foliage, leaves"],
    "flower": ["a photo of a flower", "a close-up photo of a flower", "a plant in bloom with flowers"],
    "fruit":  ["a photo of a fruit", "a close-up photo of berries and fruit", "the fruit and seeds of a plant"],
    "bark":   ["a photo of tree bark", "a close-up photo of the bark of a tree trunk", "the trunk of a tree"],
    "habit":  ["a photo of a whole plant", "a photo of a whole tree outdoors", "a shrub growing in a landscape"],
}
ORGANS = list(ORGAN_PROMPTS)
ORGAN_PROTOS = np.stack([proto(ORGAN_PROMPTS[o]) for o in ORGANS])  # [5, 768]


# ── Test A: 随机抽样部位分桶 ──────────────────────────────────────
print("\n=== Test A: zero-shot 部位分类（随机 200 张 iNat 研究级图）===")
all_species = sorted(glob.glob(f"{TEST_IMAGES}/*/"))
random.shuffle(all_species)
sample_paths = []
for sp in all_species:
    imgs = glob.glob(sp + "*.jpg")
    if imgs:
        sample_paths.append(random.choice(imgs))
    if len(sample_paths) >= 200:
        break

emb = enc_images(sample_paths)
scores = emb @ ORGAN_PROTOS.T            # [200, 5]
pred = scores.argmax(1)
dist = collections.Counter(ORGANS[i] for i in pred)
print("分桶分布:", dict(dist))
margin = np.sort(scores, 1)[:, -1] - np.sort(scores, 1)[:, -2]
print(f"top1-top2 margin: mean={margin.mean():.4f}  median={np.median(margin):.4f}  "
      f"<0.01 占比={np.mean(margin < 0.01):.2%}")
for oi, o in enumerate(ORGANS):
    ex = [os.path.basename(sample_paths[j]) for j in np.where(pred == oi)[0][:4]]
    print(f"  [{o:6}] {len(np.where(pred==oi)[0]):3d} 张  例: {ex}")


# ── Test B: 物种 + 部位 检索 ─────────────────────────────────────
print("\n=== Test B: 物种+部位 检索（pool 一个属的所有图）===")
GENUS = "Acer"
genus_dirs = [d for d in all_species if f"_{GENUS} " in d or d.rstrip("/").endswith(f"_{GENUS}")]
pool_paths, pool_species = [], []
for d in genus_dirs:
    name = os.path.basename(d.rstrip("/")).split("_", 1)[1]
    for p in glob.glob(d + "*.jpg"):
        pool_paths.append(p); pool_species.append(name)
print(f"{GENUS} 属: {len(genus_dirs)} 物种, {len(pool_paths)} 张图")
if len(pool_paths) < 8:
    print("池子太小，跳过 Test B"); sys.exit(0)

pool_emb = enc_images(pool_paths)
TARGET = pool_species[0] if "Acer rubrum" not in pool_species else "Acer rubrum"
common = {"Acer rubrum": "red maple", "Acer saccharum": "sugar maple"}.get(TARGET, TARGET)
sp_proto = proto([f"a photo of {TARGET}, a species of plant", f"a photo of {common}"])
sp_score = pool_emb @ sp_proto
print(f"目标物种: {TARGET}  (池中该种 {pool_species.count(TARGET)} 张)")

for part in ORGANS:
    q = proto([
        f"a close-up photo of the {part} of {TARGET}",
        f"the {part} of {common}",
        f"{TARGET} {part}",
    ])
    ps = pool_emb @ q
    order = np.argsort(ps)[::-1][:3]
    print(f"\n  部位=[{part}] top3:")
    for r in order:
        hit = "✓" if pool_species[r] == TARGET else " "
        print(f"    {hit} part={ps[r]:.4f}  sp={sp_score[r]:.4f}  "
              f"{pool_species[r][:22]:22}  {os.path.basename(pool_paths[r])}")
