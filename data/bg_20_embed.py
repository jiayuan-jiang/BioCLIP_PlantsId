"""
温室测试集图像 → BioCLIP 2 embedding (Botanical Garden pipeline · step 20)
==========================================================================
与线上 demo/inference_core.py 完全同口径：open_clip ViT-L-14 + 本地 bioclip2 权重、
模型自带 preprocess、L2 归一化。只做 embedding，评估见 bg_21_eval.py。

输入：BotanicalGarden/gh_images/manifest.csv + images/
输出：BotanicalGarden/gh_eval/emb.f16.npy   (N, 768) 已 L2 归一化
      BotanicalGarden/gh_eval/emb_files.csv (与 emb 行对齐：accepted_name, file, source, license...)
可断点续跑（分块写 chunks/）。

用法：/opt/miniconda3/bin/python bg_20_embed.py [--limit N] [--batch 64]
"""

import os
import glob
import time
import argparse

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

GH = "./BotanicalGarden/gh_images"
OUT = "./BotanicalGarden/gh_eval"
WEIGHTS = "./weights/bioclip2/open_clip_model.safetensors"
ARCH = "ViT-L-14"
CHUNK = 1024


class DS(torch.utils.data.Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.pp = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return self.pp(Image.open(self.paths[i]).convert("RGB")), i, True
        except Exception:
            return torch.zeros(3, 224, 224), i, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--species-file", default="", help="只处理该文件里列出的物种（一行一个 accepted_name）")
    ap.add_argument("--out-dir", default="", help="输出目录（默认 BotanicalGarden/gh_eval）")
    args = ap.parse_args()

    global OUT
    if args.out_dir:
        OUT = args.out_dir
    if args.limit:
        OUT = OUT + "_smoke"            # 冒烟测试不污染正式 chunks
    os.makedirs(f"{OUT}/chunks", exist_ok=True)
    m = pd.read_csv(f"{GH}/manifest.csv")
    m = m[m.source != "none"].reset_index(drop=True)
    m["path"] = [f"{GH}/images/{n.replace(' ', '_')}/{f}" for n, f in zip(m.accepted_name, m.file)]
    if args.species_file:
        want = {x.strip() for x in open(args.species_file, encoding="utf-8") if x.strip()}
        m = m[m.accepted_name.isin(want)].reset_index(drop=True)
    m = m[[os.path.exists(p) for p in m.path]].reset_index(drop=True)   # 清理时删掉的文件不再评
    if args.limit:
        m = m.sample(args.limit, random_state=0).reset_index(drop=True)
    m.drop(columns=["path"]).to_csv(f"{OUT}/emb_files.csv", index=False)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"图片 {len(m)}；device={device}", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=WEIGHTS)
    model = model.to(device).eval()

    t0 = time.time()
    for c0 in range(0, len(m), CHUNK):
        cf = f"{OUT}/chunks/{c0:06d}.npy"
        if os.path.exists(cf) and not args.limit:
            continue
        sub = m.iloc[c0:c0 + CHUNK]
        dl = torch.utils.data.DataLoader(DS(sub.path.tolist(), preprocess), batch_size=args.batch,
                                         num_workers=args.workers)
        embs = np.zeros((len(sub), 768), np.float16)
        bad = 0
        with torch.no_grad():
            for x, idx, ok in dl:
                e = model.encode_image(x.to(device))
                e = e / e.norm(dim=-1, keepdim=True)
                e = e.float().cpu().numpy()
                e[~ok.numpy()] = 0
                bad += int((~ok).sum())
                embs[idx.numpy()] = e.astype(np.float16)
        np.save(cf, embs)
        done = c0 + len(sub)
        print(f"[{done}/{len(m)}] {time.time()-t0:.0f}s  坏图 {bad}  "
              f"{done/(time.time()-t0 + 1e-9):.1f} img/s", flush=True)

    allc = [np.load(f) for f in sorted(glob.glob(f"{OUT}/chunks/*.npy"))]
    E = np.concatenate(allc)
    assert len(E) == len(m), (len(E), len(m))
    np.save(f"{OUT}/emb.f16.npy", E)
    print("完成", E.shape)


if __name__ == "__main__":
    main()
