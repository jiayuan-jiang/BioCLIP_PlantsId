"""S0 — Stage 1 前的零样本评测 harness。

对一个 open_clip 模型（BioCLIP 2 / DALIP ckpt / 我们的 BioCLIP2-Plant）在:
  植物基准  : plantnet300k, rare_species (+ data_root/extra/* 通用 imagefolder)
  通用留存  : imagenet1k, nabirds
上做零样本分类，产出 scoreboard.csv（一行一个 model×benchmark）。

用法（pod 上）:
  python -m eval_harness.zeroshot_eval \
      --model hf-hub:imageomics/bioclip-2 \
      --data-root /workspace/eval_data \
      --out eval_harness/scoreboard.csv

  # 本地 checkpoint:
  python -m eval_harness.zeroshot_eval --arch ViT-L-14 \
      --pretrained /path/open_clip_pytorch_model.bin --model-id bioclip2-plant-r32 ...

  # CPU 冒烟:
  python -m eval_harness.zeroshot_eval --limit 64 --batch-size 8 --workers 0 ...
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import benchmarks as B
from . import prompts as P
from .metrics import expected_calibration_error, macro_scores, softmax, topk_acc

try:
    import open_clip
except ImportError as e:  # pragma: no cover
    raise SystemExit("需要 open_clip_torch：pip install open_clip_torch") from e


# ─────────────────────────────────────────────
# 模型
# ─────────────────────────────────────────────

def load_model(model: str, arch: str | None, pretrained: str | None, device: str):
    if pretrained:
        if not arch:
            raise SystemExit("--pretrained 需同时指定 --arch（如 ViT-L-14）")
        m, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=device
        )
        tok = open_clip.get_tokenizer(arch)
        model_id = arch
    else:
        m, _, preprocess = open_clip.create_model_and_transforms(model, device=device)
        tok = open_clip.get_tokenizer(model)
        model_id = model
    m = m.eval().to(device)
    return m, preprocess, tok, model_id


_TEXT_CACHE = Path("eval_harness/.text_cache")


def _text_cache_key(model_id: str, bench: B.Benchmark) -> Path:
    import hashlib
    payload = "\x1f".join(
        P.render_class(n, bench.template_set, (bench.common_names or [None] * len(bench.classnames))[i])[0]
        if False else f"{n}|{(bench.common_names or [None]*len(bench.classnames))[i]}"
        for i, n in enumerate(bench.classnames)
    ) + f"||{bench.template_set}"
    h = hashlib.sha1(payload.encode()).hexdigest()[:16]
    safe = model_id.replace("/", "_").replace(":", "_")
    return _TEXT_CACHE / f"{safe}__{bench.name}__{h}.npy"


@torch.no_grad()
def build_text_classifier(model, tokenizer, bench: B.Benchmark, device: str,
                          batch: int = 256, model_id: str = "model",
                          use_cache: bool = True) -> torch.Tensor:
    """返回 [n_classes, dim] 的 L2 归一化文本原型矩阵。按 (模型, 基准, 类名+模板) 缓存。"""
    ck = _text_cache_key(model_id, bench)
    if use_cache and ck.is_file():
        print(f"    文本原型命中缓存 {ck.name}")
        return torch.from_numpy(np.load(ck))

    protos = []
    commons = bench.common_names or [None] * len(bench.classnames)
    n = len(bench.classnames)
    for start in range(0, n, batch):
        chunk_names = bench.classnames[start:start + batch]
        chunk_common = commons[start:start + batch]
        flat, spans = [], []
        for nm, cm in zip(chunk_names, chunk_common):
            ps = P.render_class(nm, bench.template_set, cm)
            spans.append((len(flat), len(flat) + len(ps)))
            flat.extend(ps)
        tokens = tokenizer(flat).to(device)
        feats = model.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        for a, b in spans:
            v = feats[a:b].mean(dim=0)
            protos.append((v / v.norm()).cpu())
        if start and start % (batch * 5) == 0:
            print(f"    文本编码 {start}/{n} 类", flush=True)
    W = torch.stack(protos)  # [C, D]
    if use_cache:
        _TEXT_CACHE.mkdir(parents=True, exist_ok=True)
        np.save(ck, W.numpy())
    return W


# ─────────────────────────────────────────────
# 图像特征
# ─────────────────────────────────────────────

class ImgDS(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            return self.preprocess(img), i, 1
        except Exception:  # noqa: BLE001  损坏图 -> 占位，评测时丢弃
            return torch.zeros(3, 224, 224), i, 0


@torch.no_grad()
def encode_images(model, preprocess, paths, device, batch_size, workers):
    ds = ImgDS(paths, preprocess)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers,
                    pin_memory=(device == "cuda"))
    feats = torch.zeros(len(paths), model.visual.output_dim if hasattr(model.visual, "output_dim") else 512)
    ok = np.zeros(len(paths), dtype=bool)
    dim_set = False
    done = 0
    t0 = time.time()
    for imgs, idxs, valid in dl:
        imgs = imgs.to(device, non_blocking=True)
        f = model.encode_image(imgs).float()
        f = f / f.norm(dim=-1, keepdim=True)
        if not dim_set:
            feats = torch.zeros(len(paths), f.shape[1])
            dim_set = True
        feats[idxs] = f.cpu()
        ok[idxs.numpy()] = valid.numpy().astype(bool)
        done += len(idxs)
        if done % (batch_size * 20) < batch_size:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"    {done}/{len(paths)}  ({rate:.0f} img/s)", flush=True)
    return feats, ok


# ─────────────────────────────────────────────
# 单基准评测
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, preprocess, tokenizer, bench: B.Benchmark, device, args) -> dict:
    print(f"\n=== {bench.name} ({bench.kind}) — {len(bench.image_paths)} imgs / {len(bench.classnames)} cls ===")
    text_w = build_text_classifier(model, tokenizer, bench, device,
                                   model_id=args.model_id,
                                   use_cache=not args.no_text_cache)          # [C, D]
    img_f, ok = encode_images(model, preprocess, bench.image_paths, device,
                              args.batch_size, args.workers)
    if (~ok).any():
        print(f"    丢弃损坏图 {int((~ok).sum())} 张")
    img_f, labels = img_f[ok], bench.labels[ok]

    logit_scale = float(model.logit_scale.exp().detach().cpu()) if hasattr(model, "logit_scale") else 100.0
    logits = (logit_scale * img_f @ text_w.cpu().T).numpy().astype(np.float32)   # [N, C]
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = len(bench.classnames)

    tk = topk_acc(logits, labels, ks=(1, 5))
    top5 = tk.get("top5", tk.get(f"top{min(5, n_classes)}", tk["top1"]))
    pred = logits.argmax(axis=1)
    mac = macro_scores(pred, labels, n_classes)
    ece = expected_calibration_error(softmax(logits), labels)

    if args.per_class_dir:
        pcd = Path(args.per_class_dir)
        pcd.mkdir(parents=True, exist_ok=True)
        safe = args.model_id.replace("/", "_").replace(":", "_")
        with open(pcd / f"{safe}__{bench.name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_idx", "classname", "n", "recall"])
            for c in range(n_classes):
                m = labels == c
                w.writerow([c, bench.classnames[c], int(m.sum()),
                            "" if np.isnan(mac["_per_class_recall"][c]) else f"{mac['_per_class_recall'][c]:.4f}"])

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": args.model_id,
        "benchmark": bench.name,
        "kind": bench.kind,
        "n_images": int(len(labels)),
        "n_classes": n_classes,
        "top1": round(tk["top1"], 4),
        "top5": round(top5, 4),
        "macro_recall": round(mac["macro_recall"], 4),
        "macro_f1": round(mac["macro_f1"], 4),
        "ece": round(ece, 4),
        "templates": bench.template_set,
        "notes": bench.notes,
    }
    print(f"    top1={row['top1']}  top5={row['top5']}  macro_recall={row['macro_recall']}  "
          f"macro_f1={row['macro_f1']}  ece={row['ece']}")
    return row


# ─────────────────────────────────────────────
# scoreboard
# ─────────────────────────────────────────────

FIELDS = ["timestamp", "model_id", "benchmark", "kind", "n_images", "n_classes",
          "top1", "top5", "macro_recall", "macro_f1", "ece", "templates", "notes"]


def append_scoreboard(path: str, rows: list[dict]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n✓ 追加 {len(rows)} 行 -> {p}")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hf-hub:imageomics/bioclip-2",
                    help="open_clip 模型名 / hf-hub:org/repo")
    ap.add_argument("--arch", default=None, help="配 --pretrained 用，如 ViT-L-14")
    ap.add_argument("--pretrained", default=None, help="本地权重 .bin/.safetensors")
    ap.add_argument("--model-id", default=None, help="scoreboard 里记录的名字，默认取 --model/--arch")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--benchmarks", default="all",
                    help="all 或逗号分隔：plantnet300k,rare_species,imagenet1k,nabirds")
    ap.add_argument("--out", default="eval_harness/scoreboard.csv")
    ap.add_argument("--per-class-dir", default="eval_harness/per_class")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help=">0 时每个基准随机抽样这么多张（冒烟用）")
    ap.add_argument("--no-text-cache", action="store_true", help="不使用文本原型缓存")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "mps":
        # 部分 open_clip 算子在 MPS 上可能缺实现，允许回退到 CPU
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    want = None if args.benchmarks == "all" else [x.strip() for x in args.benchmarks.split(",")]
    print(f"发现基准  data_root={args.data_root}")
    benches = B.discover(args.data_root, want)
    if not benches:
        raise SystemExit("未发现任何基准数据，检查 --data-root 布局（见 benchmarks.py 顶部注释）")
    if args.limit:
        benches = [b.subsample(args.limit) for b in benches]

    print(f"\n加载模型  {args.model if not args.pretrained else args.pretrained}  device={device}")
    model, preprocess, tokenizer, model_id = load_model(
        args.model, args.arch, args.pretrained, device)
    args.model_id = args.model_id or model_id
    print(f"  model_id = {args.model_id}")

    rows = [evaluate(model, preprocess, tokenizer, b, device, args) for b in benches]

    # 汇总
    plant = [r for r in rows if r["kind"] == "plant"]
    gen = [r for r in rows if r["kind"] == "general"]
    if plant:
        print(f"\n植物基准 top1 均值 : {np.mean([r['top1'] for r in plant]):.4f}  "
              f"macro_recall 均值 : {np.mean([r['macro_recall'] for r in plant]):.4f}")
    if gen:
        print(f"通用留存 top1 均值 : {np.mean([r['top1'] for r in gen]):.4f}")

    append_scoreboard(args.out, rows)


if __name__ == "__main__":
    main()
