"""Stage 1 训练 Dataset —— 读 Phase 1 预计算产物。

输入:
  manifest  : data/stage1/index.parquet  (row, shard, key, caption, binomial, family, phash, is_general)
  emb_path  : data/stage1/text_emb.f16.npy  memmap [M, 768] fp16，行号 = manifest.row
  img       : data/stage1/img/{shard}/{key}.jpg  (短边 256, JPEG q85)

yields (img_tensor[3,224,224], emb_idx:int, meta:dict)
  meta = {row, binomial, family, is_general, phash}

采样: 温度 τ（n_binomial^-τ）+ repeat-factor，折进一个 per-row 权重向量，配 WeightedRandomSampler。
增强: RandomResizedCrop(224, scale=(0.5,1.0)) + hflip 打底；field-conditions（低光/闪光/噪声/模糊/低质JPEG/cutout）中等概率命中其一。

依据: doc/stage1-spec.md §6（预处理）、§7（鲁棒性增强）、§8（长尾采样）。
"""

from __future__ import annotations

import io
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset

from .config import Stage1Config

# BioCLIP 2 = OpenAI CLIP 归一化，必须原样（文本塔冻结，图像分布一偏就废）
CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ─────────────────────────────────────────────────────────────
# field-conditions 增强（PIL 域，不依赖 kornia）
# ─────────────────────────────────────────────────────────────

def _low_light(img: Image.Image, rng: random.Random) -> Image.Image:
    f = rng.uniform(0.3, 0.7)
    img = ImageEnhance.Brightness(img).enhance(f)
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 0.95))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr ** rng.uniform(1.1, 1.6)                     # gamma up (更暗)
    # 色温蓝移
    arr[..., 2] = np.clip(arr[..., 2] * rng.uniform(1.05, 1.20), 0, 1)
    arr[..., 0] = np.clip(arr[..., 0] * rng.uniform(0.85, 0.98), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _flash(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (0.5 * math.hypot(h, w))
    glow = np.clip(1.0 - d, 0, 1)[..., None] ** 2
    arr = arr + glow * rng.uniform(60, 160)
    arr = arr * (1.0 - 0.15 * (1 - glow))                  # 边角压暗 -> 硬阴影感
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _sensor_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    # Poisson（散粒噪声）
    scale = rng.uniform(0.4, 1.0)
    arr = np.random.poisson(np.clip(arr, 0, None) * scale) / max(scale, 1e-6)
    # 叠加 Gaussian
    arr = arr + np.random.randn(*arr.shape) * rng.uniform(4, 14)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _blur(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        # defocus
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.0, 3.0)))
    # motion: 沿随机方向做 1-D box blur（近似）
    k = rng.choice([5, 7, 9, 11])
    arr = np.asarray(img).astype(np.float32)
    ker = np.zeros((k, k), np.float32)
    if rng.random() < 0.5:
        ker[k // 2, :] = 1.0 / k
    else:
        ker[:, k // 2] = 1.0 / k
    from numpy.lib.stride_tricks import sliding_window_view
    pad = k // 2
    p = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    for c in range(3):
        win = sliding_window_view(p[..., c], (k, k))
        out[..., c] = (win * ker).sum(axis=(-1, -2))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _jpeg(img: Image.Image, rng: random.Random) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=rng.randint(15, 45))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _cutout(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    for _ in range(rng.randint(1, 3)):
        ch, cw = rng.randint(h // 8, h // 3), rng.randint(w // 8, w // 3)
        y, x = rng.randint(0, h - ch), rng.randint(0, w - cw)
        arr[y:y + ch, x:x + cw] = rng.randint(0, 255)
    return Image.fromarray(arr)


_FIELD_AUGS = [_low_light, _flash, _sensor_noise, _blur, _jpeg, _cutout]


def apply_field_condition(img: Image.Image, rng: random.Random) -> Image.Image:
    return rng.choice(_FIELD_AUGS)(img, rng)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class Stage1Dataset(Dataset):
    def __init__(self, manifest, emb_path, cfg: Stage1Config, train: bool = True,
                 shards: list[int] | None = None):
        import pandas as pd

        self.cfg = cfg
        self.train = train
        self.root = Path(cfg.data_root)
        self.img_dir = Path(cfg.img_dir)

        self.df = pd.read_parquet(manifest) if str(manifest).endswith(".parquet") \
            else pd.read_csv(manifest)
        # 分片子集过滤（消融回落 16 片等）。text_emb 池仍用全量 row（记忆库负样本更多，无害）。
        if shards is not None:
            keep = set(str(s) for s in shards)
            self.df = self.df[self.df["shard"].astype(str).isin(keep)]
        self.df = self.df.reset_index(drop=True)
        self.rows = self.df["row"].to_numpy(dtype=np.int64)
        self.shards = self.df["shard"].astype(str).to_numpy()
        self.keys = self.df["key"].astype(str).to_numpy()
        self.binomial = self.df.get("binomial", "").astype(str).fillna("").to_numpy() \
            if "binomial" in self.df else np.array([""] * len(self.df))
        self.family = self.df.get("family", "").astype(str).fillna("").to_numpy() \
            if "family" in self.df else np.array([""] * len(self.df))
        self.is_general = self.df.get("is_general", 0).astype(int).to_numpy() \
            if "is_general" in self.df else np.zeros(len(self.df), dtype=int)
        self.phash = self.df.get("phash", "").astype(str).fillna("").to_numpy() \
            if "phash" in self.df else np.array([""] * len(self.df))

        M = int(self.rows.max()) + 1 if len(self.rows) else 0
        self.text_emb = np.load(emb_path, mmap_mode="r")           # [M, 768] fp16
        assert self.text_emb.shape[0] >= M, \
            f"text_emb 行数 {self.text_emb.shape[0]} < manifest 最大 row {M}"
        self.dim = self.text_emb.shape[1]

        self._build_sampling_weights()

    # ── 采样权重: 温度 τ + repeat-factor 折进一个向量 ──────────
    def _build_sampling_weights(self):
        tau = self.cfg.tail_tau
        has_bin = self.binomial != ""
        w = np.ones(len(self.df), dtype=np.float64)
        if tau > 0 and has_bin.any():
            uniq, inv, cnt = np.unique(self.binomial[has_bin], return_inverse=True,
                                       return_counts=True)
            freq = cnt / cnt.sum()
            cls_w = freq ** (-tau)                                 # n_c^{-tau}
            if self.cfg.repeat_factor:
                t = self.cfg.repeat_thresh
                rf = np.maximum(1.0, np.sqrt(t / np.maximum(freq, 1e-12)))
                cls_w = cls_w * rf
            cls_w = cls_w / cls_w.mean()
            w[has_bin] = cls_w[inv]
        # 无 binomial 的按自然频率（w=1）
        self.sample_weights = torch.as_tensor(w, dtype=torch.double)

    def make_sampler(self, num_samples: int | None = None):
        from torch.utils.data import WeightedRandomSampler
        n = num_samples or len(self.df)
        return WeightedRandomSampler(self.sample_weights, num_samples=n, replacement=True)

    # ── 图像变换 ─────────────────────────────────────────────
    def _load_img(self, i: int) -> Image.Image:
        p = self.img_dir / self.shards[i] / f"{self.keys[i]}.jpg"
        img = Image.open(p)
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img

    def _transform(self, img: Image.Image, rng: random.Random) -> torch.Tensor:
        S = self.cfg.image_size
        if self.train:
            # RandomResizedCrop scale=(0.5,1.0)
            lo, hi = self.cfg.rrc_scale
            w, h = img.size
            area = w * h
            for _ in range(10):
                ta = random.uniform(lo, hi) * area
                ar = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
                cw, ch = int(round(math.sqrt(ta * ar))), int(round(math.sqrt(ta / ar)))
                if cw <= w and ch <= h:
                    x, y = random.randint(0, w - cw), random.randint(0, h - ch)
                    img = img.crop((x, y, x + cw, y + ch))
                    break
            img = img.resize((S, S), Image.BICUBIC)
            if self.cfg.hflip and rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if rng.random() < self.cfg.aug_prob:
                try:
                    img = apply_field_condition(img, rng)
                except Exception:
                    pass
        else:
            img = ImageOps.fit(img, (S, S), Image.BICUBIC)

        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        arr = (arr - np.asarray(CLIP_MEAN, np.float32)) / np.asarray(CLIP_STD, np.float32)
        return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        rng = random.Random()
        for _ in range(4):
            try:
                img = self._load_img(i)
                x = self._transform(img, rng)
                break
            except Exception:
                i = random.randint(0, len(self.df) - 1)
        else:
            x = torch.zeros(3, self.cfg.image_size, self.cfg.image_size)

        emb_idx = int(self.rows[i])
        meta = {
            "row": emb_idx,
            "binomial": str(self.binomial[i]),
            "family": str(self.family[i]),
            "is_general": int(self.is_general[i]),
            "phash": str(self.phash[i]),
        }
        return x, emb_idx, meta


def load_text_emb_pool(emb_path: str, device: str = "cpu",
                       dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """把整个预编码文本 emb 池读进（GPU）内存做记忆库。prepare.py 已 L2 归一化；
    这里在 CPU 上 fp32 重归一化（消 fp16 round-trip 误差）再转 dtype 上 GPU，
    避免在 GPU 上开一份 M×768 fp32 的瞬时大张量。"""
    arr = np.load(emb_path).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(1e-12)
    return torch.from_numpy(arr.astype(np.float16) if dtype == torch.float16
                            else arr).to(device=device, dtype=dtype)
