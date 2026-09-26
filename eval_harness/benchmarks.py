"""基准数据集加载器。

每个加载器返回 Benchmark：图像路径列表 + 整数标签 + 类名列表（+ 可选俗名）。
加载器对"数据不存在"要宽容——discover() 只返回目录里真实存在的基准。

期望的目录布局（都放在 --data-root 下）：

  data_root/
    plantnet-300k/
      images/test/<species_id>/<img>.jpg          # 或 images_test/...
      plantnet300K_species_id_2_name.json
    rare-species/                                  # ImageFolder：<sci_name>/<img>.jpg
      <Genus species>/*.jpg
    imagenet-1k/
      val/<wnid>/*.JPEG
      imagenet_class_index.json                    # {"0": ["n01440764", "tench"], ...}
    nabirds/
      images/<class_id>/<img>.jpg
      classes.txt  image_class_labels.txt  images.txt  train_test_split.txt
    <any-folder>/                                  # 通用 ImageFolder 兜底
      <class_name>/*.jpg
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPEG", ".JPG", ".PNG")


@dataclass
class Benchmark:
    name: str
    kind: str                       # "plant" | "general"
    template_set: str               # prompts.py 的模板键
    image_paths: list[str]
    labels: np.ndarray              # int64, [N]
    classnames: list[str]           # 用于渲染 prompt
    common_names: list[str] | None = None
    notes: str = ""
    meta: dict = field(default_factory=dict)

    def subsample(self, limit: int, seed: int = 0) -> "Benchmark":
        if limit <= 0 or limit >= len(self.image_paths):
            return self
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.image_paths), size=limit, replace=False)
        idx.sort()
        return Benchmark(
            name=self.name, kind=self.kind, template_set=self.template_set,
            image_paths=[self.image_paths[i] for i in idx],
            labels=self.labels[idx],
            classnames=self.classnames, common_names=self.common_names,
            notes=self.notes + f" [subsampled {limit}]", meta=self.meta,
        )


def _list_imagefolder(root: Path):
    """<root>/<class>/<img> -> (paths, labels, class_dirnames)"""
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    cls2idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        for p in sorted((root / c).iterdir()):
            if p.suffix in IMG_EXT:
                paths.append(str(p))
                labels.append(cls2idx[c])
    return paths, np.asarray(labels, dtype=np.int64), classes


# ─────────────────────────────────────────────
# 具体基准
# ─────────────────────────────────────────────

def load_plantnet300k(root: Path, split: str = "test") -> Benchmark | None:
    img_root = None
    for cand in (root / "images" / split, root / f"images_{split}", root / split):
        if cand.is_dir():
            img_root = cand
            break
    if img_root is None:
        return None

    paths, labels, class_ids = _list_imagefolder(img_root)
    if not paths:
        return None

    # species_id -> 学名
    name_map = {}
    for jn in ("plantnet300K_species_id_2_name.json", "species_id_2_name.json"):
        jp = root / jn
        if jp.is_file():
            name_map = json.loads(jp.read_text())
            break
    classnames = [name_map.get(cid, cid) for cid in class_ids]
    return Benchmark(
        name="plantnet300k", kind="plant", template_set="plant",
        image_paths=paths, labels=labels, classnames=classnames,
        notes=f"split={split}, {len(class_ids)} classes",
        meta={"class_ids": class_ids},
    )


def load_rare_species(root: Path) -> Benchmark | None:
    # 直接是 ImageFolder（<Genus species>/*.jpg）
    sub = root
    if (root / "images").is_dir() and not any(d.is_dir() for d in root.iterdir() if d.name != "images"):
        sub = root / "images"
    if not sub.is_dir():
        return None
    paths, labels, classes = _list_imagefolder(sub)
    if not paths:
        return None
    # imageomics/rare-species 这个 HF 副本实为 400 个动物物种 → 归入通用/跨生物留存轴
    return Benchmark(
        name="rare_species", kind="general", template_set="species",
        image_paths=paths, labels=labels, classnames=classes,
        notes=f"{len(classes)} classes (rare-species, non-plant; retention check)",
    )


def load_imagenet_val(root: Path) -> Benchmark | None:
    val = root / "val"
    if not val.is_dir():
        return None
    paths, labels, wnids = _list_imagefolder(val)
    if not paths:
        return None
    ci = root / "imagenet_class_index.json"
    classnames = wnids
    if ci.is_file():
        idx = json.loads(ci.read_text())            # {"0": ["n01440764","tench"], ...}
        wnid2name = {v[0]: v[1].replace("_", " ") for v in idx.values()}
        classnames = [wnid2name.get(w, w).replace("_", " ") for w in wnids]
    return Benchmark(
        name="imagenet1k", kind="general", template_set="imagenet",
        image_paths=paths, labels=labels, classnames=classnames,
        notes=f"{len(wnids)} classes",
    )


def load_imagenetv2(root: Path) -> Benchmark | None:
    """ImageNetV2（ungated）：目录名是 0..999 的整数；用 imagenet_class_index.json 映射到类名。

    期望: imagenetv2/<0..999>/*.jpeg  +  imagenetv2/imagenet_class_index.json
    （class_index: {"0": ["n01440764","tench"], ...}）
    """
    # 找到实际含数字子目录的层
    base = root
    subs = [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()] if root.is_dir() else []
    if not subs:
        for d in (root.iterdir() if root.is_dir() else []):
            if d.is_dir() and any(x.is_dir() and x.name.isdigit() for x in d.iterdir()):
                base = d
                break
        subs = [d for d in base.iterdir() if d.is_dir() and d.name.isdigit()] if base.is_dir() else []
    if not subs:
        return None
    ci = None
    for cand in (root / "imagenet_class_index.json", base / "imagenet_class_index.json"):
        if cand.is_file():
            ci = json.loads(cand.read_text())
            break
    order = sorted(subs, key=lambda d: int(d.name))
    paths, labels = [], []
    for d in order:
        for p in sorted(d.iterdir()):
            if p.suffix in IMG_EXT:
                paths.append(str(p))
                labels.append(int(d.name))
    if not paths:
        return None
    if ci:
        classnames = [ci[str(i)][1].replace("_", " ") for i in range(1000)]
    else:
        classnames = [str(i) for i in range(1000)]
    return Benchmark("imagenetv2", "general", "imagenet", paths,
                     np.asarray(labels, dtype=np.int64), classnames,
                     notes=f"{len(order)} classes (imagenetv2)")


def load_nabirds(root: Path) -> Benchmark | None:
    need = ["images.txt", "image_class_labels.txt", "classes.txt"]
    if not all((root / n).is_file() for n in need):
        # 也接受纯 ImageFolder
        imgdir = root / "images"
        if imgdir.is_dir():
            paths, labels, classes = _list_imagefolder(imgdir)
            if paths:
                classes = [c.split(".", 1)[-1].replace("_", " ").strip() for c in classes]
                return Benchmark("nabirds", "general", "bird", paths, labels, classes,
                                 notes=f"{len(classes)} classes (imagefolder)")
        return None

    id2path = dict(l.split() for l in (root / "images.txt").read_text().splitlines() if l.strip())
    id2cls = {a: int(b) for a, b in (l.split() for l in (root / "image_class_labels.txt").read_text().splitlines() if l.strip())}
    cls_names = {}
    for l in (root / "classes.txt").read_text().splitlines():
        if not l.strip():
            continue
        cid, name = l.split(" ", 1)
        cls_names[int(cid)] = name.strip()

    raw_cls = sorted(set(id2cls.values()))
    remap = {c: i for i, c in enumerate(raw_cls)}
    img_base = root / "images"
    paths, labels = [], []
    for iid, rel in id2path.items():
        p = img_base / rel
        if p.is_file() and iid in id2cls:
            paths.append(str(p))
            labels.append(remap[id2cls[iid]])
    classnames = [cls_names.get(c, str(c)) for c in raw_cls]
    return Benchmark("nabirds", "general", "bird", paths,
                     np.asarray(labels, dtype=np.int64), classnames,
                     notes=f"{len(raw_cls)} classes")


def load_taxon_imagefolder(root: Path, name: str, taxonomy_csv: Path | None = None,
                           kind: str = "plant") -> Benchmark | None:
    """目录名形如 `<taxon_id>_<Genus species>`；类名取第一个下划线之后的学名。

    有 taxonomy_csv（列 scientific_name, common_name）时补俗名。
    """
    if not root.is_dir():
        return None
    dirs = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not dirs:
        return None
    cls2idx = {d: i for i, d in enumerate(dirs)}
    paths, labels = [], []
    for d in dirs:
        for p in sorted((root / d).iterdir()):
            if p.suffix in IMG_EXT:
                paths.append(str(p))
                labels.append(cls2idx[d])
    if not paths:
        return None

    def sci_of(dirname: str) -> str:
        return dirname.split("_", 1)[1] if "_" in dirname else dirname

    classnames = [sci_of(d) for d in dirs]
    commons = None
    if taxonomy_csv and taxonomy_csv.is_file():
        import csv as _csv
        m = {}
        with open(taxonomy_csv, encoding="utf-8-sig") as f:
            for r in _csv.DictReader(f):
                if r.get("scientific_name"):
                    m[r["scientific_name"].strip()] = (r.get("common_name") or "").strip()
        commons = [m.get(c) or None for c in classnames]

    return Benchmark(
        name=name, kind=kind, template_set="plant",
        image_paths=paths, labels=np.asarray(labels, dtype=np.int64),
        classnames=classnames, common_names=commons,
        notes=f"{len(dirs)} classes (taxon imagefolder)",
    )


def load_generic_imagefolder(root: Path, name: str, kind: str = "plant") -> Benchmark | None:
    if not root.is_dir():
        return None
    paths, labels, classes = _list_imagefolder(root)
    if not paths:
        return None
    tpl = "plant" if kind == "plant" else "imagenet"
    classes = [c.replace("_", " ") for c in classes]
    return Benchmark(name, kind, tpl, paths, labels, classes,
                     notes=f"{len(classes)} classes (generic imagefolder)")


# ─────────────────────────────────────────────
# 发现
# ─────────────────────────────────────────────

def _load_local_testset(r: Path) -> Benchmark | None:
    for cand in (r / "test_images", r / "testimages"):
        if cand.is_dir():
            return load_taxon_imagefolder(cand, "plant_head5k", r / "taxonomy_enriched.csv")
    return None


_KNOWN = {
    "plantnet300k":  lambda r: load_plantnet300k(r / "plantnet-300k"),
    "rare_species":  lambda r: load_rare_species(r / "rare-species"),
    "imagenet1k":    lambda r: load_imagenet_val(r / "imagenet-1k"),
    "imagenetv2":    lambda r: load_imagenetv2(r / "imagenetv2"),
    "nabirds":       lambda r: load_nabirds(r / "nabirds"),
    "plant_head5k":  _load_local_testset,
}


def discover(data_root: str | Path, want: list[str] | None = None) -> list[Benchmark]:
    root = Path(data_root)
    out = []
    for key, fn in _KNOWN.items():
        if want and key not in want:
            continue
        try:
            b = fn(root)
        except Exception as e:                       # noqa: BLE001
            print(f"  [warn] 基准 {key} 加载失败：{e}")
            b = None
        if b is not None and len(b.image_paths):
            out.append(b)
            print(f"  ✓ {b.name:14s} {len(b.image_paths):>7d} imgs  {len(b.classnames):>5d} cls  ({b.kind})")
        elif want and key in want:
            print(f"  – {key:14s} 未找到数据（跳过）")
    # 额外：data_root/extra/<name> 下的通用 imagefolder
    extra = root / "extra"
    if extra.is_dir():
        for d in sorted(extra.iterdir()):
            if d.is_dir() and (not want or d.name in want):
                b = load_generic_imagefolder(d, d.name, "plant")
                if b:
                    out.append(b)
                    print(f"  ✓ {b.name:14s} {len(b.image_paths):>7d} imgs  {len(b.classnames):>5d} cls  (extra)")
    return out
