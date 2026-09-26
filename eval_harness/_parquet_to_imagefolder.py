"""把 HF parquet 数据集导出成 eval_harness 期望的 ImageFolder 布局（pod 上一次性用）。

用法:
  python _parquet_to_imagefolder.py plantnet300k  <pn300k-hf 目录>  <输出 eval_data 目录>
  python _parquet_to_imagefolder.py rare_species   <rare-species-raw 目录>  <输出 eval_data 目录>
"""
import csv
import glob
import io
import json
import os
import sys

import pyarrow.parquet as pq
from PIL import Image


def _write_img(rec, path):
    if os.path.exists(path):
        return
    b = rec["bytes"] if isinstance(rec, dict) else rec
    if b is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        Image.open(io.BytesIO(b)).convert("RGB").save(path, "JPEG", quality=92)
    except Exception as e:  # noqa: BLE001
        print("  skip bad image:", e)


def do_plantnet300k(src, out):
    dst = os.path.join(out, "plantnet-300k")
    img_root = os.path.join(dst, "images", "test")
    os.makedirs(img_root, exist_ok=True)

    # metadata.csv: label -> (species_id, scientific_name)
    meta = os.path.join(src, "data", "test", "metadata.csv")
    lab2sid, lab2name = {}, {}
    with open(meta, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab2sid[int(r["label"])] = r["class"]
            lab2name[int(r["label"])] = r["species"]
    json.dump({sid: lab2name[lab] for lab, sid in lab2sid.items()},
              open(os.path.join(dst, "plantnet300K_species_id_2_name.json"), "w"),
              ensure_ascii=False)
    print(f"  {len(lab2sid)} classes")

    n = 0
    for pf_path in sorted(glob.glob(os.path.join(src, "data", "test-*.parquet"))):
        pf = pq.ParquetFile(pf_path)
        for bi, batch in enumerate(pf.iter_batches(batch_size=256)):
            d = batch.to_pydict()
            for img, lab in zip(d["image"], d["label"]):
                sid = lab2sid.get(lab, str(lab))
                _write_img(img, os.path.join(img_root, sid, f"{n:07d}.jpg"))
                n += 1
        print(f"  {os.path.basename(pf_path)} done, total {n}")
    print(f"  wrote {n} test images -> {img_root}")


def do_rare_species(src, out, kingdom_filter="Plantae"):
    dst = os.path.join(out, "rare-species")
    os.makedirs(dst, exist_ok=True)
    n, skip = 0, 0
    for pf_path in sorted(glob.glob(os.path.join(src, "data", "*.parquet"))):
        pf = pq.ParquetFile(pf_path)
        for batch in pf.iter_batches(batch_size=256):
            d = batch.to_pydict()
            for i in range(len(d["file_name"])):
                if kingdom_filter and d.get("kingdom", [None] * len(d["file_name"]))[i] != kingdom_filter:
                    skip += 1
                    continue
                sci = (d["sciName"][i] or "").strip().replace("/", "_")
                if not sci:
                    skip += 1
                    continue
                _write_img(d["file_name"][i], os.path.join(dst, sci, f"{n:07d}.jpg"))
                n += 1
        print(f"  {os.path.basename(pf_path)} done, kept {n} skip {skip}")
    print(f"  wrote {n} images ({kingdom_filter} only) -> {dst}")


if __name__ == "__main__":
    which, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    if which == "plantnet300k":
        do_plantnet300k(src, out)
    elif which == "rare_species":
        kf = sys.argv[4] if len(sys.argv) > 4 else "Plantae"
        do_rare_species(src, out, kingdom_filter=(None if kf in ("", "none", "all") else kf))
    else:
        raise SystemExit(f"unknown: {which}")
