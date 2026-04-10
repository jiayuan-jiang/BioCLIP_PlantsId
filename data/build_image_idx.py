"""
BioCLIP 2 图像向量库构建脚本
==============================
读取 test_images/ 下所有物种目录中的图片，
用 BioCLIP 2 图像编码器逐批编码，
对每个物种取所有图片嵌入的均值向量（L2 归一化），
输出图像向量库 image_index.npz。

推理时直接用图像均值向量替代文本向量做余弦相似度匹配，
完全绕开文本-图像对齐偏差问题。

依赖：pip install torch open-clip-torch numpy pandas tqdm pillow

输出文件：image_index.npz
  embeddings   (N, D) float32  每个物种的均值图像嵌入（已归一化）
  taxon_ids    (N,)   str
  sci_names    (N,)   str
  common_names (N,)   str
  img_counts   (N,)   int      每个物种实际参与平均的图片数量
"""

import os
import re
import math
import queue
import threading

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import open_clip


# ══════════════════════════════════════════════
#  修改这里的变量
# ══════════════════════════════════════════════

TAXONOMY_CSV   = "taxonomy_enriched.csv"
IMAGE_DIR      = "./test_images"
OUTPUT_NPZ     = "image_index.npz"

WEIGHTS_DIR    = "./weights/bioclip2"
HF_REPO_ID     = "imageomics/bioclip-2"
OPEN_CLIP_ARCH = "ViT-L-14"

BATCH_SIZE         = 64    # 图像编码批大小，显存不足时调小
MAX_IMAGES_PER_SPP = None  # None = 全部；设整数则每物种最多取前 N 张
NUM_IO_WORKERS     = 8     # 并行读图线程数（I/O 密集，可以开多一些）
PREFETCH_BATCHES   = 4     # 预加载队列深度（内存充足可适当调大）

# ══════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LOCAL_WEIGHT_CANDIDATES = ["open_clip_pytorch_model.bin", "model.safetensors"]


# ─────────────────────────────────────────────
# 加载模型
# ─────────────────────────────────────────────

def load_model():
    weight_file = None
    if WEIGHTS_DIR and os.path.isdir(WEIGHTS_DIR):
        for c in _LOCAL_WEIGHT_CANDIDATES:
            p = os.path.join(WEIGHTS_DIR, c)
            if os.path.isfile(p):
                weight_file = p
                break

    if weight_file:
        print(f"加载模型（本地权重）：{weight_file}  device={DEVICE}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            OPEN_CLIP_ARCH, pretrained=weight_file
        )
    else:
        hf_name = f"hf-hub:{HF_REPO_ID}"
        print(f"加载模型（hf-hub）：{hf_name}  device={DEVICE}")
        model, _, preprocess = open_clip.create_model_and_transforms(hf_name)

    model = model.to(DEVICE).eval()
    print("✓ 模型加载完成")
    return model, preprocess


# ─────────────────────────────────────────────
# 扫描图片目录
# ─────────────────────────────────────────────

def scan_image_dir(image_dir: str, df: pd.DataFrame) -> dict:
    """
    扫描目录，返回 {taxon_id: [image_path, ...]}
    子目录名格式：{taxon_id}_{sci_name}
    """
    tid_set = set(df["taxon_id"].astype(str).tolist())
    result  = {}

    for sub in sorted(os.listdir(image_dir)):
        sub_path = os.path.join(image_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        tid = sub.split("_", 1)[0]
        if tid not in tid_set:
            continue
        imgs = sorted([
            os.path.join(sub_path, f)
            for f in os.listdir(sub_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ])
        if MAX_IMAGES_PER_SPP:
            imgs = imgs[:MAX_IMAGES_PER_SPP]
        if imgs:
            result[tid] = imgs

    print(f"✓ 扫描完成：{len(result)} 个物种有图片，"
          f"共 {sum(len(v) for v in result.values())} 张")
    return result


# ─────────────────────────────────────────────
# 并行读图 + 流水线推理
# ─────────────────────────────────────────────

_SENTINEL = object()   # 队列终止信号


def _load_one(path: str, preprocess) -> torch.Tensor | None:
    """读取单张图片并预处理，失败返回 None。"""
    try:
        return preprocess(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _io_worker(paths: list, preprocess,
               out_q: queue.Queue, batch_size: int):
    """
    后台线程：用线程池并行读图，攒够 batch_size 后放入 out_q。
    结束时放入 _SENTINEL。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    buf = []
    with ThreadPoolExecutor(max_workers=NUM_IO_WORKERS) as ex:
        futs = {ex.submit(_load_one, p, preprocess): p for p in paths}
        for fut in as_completed(futs):
            t = fut.result()
            if t is not None:
                buf.append(t)
            if len(buf) >= batch_size:
                out_q.put(buf[:batch_size])
                buf = buf[batch_size:]
    if buf:
        out_q.put(buf)
    out_q.put(_SENTINEL)


def encode_image_list(model, preprocess, paths: list) -> np.ndarray:
    """
    并行 I/O + 流水线推理：后台线程读图，主线程做 GPU 推理。
    返回 (N, D) 归一化嵌入。
    """
    if not paths:
        return np.zeros((0, 1), dtype=np.float32)

    q = queue.Queue(maxsize=PREFETCH_BATCHES)
    t = threading.Thread(
        target=_io_worker,
        args=(paths, preprocess, q, BATCH_SIZE),
        daemon=True,
    )
    t.start()

    all_embs = []
    with torch.no_grad():
        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            tensor = torch.stack(item).to(DEVICE)
            embs = model.encode_image(tensor)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().float().numpy())

    t.join()
    if not all_embs:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(all_embs, axis=0)


# ─────────────────────────────────────────────
# 构建每物种均值向量
# ─────────────────────────────────────────────

def build_species_embeddings(model, preprocess,
                             tid2paths: dict, df: pd.DataFrame):
    """
    对每个物种的所有图片编码后取均值，再归一化。
    返回与 df 行顺序对齐的 embeddings、img_counts。
    """
    df = df.copy()
    df["taxon_id"] = df["taxon_id"].astype(str)
    tid2idx = {row["taxon_id"]: i for i, row in df.iterrows()}

    # 把所有图片打平成一个大列表，记录每张图属于哪个物种
    all_paths = []
    path_tids  = []
    for tid, paths in tid2paths.items():
        all_paths.extend(paths)
        path_tids.extend([tid] * len(paths))

    total_imgs = len(all_paths)
    print(f"  共 {total_imgs} 张图片，{NUM_IO_WORKERS} 线程并行读图，"
          f"batch={BATCH_SIZE}，prefetch={PREFETCH_BATCHES}")

    # 一次性编码全部图片（流水线）
    q = queue.Queue(maxsize=PREFETCH_BATCHES)
    io_t = threading.Thread(
        target=_io_worker,
        args=(all_paths, preprocess, q, BATCH_SIZE),
        daemon=True,
    )
    io_t.start()

    # 收集每张图的嵌入，按 path 索引
    path_embs = []   # list of (batch_emb, start_idx)
    processed = 0
    pbar = tqdm(total=total_imgs, desc="编码图片", unit="img")

    with torch.no_grad():
        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            tensor = torch.stack(item).to(DEVICE)
            embs = model.encode_image(tensor)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            path_embs.append((processed, embs.cpu().float().numpy()))
            processed += len(item)
            pbar.update(len(item))

    pbar.close()
    io_t.join()

    # 拼成完整数组
    if not path_embs:
        raise RuntimeError("未能编码任何图片，请检查 IMAGE_DIR 路径。")

    # 先确定维度
    dim = path_embs[0][1].shape[1]
    all_emb_arr = np.zeros((total_imgs, dim), dtype=np.float32)
    for start_idx, emb_batch in path_embs:
        end_idx = start_idx + emb_batch.shape[0]
        all_emb_arr[start_idx:end_idx] = emb_batch

    # 按物种聚合均值
    embeddings = np.zeros((len(df), dim), dtype=np.float32)
    img_counts = np.zeros(len(df), dtype=np.int32)

    from collections import defaultdict
    tid_embs = defaultdict(list)
    for i, tid in enumerate(path_tids):
        if i < all_emb_arr.shape[0]:
            tid_embs[tid].append(all_emb_arr[i])

    for tid, emb_list in tid_embs.items():
        idx = tid2idx.get(tid)
        if idx is None:
            continue
        mean_emb = np.stack(emb_list).mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-8:
            mean_emb = mean_emb / norm
        embeddings[idx] = mean_emb
        img_counts[idx] = len(emb_list)

    covered = (img_counts > 0).sum()
    print(f"✓ 完成：{covered} / {len(df)} 个物种有图像嵌入，"
          f"{len(df) - covered} 个无图片（零向量兜底）")
    return embeddings, img_counts


# ─────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────

def save_index(embeddings: np.ndarray, img_counts: np.ndarray,
               df: pd.DataFrame, out_path: str):
    np.savez_compressed(
        out_path,
        embeddings   = embeddings.astype(np.float32),
        taxon_ids    = df["taxon_id"].fillna("").astype(str).values,
        sci_names    = df["scientific_name"].fillna("").astype(str).values,
        common_names = (df["common_name"].fillna("").astype(str).values
                        if "common_name" in df.columns
                        else np.array([""] * len(df))),
        img_counts   = img_counts,
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"✓ 图像向量库已保存：{out_path}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    # 1. 读取分类学 CSV（保证 taxon_id 顺序与向量库对齐）
    df = pd.read_csv(TAXONOMY_CSV, dtype={"taxon_id": str})
    print(f"✓ 读取 {len(df)} 个物种  [{TAXONOMY_CSV}]")

    # 2. 扫描图片目录
    tid2paths = scan_image_dir(IMAGE_DIR, df)

    # 3. 加载模型
    model, preprocess = load_model()

    # 4. 逐物种编码并取均值
    embeddings, img_counts = build_species_embeddings(
        model, preprocess, tid2paths, df)

    # 5. 保存
    save_index(embeddings, img_counts, df, OUTPUT_NPZ)

    print(f"\n构建完成！")
    print(f"  输出：{OUTPUT_NPZ}")
    print(f"\n使用方式：")
    print(f"  在 infer.py 中把 INDEX_NPZ 改为 '{OUTPUT_NPZ}' 即可切换到图像向量库")


if __name__ == "__main__":
    main()