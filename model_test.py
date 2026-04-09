"""
BioCLIP 2 零样本分类评测脚本
==============================
输入：
  - plant_text_index.npz   （向量库）
  - ./test_images/          （下载好的测试图片目录）

输出（保存到 OUTPUT_DIR）：
  - results.csv             逐图预测明细
  - metrics_summary.txt     汇总指标
  - plot_accuracy_topk.png  Top-1/3/5 accuracy 柱状图
  - plot_confusion_top50.png Top-50 物种混淆矩阵
  - plot_roc_auc.png         各物种 OvR ROC + macro AUC
  - plot_score_dist.png      正确/错误预测的相似度分布
  - plot_per_species_acc.png 每物种 Top-1 accuracy 分布（直方图 + 尾部排名）
  - plot_tsne.png            图像嵌入 t-SNE（随机抽样 2000 张）

依赖：
  pip install torch open-clip-torch numpy pandas tqdm pillow
  pip install scikit-learn matplotlib seaborn
"""

import os
import re
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import open_clip

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════
#  修改这里的变量
# ══════════════════════════════════════════════

INDEX_NPZ      = "./data/plant_text_index.npz"   # 向量库
TEST_IMAGE_DIR = "./data/test_images"          # 下载的图片根目录（每子目录为一个物种）
OUTPUT_DIR     = "./eval_results"         # 评测结果保存目录

# 权重加载（与 build_vector.py 保持一致）
WEIGHTS_DIR    = "./data/weights/bioclip2"
HF_REPO_ID     = "imageomics/bioclip-2"
OPEN_CLIP_ARCH = "ViT-L-14"

BATCH_SIZE     = 64      # 图像编码批大小，显存不足时调小
MAX_IMAGES     = None    # None = 全部；设置整数可限制评测图片总数（调试用）
TSNE_SAMPLE    = 2000    # t-SNE 最多使用的图像数
CONFUSION_TOPN = 50      # 混淆矩阵展示观测数最多的 N 个物种
RANDOM_SEED    = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

_LOCAL_WEIGHT_CANDIDATES = ["open_clip_pytorch_model.bin", "model.safetensors"]


# ─────────────────────────────────────────────
# 1. 加载向量库
# ─────────────────────────────────────────────

def load_index(path: str):
    print(f"加载向量库：{path}")
    idx = np.load(path, allow_pickle=True)
    embeddings   = idx["embeddings"].astype(np.float32)   # (N, D)
    taxon_ids    = idx["taxon_ids"].tolist()
    sci_names    = idx["sci_names"].tolist()
    common_names = idx["common_names"].tolist() if "common_names" in idx else [""] * len(taxon_ids)
    print(f"  物种数：{len(taxon_ids)}，嵌入维度：{embeddings.shape[1]}")
    return embeddings, taxon_ids, sci_names, common_names


# ─────────────────────────────────────────────
# 2. 加载模型（本地优先，缺失则在线）
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
        try:
            tokenizer = open_clip.get_tokenizer(f"hf-hub:{HF_REPO_ID}")
        except Exception:
            tokenizer = open_clip.get_tokenizer(OPEN_CLIP_ARCH)
    else:
        hf_name = f"hf-hub:{HF_REPO_ID}"
        print(f"加载模型（hf-hub）：{hf_name}  device={DEVICE}")
        model, _, preprocess = open_clip.create_model_and_transforms(hf_name)
        tokenizer = open_clip.get_tokenizer(hf_name)

    model = model.to(DEVICE).eval()
    print("✓ 模型加载完成")
    return model, preprocess


# ─────────────────────────────────────────────
# 3. 扫描测试图片
# ─────────────────────────────────────────────

def scan_test_images(image_dir: str, taxon_ids: list, sci_names: list):
    """
    扫描目录，子目录名格式：{taxon_id}_{sci_name}
    返回 list of {path, taxon_id, label_idx}
    """
    tid2idx = {tid: i for i, tid in enumerate(taxon_ids)}
    records = []
    skipped_dirs = 0

    for sub in sorted(os.listdir(image_dir)):
        sub_path = os.path.join(image_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        parts = sub.split("_", 1)
        tid = parts[0]
        if tid not in tid2idx:
            skipped_dirs += 1
            continue
        label_idx = tid2idx[tid]
        for fname in os.listdir(sub_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                records.append({
                    "path":      os.path.join(sub_path, fname),
                    "taxon_id":  tid,
                    "label_idx": label_idx,
                    "sci_name":  sci_names[label_idx],
                })

    if skipped_dirs:
        print(f"  ⚠ {skipped_dirs} 个子目录的 taxon_id 不在向量库中，已跳过")

    if MAX_IMAGES and len(records) > MAX_IMAGES:
        random.shuffle(records)
        records = records[:MAX_IMAGES]

    print(f"  找到 {len(records)} 张图片，覆盖 "
          f"{len({r['taxon_id'] for r in records})} 个物种")
    return records


# ─────────────────────────────────────────────
# 4. 图像编码
# ─────────────────────────────────────────────

def encode_images(model, preprocess, records: list):
    all_embs = []
    n_batches = math.ceil(len(records) / BATCH_SIZE)

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="图像编码"):
            batch = records[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            imgs = []
            valid_idx = []
            for j, rec in enumerate(batch):
                try:
                    img = preprocess(Image.open(rec["path"]).convert("RGB"))
                    imgs.append(img)
                    valid_idx.append(i * BATCH_SIZE + j)
                except Exception:
                    pass
            if not imgs:
                continue
            tensor = torch.stack(imgs).to(DEVICE)
            embs = model.encode_image(tensor)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)   # (M, D)


# ─────────────────────────────────────────────
# 5. 推理：余弦相似度打分
# ─────────────────────────────────────────────

def run_inference(img_embs: np.ndarray, text_embs: np.ndarray):
    """
    返回 sim_matrix (M, N_classes)，已 softmax 归一化为概率
    """
    sims = img_embs @ text_embs.T   # (M, N)
    # 转为概率（温度缩放后 softmax）
    temp = 0.01
    sims_scaled = sims / temp
    sims_scaled -= sims_scaled.max(axis=1, keepdims=True)
    exp = np.exp(sims_scaled)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return sims, probs


# ─────────────────────────────────────────────
# 6. 计算指标
# ─────────────────────────────────────────────

def compute_metrics(sims, probs, labels: np.ndarray, n_classes: int):
    pred_top1 = sims.argmax(axis=1)

    top1 = (pred_top1 == labels).mean()
    top3 = top_k_accuracy_score(labels, sims, k=3,
                                 labels=list(range(n_classes)))
    top5 = top_k_accuracy_score(labels, sims, k=5,
                                 labels=list(range(n_classes)))

    # per-species top-1
    spp_acc = {}
    for cls in np.unique(labels):
        mask = labels == cls
        spp_acc[int(cls)] = (pred_top1[mask] == cls).mean()

    # macro AUC（仅对出现在测试集中的类）
    present = sorted(np.unique(labels).tolist())
    if len(present) >= 2:
        y_bin = label_binarize(labels, classes=list(range(n_classes)))[:, present]
        p_sub = probs[:, present]
        try:
            macro_auc = auc(*[x.mean() for x in  # 简化：用 per-class AUC 均值
                               [np.array([0]), np.array([0])]])
        except Exception:
            macro_auc = float("nan")
        # 真实 macro AUC
        aucs = []
        for i, cls in enumerate(present):
            fpr, tpr, _ = roc_curve(y_bin[:, i], p_sub[:, i])
            aucs.append(auc(fpr, tpr))
        macro_auc = float(np.mean(aucs))
    else:
        macro_auc = float("nan")
        aucs = []
        present = []

    return {
        "top1": top1, "top3": top3, "top5": top5,
        "macro_auc": macro_auc,
        "per_species_acc": spp_acc,
        "per_class_auc":   dict(zip(present, aucs)) if aucs else {},
        "pred_top1":       pred_top1,
        "present_classes": present,
    }


# ─────────────────────────────────────────────
# 7. 绘图
# ─────────────────────────────────────────────

PALETTE = {
    "green":  "#1D9E75",
    "blue":   "#378ADD",
    "coral":  "#D85A30",
    "purple": "#7F77DD",
    "amber":  "#BA7517",
    "gray":   "#888780",
}

def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.color":       "#EBEBEB",
        "grid.linewidth":   0.6,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
    })


def plot_accuracy_topk(metrics: dict, out_dir: str):
    ks     = ["Top-1", "Top-3", "Top-5"]
    values = [metrics["top1"], metrics["top3"], metrics["top5"]]
    colors = [PALETTE["green"], PALETTE["blue"], PALETTE["purple"]]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(ks, [v * 100 for v in values], color=colors,
                  width=0.45, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val*100:.2f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                color=bar.get_facecolor())
    ax.set_ylim(0, max(v * 100 for v in values) * 1.18)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Zero-shot Classification Accuracy (BioCLIP 2)", pad=12)
    macro_auc = metrics["macro_auc"]
    ax.text(0.98, 0.04, f"Macro AUC = {macro_auc:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color=PALETTE["gray"])
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_accuracy_topk.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_score_dist(sims: np.ndarray, labels: np.ndarray, out_dir: str):
    pred   = sims.argmax(axis=1)
    correct_scores = sims[np.arange(len(labels)), labels]
    wrong_mask     = pred != labels
    wrong_scores   = sims[wrong_mask, pred[wrong_mask]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(
        min(correct_scores.min(), wrong_scores.min()) - 0.01,
        max(correct_scores.max(), wrong_scores.max()) + 0.01,
        60,
    )
    ax.hist(correct_scores, bins=bins, alpha=0.7,
            color=PALETTE["green"], label=f"Correct (n={len(correct_scores):,})")
    ax.hist(wrong_scores,   bins=bins, alpha=0.7,
            color=PALETTE["coral"], label=f"Incorrect (n={len(wrong_scores):,})")
    ax.axvline(correct_scores.mean(), color=PALETTE["green"],
               linestyle="--", linewidth=1.5,
               label=f"Correct mean = {correct_scores.mean():.3f}")
    ax.axvline(wrong_scores.mean(), color=PALETTE["coral"],
               linestyle="--", linewidth=1.5,
               label=f"Incorrect mean = {wrong_scores.mean():.3f}")
    ax.set_xlabel("Cosine Similarity (ground-truth class)")
    ax.set_ylabel("Count")
    ax.set_title("Similarity Score Distribution: Correct vs Incorrect")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_score_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_roc_auc(metrics: dict, probs: np.ndarray, labels: np.ndarray,
                 n_classes: int, sci_names: list, out_dir: str):
    present = metrics["present_classes"]
    if len(present) < 2:
        print("  ⚠ 类别数不足，跳过 ROC 图")
        return

    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    p_sub = probs[:, present]

    # 只绘制 AUC 最高和最低各 5 条 + macro 平均
    per_auc = metrics["per_class_auc"]
    sorted_cls = sorted(per_auc, key=per_auc.get)
    show_cls = sorted_cls[:5] + sorted_cls[-5:]   # 最低5 + 最高5

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")

    for i, cls in enumerate(show_cls):
        pi = present.index(cls)
        fpr, tpr, _ = roc_curve(y_bin[:, pi], p_sub[:, pi])
        a = per_auc[cls]
        name = sci_names[cls][:28]
        ax.plot(fpr, tpr, lw=1.2, color=cmap(i % 10),
                label=f"{name} ({a:.3f})", alpha=0.75)

    # macro 均值
    all_fpr = np.linspace(0, 1, 300)
    mean_tpr = np.zeros(300)
    for pi, cls in enumerate(present):
        fpr_c, tpr_c, _ = roc_curve(y_bin[:, pi], p_sub[:, pi])
        mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
    mean_tpr /= len(present)
    macro_auc = metrics["macro_auc"]
    ax.plot(all_fpr, mean_tpr, color="black", lw=2.5, linestyle="--",
            label=f"Macro avg (AUC = {macro_auc:.4f})")

    ax.plot([0, 1], [0, 1], ":", color=PALETTE["gray"], lw=1)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Bottom-5 & Top-5 AUC Species")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.85)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_roc_auc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_confusion(sims: np.ndarray, labels: np.ndarray,
                   sci_names: list, out_dir: str, topn: int = 50):
    pred = sims.argmax(axis=1)
    unique, counts = np.unique(labels, return_counts=True)
    # 取观测最多的 topn 物种
    top_cls = unique[np.argsort(-counts)][:topn].tolist()
    mask    = np.isin(labels, top_cls)
    l_sub   = labels[mask]
    p_sub   = pred[mask]

    cls_order = sorted(top_cls)
    cm = confusion_matrix(l_sub, p_sub, labels=cls_order)
    # 归一化为行百分比
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    tick_labels = [sci_names[c][:20] for c in cls_order]
    fig_size = max(12, topn * 0.28)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
    sns.heatmap(
        cm_norm, ax=ax,
        xticklabels=tick_labels, yticklabels=tick_labels,
        cmap="YlOrRd", vmin=0, vmax=1,
        linewidths=0.3, linecolor="#EBEBEB",
        cbar_kws={"shrink": 0.6, "label": "Recall"},
    )
    ax.set_xlabel("Predicted", labelpad=8)
    ax.set_ylabel("True", labelpad=8)
    ax.set_title(f"Confusion Matrix — Top-{topn} Most Observed Species (row-normalized)",
                 pad=12, fontsize=12)
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.tick_params(axis="y", rotation=0,  labelsize=6)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_confusion_top50.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_per_species_acc(metrics: dict, sci_names: list, out_dir: str):
    spp_acc = metrics["per_species_acc"]
    accs = list(spp_acc.values())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：分布直方图
    ax = axes[0]
    ax.hist(accs, bins=20, color=PALETTE["blue"], edgecolor="white", zorder=3)
    ax.axvline(np.mean(accs), color=PALETTE["coral"], linestyle="--",
               linewidth=1.8, label=f"Mean = {np.mean(accs):.3f}")
    ax.axvline(np.median(accs), color=PALETTE["amber"], linestyle="--",
               linewidth=1.8, label=f"Median = {np.median(accs):.3f}")
    ax.set_xlabel("Top-1 Accuracy per Species")
    ax.set_ylabel("Species Count")
    ax.set_title("Per-species Top-1 Accuracy Distribution")
    ax.legend()

    # 右：尾部最差 20 物种
    ax2 = axes[1]
    sorted_items = sorted(spp_acc.items(), key=lambda x: x[1])[:20]
    cls_list = [x[0] for x in sorted_items]
    acc_list = [x[1] for x in sorted_items]
    names = [sci_names[c][:30] for c in cls_list]
    colors = [PALETTE["coral"] if a < 0.2 else
              PALETTE["amber"] if a < 0.5 else
              PALETTE["green"] for a in acc_list]
    bars = ax2.barh(names, acc_list, color=colors, height=0.65, zorder=3)
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("Top-1 Accuracy")
    ax2.set_title("Bottom-20 Species by Accuracy")
    ax2.tick_params(axis="y", labelsize=8)
    for bar, val in zip(bars, acc_list):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0%}", va="center", fontsize=8)

    fig.tight_layout(pad=2)
    path = os.path.join(out_dir, "plot_per_species_acc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_tsne(img_embs: np.ndarray, labels: np.ndarray,
              sci_names: list, out_dir: str, n_sample: int = 2000):
    N = len(img_embs)
    if N > n_sample:
        idx = np.random.choice(N, n_sample, replace=False)
        embs_s = img_embs[idx]
        lbls_s = labels[idx]
    else:
        embs_s, lbls_s = img_embs, labels

    # 保留全部有图片的物种（无最小样本量限制）
    unique, counts = np.unique(lbls_s, return_counts=True)
    keep_cls = set(unique[counts >= 1].tolist())
    mask = np.array([l in keep_cls for l in lbls_s])
    embs_s = embs_s[mask]
    lbls_s = lbls_s[mask]

    if len(embs_s) == 0:
        print("  ⚠ t-SNE：无有效样本，跳过")
        return

    # perplexity 必须小于样本数
    perplexity = min(40, max(5, len(embs_s) // 10))
    print(f"  t-SNE 降维中（{len(embs_s)} 点，perplexity={perplexity}）…")

    # scikit-learn >= 1.2 将 n_iter 重命名为 max_iter，兼容两者
    import sklearn
    sk_ver = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
    tsne_kw = dict(n_components=2, perplexity=perplexity,
                   random_state=RANDOM_SEED, verbose=0)
    if sk_ver >= (1, 2):
        tsne_kw['max_iter'] = 1000
    else:
        tsne_kw['n_iter'] = 1000
    tsne = TSNE(**tsne_kw)
    coords = tsne.fit_transform(embs_s)

    unique_cls = np.unique(lbls_s)
    n_cls = len(unique_cls)
    cmap  = plt.get_cmap("tab20" if n_cls <= 20 else "hsv")
    cls2color = {c: cmap(i / max(n_cls - 1, 1)) for i, c in enumerate(unique_cls)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in unique_cls:
        m = lbls_s == cls
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=[cls2color[cls]], s=8, alpha=0.6,
                   label=sci_names[cls][:22] if n_cls <= 20 else None,
                   linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t-SNE of Image Embeddings — {len(embs_s)} samples, "
                 f"{n_cls} species", pad=10)
    if n_cls <= 20:
        ax.legend(fontsize=7, markerscale=2, loc="best",
                  framealpha=0.7, ncol=2)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_tsne.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────
# 8. 保存结果 CSV 和汇总
# ─────────────────────────────────────────────

def save_results(records: list, pred_top1: np.ndarray,
                 sims: np.ndarray, labels: np.ndarray,
                 sci_names: list, out_dir: str):
    rows = []
    for i, rec in enumerate(records):
        gt_idx  = labels[i]
        p1_idx  = pred_top1[i]
        top5_idx = np.argsort(sims[i])[::-1][:5].tolist()
        rows.append({
            "image_path":       rec["path"],
            "true_taxon_id":    rec["taxon_id"],
            "true_sci_name":    sci_names[gt_idx],
            "pred_sci_name":    sci_names[p1_idx],
            "correct_top1":     int(gt_idx == p1_idx),
            "gt_sim_score":     round(float(sims[i, gt_idx]), 5),
            "top1_sim_score":   round(float(sims[i, p1_idx]), 5),
            "top5_predictions": "|".join(sci_names[j] for j in top5_idx),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ {csv_path}")
    return df


def save_summary(metrics: dict, n_images: int, n_species: int, out_dir: str):
    lines = [
        "BioCLIP 2 Zero-shot Evaluation Summary",
        "=" * 45,
        f"Images evaluated : {n_images:,}",
        f"Species covered  : {n_species:,}",
        "",
        f"Top-1 Accuracy   : {metrics['top1']*100:.3f}%",
        f"Top-3 Accuracy   : {metrics['top3']*100:.3f}%",
        f"Top-5 Accuracy   : {metrics['top5']*100:.3f}%",
        f"Macro AUC        : {metrics['macro_auc']:.5f}",
        "",
        "Per-species Top-1 Accuracy:",
        f"  Mean   : {np.mean(list(metrics['per_species_acc'].values()))*100:.3f}%",
        f"  Median : {np.median(list(metrics['per_species_acc'].values()))*100:.3f}%",
        f"  Min    : {min(metrics['per_species_acc'].values())*100:.3f}%",
        f"  Max    : {max(metrics['per_species_acc'].values())*100:.3f}%",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    path = os.path.join(out_dir, "metrics_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  ✓ {path}")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_style()

    # 1. 加载向量库
    text_embs, taxon_ids, sci_names, common_names = load_index(INDEX_NPZ)
    n_classes = len(taxon_ids)

    # 2. 加载模型
    model, preprocess = load_model()

    # 3. 扫描测试图片
    print(f"\n扫描测试图片目录：{TEST_IMAGE_DIR}")
    records = scan_test_images(TEST_IMAGE_DIR, taxon_ids, sci_names)
    if not records:
        raise RuntimeError("未找到任何测试图片，请检查 TEST_IMAGE_DIR 路径。")

    # 4. 编码图像
    print(f"\n开始图像编码（{len(records)} 张，batch={BATCH_SIZE}，device={DEVICE}）…")
    img_embs = encode_images(model, preprocess, records)

    # 5. 推理
    labels = np.array([r["label_idx"] for r in records])
    print(f"\n计算相似度…")
    sims, probs = run_inference(img_embs, text_embs)

    # 6. 指标
    print("计算指标…")
    metrics = compute_metrics(sims, probs, labels, n_classes)

    # 7. 绘图
    print(f"\n生成图表 → {OUTPUT_DIR}/")
    plot_accuracy_topk(metrics, OUTPUT_DIR)
    plot_score_dist(sims, labels, OUTPUT_DIR)
    plot_roc_auc(metrics, probs, labels, n_classes, sci_names, OUTPUT_DIR)
    plot_confusion(sims, labels, sci_names, OUTPUT_DIR, topn=CONFUSION_TOPN)
    plot_per_species_acc(metrics, sci_names, OUTPUT_DIR)
    plot_tsne(img_embs, labels, sci_names, OUTPUT_DIR, n_sample=TSNE_SAMPLE)

    # 8. 保存结果
    print(f"\n保存结果…")
    save_results(records, metrics["pred_top1"], sims, labels, sci_names, OUTPUT_DIR)
    save_summary(metrics, len(records), len(np.unique(labels)), OUTPUT_DIR)

    print(f"\n全部完成！结果保存在：{os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()