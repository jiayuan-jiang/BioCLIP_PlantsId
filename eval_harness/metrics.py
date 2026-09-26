"""评测指标：top-k / macro recall(=balanced acc) / macro F1 / ECE。"""

import numpy as np


def topk_acc(logits: np.ndarray, labels: np.ndarray, ks=(1, 5)) -> dict:
    order = np.argsort(-logits, axis=1)
    out = {}
    for k in ks:
        k = min(k, logits.shape[1])
        hit = (order[:, :k] == labels[:, None]).any(axis=1)
        out[f"top{k}"] = float(hit.mean())
    return out


def per_class_recall(pred: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    rec = np.full(n_classes, np.nan, dtype=np.float64)
    for c in range(n_classes):
        m = labels == c
        if m.any():
            rec[c] = (pred[m] == c).mean()
    return rec


def macro_scores(pred: np.ndarray, labels: np.ndarray, n_classes: int) -> dict:
    rec = per_class_recall(pred, labels, n_classes)
    # macro F1
    f1 = np.full(n_classes, np.nan)
    for c in range(n_classes):
        tp = np.sum((pred == c) & (labels == c))
        fp = np.sum((pred == c) & (labels != c))
        fn = np.sum((pred != c) & (labels == c))
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1[c] = 2 * tp / denom
    return {
        "macro_recall": float(np.nanmean(rec)),   # balanced accuracy
        "macro_f1": float(np.nanmean(f1)),
        "n_classes_present": int(np.sum(~np.isnan(rec))),
        "_per_class_recall": rec,
    }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins=15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def softmax(x: np.ndarray, axis=1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
