"""Stage 1 损失。

主损失: locked_text_infonce（图→文单向 CE，负样本从预计算静态文本 emb 池采样）。
消融项: feature_distill / mbdc / proto_ce —— stub / 轻实现，config flag 控。

依据: doc/stage1-spec.md §3（损失）、§4（记忆库 — 无 MoCo momentum encoder）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 主损失: locked-text InfoNCE（图 → 文）
# ─────────────────────────────────────────────────────────────

def sample_negative_bank(text_emb_pool: torch.Tensor, k: int, exclude: torch.Tensor,
                         generator: torch.Generator | None = None) -> torch.Tensor:
    """从全局文本 emb 池 [M, D] 采 K 行做记忆库负样本。

    text_emb_pool: L2 归一化的 fp16/fp32，GPU 常驻。
    exclude: [B] 本 batch 正样本在池中的行号（尽量不采到自己）。
    返回被采行号 [K]（在 pool 上 index_select 得到 [K, D]）。
    纯 index_select，无前向 —— 记忆库“免费”。
    """
    m = text_emb_pool.shape[0]
    k = min(k, m)
    idx = torch.randint(0, m, (k,), device=text_emb_pool.device, generator=generator)
    # 简单排除: 命中正样本行的重采一次（碰撞概率极低，够用）
    if exclude.numel():
        bad = torch.isin(idx, exclude)
        if bad.any():
            idx[bad] = torch.randint(0, m, (int(bad.sum()),), device=idx.device,
                                     generator=generator)
    return idx


def locked_text_infonce(
    z_img: torch.Tensor,          # [B, D]  图像塔输出（未归一化）
    t_pos: torch.Tensor,          # [B, D]  正样本 caption 预编码 emb（已归一化）
    t_neg: torch.Tensor,          # [K, D]  记忆库负样本 emb（已归一化）
    logit_scale: float,           # 标量，用 BioCLIP2 的 exp(logit_scale)=100
    fn_cos_thresh: float = 0.9,   # false-negative 屏蔽: neg 与 pos cos > thr 的置 -inf
) -> tuple[torch.Tensor, dict]:
    """target 恒为 0（每行第 0 列是正样本），CE。

    logits[i] = s · z_i · [t_pos_i ; t_neg]ᵀ  ∈ ℝ^{1+K}
    """
    z = F.normalize(z_img.float(), dim=-1)
    tp = t_pos.float()
    tn = t_neg.float()

    pos_logit = (z * tp).sum(-1, keepdim=True)                 # [B, 1]
    neg_logit = z @ tn.t()                                     # [B, K]

    # false-negative 屏蔽: 负样本文本与正样本文本太像 -> 不当负样本
    with torch.no_grad():
        sim_pn = tp @ tn.t()                                   # [B, K]
        fn_mask = sim_pn > fn_cos_thresh
    neg_logit = neg_logit.masked_fill(fn_mask, float("-inf"))

    logits = logit_scale * torch.cat([pos_logit, neg_logit], dim=1)   # [B, 1+K]
    target = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
    loss = F.cross_entropy(logits, target)

    with torch.no_grad():
        acc = (logits.argmax(-1) == 0).float().mean()
        n_fn = fn_mask.float().mean()
    return loss, {"infonce_acc": float(acc), "fn_frac": float(n_fn)}


# ─────────────────────────────────────────────────────────────
# 消融项
# ─────────────────────────────────────────────────────────────

def feature_distill(f_student: torch.Tensor, f_teacher: torch.Tensor) -> torch.Tensor:
    """1 − cos(f_student, f_teacher)。teacher = 冻结 BioCLIP2 visual（无梯度）。

    stage1-spec §3: 默认 off，仅消融。调用方负责只在 ~distill_frac 样本上算。
    """
    fs = F.normalize(f_student.float(), dim=-1)
    ft = F.normalize(f_teacher.float().detach(), dim=-1)
    return (1.0 - (fs * ft).sum(-1)).mean()


def mbdc(tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> torch.Tensor:
    """DALIP 的二阶分布对齐（Multi-head Brownian Distance Covariance）。

    STUB —— 需要 token 级特征 + 我们 locked-text 与 DALIP 双塔训练结构不同。
    仅当 cfg.use_mbdc 时启用；消融 λ_mbdc ∈ {0, 0.5, 1}。
    真实现: 对每个 head 的 token 特征算 Brownian distance covariance 矩阵，
    对齐图/文两侧。先留 0 占位，进入该消融时再补。
    """
    raise NotImplementedError("mbdc: 进入 λ_mbdc 消融时实现（stage1-spec §3 / §10）")


def proto_ce(z_img: torch.Tensor, species_proto: torch.Tensor,
             species_label: torch.Tensor, logit_scale: float,
             logit_adjust: torch.Tensor | None = None) -> torch.Tensor:
    """species 原型交叉熵（对 ~66% 以 binomial 开头的样本）。

    z_img: [B, D] 未归一化；species_proto: [C, D] 已归一化（该 batch 覆盖到的 species 文本原型）;
    species_label: [B] ∈ [0, C)（无 binomial 的样本调用方需先剔除）;
    logit_adjust: [C] 可选 −log π_c（Menon logit adjustment，长尾）。

    stage1-spec §3 / §8: 默认 off，消融 λ_p ∈ {0, 0.3}。
    """
    z = F.normalize(z_img.float(), dim=-1)
    logits = logit_scale * z @ species_proto.float().t()      # [B, C]
    if logit_adjust is not None:
        logits = logits + logit_adjust.float().to(logits.device)
    return F.cross_entropy(logits, species_label)
