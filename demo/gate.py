"""
非植物拦截（OOD gate）v1 — energy-only。

原理：把分类器的 logits 边缘化掉类别，logsumexp(logits) 正比于输入 x 在模型
隐式定义的分布下的对数密度（Liu et al. 2020, "Energy-based Out-of-distribution
Detection"，NeurIPS 2020）。真实植物图会和某个具体物种的相似度冒尖 → energy
高；OOD 图在 35 万个物种上相似度都偏平 → energy 低。

阈值标定（2026-09-26，独立测试站实测，未跑进本文件）：data/test_images 随机
300 个物种各取 1 张真实植物图，算 energy 后取 2 分位数 → 98% TPR。放宽到 98%
（而非更保守的 95%）是产品侧明确要求：宁可放过非植物（precision 可以低），
也不能拦真植物（recall 优先）——95% TPR 阈值下曾把一张明显的枫叶特写误拦。

已知限制：同批 60 张合成截图代理图测试显示，98% TPR 阈值下 FPR(截图)=100%，
即这版完全拦不住截图/文档类结构化非植物图，只挡"什么都不像"的极端输入
（纯色图、噪声图等）。energy 单信号无法同时兼顾"植物召回高"和"截图拦得住"
——TPR 从 95%→98%，FPR(截图) 从 35% 直接跳到 100%，是悬崖不是平滑 trade-off。

后续（v1.1/v2）：负概念库 zero-shot 或独立训练的小模型专门补上截图这块短板，
设计见本地 doc/ood-detection-notes.md + doc/demo-engineering-spec.md §B（不入库）。
"""

import numpy as np

THRESHOLD = 68.537  # 98% TPR（n=300 校准，见上）


def energy_score(sims: np.ndarray, logit_scale: float) -> float:
    """sims: 图像 embedding 与全量物种文字原型的余弦相似度数组（infer() 里已经算好，直接传入）。"""
    z = logit_scale * sims
    m = float(z.max())
    return m + float(np.log(np.sum(np.exp(z - m))))


def is_plant(energy: float) -> bool:
    return energy >= THRESHOLD
