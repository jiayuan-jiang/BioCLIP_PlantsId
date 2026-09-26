"""极简手写 LoRA —— 不依赖 peft。

兼容 open_clip ViT-L/14 visual 塔的两种层:
  1. `nn.Linear`（`mlp.c_fc` / `mlp.c_proj`）           -> `LoRALinear`
  2. `nn.MultiheadAttention`（打包 QKV `in_proj_weight`）-> `LoRAMultiheadAttention`
     （对 vision 自注意力重写 forward，用 `F.scaled_dot_product_attention`，
      QKV / out_proj 各挂一个 `LoRALinear`，所以 dropout 正常生效。）

设计依据: doc/spec/stage1-execution.md「LoRA 注入细节」段。

用法:
    inject_lora(model.visual, blocks=range(20), r=32, alpha=64, dropout=0.05)
    ...train...
    merge_lora_(model.visual)   # 原地换回标准 nn.Linear / nn.MultiheadAttention

merge 后 `model.visual` 的 state_dict key 与原始 ViT-L-14 一致 ->
可直接 load 进 `open_clip.create_model_and_transforms("ViT-L-14")` 或写回 BioCLIP2。
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# LoRALinear
# ─────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """冻结的 base 线性层 + 可训练低秩增量 (alpha/r)·B@A。

    y = F.linear(x, W0, b0) + scaling · (dropout(x) @ Aᵀ) @ Bᵀ
    A: [r, in]，B: [out, r]，B 零初始化 -> 训练起点等价 base。
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.scaling = alpha / r

        # base 权重冻结，作为 buffer 保留（不进 optimizer，但进 state_dict / .to / autocast）
        self.register_buffer("weight", base.weight.detach().clone())
        if base.bias is not None:
            self.register_buffer("bias", base.bias.detach().clone())
        else:
            self.bias = None

        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base + self.scaling * lora.to(base.dtype)

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        delta = self.scaling * (self.lora_B @ self.lora_A)
        return self.weight + delta.to(self.weight.dtype)

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        lin.weight.copy_(self.merged_weight())
        if self.bias is not None:
            lin.bias.copy_(self.bias)
        return lin


# ─────────────────────────────────────────────────────────────
# LoRAMultiheadAttention
# ─────────────────────────────────────────────────────────────

class LoRAMultiheadAttention(nn.Module):
    """替换 open_clip resblock 里的 `nn.MultiheadAttention`（batch_first, 自注意力）。

    call signature 与 nn.MultiheadAttention 一致，返回 (out, None)，
    以便 `ResidualAttentionBlock.attention()` 的 `[0]` 取值不变。
    """

    def __init__(self, mha: nn.MultiheadAttention, r: int, alpha: int, dropout: float):
        super().__init__()
        assert isinstance(mha, nn.MultiheadAttention)
        assert mha.batch_first, "open_clip 用 batch_first=True"
        assert mha._qkv_same_embed_dim, "期望打包 in_proj_weight"

        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads

        # 打包 QKV: [3E, E]，当作一个 in->3E 的线性层挂 LoRA
        qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=mha.in_proj_bias is not None)
        with torch.no_grad():
            qkv.weight.copy_(mha.in_proj_weight)
            if mha.in_proj_bias is not None:
                qkv.bias.copy_(mha.in_proj_bias)
        self.qkv = LoRALinear(qkv, r, alpha, dropout)

        # out_proj: nn.Linear（torch 里是 NonDynamicallyQuantizableLinear，鸭子类型可用）
        out = nn.Linear(self.embed_dim, self.embed_dim, bias=mha.out_proj.bias is not None)
        with torch.no_grad():
            out.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                out.bias.copy_(mha.out_proj.bias)
        self.out_proj = LoRALinear(out, r, alpha, dropout)

    def forward(self, query, key=None, value=None, need_weights: bool = False,
                attn_mask=None, **_):
        # vision 自注意力: query is key is value, attn_mask 恒为 None
        x = query
        N, L, E = x.shape
        H, D = self.num_heads, self.head_dim
        qkv = self.qkv(x)                                  # [N, L, 3E]
        q, k, v = qkv.split(E, dim=-1)

        def split_heads(t):
            return t.view(N, L, H, D).transpose(1, 2)      # [N, H, L, D]

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0)   # [N, H, L, D]
        o = o.transpose(1, 2).contiguous().view(N, L, E)
        o = self.out_proj(o)
        return o, None

    @torch.no_grad()
    def to_mha(self) -> nn.MultiheadAttention:
        mha = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        mha.in_proj_weight.copy_(self.qkv.merged_weight())
        if mha.in_proj_bias is not None and self.qkv.bias is not None:
            mha.in_proj_bias.copy_(self.qkv.bias)
        mha.out_proj.weight.copy_(self.out_proj.merged_weight())
        if mha.out_proj.bias is not None and self.out_proj.bias is not None:
            mha.out_proj.bias.copy_(self.out_proj.bias)
        return mha


# ─────────────────────────────────────────────────────────────
# 注入 / 合并
# ─────────────────────────────────────────────────────────────

def _resblocks(visual: nn.Module):
    return visual.transformer.resblocks


def inject_lora(visual: nn.Module, blocks: Iterable[int], r: int, alpha: int,
                dropout: float) -> int:
    """在 `visual.transformer.resblocks[i]`（i ∈ blocks）的 attn + mlp 上挂 LoRA。

    只动 blocks 列出的 block；其余 block 不碰（末段全量解冻在 model.py 处理）。
    返回注入的 LoRA 模块数。
    """
    rb = _resblocks(visual)
    blocks = set(int(b) for b in blocks)
    n = 0
    for i, blk in enumerate(rb):
        if i not in blocks:
            continue
        blk.attn = LoRAMultiheadAttention(blk.attn, r, alpha, dropout)
        blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, r, alpha, dropout)
        blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, r, alpha, dropout)
        n += 3
    return n


@torch.no_grad()
def merge_lora_(visual: nn.Module) -> int:
    """原地把所有 LoRA 模块换回标准 nn.Linear / nn.MultiheadAttention（权重已合并）。

    返回合并的模块数。merge 后 state_dict 与原始 ViT-L-14 对齐。
    """
    rb = _resblocks(visual)
    n = 0
    for blk in rb:
        if isinstance(blk.attn, LoRAMultiheadAttention):
            blk.attn = blk.attn.to_mha()
            n += 1
        if isinstance(blk.mlp.c_fc, LoRALinear):
            blk.mlp.c_fc = blk.mlp.c_fc.to_linear()
            n += 1
        if isinstance(blk.mlp.c_proj, LoRALinear):
            blk.mlp.c_proj = blk.mlp.c_proj.to_linear()
            n += 1
    return n


def lora_parameter_names(model: nn.Module) -> list[str]:
    return [n for n, _ in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def is_lora_param(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name
