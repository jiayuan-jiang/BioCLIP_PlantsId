"""构建 Stage 1 训练模型: load BioCLIP2 -> freeze/unfreeze -> 注入 LoRA -> grad-ckpt。

冻结/解冻表（doc/stage1-spec.md §2）:
  文本塔全部        冻结
  visual.conv1      冻结
  visual.ln_pre     冻结
  logit_scale       冻结（用 BioCLIP2 的值）
  resblocks 0..19   base 冻结 + LoRA(r=32, α=64, drop=0.05) 挂 attn(QKV+out) / mlp(c_fc,c_proj)
  resblocks 20..23  全量解冻
  visual.ln_post    全量解冻
  visual.proj       全量解冻

期望可训练参数 ≈ 55–65M（LoRA 20 blk ≈ 10.5M + 末 4 blk 全量 ≈ 50M + ln_post/proj ≈ 0.8M）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Stage1Config
from .lora import inject_lora, is_lora_param


def _freeze_all(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def build_model(cfg: Stage1Config, device: str = "cpu"):
    """返回 (model, preprocess_val, param_groups)。

    param_groups = [{"params": [...LoRA...], "lr": cfg.lr_lora, "name": "lora"},
                    {"params": [...解冻...], "lr": cfg.lr_unfrozen, "name": "unfrozen"}]
    """
    import open_clip

    model, _, preprocess_val = open_clip.create_model_and_transforms(cfg.model_name)
    model = model.to(device)

    # 1) 先整体冻结
    _freeze_all(model)

    visual = model.visual
    rb = visual.transformer.resblocks
    assert len(rb) == 24, f"期望 ViT-L/14 24 blocks，实际 {len(rb)}"

    # 2) 注入 LoRA（blocks 0..19），base 权重保持冻结，只有 lora_A/lora_B 可训
    n_lora_mods = inject_lora(visual, cfg.lora_blocks, cfg.lora_r, cfg.lora_alpha,
                              cfg.lora_dropout)

    # 3) 末 4 block 全量解冻
    for i in cfg.unfrozen_blocks:
        for p in rb[i].parameters():
            p.requires_grad_(True)

    # 4) ln_post + proj 全量解冻
    for p in visual.ln_post.parameters():
        p.requires_grad_(True)
    if isinstance(visual.proj, nn.Parameter):
        visual.proj.requires_grad_(True)

    # 5) 恒冻结项再确认（conv1 / ln_pre / class_embedding / positional_embedding / 文本塔 / logit_scale）
    _freeze_all(visual.conv1)
    _freeze_all(visual.ln_pre)
    visual.class_embedding.requires_grad_(False)
    visual.positional_embedding.requires_grad_(False)
    if cfg.logit_scale_fixed:
        model.logit_scale.requires_grad_(False)
    # 文本塔: open_clip CLIP 的 token_embedding / positional_embedding / transformer / ln_final / text_projection
    for attr in ("token_embedding", "ln_final"):
        if hasattr(model, attr):
            _freeze_all(getattr(model, attr))
    if hasattr(model, "transformer"):      # 文本 transformer（与 visual.transformer 不同对象）
        _freeze_all(model.transformer)
    for attr in ("text_projection", "positional_embedding", "logit_bias"):
        t = getattr(model, attr, None)
        if isinstance(t, nn.Parameter):
            t.requires_grad_(False)

    # 6) grad checkpointing（仅可训 block 受益，但整体开销可接受）
    if cfg.grad_checkpoint:
        visual.transformer.grad_checkpointing = True

    model = model.to(device)
    pgroups = param_groups(model, cfg)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[build_model] LoRA 模块 {n_lora_mods}  可训练 {n_train/1e6:.1f}M / {n_total/1e6:.1f}M "
          f"({100*n_train/n_total:.1f}%)")
    return model, preprocess_val, pgroups


def param_groups(model: nn.Module, cfg: Stage1Config) -> list[dict]:
    lora, unfrozen = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (lora if is_lora_param(name) else unfrozen).append(p)
    groups = []
    if lora:
        groups.append({"params": lora, "lr": cfg.lr_lora, "name": "lora"})
    if unfrozen:
        groups.append({"params": unfrozen, "lr": cfg.lr_unfrozen, "name": "unfrozen"})
    return groups


def trainable_state_dict(model: nn.Module) -> dict:
    """只取 requires_grad 的参数（LoRA A/B + 末段全量 + ln_post/proj），用于轻量 ckpt。"""
    train_names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {n: v.detach().cpu() for n, v in model.state_dict().items() if n in train_names}


def load_trainable_(model: nn.Module, sd: dict):
    missing = model.load_state_dict(sd, strict=False)
    return missing


@torch.no_grad()
def summarize_freeze(model: nn.Module) -> dict:
    """调试用: 分组统计冻结/可训。"""
    buckets: dict[str, list[int]] = {}
    for name, p in model.named_parameters():
        if name.startswith("visual.transformer.resblocks."):
            key = "visual.resblocks." + name.split(".")[3]
        elif name.startswith("visual."):
            key = "visual." + name.split(".")[1]
        else:
            key = name.split(".")[0]
        buckets.setdefault(key, [0, 0])
        buckets[key][0 if p.requires_grad else 1] += p.numel()
    return {k: {"trainable": v[0], "frozen": v[1]} for k, v in sorted(buckets.items())}
