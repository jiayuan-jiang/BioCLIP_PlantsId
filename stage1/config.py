"""Stage 1 配置 + 分片列表 + 默认超参。

依据: doc/stage1-spec.md §11（起始超参）、doc/spec/stage1-execution.md「不要重新讨论」表。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

# PlantMix-13M 分片 0..23（避开 79GB 异常分片 101.tar）。
# 消融迭代可回落到前 16 片省钱（见 stage1-spec §10）。
SHARDS: list[int] = list(range(24))
PLANTMIX_URL_TMPL = "https://huggingface.co/datasets/HEART77/plantmix/resolve/main/{shard}.tar"

# BioCLIP 2 = OpenCLIP ViT-L/14: width 1024, 24 blocks, mlp hidden 4096, proj 1024->768.
N_VISUAL_BLOCKS = 24


@dataclass
class Stage1Config:
    # ── 模型 ──────────────────────────────────────────
    model_name: str = "hf-hub:imageomics/bioclip-2"
    # 前 20 block 挂 LoRA；末 4 block 全量解冻。
    lora_blocks: tuple[int, ...] = tuple(range(20))          # 0..19
    unfrozen_blocks: tuple[int, ...] = (20, 21, 22, 23)
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    grad_checkpoint: bool = True
    # 冻结项（恒定，不消融）: 整个文本塔 + conv1 + ln_pre + logit_scale。

    # ── 数据 ──────────────────────────────────────────
    data_root: str = "data/stage1"
    manifest: str = "data/stage1/index.parquet"
    text_emb_path: str = "data/stage1/text_emb.f16.npy"
    img_dir: str = "data/stage1/img"
    image_size: int = 224
    rrc_scale: tuple[float, float] = (0.5, 1.0)              # crop 下限 0.5，非 0.08
    hflip: bool = True
    aug_prob: float = 0.5                                    # field-conditions 增强命中概率
    # 长尾采样
    tail_tau: float = 0.5                                    # n_c^(-tau)，sqrt 折中
    repeat_factor: bool = True                               # LVIS 式 max(1, sqrt(t/f_c))
    repeat_thresh: float = 1e-3

    # ── 损失（括号项均消融，默认只有 L_clip）──────────
    logit_scale_fixed: bool = True                           # 用 BioCLIP2 的值，不训
    memory_bank_k: int = 32_000
    fn_cos_thresh: float = 0.9                               # false-negative 屏蔽阈值
    use_distill: bool = False                                # L_distill，默认 off
    distill_lambda: float = 1.0
    distill_frac: float = 0.25                               # 仅 ~25% 样本算蒸馏
    use_mbdc: bool = False                                   # DALIP 二阶项
    mbdc_lambda: float = 0.0
    use_proto_ce: bool = False                               # species 原型 CE
    proto_ce_lambda: float = 0.0
    replay_ratio: float = 0.0                                # 通用回放比，默认 0；消融 {0,0.1,0.23}

    # ── 优化 ──────────────────────────────────────────
    lr_lora: float = 1e-4
    lr_unfrozen: float = 2e-5
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-6
    warmup_steps: int = 200
    epochs: int = 4
    batch_size: int = 256
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    amp_dtype: str = "bfloat16"

    # ── 训练期评测 / ckpt ─────────────────────────────
    eval_every: int = 2000
    eval_limit: int = 8000                                   # plantnet300k 评测抽样加速
    eval_data_root: str = "/workspace/eval_data"
    early_stop_patience: int = 2                             # plantnet300k top1 连续不升次数
    scoreboard: str = "eval_harness/scoreboard.csv"
    out_dir: str = "runs/stage1/baseline"
    seed: int = 0

    # ── 消融标签（写进 scoreboard model_id / ckpt 名）──
    run_tag: str = "baseline"

    def to_dict(self) -> dict:
        return asdict(self)
