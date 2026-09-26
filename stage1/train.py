"""Stage 1 训练循环。pod 上跑。

param groups -> AdamW -> cosine+warmup -> bf16 autocast -> grad-checkpoint
每 N step 调 eval_harness 跑 plantnet300k（+ rare/imagenetv2 诊断）-> scoreboard -> 早停
ckpt 只存 requires_grad 参数（LoRA A/B + 末段全量 + ln_post/proj）。

依据: doc/spec/stage1-execution.md Phase 2 · doc/stage1-spec.md §11（超参）。

用法:
  python -m stage1.train --run-tag baseline --shards 24 \
    --data-root data/stage1 --eval-data-root /workspace/eval_data \
    --out runs/stage1/baseline
  # 冒烟: --max-steps 20 --eval-every 0 --batch-size 8 --device cpu
"""

from __future__ import annotations

import os

# DataLoader worker 里的 numpy 增强别让 BLAS 按 96 核开线程池（会和 24 个 worker 互相超订）。
# GPU 训练主进程的算力在 CUDA 上，不受影响。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Stage1Config
from .data import Stage1Dataset, load_text_emb_pool
from .losses import feature_distill, locked_text_infonce, proto_ce, sample_negative_bank
from .model import build_model, trainable_state_dict


# ─────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────

def lr_lambda(step: int, warmup: int, total: int):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


# ─────────────────────────────────────────────────────────────
# 训练期评测（in-process 复用 eval_harness）
# ─────────────────────────────────────────────────────────────

def run_eval(model, preprocess, tokenizer, cfg: Stage1Config, step: int,
             model_id: str) -> dict:
    """in-process 复用 eval_harness。

    文本塔全程冻结（== BioCLIP2）→ 所有 Stage-1 评测共用同一套文本原型：
    用**稳定** text-cache id 让 `evaluate` 只在首次评测编码一次，之后秒出；
    scoreboard 行再改写成 step-tagged id。
    任一基准评测异常不致命（不中断训练）。
    """
    from eval_harness import benchmarks as B
    from eval_harness.zeroshot_eval import append_scoreboard, evaluate

    want = ["plantnet300k", "rare_species", "imagenetv2"]
    try:
        benches = B.discover(cfg.eval_data_root, want)
    except Exception as e:
        print(f"  [eval] discover 失败: {e}")
        return {}
    if not benches:
        print("  [eval] 未发现 eval 数据，跳过")
        return {}
    if cfg.eval_limit:
        benches = [b.subsample(cfg.eval_limit) if b.name == "plantnet300k" else b
                   for b in benches]

    class A:
        pass
    a = A()
    a.model_id = "stage1-textfrozen"        # 稳定 → 文本原型缓存复用
    a.batch_size = 256
    a.workers = 8
    a.per_class_dir = "eval_harness/per_class"
    a.no_text_cache = False

    device = next(model.parameters()).device.type
    was_training = model.training
    model.eval()
    rows = {}
    with torch.no_grad():
        for b in benches:
            try:
                r = evaluate(model, preprocess, tokenizer, b, device, a)
                r["model_id"] = f"{model_id}@{step}"     # scoreboard 行按 step 标注
                rows[b.name] = r
            except Exception as e:
                print(f"  [eval] {b.name} 失败: {e}")
    if was_training:
        model.train()

    if rows:
        append_scoreboard(cfg.scoreboard, list(rows.values()))
    return rows


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def build_cfg(args) -> Stage1Config:
    cfg = Stage1Config()
    over = dict(
        run_tag=args.run_tag,
        data_root=args.data_root,
        manifest=str(Path(args.data_root) / "index.parquet"),
        text_emb_path=str(Path(args.data_root) / "text_emb.f16.npy"),
        img_dir=str(Path(args.data_root) / "img"),
        eval_data_root=args.eval_data_root,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        eval_every=args.eval_every,
        memory_bank_k=args.memory_k,
        lora_r=args.lora_r,
        seed=args.seed,
        grad_checkpoint=bool(args.grad_ckpt),
        early_stop_patience=args.patience,
    )
    if args.lr_lora is not None:
        over["lr_lora"] = args.lr_lora
    if args.lr_unfrozen is not None:
        over["lr_unfrozen"] = args.lr_unfrozen
    if args.tail_tau is not None:
        over["tail_tau"] = args.tail_tau
    if args.aug_prob is not None:
        over["aug_prob"] = args.aug_prob
    if args.unfreeze_depth is not None:
        over["unfrozen_blocks"] = tuple(range(24 - args.unfreeze_depth, 24))
        over["lora_blocks"] = tuple(range(24 - args.unfreeze_depth))
    if args.replay_ratio is not None:
        over["replay_ratio"] = args.replay_ratio
    if args.use_distill:
        over["use_distill"] = True
    cfg = replace(cfg, **over)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="baseline")
    ap.add_argument("--data-root", default="data/stage1")
    ap.add_argument("--eval-data-root", default="/workspace/eval_data")
    ap.add_argument("--out", default="runs/stage1/baseline")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--memory-k", type=int, default=32000)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--grad-ckpt", type=int, default=1, help="1=grad checkpointing on（省显存慢）, 0=off（快，需显存够）")
    ap.add_argument("--lr-lora", type=float, default=None)
    ap.add_argument("--lr-unfrozen", type=float, default=None)
    ap.add_argument("--patience", type=int, default=2, help="plantnet300k top1 连续不升多少次评测就早停")
    ap.add_argument("--proto-ce-lambda", type=float, default=0.0,
                    help=">0 时对含 binomial 的样本加 λ·CE(image, 干净 species 原型)（stage1-spec §3）")
    ap.add_argument("--proto-logit-scale", type=float, default=0.0,
                    help=">0 时 protoCE 用独立（较低，如 30）的 logit_scale，替代主 100")
    ap.add_argument("--proto-logit-adjust", type=int, default=0,
                    help="1=protoCE logits 加 −log π_c（Menon logit adjustment，长尾）")
    ap.add_argument("--tail-tau", type=float, default=None, help="温度采样 n_c^(-tau)（消融 0/0.5/1）")
    ap.add_argument("--aug-prob", type=float, default=None, help="field-conditions 增强命中概率（消融）")
    ap.add_argument("--unfreeze-depth", type=int, default=None, help="末段全量解冻 block 数（消融）")
    ap.add_argument("--replay-ratio", type=float, default=None)
    ap.add_argument("--use-distill", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help=">0 时限制总 step（冒烟）")
    ap.add_argument("--shards", default=None,
                    help="训练用分片子集，如 '0-15'（消融回落）；默认全部")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = build_cfg(args)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    log = open(out / "train.log", "a")

    def P(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    P(f"=== Stage 1 train  tag={cfg.run_tag}  device={device}  {time.strftime('%F %T')} ===")
    P("config:", json.dumps(cfg.to_dict(), default=str))

    # 模型
    model, preprocess, pgroups = build_model(cfg, device=device)
    import open_clip
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    model.train()

    teacher = None
    if cfg.use_distill:
        t_model, _, _ = open_clip.create_model_and_transforms(cfg.model_name)
        teacher = t_model.visual.eval().to(device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        P("[distill] teacher visual 已加载（冻结）")

    # 数据
    shard_subset = None
    if args.shards:
        a, b = (args.shards.split("-") + [args.shards])[:2]
        shard_subset = list(range(int(a), int(b) + 1))
    ds = Stage1Dataset(cfg.manifest, cfg.text_emb_path, cfg, train=True, shards=shard_subset)
    P(f"[data] manifest {len(ds)} 行  dim={ds.dim}  shards={shard_subset or 'ALL'}")
    steps_per_epoch = max(1, len(ds) // (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sampler = ds.make_sampler(num_samples=cfg.batch_size * cfg.grad_accum * total_steps)
    dl = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler,
                    num_workers=args.workers, pin_memory=(device == "cuda"),
                    drop_last=True, persistent_workers=(args.workers > 0))

    # species 原型（protoCE，stage1-spec §3）—— 冻结文本塔一次性编码全部 unique binomial 的干净模板
    species_proto = None
    binom2idx = {}
    proto_logit_adj = None
    proto_scale = args.proto_logit_scale if args.proto_logit_scale else None  # None -> 用主 logit_scale
    if args.proto_ce_lambda > 0:
        from eval_harness import prompts as _P
        binoms_all = ds.binomial.tolist()
        uniq = sorted({b for b in binoms_all if b})
        binom2idx = {b: i for i, b in enumerate(uniq)}
        if args.proto_logit_adjust:                    # Menon logit adjustment: logits += −log π_c
            import collections
            cnt = collections.Counter(b for b in binoms_all if b)
            freq = np.array([cnt[b] for b in uniq], dtype=np.float64)
            freq = freq / freq.sum()
            proto_logit_adj = torch.tensor(-np.log(np.clip(freq, 1e-12, None)),
                                           dtype=torch.float32, device=device)
            P(f"[protoCE] logit adjustment on (−log π_c, range [{proto_logit_adj.min():.2f},{proto_logit_adj.max():.2f}])")
        P(f"[protoCE] {len(uniq)} unique species, λ={args.proto_ce_lambda} "
          f"proto_scale={proto_scale or 'main(100)'}  编码原型…")
        protos = []
        with torch.no_grad():
            for s in range(0, len(uniq), 384):                  # 384 species × 4 模板 = 1536 seq/批（文本塔 MLP 激活别爆）
                chunk = uniq[s:s + 384]
                flat, spans = [], []
                for nm in chunk:
                    ps = _P.render_class(nm, "plant", None)
                    spans.append((len(flat), len(flat) + len(ps)))
                    flat.extend(ps)
                tok = tokenizer(flat).to(device)
                f = model.encode_text(tok).float()
                f = torch.nn.functional.normalize(f, dim=-1)
                for a2, b2 in spans:
                    v = f[a2:b2].mean(0)
                    protos.append((v / v.norm()).half().cpu())
                if s % 7680 == 0:
                    P(f"  [protoCE] {s}/{len(uniq)}")
        species_proto = torch.stack(protos).to(device)          # [C, 768] fp16, normalized
        torch.cuda.empty_cache()
        P(f"[protoCE] species_proto {tuple(species_proto.shape)} "
          f"{species_proto.element_size()*species_proto.nelement()/1e6:.0f} MB")

    # 文本 emb 池（记忆库）
    pool = load_text_emb_pool(cfg.text_emb_path, device=device, dtype=torch.float16)
    P(f"[bank] text_emb pool {tuple(pool.shape)}  {pool.element_size()*pool.nelement()/1e9:.2f} GB on {device}")

    logit_scale = float(torch.tensor(4.6052).exp())     # BioCLIP2 固定值 = 100.0

    opt = torch.optim.AdamW(pgroups, betas=cfg.betas, eps=cfg.eps,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: lr_lambda(s, cfg.warmup_steps, total_steps))
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16

    start_step = 0
    if args.resume and Path(args.resume).is_file():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step = ck["step"]
        P(f"[resume] 从 step {start_step} 恢复")

    # 早停状态
    best_top1 = -1.0
    bad_evals = 0
    P(f"[plan] steps_per_epoch={steps_per_epoch}  total_steps={total_steps}")

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)
    t0 = time.time()
    step = start_step
    opt.zero_grad(set_to_none=True)
    accum = 0
    running = {}

    for img, emb_idx, meta in dl:
        img = img.to(device, non_blocking=True)
        emb_idx = emb_idx.to(device, non_blocking=True)

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
            z = model.encode_image(img)                       # [B, 768]
            t_pos = pool[emb_idx].float()                     # [B, 768]
            neg_idx = sample_negative_bank(pool, cfg.memory_bank_k, emb_idx, generator=None)
            t_neg = pool.index_select(0, neg_idx).float()
            loss, stats = locked_text_infonce(z, t_pos, t_neg, logit_scale,
                                              cfg.fn_cos_thresh)
            if teacher is not None and cfg.distill_lambda > 0:
                b = img.shape[0]
                k = max(1, int(b * cfg.distill_frac))
                with torch.no_grad():
                    f_t = teacher(img[:k])
                f_s = model.visual(img[:k])
                dl_loss = feature_distill(f_s, f_t)
                loss = loss + cfg.distill_lambda * dl_loss
                stats["distill"] = float(dl_loss)

            if species_proto is not None:
                sp_idx = torch.tensor([binom2idx.get(bn, -1) for bn in meta["binomial"]],
                                      device=device)
                m = sp_idx >= 0
                if m.any():
                    pce = proto_ce(z[m], species_proto, sp_idx[m],
                                   proto_scale if proto_scale else logit_scale,
                                   logit_adjust=proto_logit_adj)
                    loss = loss + args.proto_ce_lambda * pce
                    stats["proto_ce"] = float(pce)

        (loss / cfg.grad_accum).backward()
        accum += 1
        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v
        running["loss"] = running.get("loss", 0.0) + float(loss)

        if accum < cfg.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(
            [p for g in pgroups for p in g["params"]], cfg.max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        accum = 0
        step += 1

        if step % 50 == 0:
            n = 50 * cfg.grad_accum
            rate = (step - start_step) * cfg.batch_size * cfg.grad_accum / (time.time() - t0)
            msg = "  ".join(f"{k}={v/n:.4f}" for k, v in sorted(running.items()))
            P(f"step {step}/{total_steps}  lr={sched.get_last_lr()[0]:.2e}  "
              f"{rate:.0f} img/s  {msg}")
            running = {}

        # 训练期评测 + 早停
        if cfg.eval_every and step % cfg.eval_every == 0:
            P(f"--- eval @ step {step} ---")
            rows = run_eval(model, preprocess, tokenizer, cfg, step,
                            model_id=f"stage1-{cfg.run_tag}")
            pn = rows.get("plantnet300k", {})
            top1 = pn.get("top1", 0.0)
            P(f"  plantnet300k top1={top1}  macro_recall={pn.get('macro_recall')}  "
              f"rare={rows.get('rare_species',{}).get('top1')}  "
              f"inv2={rows.get('imagenetv2',{}).get('top1')}")
            # 存 ckpt
            ckpt = out / f"ckpt_step{step}.pt"
            torch.save({"step": step, "model": trainable_state_dict(model),
                        "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "eval": {k: v for k, v in rows.items()}}, ckpt)
            P(f"  saved {ckpt}")
            if top1 > best_top1 + 1e-4:
                best_top1 = top1
                bad_evals = 0
                torch.save({"step": step, "model": trainable_state_dict(model),
                            "eval": rows}, out / "best.pt")
                P(f"  new best top1={best_top1} -> best.pt")
            else:
                bad_evals += 1
                P(f"  no improve ({bad_evals}/{cfg.early_stop_patience})")
                if bad_evals >= cfg.early_stop_patience:
                    P("  *** 早停 ***")
                    break

        if total_steps and step >= total_steps:
            break

    # 收尾
    torch.save({"step": step, "model": trainable_state_dict(model)}, out / "final.pt")
    P(f"=== 训练结束 step={step}  best_plantnet_top1={best_top1}  "
      f"wall={(time.time()-t0)/3600:.2f}h ===")
    log.close()


if __name__ == "__main__":
    main()
