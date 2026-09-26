"""WiSE-FT —— merge LoRA -> 对全部 visual 参数 θ = α·θ_tuned + (1−α)·θ_bioclip2 扫 α。

每个 α 跑 eval_harness scoreboard -> 选 plantnet300k 最优且诊断列不崩的 α。
最终产出 BioCLIP2-Plant（merged 全模型权重，文本塔原样 -> bioclip_full_index.npz 仍有效）。

依据: doc/spec/stage1-execution.md Phase 3 · doc/stage1-spec.md §2 末（训练后 WiSE-FT）。

用法:
  python -m stage1.wise_ft --ckpt runs/stage1/baseline/best.pt \
    --alphas 0.5 0.6 0.7 0.8 0.9 \
    --eval-data-root /workspace/eval_data \
    --out runs/stage1/wiseft --save-best runs/stage1/BioCLIP2-Plant.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import Stage1Config
from .lora import merge_lora_
from .model import build_model


def _tuned_visual_state(ckpt_path: str, cfg: Stage1Config, device: str) -> dict:
    """load base + LoRA/解冻 ckpt -> merge_lora_ -> 返回 merged visual state_dict（标准 key）。"""
    model, _, _ = build_model(cfg, device=device)
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    merge_lora_(model.visual)
    return {k: v.detach().cpu().clone() for k, v in model.visual.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Phase 2 产出的 best.pt / final.pt")
    ap.add_argument("--model", default="hf-hub:imageomics/bioclip-2")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--eval-data-root", default="/workspace/eval_data")
    ap.add_argument("--out", default="runs/stage1/wiseft")
    ap.add_argument("--save-best", default="runs/stage1/BioCLIP2-Plant.pt")
    ap.add_argument("--scoreboard", default="eval_harness/scoreboard.csv")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--unfreeze-depth", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = Stage1Config()
    from dataclasses import replace
    cfg = replace(cfg, lora_r=args.lora_r,
                  lora_blocks=tuple(range(24 - args.unfreeze_depth)),
                  unfrozen_blocks=tuple(range(24 - args.unfreeze_depth, 24)),
                  eval_data_root=args.eval_data_root, scoreboard=args.scoreboard)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import open_clip
    from eval_harness import benchmarks as B
    from eval_harness.zeroshot_eval import append_scoreboard, evaluate

    # base visual state
    base_model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    base_model = base_model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)
    base_vis = {k: v.detach().cpu().clone() for k, v in base_model.visual.state_dict().items()}

    # tuned visual state (merged)
    tuned_vis = _tuned_visual_state(args.ckpt, cfg, device)
    common = [k for k in base_vis if k in tuned_vis and base_vis[k].shape == tuned_vis[k].shape]
    print(f"[wise-ft] 可插值 visual 参数 key: {len(common)} / base {len(base_vis)}")

    want = ["plantnet300k", "rare_species", "imagenetv2"]

    class A:
        pass

    results = []
    for alpha in args.alphas:
        merged = {}
        for k in base_vis:
            if k in common:
                merged[k] = alpha * tuned_vis[k].float() + (1 - alpha) * base_vis[k].float()
                merged[k] = merged[k].to(base_vis[k].dtype)
            else:
                merged[k] = base_vis[k]
        base_model.visual.load_state_dict(merged, strict=True)
        base_model.eval()

        benches = B.discover(args.eval_data_root, want)
        benches = [b.subsample(0) for b in benches]
        a = A()
        a.model_id = f"stage1-wiseft-a{alpha:.2f}"
        a.batch_size = 256
        a.workers = 8
        a.per_class_dir = "eval_harness/per_class"
        a.no_text_cache = False
        rows = []
        with torch.no_grad():
            for b in benches:
                rows.append(evaluate(base_model, preprocess, tokenizer, b, device, a))
        append_scoreboard(args.scoreboard, rows)
        by = {r["benchmark"]: r for r in rows}
        pn = by.get("plantnet300k", {})
        rec = {"alpha": alpha, "plantnet_top1": pn.get("top1"),
               "plantnet_macro_recall": pn.get("macro_recall"),
               "rare_top1": by.get("rare_species", {}).get("top1"),
               "inv2_top1": by.get("imagenetv2", {}).get("top1")}
        results.append(rec)
        print(f"[alpha {alpha}] {rec}")

    (out / "wiseft_sweep.json").write_text(json.dumps(results, indent=2))

    # 选 α: plantnet top1 最优
    best = max(results, key=lambda r: (r["plantnet_top1"] or 0))
    print(f"\n[wise-ft] best alpha = {best['alpha']}  {best}")

    # 存 BioCLIP2-Plant（完整模型: 插值后的 visual + 原样文本塔）
    alpha = best["alpha"]
    merged = {}
    for k in base_vis:
        if k in common:
            merged[k] = (alpha * tuned_vis[k].float() + (1 - alpha) * base_vis[k].float()).to(base_vis[k].dtype)
        else:
            merged[k] = base_vis[k]
    base_model.visual.load_state_dict(merged, strict=True)
    Path(args.save_best).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": base_model.state_dict(),
                "arch": "ViT-L-14", "wiseft_alpha": alpha,
                "source_ckpt": args.ckpt, "sweep": results}, args.save_best)
    print(f"✓ BioCLIP2-Plant -> {args.save_best}  (alpha={alpha})")


if __name__ == "__main__":
    main()
