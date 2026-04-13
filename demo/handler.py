"""
BioCLIP 2 — RunPod Serverless Handler
======================================
调用方式（JSON input）：
  {
    "input": {
      "image_base64": "<base64 编码的图片>",
      "topk": 5          // 可选，默认 5，范围 1–20
    }
  }

响应：
  {
    "results": [ { "rank": 1, "similarity": 0.42, ... }, ... ]
  }
"""

import base64
import io
import logging

import runpod
from PIL import Image

from inference_core import DEFAULT_TOPK, load_resources, infer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("bioclip")

# ── 冷启动时加载模型 & 向量库（每个 Worker 只执行一次）────────────────
load_resources()


def handler(job: dict) -> dict:
    """RunPod Serverless 入口函数，每次请求调用一次。"""
    job_input = job.get("input", {})

    # ── 解析输入 ──────────────────────────────────────────────────────
    image_b64 = job_input.get("image_base64")
    if not image_b64:
        return {"error": "缺少字段 image_base64"}

    topk = int(job_input.get("topk", DEFAULT_TOPK))
    topk = max(1, min(20, topk))

    # ── 解码图片 ──────────────────────────────────────────────────────
    try:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"图片解析失败：{e}"}

    # ── 推理 ──────────────────────────────────────────────────────────
    try:
        results = infer(img, topk)
    except Exception as e:
        logger.error("推理异常", exc_info=True)
        return {"error": f"推理失败：{e}"}

    return {"results": results}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
