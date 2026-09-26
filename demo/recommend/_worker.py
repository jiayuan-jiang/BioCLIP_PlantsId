"""配图推荐进程池 worker —— 独立小模块，**不 import torch / inference_core**。

ProcessPoolExecutor(spawn) 的 worker 只会 import 这个模块 + recommend.engine
（numpy + faiss），不碰识图那套 1.7GB 模型。
"""
import os

_ENGINE = None


def init_worker(artifacts_dir: str, ef_search: int = 384):
    global _ENGINE
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    from recommend.engine import RecommendEngine

    _ENGINE = RecommendEngine()
    _ENGINE.load(artifacts_dir, ef_search=ef_search)


def run_job(kwargs: dict):
    return _ENGINE.run(**kwargs)
