"""图像 embedding 共享节点。

所有下游（公版物种识别 /predict、温室闭集 /matthaei/predict、以后的部位识别…）只消费这里的输出，
不再各自编码图像。节点做三件事：
  1. 去重：同一张图（字节 sha1 相同）正在算时，后来者等同一个结果，不重复算；
  2. 缓存：算过的 embedding 放 LRU，同一张图再来直接取；
  3. 微批：队列里的图凑成一批（最多 EMBED_MAX_BATCH 张，最多等 EMBED_BATCH_WAIT_MS 毫秒）一次前向，
     由单线程执行器串行跑——ViT-L 前向本身会吃满全部核，多个前向同时跑只会互相抢核。
"""
import asyncio
import hashlib
import io
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from inference_core import DEVICE, _res

MAX_BATCH = int(os.environ.get("EMBED_MAX_BATCH", 8))
BATCH_WAIT = float(os.environ.get("EMBED_BATCH_WAIT_MS", 30)) / 1000
CACHE_SIZE = int(os.environ.get("EMBED_CACHE", 256))


class ImageDecodeError(Exception):
    pass


class Emb:
    """一张图的 embedding；energy 是 gate 用的（对全量索引的 energy），算一次后各下游共用。"""
    __slots__ = ("vec", "energy")

    def __init__(self, vec: np.ndarray):
        self.vec = vec
        self.energy = None


class EmbedNode:
    def __init__(self):
        self._q = None
        self._worker = None
        self._cache: OrderedDict = OrderedDict()
        self._inflight: dict = {}
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")
        self.stats = {"requests": 0, "cache_hit": 0, "dedup": 0, "computed": 0, "batches": 0, "max_batch_seen": 0}

    def _ensure(self):
        if self._q is None:
            self._q = asyncio.Queue()
            self._worker = asyncio.get_running_loop().create_task(self._run())

    async def embed(self, data: bytes):
        """返回 (Emb, info)。info: cache=hit|dedup|miss，miss 时带 batch / embed_ms / queue_ms。"""
        self._ensure()
        self.stats["requests"] += 1
        key = hashlib.sha1(data).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            self.stats["cache_hit"] += 1
            return self._cache[key], {"cache": "hit"}
        if key in self._inflight:
            self.stats["dedup"] += 1
            emb, info = await asyncio.shield(self._inflight[key])
            return emb, {"cache": "dedup", "batch": info.get("batch")}
        fut = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        await self._q.put((key, data, fut, time.perf_counter()))
        try:
            emb, info = await asyncio.shield(fut)
        finally:
            self._inflight.pop(key, None)
        return emb, {"cache": "miss", **info}

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self._q.get()]
            deadline = loop.time() + BATCH_WAIT
            while len(batch) < MAX_BATCH:
                left = deadline - loop.time()
                if left <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._q.get(), left))
                except asyncio.TimeoutError:
                    break
            t_start = time.perf_counter()
            try:
                results = await loop.run_in_executor(self._exec, self._encode_batch, [b[1] for b in batch])
            except Exception as e:  # 整批失败（极少见）：全部报错
                results = [e] * len(batch)
            embed_ms = round((time.perf_counter() - t_start) * 1000, 1)
            self.stats["batches"] += 1
            self.stats["max_batch_seen"] = max(self.stats["max_batch_seen"], len(batch))
            for (key, _data, fut, t_enq), r in zip(batch, results):
                if isinstance(r, Exception):
                    fut.set_exception(r)
                    continue
                emb = Emb(r)
                self._cache[key] = emb
                while len(self._cache) > CACHE_SIZE:
                    self._cache.popitem(last=False)
                self.stats["computed"] += 1
                fut.set_result((emb, {"batch": len(batch), "embed_ms": embed_ms,
                                      "queue_ms": round((t_start - t_enq) * 1000, 1)}))

    @staticmethod
    def _encode_batch(datas: list) -> list:
        """在执行器线程里：逐张解码+预处理，成功的凑一批前向；解码失败的只让那一张报错。"""
        pre, model = _res["preprocess"], _res["model"]
        out, tensors, where = [None] * len(datas), [], []
        for i, d in enumerate(datas):
            try:
                tensors.append(pre(Image.open(io.BytesIO(d)).convert("RGB")))
                where.append(i)
            except Exception as e:
                out[i] = ImageDecodeError(f"图片解析失败：{e}")
        if tensors:
            with torch.no_grad():
                e = model.encode_image(torch.stack(tensors).to(DEVICE))
                e = (e / e.norm(dim=-1, keepdim=True)).cpu().float().numpy()
            for i, v in zip(where, e):
                out[i] = v
        return out


node = EmbedNode()
