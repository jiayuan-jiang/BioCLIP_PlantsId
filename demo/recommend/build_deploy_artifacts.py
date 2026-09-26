r"""配图推荐 —— 部署产物构建（Mac，一次性）。见 doc/spec/recommend-deploy-execution.md §2。

输入（默认路径，可 --pool / --proto / --organ 覆盖）：
  inat_pool.npz            emb[N,768 fp16] · species_idx[N] · dhash[N,8] · url/license/observer_login/photo_id
  inat_species_proto.npz   proto[20000,768 fp16 归一] · species[20000]   （按 species_idx 对齐）
  organ_tagger_v3.npz      coef[5,768] · intercept[5] · classes[5]

产出 --out 目录（默认 demo/deploy_artifacts/）—— 全部单文件 .npy，运行时 mmap（不进内存）：
  recommend_emb.f16.npy       [N,768] fp16
  recommend_protos.f16.npy    [20000,768] fp16      物种原型（= proto 重归一化，小，运行时全读）
  recommend_topm_idx.npy      [N,32] uint16         每照片 top-32 种 ID
  recommend_topm_cos.npy      [N,32] fp16           对应 cos
  recommend_photo_id.npy      [N] int32
  recommend_species_idx.npy   [N] int32
  recommend_dhash.npy         [N,8] uint8
  recommend_ext.npy           [N] uint8             → aux["ext_vocab"]
  recommend_license.npy       [N] uint8             → aux["license_vocab"]
  recommend_observer.npy      [N] int32             → recommend_observer_vocab.json
  recommend_aux.json          {ext_vocab, license_vocab, url_template, url_exceptions:{idx:url}}
  recommend_observer_vocab.json  [~3万 个 login]
  recommend_species.json      [20000] 名字，index = species_idx
  organ_tagger_v3.npz         原样拷贝
  recommend_index.faiss       IndexHNSWPQ(d=768, pq_m=384, M=32, nbits=8), METRIC_L2, efC=40

已存在的产物默认跳过（--force 强制重建）。
跑： conda run -n siglip_env python demo/recommend/build_deploy_artifacts.py
"""
import argparse
import gc
import json
import os
import re
import shutil
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = "/Volumes/Jiayuan_2T/to_mac/demo_recommend"
URL_RX = re.compile(r"^https://inaturalist-open-data\.s3\.amazonaws\.com/photos/(\d+)/medium\.([a-z0-9]+)$")
URL_TMPL = "https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/medium.{ext}"

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default=os.path.join(EXT, "inat_pool.npz"))
ap.add_argument("--proto", default=os.path.join(EXT, "inat_species_proto.npz"))
ap.add_argument("--organ", default=os.path.join(HERE, "..", "..", "eval_harness", "organ_probe", "organ_tagger_v3.npz"))
ap.add_argument("--out", default=os.path.join(HERE, "..", "deploy_artifacts"))
ap.add_argument("--force", action="store_true", help="重建已存在的产物")
ap.add_argument("--pq-m", type=int, default=384)
ap.add_argument("--hnsw-m", type=int, default=32)
ap.add_argument("--ef-construction", type=int, default=40)
ap.add_argument("--train-sample", type=int, default=200_000)
ap.add_argument("--topm", type=int, default=32)
ap.add_argument("--chunk", type=int, default=20_000)
A = ap.parse_args()

os.makedirs(A.out, exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def O(name):
    return os.path.join(A.out, name)


def need(*names):
    """任一目标缺失（或 --force）→ 需要这一步。"""
    return A.force or any(not os.path.exists(O(n)) for n in names)


emb_path = O("recommend_emb.f16.npy")
proto_out = O("recommend_protos.f16.npy")

# ── pool（除 emb / faiss 外都要它）─────────────────────────────────────
_pool = None


def pool():
    global _pool
    if _pool is None:
        log(f"load pool: {A.pool}")
        _pool = np.load(A.pool, allow_pickle=True)
    return _pool


# ── 1. emb 落盘 ──────────────────────────────────────────────────────
if need("recommend_emb.f16.npy"):
    emb = pool()["emb"]
    assert emb.dtype == np.float16 and emb.shape[1] == 768, emb.shape
    np.save(emb_path, emb)
    log(f"saved emb {emb.shape}")
    del emb
    gc.collect()
else:
    log("skip emb（已存在）")
N = np.load(emb_path, mmap_mode="r").shape[0]
log(f"N = {N:,}")

# ── 2. protos ───────────────────────────────────────────────────────
pz = np.load(A.proto, allow_pickle=True)
names = [str(s) for s in pz["species"]]
S = len(names)
if need("recommend_protos.f16.npy"):
    Q = pz["proto"].astype(np.float32)
    assert Q.shape == (S, 768), Q.shape
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    np.save(proto_out, Q.astype(np.float16))
    log(f"saved protos ({S})")
    del Q
del pz
gc.collect()

# ── 3. species.json + organ tagger ─────────────────────────────────
if need("recommend_species.json"):
    json.dump(names, open(O("recommend_species.json"), "w"))
    log("saved species.json")
if need("organ_tagger_v3.npz"):
    shutil.copyfile(os.path.abspath(A.organ), O("organ_tagger_v3.npz"))
    _og = np.load(O("organ_tagger_v3.npz"), allow_pickle=True)
    log(f"copied organ_tagger_v3.npz classes={list(_og['classes'])}")
    del _og

# ── 4. meta 拆成单文件 .npy + 字典编码 ─────────────────────────────
META = ["recommend_photo_id.npy", "recommend_species_idx.npy", "recommend_dhash.npy",
        "recommend_ext.npy", "recommend_license.npy", "recommend_observer.npy",
        "recommend_aux.json", "recommend_observer_vocab.json"]
if need(*META):
    P = pool()
    photo_id = P["photo_id"].astype(np.int32)
    species_idx = P["species_idx"].astype(np.int32)
    dhash = P["dhash"].astype(np.uint8)
    url = P["url"].astype(str)
    lic = P["license"].astype(str)
    obs = P["observer_login"].astype(str)
    assert 0 <= species_idx.min() and species_idx.max() < S

    np.save(O("recommend_photo_id.npy"), photo_id)
    np.save(O("recommend_species_idx.npy"), species_idx)
    np.save(O("recommend_dhash.npy"), dhash)

    # ext：URL 结尾扩展名（不匹配模板的进 url_exceptions）
    ext_vocab, ext_i = [], {}
    ext_code = np.zeros(N, np.uint8)
    url_exc = {}
    for i, (u, pid) in enumerate(zip(url, photo_id)):
        m = URL_RX.match(u)
        if m and int(m.group(1)) == int(pid):
            e = m.group(2)
        else:
            url_exc[str(i)] = u
            e = "jpg"  # 占位；真正用 exceptions
        if e not in ext_i:
            ext_i[e] = len(ext_vocab)
            ext_vocab.append(e)
        ext_code[i] = ext_i[e]
    np.save(O("recommend_ext.npy"), ext_code)
    log(f"ext_vocab={ext_vocab}  url_exceptions={len(url_exc)}")

    # license：3 值 → uint8
    lic_vocab, lic_i = [], {}
    lic_code = np.zeros(N, np.uint8)
    for i, x in enumerate(lic):
        if x not in lic_i:
            lic_i[x] = len(lic_vocab)
            lic_vocab.append(x)
        lic_code[i] = lic_i[x]
    np.save(O("recommend_license.npy"), lic_code)
    log(f"license_vocab={lic_vocab}")

    # observer：~3万 unique → int32 code + vocab
    obs_vocab, obs_i = [], {}
    obs_code = np.zeros(N, np.int32)
    for i, x in enumerate(obs):
        if x not in obs_i:
            obs_i[x] = len(obs_vocab)
            obs_vocab.append(x)
        obs_code[i] = obs_i[x]
    np.save(O("recommend_observer.npy"), obs_code)
    json.dump(obs_vocab, open(O("recommend_observer_vocab.json"), "w"))
    log(f"observer vocab {len(obs_vocab)}")

    json.dump({"ext_vocab": ext_vocab, "license_vocab": lic_vocab,
               "url_template": URL_TMPL, "url_exceptions": url_exc},
              open(O("recommend_aux.json"), "w"))
    log("saved recommend_aux.json + meta .npy ×6")
    del photo_id, species_idx, dhash, url, lic, obs, ext_code, lic_code, obs_code
    gc.collect()
else:
    log("skip meta（已存在）")

# ── 5. topm：idx + cos 各一个 .npy ─────────────────────────────────
if need("recommend_topm_idx.npy", "recommend_topm_cos.npy"):
    if os.path.exists(O("recommend_topm.npz")) and not A.force:
        log("从旧 recommend_topm.npz 转 .npy")
        z = np.load(O("recommend_topm.npz"))
        np.save(O("recommend_topm_idx.npy"), z["topm_idx"].astype(np.uint16))
        np.save(O("recommend_topm_cos.npy"), z["topm_cos"].astype(np.float16))
        del z
    else:
        log(f"topm: E @ Q.T  chunk={A.chunk}")
        Q = np.load(proto_out).astype(np.float32)
        Qt = np.ascontiguousarray(Q.T)
        emb_mm = np.load(emb_path, mmap_mode="r")
        topm_idx = np.zeros((N, A.topm), np.uint16)
        topm_cos = np.zeros((N, A.topm), np.float16)
        for s in range(0, N, A.chunk):
            e = min(s + A.chunk, N)
            C = emb_mm[s:e].astype(np.float32) @ Qt
            part = np.argpartition(-C, A.topm, axis=1)[:, :A.topm]
            rows = np.arange(e - s)[:, None]
            order = np.argsort(-C[rows, part], axis=1)
            part = part[rows, order]
            topm_idx[s:e] = part.astype(np.uint16)
            topm_cos[s:e] = C[rows, part].astype(np.float16)
            if (s // A.chunk) % 10 == 0:
                log(f"  topm {e:,}/{N:,}")
            del C, part, rows, order
        np.save(O("recommend_topm_idx.npy"), topm_idx)
        np.save(O("recommend_topm_cos.npy"), topm_cos)
        del topm_idx, topm_cos, emb_mm, Q, Qt
    gc.collect()
    log("saved topm .npy ×2")
else:
    log("skip topm（已存在）")

# ── 6. faiss HNSW+PQ ──────────────────────────────────────────────
if need("recommend_index.faiss"):
    import faiss  # noqa: E402
    log("faiss: materialize fp32 E …")
    E = np.ascontiguousarray(np.load(emb_path).astype(np.float32))
    log(f"E {E.shape}  ({E.nbytes / 1e9:.1f} GB)")
    index = faiss.IndexHNSWPQ(768, A.pq_m, A.hnsw_m, 8)
    index.metric_type = faiss.METRIC_L2
    index.hnsw.efConstruction = A.ef_construction
    rng = np.random.default_rng(0)
    sub = np.ascontiguousarray(E[rng.choice(len(E), min(A.train_sample, len(E)), replace=False)])
    log(f"train PQ on {len(sub):,} …")
    index.train(sub)
    del sub
    gc.collect()
    log("add all …")
    index.add(E)
    faiss.write_index(index, O("recommend_index.faiss"))
    log(f"saved faiss  ntotal={index.ntotal:,}")
    del E, index
    gc.collect()
else:
    log("skip faiss（已存在）")

# ── 自检 ──────────────────────────────────────────────────────────
import faiss  # noqa: E402
idx = faiss.read_index(O("recommend_index.faiss"))
idx.hnsw.efSearch = 256
Qf = np.load(proto_out).astype(np.float32)
si = np.load(O("recommend_species_idx.npy"), mmap_mode="r")
rng = np.random.default_rng(1)
probe = rng.choice(len(Qf), 200, replace=False)
_, I = idx.search(np.ascontiguousarray(Qf[probe]), 1)
hit = np.mean([si[I[k, 0]] == probe[k] for k in range(len(probe))])
log(f"自检: 200 个物种原型 top-1 照片命中该种 = {hit:.3f}  (应 >0.9)")

# 清理旧的 .npz（已被 .npy 取代）
for old in ("recommend_meta.npz", "recommend_topm.npz"):
    if os.path.exists(O(old)):
        os.remove(O(old))
        log(f"removed 旧 {old}")

sz = sum(os.path.getsize(O(f)) for f in os.listdir(A.out))
log(f"DONE  产物目录 {sz / 1e9:.2f} GB")
