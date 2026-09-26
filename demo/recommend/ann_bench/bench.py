"""HNSW / HNSW+PQ recall@K 实测 —— 在真实 inat_pool 上（子采样，机器内存紧张见下）。

机器起点就有 swap 压力（跑之前 swap 已用 12.5/13.3GB），加上之前 BarkNet 编码把 swap 撑炸过一次，
这次保守：**子采样 500k / 全量 1.4M** 建索引（ground truth 也在同一个子集上算，否则 recall 没意义）。
每步之间显式 del + gc.collect()，进度写 checkpoint json，中途挂了不丢已跑完的。

阶段：sample -> gt(ground truth, 全暴力) -> 各 index 的 build+sweep efSearch -> 汇总

用法：python bench.py
输出：results.json（表）、recall_curves.png（图）
"""
import os, sys, json, time, gc, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
POOL_NPZ = "/Volumes/Jiayuan_2T/to_mac/demo_recommend/inat_pool.npz"
PROTO_NPZ = "/Volumes/Jiayuan_2T/to_mac/demo_recommend/inat_species_proto.npz"

N_SUB = 500_000          # 子采样数据库大小（资源安全，见上）
N_QUERY = 300            # 抽多少个物种当 query
K_MAX = 1000             # ground truth 存 top-1000，recall@10/100/1000 都从这里切
K_REPORT = [10, 100, 1000]
EF_SEARCH_LIST = [16, 32, 64, 128, 256, 512]
CHUNK = 100_000          # 暴力搜索时的分块大小
SEED = 0

RESULTS = os.path.join(HERE, "results.json")
STATE = os.path.join(HERE, "_state")
os.makedirs(STATE, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_results():
    if os.path.exists(RESULTS):
        return json.load(open(RESULTS))
    return {"configs": []}


def save_results(r):
    json.dump(r, open(RESULTS, "w"), indent=2)


# ── 1. 子采样（缓存到磁盘，下次直接读） ─────────────────────────────
sub_emb_path = os.path.join(STATE, "sub_emb.npy")
sub_meta_path = os.path.join(STATE, "sub_meta.npz")
if os.path.exists(sub_emb_path) and os.path.exists(sub_meta_path):
    log(f"子采样缓存已存在，跳过重建")
else:
    log("加载 inat_pool.npz …")
    P = np.load(POOL_NPZ, allow_pickle=True)
    N = len(P["species_idx"])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(N, size=N_SUB, replace=False)
    idx.sort()  # 顺序读，省 npz 内部解压开销
    sub_emb = P["emb"][idx].astype(np.float32)   # 500k*768*4 = 1.5GB
    sub_spi = P["species_idx"][idx]
    np.save(sub_emb_path, sub_emb)
    np.savez(sub_meta_path, species_idx=sub_spi, orig_idx=idx)
    log(f"子采样完成：{len(sub_emb)} 行，存到 {sub_emb_path}")
    del P, sub_emb, sub_spi
    gc.collect()

# ── 2. query：抽 N_QUERY 个物种原型 ──────────────────────────────
q_path = os.path.join(STATE, "queries.npy")
if os.path.exists(q_path):
    Q = np.load(q_path)
    log(f"query 缓存已存在：{len(Q)} 条")
else:
    SP = np.load(PROTO_NPZ, allow_pickle=True)
    rng = np.random.default_rng(SEED + 1)
    qidx = rng.choice(len(SP["proto"]), size=N_QUERY, replace=False)
    Q = SP["proto"][qidx].astype(np.float32)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    np.save(q_path, Q)
    log(f"query 采样完成：{len(Q)} 条，存到 {q_path}")
    del SP
    gc.collect()

sub_emb = np.load(sub_emb_path, mmap_mode="r")
N = len(sub_emb)
d = sub_emb.shape[1]
log(f"数据库子集：{N} × {d}，query {len(Q)} 条")

# ── 3. ground truth：全暴力，分块，缓存 ────────────────────────────
gt_path = os.path.join(STATE, "gt_top1000.npy")
if os.path.exists(gt_path):
    GT = np.load(gt_path)
    log(f"ground truth 缓存已存在：{GT.shape}")
else:
    log(f"算 ground truth（暴力，分块 {CHUNK}）…")
    t0 = time.time()
    best_val = np.full((len(Q), K_MAX), -1e9, dtype=np.float32)
    best_idx = np.full((len(Q), K_MAX), -1, dtype=np.int64)
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        chunk = np.ascontiguousarray(sub_emb[start:end])          # fp32, 已归一
        sims = Q @ chunk.T                                        # [nq, chunk] inner product = cos
        # 合并当前 best 和这块的结果，取每行 top-K_MAX
        merged_val = np.concatenate([best_val, sims], axis=1)
        merged_idx = np.concatenate([best_idx, np.arange(start, end)[None, :].repeat(len(Q), 0)], axis=1)
        top = np.argpartition(-merged_val, K_MAX - 1, axis=1)[:, :K_MAX]
        rows = np.arange(len(Q))[:, None]
        best_val = merged_val[rows, top]
        best_idx = merged_idx[rows, top]
        del chunk, sims, merged_val, merged_idx, top
        if start % (CHUNK * 4) == 0:
            log(f"  gt {end}/{N}  {time.time()-t0:.0f}s")
    # 每行按分数排序（前面只是 top-K 无序）
    order = np.argsort(-best_val, axis=1)
    rows = np.arange(len(Q))[:, None]
    GT = best_idx[rows, order]
    np.save(gt_path, GT)
    log(f"ground truth 完成，{time.time()-t0:.0f}s，存到 {gt_path}")
    del best_val, best_idx, order, rows
    gc.collect()

GT_SETS = {k: [set(GT[i, :k].tolist()) for i in range(len(Q))] for k in K_REPORT}
log("ground truth 就绪，开始建索引")

# ── 4. 各 index config：build + sweep efSearch ─────────────────────
import faiss

CONFIGS = [
    dict(name="HNSWFlat_M32", kind="flat", M=32, efC=40),
    dict(name="HNSWPQ_M32_m48", kind="pq", M=32, efC=40, pq_m=48, pq_nbits=8),
    dict(name="HNSWPQ_M32_m96", kind="pq", M=32, efC=40, pq_m=96, pq_nbits=8),
    dict(name="HNSWPQ_M32_m128", kind="pq", M=32, efC=40, pq_m=128, pq_nbits=8),
    dict(name="HNSWPQ_M32_m192", kind="pq", M=32, efC=40, pq_m=192, pq_nbits=8),
    dict(name="HNSWPQ_M32_m384", kind="pq", M=32, efC=40, pq_m=384, pq_nbits=8),
    dict(name="HNSWPQ_M32_m768", kind="pq", M=32, efC=40, pq_m=768, pq_nbits=8),
]

results = load_results()
done_names = {c["name"] for c in results["configs"]}

for cfg in CONFIGS:
    if cfg["name"] in done_names:
        log(f"跳过已完成: {cfg['name']}")
        continue
    log(f"=== 建索引: {cfg['name']} ===")
    t0 = time.time()
    if cfg["kind"] == "flat":
        index = faiss.IndexHNSWFlat(d, cfg["M"], faiss.METRIC_L2)
    else:
        index = faiss.IndexHNSWPQ(d, cfg["pq_m"], cfg["M"], cfg["pq_nbits"])
        index.metric_type = faiss.METRIC_L2
    index.hnsw.efConstruction = cfg["efC"]

    full = np.ascontiguousarray(sub_emb[:])   # 需要连续内存喂 faiss，这里才真正物化整份 fp32(1.5GB)
    if cfg["kind"] == "pq":
        log("  PQ 训练 codebook …")
        index.train(full)
    index.add(full)
    build_s = time.time() - t0
    log(f"  build 完成 {build_s:.0f}s，index.ntotal={index.ntotal}")
    del full
    gc.collect()

    cfg_result = dict(name=cfg["name"], kind=cfg["kind"], M=cfg["M"], build_s=round(build_s, 1),
                       pq_m=cfg.get("pq_m"), pq_nbits=cfg.get("pq_nbits"), sweeps=[])
    for ef in EF_SEARCH_LIST:
        index.hnsw.efSearch = ef
        t0 = time.time()
        D, I = index.search(Q, K_MAX)
        lat_ms = (time.time() - t0) / len(Q) * 1000
        recalls = {}
        for k in K_REPORT:
            hits = 0
            for i in range(len(Q)):
                found = set(I[i, :k].tolist()) - {-1}
                hits += len(found & GT_SETS[k][i]) / k
            recalls[str(k)] = round(hits / len(Q), 4)
        row = dict(ef_search=ef, latency_ms=round(lat_ms, 2), **{f"recall@{k}": recalls[str(k)] for k in K_REPORT})
        cfg_result["sweeps"].append(row)
        log(f"  efSearch={ef:4d}  lat={lat_ms:6.2f}ms  " + "  ".join(f"R@{k}={recalls[str(k)]}" for k in K_REPORT))
        del D, I
    results["configs"].append(cfg_result)
    save_results(results)
    del index
    gc.collect()
    log(f"=== {cfg['name']} 完成，已存 checkpoint ===\n")

log("全部 config 跑完")

# ── 5. 画图 ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(K_REPORT), figsize=(15, 4.5))
for ax, k in zip(axes, K_REPORT):
    for cfg_r in results["configs"]:
        xs = [s["ef_search"] for s in cfg_r["sweeps"]]
        ys = [s[f"recall@{k}"] for s in cfg_r["sweeps"]]
        ax.plot(xs, ys, marker="o", label=cfg_r["name"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("efSearch")
    ax.set_ylabel(f"recall@{k}")
    ax.set_title(f"recall@{k} vs efSearch")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "recall_curves.png"), dpi=130)
log(f"图存到 {os.path.join(HERE, 'recall_curves.png')}")

# latency vs recall@100 (Pareto)
fig2, ax = plt.subplots(figsize=(6, 4.5))
for cfg_r in results["configs"]:
    xs = [s["latency_ms"] for s in cfg_r["sweeps"]]
    ys = [s["recall@100"] for s in cfg_r["sweeps"]]
    ax.plot(xs, ys, marker="o", label=cfg_r["name"])
ax.set_xlabel("query latency (ms)")
ax.set_ylabel("recall@100")
ax.set_title("速度/精度 权衡 (recall@100)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "latency_vs_recall.png"), dpi=130)
log("done")
