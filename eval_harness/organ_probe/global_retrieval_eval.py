"""全局检索架构 eval（Mac，纯 numpy）。验证：全库尺度下 score 能不能把对的物种排前，
quality 乘数导致的跨物种替换是有用还是有害，扫 a/b/c 权重。

score(photo) = cos(q_S, emb)^a · p_organ[P]^b · quality^c   →  对全池排序取 top-k

用法：python global_retrieval_eval.py   （需 global_pool.npz + global_species_proto.npz）
"""
import os, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--pool", default=os.path.join(H, "global_pool.npz"))
ap.add_argument("--proto", default=os.path.join(H, "global_species_proto.npz"))
ARG = ap.parse_args()
P = np.load(ARG.pool, allow_pickle=True)
SP = np.load(ARG.proto, allow_pickle=True)
EMB = P["emb"].astype(np.float32)                     # [N,768]
SPI = P["species_idx"]; PORG = P["p_organ"]           # [N] , [N,4]
CLS = list(P["classes"]); SPECIES = list(SP["species"]); PROTO = SP["proto"].astype(np.float32)
N = len(EMB)
GENUS = np.array([s.split(" ")[0] for s in SPECIES])

# quality → (0,1]：分辨率 + 清晰度各自 z-score，tanh 压到 (0,1)，取均值
def z(x): x = np.log1p(np.maximum(x, 0)); return (x - x.mean()) / (x.std() + 1e-6)
QUAL = 0.5 + 0.5 * np.tanh((z(P["q_res"]) + z(P["q_sharp"])) / 2)   # [N]

rng = np.random.default_rng(0)
# 查询物种：池里有 ≥3 张的
cnt = np.bincount(SPI, minlength=len(SPECIES))
cand_sp = np.where(cnt >= 3)[0]
q_sp = rng.choice(cand_sp, size=min(400, len(cand_sp)), replace=False)

def score(si, pi, a, b, c):
    cos = np.clip(EMB @ PROTO[si], 1e-4, 1.0)          # 每 query 现算，2M 也就 ~1.5 GFLOP
    return cos**a * np.clip(PORG[:, pi], 1e-4, 1.0)**b * QUAL**c

def eval_cfg(a, b, c, k=5):
    pur, r1S, sPres = [], [], []
    swaps_useful, swaps_genus, swaps = 0, 0, 0
    for si in q_sp:
        own = np.where(SPI == si)[0]
        best_own_q = QUAL[own].max()
        for pi in range(4):
            s = score(si, pi, a, b, c)
            top = np.argsort(-s)[:k]
            is_S = SPI[top] == si
            pur.append(is_S.mean())
            r1S.append(bool(is_S[0]))
            sPres.append(bool(is_S.any()))
            if not is_S[0]:                              # top-1 不是 S → 一次"替换"
                swaps += 1
                t0 = top[0]
                if QUAL[t0] > best_own_q: swaps_useful += 1
                if GENUS[SPI[t0]] == GENUS[si]:         swaps_genus += 1
    return (np.mean(pur), np.mean(r1S), np.mean(sPres),
            swaps_useful / max(swaps, 1), swaps_genus / max(swaps, 1), swaps)

print(f"池 {N} 张 / {len(SPECIES)} 种；查询 {len(q_sp)} 种 × 4 部位 = {len(q_sp)*4} query\n")
print(f"{'a':>3}{'b':>4}{'c':>4} | {'purity@5':>9} {'rank1=S':>8} {'S∈top5':>7} | {'swap:useful':>11} {'swap:genus':>10}")
for a in (1, 2, 3):
    for b in (1, 2, 4):
        for c in (0, 0.5, 1, 2):
            pu, r1, sp, su, sg, ns = eval_cfg(a, b, c)
            print(f"{a:>3}{b:>4}{c:>4} | {pu:9.3f} {r1:8.3f} {sp:7.3f} | {su:11.2f} {sg:10.2f}")

print("\n读法：purity@5 / rank1=S 越高越守物种；c(quality 权重)拉高 → 跨物种替换增多，"
      "看 swap:useful（替换成更清晰的比例）、swap:genus（替换成同属的比例）是否也高 → 有用 vs 有害。")
