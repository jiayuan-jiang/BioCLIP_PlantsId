"""配图推荐 —— 打分形式 ablation（Mac，纯 numpy）。

背景：recommend.py / retrieval_eval.py 原用 `cos^a · p_organ^b · quality^c`（乘积 + 指数）。
本脚本验证该形式的问题并给出候选替代。结论见同目录 SCORE_ABLATION.md。

四块：
  1) ground-truth 覆盖率     —— 原 eval 有多少 (种,器官) query 池里根本没有正确答案
  2) 纯 score1（cos）基线     —— 物种检索本身行不行
  3) 打分形式 ablation        —— raw 乘积 / z-logit / softmax(cos/T)，× p_organ (± quality)
  4) 无正确答案时的行为        —— 剔除本种照片，看 softmax 捧出来的赢家 + 裸 cos 阈值能否兜底

用法：python retrieval_score_ablation.py            （需同目录 global_pool.npz + global_species_proto.npz）
"""
import os, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()

P = np.load(os.path.join(HERE, "global_pool.npz"), allow_pickle=True)
SP = np.load(os.path.join(HERE, "global_species_proto.npz"), allow_pickle=True)
SPI = P["species_idx"]
PORG = P["p_organ"].astype(np.float32)                       # [N,4]  leaf/flower/fruit/bark
EMB = P["emb"].astype(np.float32)                            # [N,768] 已归一
PROTO = SP["proto"].astype(np.float32)                       # [S,768]
SPECIES = list(SP["species"])
GEN = np.array([s.split(" ")[0] for s in SPECIES])
N, S = len(EMB), len(SPECIES)
CLASSES = list(P["classes"])

def z(x):
    x = np.log1p(np.maximum(x, 0)); return (x - x.mean()) / (x.std() + 1e-6)
QUAL = (0.5 + 0.5 * np.tanh((z(P["q_res"]) + z(P["q_sharp"])) / 2)).astype(np.float32)

cnt = np.bincount(SPI, minlength=S)
rng = np.random.default_rng(0)
Q = rng.choice(np.where(cnt >= 3)[0], size=min(400, int((cnt >= 3).sum())), replace=False)

COS = (EMB @ PROTO.T).astype(np.float32)                     # [N,S]
print(f"池 {N} 张 / {S} 种；查询 {len(Q)} 种 × 4 器官；COS_ALL {COS.shape}  {time.time()-t0:.1f}s\n")

# ── 1) ground-truth 覆盖率 ───────────────────────────────────────────
def organ_mask(k):
    m = np.zeros((len(Q), 4), bool)
    for i, si in enumerate(Q):
        own = np.where(SPI == si)[0]
        for pi in range(4):
            m[i, pi] = (PORG[own, pi] > 0.5).sum() >= k
    return m
HAS1, HAS3 = organ_mask(1), organ_mask(3)
print("[1] 原 eval 的 1600 个 (种,器官) query 覆盖率")
print(f"    池内 ≥1 张该器官正确照片：{HAS1.sum():4d}/{HAS1.size}  ({HAS1.mean()*100:.0f}%)")
print(f"    池内 ≥3 张该器官正确照片：{(HAS3).sum():4d}/{HAS3.size}  ({HAS3.mean()*100:.0f}%)")
print(f"    每种平均照片数：{cnt[Q].mean():.1f}")
print("    → 原 retrieval_eval.py 不加掩码，一半 query 结构上就得 0，把联合 purity 机械压低\n")

# ── 2) 纯 score1（cos）基线 ─────────────────────────────────────────
def eval_score(scorecol, has):
    r1, p5, g5 = [], [], []
    for i, si in enumerate(Q):
        for pi in range(4):
            if not has[i, pi]:
                continue
            s = scorecol(si, pi)
            top = np.argpartition(-s, 5)[:5]; top = top[np.argsort(-s[top])]
            isS = SPI[top] == si
            r1.append(bool(isS[0])); p5.append(isS.mean())
            g5.append((GEN[SPI[top]] == GEN[si]).mean())
    return np.mean(r1), np.mean(p5), np.mean(g5), len(r1)

ALLTRUE = np.ones((len(Q), 4), bool)
r1, p5, g5, n = eval_score(lambda si, pi: COS[:, si], ALLTRUE)   # 任一本种照片算对，不看器官
print("[2] 纯 score1：只按 cos 排，不乘 p_organ / quality，不看器官")
print(f"    rank1=S {r1:.3f}   purity@5 {p5:.3f}   genus@5 {g5:.3f}   (n={n})")
print("    → 物种检索本身没问题；是后续全局乘 p_organ/quality 把它砸下去\n")

# ── 3) 打分形式 ablation ────────────────────────────────────────────
mu, sd = COS.mean(0, keepdims=True), COS.std(0, keepdims=True)
ZC = 1.0 / (1.0 + np.exp(-(COS - mu) / (sd + 1e-9)))         # 池内 z-score → logistic
def softmax_T(T):
    e = np.exp((COS - COS.max(1, keepdims=True)) / T)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)  # P(species|photo)
PSM = {T: softmax_T(T) for T in (0.02, 0.01, 0.005)}

CFGS = {
    "raw cos^3 only":              lambda si, pi: COS[:, si] ** 3,
    "raw prod 3/1/.5":             lambda si, pi: np.clip(COS[:, si],1e-4,1)**3 * np.clip(PORG[:,pi],1e-4,1) * QUAL**0.5,
    "raw prod 1/1/1":              lambda si, pi: np.clip(COS[:, si],1e-4,1)    * np.clip(PORG[:,pi],1e-4,1) * QUAL,
    "zlogit(cos)*p_org (1/1/0)":   lambda si, pi: ZC[:, si] * PORG[:, pi],
    "zlogit(cos)*p_org*q (1/1/1)": lambda si, pi: ZC[:, si] * PORG[:, pi] * QUAL,
    "softmaxT.02 *p_org*q":        lambda si, pi: PSM[0.02][:, si] * PORG[:, pi] * QUAL,
    "softmaxT.01 *p_org*q":        lambda si, pi: PSM[0.01][:, si] * PORG[:, pi] * QUAL,
    "softmaxT.005*p_org*q":        lambda si, pi: PSM[0.005][:, si] * PORG[:, pi] * QUAL,
    "softmaxT.01 *p_org (1/1/0)":  lambda si, pi: PSM[0.01][:, si] * PORG[:, pi],
}
for tag, has in [("HAS>=1", HAS1), ("HAS>=3", HAS3)]:
    print(f"[3] 打分形式 ablation —— 只在 ({tag}) 有正确答案的 (种,器官) 上评")
    print(f"    {'config':30} {'rank1=S':>8} {'purity@5':>9} {'genus@5':>8} {'n':>5}")
    for name, fn in CFGS.items():
        a, b, g, m = eval_score(fn, has)
        print(f"    {name:30} {a:8.3f} {b:9.3f} {g:8.3f} {m:5d}")
    print()

# ── 4) 无正确答案时的行为（剔除本种照片模拟） ──────────────────────
T = 0.01
norm = dict(cos=[], isS=[], gen=[]); hold = dict(cos=[], isS=[], gen=[])
for si in Q:
    col = COS[:, si].astype(np.float32)
    e = np.exp((col - col.max()) / T); p = e / e.sum(); j = int(np.argmax(p))
    norm["cos"].append(col[j]); norm["isS"].append(SPI[j] == si); norm["gen"].append(GEN[SPI[j]] == GEN[si])
    m = col.copy(); m[SPI == si] = -1e9
    e = np.exp((m - m.max()) / T); p = e / e.sum(); j = int(np.argmax(p))
    hold["cos"].append(col[j]); hold["isS"].append(SPI[j] == si); hold["gen"].append(GEN[SPI[j]] == GEN[si])
nc, hc = np.array(norm["cos"]), np.array(hold["cos"])
print("[4] 无正确答案时 softmax 的赢家（剔除本种照片模拟）")
print(f"    正常（本种在池）：      赢家裸cos {nc.mean():.3f}   真是S {np.mean(norm['isS'])*100:3.0f}%   同属 {np.mean(norm['gen'])*100:3.0f}%")
print(f"    剔除本种（无答案）：    赢家裸cos {hc.mean():.3f}   真是S {np.mean(hold['isS'])*100:3.0f}%   同属 {np.mean(hold['gen'])*100:3.0f}%")
for tau in (0.65, 0.68, 0.70, 0.72, 0.74):
    print(f"      cos>={tau:.2f} : 正常保留 {(nc>=tau).mean()*100:3.0f}%   无答案误报 {(hc>=tau).mean()*100:3.0f}%")
print("    → softmax 永远捧赢家；裸 cos 绝对阈值区分力弱（无答案赢家 cos 也 ~0.74）。")
print("      更可靠的『无答案』判据：top-N 里 provenance==exact 的数量、genus@5 的塌陷。")
print(f"\ntotal {time.time()-t0:.1f}s")
