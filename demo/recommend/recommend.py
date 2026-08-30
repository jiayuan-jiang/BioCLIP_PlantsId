"""配图推荐 —— 全局向量检索的参考实现。无 fallback、无分类学门控。

  score(photo) = cos(q_species, emb)^a · p_organ[part]^b · quality^c
  → 对整个照片索引排序 → phash 去重 + MMR → provenance 后处理标注

索引：inat_pool.npz（iNat Open Data top-20k 种，inat_index_build.py 产出）。
物种原型：离线预算好的 inat_species_proto.npz（生产时改成在线 encode_text）。

用法：
  python recommend.py --species "Acer rubrum"                         # 单 query
  python recommend.py --pool <p>.npz --proto <s>.npz --species "..."  # 指定索引
"""
import os, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ── 权重（用 retrieval_eval.py 的 sweep 结果定；这里是占位默认）──
A, B, C = 2.0, 2.0, 1.0
MMR_LAMBDA = 0.5        # 1=纯相关，0=纯多样
DEDUP_HAMMING = 6       # dHash 汉明距离 ≤ 此值视为重复


def _popcount_xor(a, b):
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


class Recommender:
    def __init__(self, pool_npz=None, proto_npz=None, a=A, b=B, c=C):
        pool_npz = pool_npz or os.path.join(HERE, "inat_pool.npz")
        proto_npz = proto_npz or os.path.join(HERE, "inat_species_proto.npz")
        P = np.load(pool_npz, allow_pickle=True)
        SP = np.load(proto_npz, allow_pickle=True)
        self.emb = P["emb"].astype(np.float32)                    # [N,768] 已归一
        self.spi = P["species_idx"]
        self.p_organ = P["p_organ"]                               # [N,4]
        self.dhash = P["dhash"]                                   # [N,8] uint8
        self.ref = P["path"] if "path" in P.files else P["url"]   # v0=本地文件名, v1=iNat URL
        self.url = P["url"] if "url" in P.files else self.ref
        self.license = P["license"] if "license" in P.files else np.array([""] * len(self.emb))
        self.observer = P["observer_login"] if "observer_login" in P.files else np.array([""] * len(self.emb))
        self.classes = list(P["classes"])
        self.species = list(SP["species"])
        self.proto = SP["proto"].astype(np.float32)               # [S,768]
        self.genus = np.array([s.split(" ")[0] for s in self.species])
        self.sp2i = {s: i for i, s in enumerate(self.species)}
        self.a, self.b, self.c = a, b, c
        # quality → (0,1]
        def z(x):
            x = np.log1p(np.maximum(x, 0)); return (x - x.mean()) / (x.std() + 1e-6)
        self.qual = 0.5 + 0.5 * np.tanh((z(P["q_res"]) + z(P["q_sharp"])) / 2)

    # ── 物种原型（离线预算；生产时改在线 encode_text）──
    def _species_vec(self, name):
        i = self.sp2i.get(name)
        if i is None:
            raise KeyError(f"物种不在预算原型里：{name}（生产版会在线 encode_text）")
        return self.proto[i], i

    def recommend(self, species, parts=("leaf", "flower", "fruit", "bark"), per_part=5, pool_topn=40):
        q, s_idx = self._species_vec(species)
        cos = np.clip(self.emb @ q, 1e-4, 1.0)
        out = {}
        for part in parts:
            pi = self.classes.index(part)
            score = cos ** self.a * np.clip(self.p_organ[:, pi], 1e-4, 1.0) ** self.b * self.qual ** self.c
            cand = np.argsort(-score)[:pool_topn]
            # phash 去重
            kept = []
            for j in cand:
                if all(_popcount_xor(self.dhash[j], self.dhash[k]) > DEDUP_HAMMING for k in kept):
                    kept.append(j)
            # MMR 多样化
            sel = []
            pool = kept[:]
            while pool and len(sel) < per_part:
                if not sel:
                    sel.append(pool.pop(0)); continue
                best, bj = -1e9, 0
                for idx, j in enumerate(pool):
                    rel = score[j]
                    div = max(float(self.emb[j] @ self.emb[k]) for k in sel)
                    mmr = MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * div
                    if mmr > best:
                        best, bj = mmr, idx
                sel.append(pool.pop(bj))
            # provenance
            res = []
            for j in sel:
                sp_j = self.species[self.spi[j]]
                if self.spi[j] == s_idx:
                    prov = "exact"
                elif self.genus[self.spi[j]] == self.genus[s_idx]:
                    prov = f"genus-relative ({sp_j})"
                else:
                    prov = f"look-alike ({sp_j})"
                res.append(dict(ref=str(self.ref[j]), url=str(self.url[j]),
                                license=str(self.license[j]), attribution=str(self.observer[j]),
                                species=sp_j, provenance=prov,
                                score=round(float(score[j]), 4), cos=round(float(cos[j]), 3),
                                p_organ=round(float(self.p_organ[j, pi]), 3),
                                quality=round(float(self.qual[j]), 3)))
            out[part] = res
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=None)
    ap.add_argument("--proto", default=None)
    ap.add_argument("--species", default="Acer rubrum")
    ap.add_argument("--per-part", type=int, default=6)
    args = ap.parse_args()
    rec = Recommender(pool_npz=args.pool, proto_npz=args.proto)
    r = rec.recommend(args.species, per_part=args.per_part)
    for part, lst in r.items():
        print(f"\n=== {args.species} — {part} ===")
        for i, x in enumerate(lst, 1):
            print(f"  {i}. {x['species']:26} {x['provenance']:30} score={x['score']} "
                  f"cos={x['cos']} p={x['p_organ']} q={x['quality']}\n     {x['url']}")
