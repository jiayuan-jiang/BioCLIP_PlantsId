"""配图推荐引擎 —— 两段式检索 + 双温度打分。

见 doc/spec/recommend-deploy-execution.md。
· Stage 1: 物种原型 → HNSW+PQ 取 top-`pool_k` 候选
· Stage 2: 精确 cos + topm 重建 softmax 分母 Z_j → psm；实时算 v3 organ probe → t_org 温度缩放
           → score = psm · p_org → dHash 去重 + MMR → top-`topk`

无 fallback、无分类学门控。不 import torch（物种向量是查表）。
"""
import json
import os

import numpy as np

DEDUP_HAMMING = 6      # dHash 汉明距 ≤ 此值视为重复
MMR_LAMBDA = 0.5       # 1=纯相关，0=纯多样

# 打分参数默认值 + 合法区间（API clamp 用）
DEFAULTS = dict(topk=12, t_sp=0.01, t_org=1.0, pool_k=1000)
RANGES = dict(topk=(1, 40), t_sp=(0.003, 0.3), t_org=(0.1, 5.0), pool_k=(200, 3000))


def clamp(name, v):
    lo, hi = RANGES[name]
    return type(v)(min(max(v, lo), hi))


class RecommendEngine:
    def load(self, base_dir, ef_search=384):
        import faiss

        base = str(base_dir)
        mm = lambda n: np.load(os.path.join(base, n), mmap_mode="r")   # noqa: E731

        # ── 全部 mmap，不进内存（按页 fault）──
        self.emb = mm("recommend_emb.f16.npy")                # [N,768] fp16
        self.photo_id = mm("recommend_photo_id.npy")          # [N] int32
        self.species_idx = mm("recommend_species_idx.npy")    # [N] int32
        self.dhash = mm("recommend_dhash.npy")                # [N,8] uint8
        self.ext_code = mm("recommend_ext.npy")               # [N] uint8
        self.lic_code = mm("recommend_license.npy")           # [N] uint8
        self.obs_code = mm("recommend_observer.npy")          # [N] int32
        self.topm_idx = mm("recommend_topm_idx.npy")          # [N,32] uint16
        self.topm_cos_mm = mm("recommend_topm_cos.npy")       # [N,32] fp16

        # ── 小的，全读进内存 ──
        self.protos = np.load(os.path.join(base, "recommend_protos.f16.npy")).astype(np.float32)  # [S,768]
        aux = json.load(open(os.path.join(base, "recommend_aux.json")))
        self.ext_vocab = aux["ext_vocab"]
        self.lic_vocab = aux["license_vocab"]
        self.url_tmpl = aux["url_template"]
        self.url_exc = aux.get("url_exceptions", {})          # {str(row_idx): full_url}
        self.obs_vocab = json.load(open(os.path.join(base, "recommend_observer_vocab.json")))

        og = np.load(os.path.join(base, "organ_tagger_v3.npz"), allow_pickle=True)
        self.og_coef = np.ascontiguousarray(og["coef"].astype(np.float32))   # [5,768]
        self.og_icpt = og["intercept"].astype(np.float32)                    # [5]
        self.classes = [str(c) for c in og["classes"]]                       # leaf/flower/fruit/bark/stem

        self.names = json.load(open(os.path.join(base, "recommend_species.json")))
        self.sp2i = {s: i for i, s in enumerate(self.names)}
        self.genus = np.array([s.split(" ")[0] for s in self.names])

        self.index = faiss.read_index(os.path.join(base, "recommend_index.faiss"), faiss.IO_FLAG_MMAP)
        self.index.hnsw.efSearch = int(ef_search)
        try:
            faiss.omp_set_num_threads(1)
        except Exception:
            pass

    def _url(self, g):
        u = self.url_exc.get(str(g))
        if u:
            return u
        return self.url_tmpl.format(photo_id=int(self.photo_id[g]),
                                    ext=self.ext_vocab[int(self.ext_code[g])])

    # ── 主入口（在 pool worker 进程里跑，纯 CPU）─────────────────────
    def run(self, species_or_idx=None, parts=None, topk=None, t_sp=None, t_org=None, pool_k=None,
            species_name=None, query_vec=None):
        """species_or_idx: 图集内物种的 species_idx（int）或学名（str）。
        query_vec: 图集外物种的 768 维文本原型（识图索引查来的），此时 si=-1，
                   结果全是 genus-relative / look-alike，species_name 提供属名。"""
        if query_vec is not None:
            si = -1
            sp_name = species_name or "?"
            g_si = sp_name.split(" ")[0]
            qv = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
            qv = qv / (np.linalg.norm(qv) + 1e-12)
            q = np.ascontiguousarray(qv)
        else:
            si = species_or_idx if isinstance(species_or_idx, (int, np.integer)) else self.sp2i.get(species_or_idx)
            if si is None:
                return None
            si = int(si)
            sp_name = self.names[si]
            g_si = self.genus[si]
            q = np.ascontiguousarray(self.protos[si:si + 1])

        parts = list(parts) if parts else list(self.classes)
        topk = clamp("topk", int(topk if topk is not None else DEFAULTS["topk"]))
        t_sp = clamp("t_sp", float(t_sp if t_sp is not None else DEFAULTS["t_sp"]))
        t_org = clamp("t_org", float(t_org if t_org is not None else DEFAULTS["t_org"]))
        pool_k = clamp("pool_k", int(pool_k if pool_k is not None else DEFAULTS["pool_k"]))

        # ── Stage 1: ANN ────────────────────────────────────────────
        _, I = self.index.search(q, pool_k)
        cand = I[0][I[0] >= 0].astype(np.int64)                       # [K] 全局 idx
        if len(cand) == 0:
            return {"species": sp_name, "not_in_pool": True, "in_pool": si != -1,
                    "params": dict(topk=topk, t_sp=t_sp, t_org=t_org, pool_k=pool_k),
                    "results": {p: [] for p in parts}}
        E = np.asarray(self.emb[cand], dtype=np.float32)              # [K,768] mmap gather
        cos_si = E @ q[0]                                             # [K] 精确 cos

        # ── Stage 2a: 物种项 psm（topm 重建 Z_j，减最大值防 exp 溢出）─
        tc = np.asarray(self.topm_cos_mm[cand], dtype=np.float32)     # [K,32]
        ti = np.asarray(self.topm_idx[cand], dtype=np.int64)          # [K,32]
        mstar = np.maximum(cos_si, tc.max(1))                         # [K]
        num = np.exp((cos_si - mstar) / t_sp)                        # [K]
        tail = np.where(ti == si, 0.0, np.exp((tc - mstar[:, None]) / t_sp)).sum(1)
        psm = num / (num + tail)                                      # [K]  ∈ (0,1]

        # ── Stage 2b: organ probe 实时算 5 头 ───────────────────────
        logit5 = np.clip(E @ self.og_coef.T + self.og_icpt, -30.0, 30.0)  # [K,5]
        P5 = 1.0 / (1.0 + np.exp(-logit5))

        cand_sp = np.asarray(self.species_idx[cand], dtype=np.int64)  # [K]
        cand_genus = self.genus[cand_sp]
        cand_name = [self.names[k] for k in cand_sp]

        res = {}
        for part in parts:
            if part not in self.classes:
                res[part] = []
                continue
            po = P5[:, self.classes.index(part)]
            if abs(t_org - 1.0) > 1e-9:
                po = np.clip(po, 1e-6, 1.0 - 1e-6)
                po = 1.0 / (1.0 + np.exp(-np.log(po / (1.0 - po)) / t_org))
            score = psm * po                                         # [K]
            order = np.argsort(-score)
            kept = self._dedup(order, cand)                          # 去重后的 cand-局部索引
            picked = self._mmr(kept, E, score, topk)
            res[part] = [
                self._row(j, cand, cand_sp, cand_name, cand_genus, g_si, si, score, cos_si, psm, po)
                for j in picked
            ]
        return {
            "species": sp_name,
            "in_pool": si != -1,
            "not_in_pool": bool(si == -1 or int((cand_sp == si).sum()) == 0),
            "params": dict(topk=topk, t_sp=t_sp, t_org=t_org, pool_k=pool_k),
            "results": res,
        }

    # ── dHash 去重：按 score 序保留，与已留的汉明距都 > 阈值 ─────────
    def _dedup(self, order, cand, cap=None):
        cap = cap or 200
        d = np.ascontiguousarray(self.dhash[cand])               # [K,8] uint8
        h = d.view(np.uint64).reshape(-1)                        # [K]
        kept = []
        kept_h = []
        for j in order:
            hj = int(h[j])
            if all((hj ^ hk).bit_count() > DEDUP_HAMMING for hk in kept_h):
                kept.append(int(j))
                kept_h.append(hj)
                if len(kept) >= cap:
                    break
        return kept

    # ── MMR：rel = score，div = 与已选的最大 emb 余弦 ────────────────
    def _mmr(self, kept, E, score, topk):
        if not kept:
            return []
        kept = np.asarray(kept)
        rel = score[kept]
        Ek = E[kept]                                                 # [m,768]（已归一）
        sel = [int(np.argmax(rel))]
        max_sim = Ek @ Ek[sel[0]]                                    # [m]
        while len(sel) < min(topk, len(kept)):
            mmr = MMR_LAMBDA * rel - (1.0 - MMR_LAMBDA) * max_sim
            mmr[sel] = -1e9
            nxt = int(np.argmax(mmr))
            sel.append(nxt)
            max_sim = np.maximum(max_sim, Ek @ Ek[nxt])
        return [int(kept[s]) for s in sel]

    def _row(self, j, cand, cand_sp, cand_name, cand_genus, g_si, si, score, cos_si, psm, po):
        g = int(cand[j])
        sp_j = cand_name[j]
        if int(cand_sp[j]) == si:
            prov = "exact"
        elif cand_genus[j] == g_si:
            prov = f"genus-relative ({sp_j})"
        else:
            prov = f"look-alike ({sp_j})"
        return dict(
            photo_id=int(self.photo_id[g]),
            url=self._url(g),
            license=self.lic_vocab[int(self.lic_code[g])],
            attribution=self.obs_vocab[int(self.obs_code[g])],
            species=sp_j,
            provenance=prov,
            score=round(float(score[j]), 4),
            cos=round(float(cos_si[j]), 3),
            p_species=round(float(psm[j]), 3),
            p_organ=round(float(po[j]), 3),
        )

    # ── 404 时的模糊建议（主进程也会调）──────────────────────────────
    def suggest(self, name, k=8):
        q = name.lower().strip()
        hit = [s for s in self.names if q in s.lower()]
        return hit[:k]
