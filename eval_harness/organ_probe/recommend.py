"""配图推荐 —— 全局向量检索的参考实现。无 fallback、无分类学门控。

  score(photo) = cos(q_species, emb)^a · p_organ[part]^b · quality^c
  → 对整个照片索引排序 → phash 去重 + MMR → provenance 后处理标注

现在跑在 global_pool.npz（data/test_images 池）上；生产时换成 iNat Open Data 索引。
物种原型：离线预算好的 global_species_proto.npz（生产时改成在线 encode_text）。

用法：
  python recommend.py                       # 跑几个示例 query，打印
  python recommend.py --visualize           # 额外导出带标注的 contact sheet 到 runs/recommend_v2/
"""
import os, sys, math, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

# ── 权重（用 global_retrieval_eval.py 的 sweep 结果定；这里是占位默认）──
A, B, C = 2.0, 2.0, 1.0
MMR_LAMBDA = 0.5        # 1=纯相关，0=纯多样
DEDUP_HAMMING = 6       # dHash 汉明距离 ≤ 此值视为重复


def _popcount_xor(a, b):
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


class Recommender:
    def __init__(self, pool_npz=None, proto_npz=None, a=A, b=B, c=C):
        pool_npz = pool_npz or os.path.join(HERE, "global_pool.npz")
        proto_npz = proto_npz or os.path.join(HERE, "global_species_proto.npz")
        P = np.load(pool_npz, allow_pickle=True)
        SP = np.load(proto_npz, allow_pickle=True)
        self.emb = P["emb"].astype(np.float32)                    # [N,768] 已归一
        self.spi = P["species_idx"]
        self.p_organ = P["p_organ"]                               # [N,4]
        self.dhash = P["dhash"]                                   # [N,8] uint8
        self.path = P["path"]
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
                res.append(dict(path=str(self.path[j]), species=sp_j, provenance=prov,
                                score=round(float(score[j]), 4), cos=round(float(cos[j]), 3),
                                p_organ=round(float(self.p_organ[j, pi]), 3),
                                quality=round(float(self.qual[j]), 3)))
            out[part] = res
        return out


# ── 可视化（复用 mvp 的标注 + contact sheet）──
def _visualize(rec, queries, outdir):
    from PIL import Image, ImageDraw, ImageFont
    TI = os.path.join(ROOT, "data", "test_images")
    dirmap = {}
    import glob
    for d in glob.glob(f"{TI}/*"):
        b = os.path.basename(d)
        if "_" in b:
            dirmap.setdefault(b.split("_", 1)[1], []).append(d)
    def find(basename, sp):
        for d in dirmap.get(sp, []):
            p = os.path.join(d, basename)
            if os.path.exists(p):
                return p
        return None
    try:
        F1 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
        F2 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except Exception:
        F1 = F2 = ImageFont.load_default()
    os.makedirs(outdir, exist_ok=True)
    for sp, part in queries:
        res = rec.recommend(sp, parts=[part], per_part=6)[part]
        cards = []
        for r in res:
            p = find(r["path"], r["species"])
            if not p:
                continue
            im = Image.open(p).convert("RGB"); w = 380
            im = im.resize((w, int(im.height * w / im.width)))
            cv = Image.new("RGB", (w, im.height + 58), (18, 20, 28)); cv.paste(im, (0, 0))
            d = ImageDraw.Draw(cv)
            col = (74, 222, 128) if r["provenance"] == "exact" else (250, 204, 100)
            d.text((8, im.height + 5), f"{r['species']}  {r['provenance']}", font=F1, fill=col)
            d.text((8, im.height + 25), f"score={r['score']} cos={r['cos']} p_{part}={r['p_organ']} q={r['quality']}",
                   font=F2, fill=(200, 205, 215))
            cards.append(cv)
        if not cards:
            continue
        cw, ch, cols = cards[0].width, max(c.height for c in cards), 3
        rows = math.ceil(len(cards) / cols)
        sheet = Image.new("RGB", (cols * cw + 8 * (cols + 1), 44 + rows * (ch + 8)), (10, 11, 16))
        ImageDraw.Draw(sheet).text((10, 12), f"{sp}  —  {part}   (全局检索，无 fallback；绿=本种 黄=近亲)",
                                   font=F1, fill=(230, 235, 245))
        for k, c in enumerate(cards):
            sheet.paste(c, (8 + (k % cols) * (cw + 8), 44 + (k // cols) * (ch + 8)))
        sheet.save(os.path.join(outdir, f"{sp.replace(' ', '_')}__{part}.jpg"))
    print("→", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--visualize", action="store_true")
    args = ap.parse_args()
    rec = Recommender()
    demo = [("Acer rubrum", ["leaf", "bark"]), ("Asclepias syriaca", ["flower"]),
            ("Pinus strobus", ["bark"]), ("Quercus rubra", ["fruit"])]
    for sp, parts in demo:
        try:
            r = rec.recommend(sp, parts=parts, per_part=5)
        except KeyError as e:
            print(e); continue
        for part, lst in r.items():
            print(f"\n=== {sp} — {part} ===")
            for i, x in enumerate(lst, 1):
                print(f"  {i}. {x['species']:24} {x['provenance']:26} "
                      f"score={x['score']} cos={x['cos']} p={x['p_organ']} q={x['quality']}")
    if args.visualize:
        _visualize(rec, [("Acer rubrum", "leaf"), ("Acer rubrum", "bark"),
                         ("Asclepias syriaca", "flower"), ("Pinus strobus", "bark"),
                         ("Quercus rubra", "fruit")],
                   os.path.join(ROOT, "runs", "recommend_v2"))
