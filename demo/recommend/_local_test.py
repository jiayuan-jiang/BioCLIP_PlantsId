"""本地冒烟测试 —— 直接跑 RecommendEngine（不起 uvicorn）。见 spec §5。

  conda run -n siglip_env python demo/recommend/_local_test.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # demo/ 上 path，import recommend.engine
from recommend.engine import RecommendEngine          # noqa: E402

ART = os.path.join(HERE, "..", "deploy_artifacts")

print("load engine …", flush=True)
t0 = time.time()
eng = RecommendEngine()
eng.load(ART, ef_search=384)
print(f"  loaded in {time.time()-t0:.1f}s  ({len(eng.names)} species, classes={eng.classes})")

COVERED = ["Acer rubrum", "Quercus rubra", "Rosa canina"]


def summarize(r, part):
    lst = r["results"][part]
    provs = [x["provenance"].split(" ")[0] for x in lst]
    n_exact = provs.count("exact")
    n_genus = provs.count("genus-relative")
    n_look = provs.count("look-alike")
    p_org_med = float(np.median([x["p_organ"] for x in lst])) if lst else 0.0
    return f"{part:6} n={len(lst):2} exact={n_exact} genus={n_genus} look={n_look} p_org_med={p_org_med:.2f}"


# ── 1. 覆盖良好的种：exact 占比 ───────────────────────────────────
print("\n[1] 覆盖良好的种")
for sp in COVERED:
    t0 = time.time()
    r = eng.run(sp, topk=12)
    dt = (time.time() - t0) * 1000
    print(f"  {sp:16} not_in_pool={r['not_in_pool']}  {dt:.0f}ms")
    for part in eng.classes:
        print("     ", summarize(r, part))

# ── 2. 不在图集的种（识图能出但不在 20k）──────────────────────────
print("\n[2] 不在图集的种")
for sp in ["Abies magnifica", "Zzz not a species"]:
    si = eng.sp2i.get(sp)
    if si is None:
        print(f"  {sp!r}: 不在 20k 池（404），suggest={eng.suggest(sp)[:4]}")
        continue
    r = eng.run(sp, topk=12)
    print(f"  {sp:20} not_in_pool={r['not_in_pool']}")
    for part in ["leaf", "bark"]:
        print("     ", summarize(r, part))

# ── 3. t_sp 扫：近亲占比应随 t_sp 上升 ──────────────────────────
print("\n[3] t_sp 扫（Acer rubrum, leaf）")
for t_sp in [0.01, 0.03, 0.05, 0.1]:
    r = eng.run("Acer rubrum", parts=["leaf"], topk=12, t_sp=t_sp)
    lst = r["results"]["leaf"]
    non_exact = sum(1 for x in lst if not x["provenance"].startswith("exact"))
    print(f"  t_sp={t_sp:5}  非 exact = {non_exact}/{len(lst)}")

# ── 4. t_org 扫：bark p_organ 中位数应随 t_org 下降而上升 ─────────
print("\n[4] t_org 扫（Acer rubrum, bark）")
for t_org in [1.0, 0.5, 0.3, 0.2]:
    r = eng.run("Acer rubrum", parts=["bark"], topk=12, t_org=t_org)
    lst = r["results"]["bark"]
    med = float(np.median([x["p_organ"] for x in lst])) if lst else 0.0
    print(f"  t_org={t_org:4}  p_organ 中位数 = {med:.3f}")

# ── 5. 延迟：一次 5 part 请求 ───────────────────────────────────
print("\n[5] 延迟")
for _ in range(2):  # warmup + measure
    t0 = time.time()
    r = eng.run("Acer rubrum", topk=12)
    dt = (time.time() - t0) * 1000
print(f"  5-part 请求 pool_k=1000: {dt:.0f}ms")

# ── 6. 去重：结果内两两 dHash 汉明 > 6 ─────────────────────────
print("\n[6] 去重")
r = eng.run("Acer rubrum", parts=["leaf"], topk=20)
ids = [x["photo_id"] for x in r["results"]["leaf"]]
pid_all = np.asarray(eng.photo_id)
pid2row = {int(p): i for i, p in enumerate(pid_all)}
rows = [pid2row[i] for i in ids if i in pid2row]
d = np.ascontiguousarray(np.asarray(eng.dhash)[rows]).view(np.uint64).reshape(-1)
mind = min((int(d[a] ^ d[b]).bit_count() for a in range(len(d)) for b in range(a + 1, len(d))), default=99)
print(f"  leaf top-{len(ids)} 两两最小汉明距 = {mind}  (应 > {6})")

# ── 7. 打印一个完整结果样例 ─────────────────────────────────────
print("\n[样例] Acer rubrum / flower top-5:")
r = eng.run("Acer rubrum", parts=["flower"], topk=5)
print(json.dumps(r["results"]["flower"], ensure_ascii=False, indent=2)[:1600])
print("\nDONE")
