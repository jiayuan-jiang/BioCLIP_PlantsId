"""
导出 pybioclip 完整向量库
==========================
从 pybioclip 的 TreeOfLifeClassifier 直接提取预计算的
867455 个物种文本嵌入向量，保存为 infer.py 兼容的 npz 格式。

依赖：pip install pybioclip
输出：bioclip_full_index.npz
"""

import numpy as np
from bioclip import TreeOfLifeClassifier


# ══════════════════════════════════════════════
#  修改这里的变量
# ══════════════════════════════════════════════

OUTPUT_NPZ   = "bioclip_full_index.npz"
ONLY_PLANTS  = False   # True = 只保留 Plantae，False = 保留全部 867455 个物种

# ══════════════════════════════════════════════


def parse_entry(entry) -> dict:
    """
    解析 pybioclip txt_names 中的单条记录。
    格式：[[kingdom, phylum, class, order, family, genus, species], common_name]
    """
    ranks  = entry[0]   # list of 7 strings
    common = entry[1] if len(entry) > 1 else ""

    kingdom = ranks[0] if len(ranks) > 0 else ""
    phylum  = ranks[1] if len(ranks) > 1 else ""
    class_  = ranks[2] if len(ranks) > 2 else ""
    order   = ranks[3] if len(ranks) > 3 else ""
    family  = ranks[4] if len(ranks) > 4 else ""
    genus   = ranks[5] if len(ranks) > 5 else ""
    species = ranks[6] if len(ranks) > 6 else ""

    # 科学名 = genus + species epithet
    sci_parts = [p for p in [genus, species] if p]
    sci_name  = " ".join(sci_parts)

    # 分类学路径（BioCLIP 官方训练格式，用于调试）
    path_parts = [p for p in ranks if p]
    taxon_path = " ".join(path_parts)

    return {
        "kingdom":    kingdom,
        "phylum":     phylum,
        "class_":     class_,
        "order":      order,
        "family":     family,
        "genus":      genus,
        "species":    species,
        "sci_name":   sci_name,
        "common":     common,
        "taxon_path": taxon_path,
    }


def main():
    print("加载 pybioclip TreeOfLifeClassifier…")
    clf = TreeOfLifeClassifier()

    # 原始数据
    # txt_embeddings shape: (D, N) = (768, 867455)，需要转置
    raw_embs  = clf.txt_embeddings.cpu().float().numpy().T  # → (N, D)
    raw_names = clf.txt_names                               # list of N entries

    print(f"原始向量库：{raw_embs.shape[0]} 个物种，维度 {raw_embs.shape[1]}")

    # 解析所有条目
    print("解析物种名称…")
    parsed = [parse_entry(e) for e in raw_names]

    # 可选：只保留植物
    if ONLY_PLANTS:
        mask = [i for i, p in enumerate(parsed) if p["kingdom"] == "Plantae"]
        raw_embs = raw_embs[mask]
        parsed   = [parsed[i] for i in mask]
        print(f"筛选后（仅 Plantae）：{len(parsed)} 个物种")

    # 组装各列
    sci_names    = np.array([p["sci_name"]   for p in parsed], dtype=object)
    common_names = np.array([p["common"]     for p in parsed], dtype=object)
    taxon_paths  = np.array([p["taxon_path"] for p in parsed], dtype=object)
    kingdoms     = np.array([p["kingdom"]    for p in parsed], dtype=object)
    families     = np.array([p["family"]     for p in parsed], dtype=object)
    genera       = np.array([p["genus"]      for p in parsed], dtype=object)
    # pybioclip 没有 iNat taxon_id，留空（infer.py 兼容空值）
    taxon_ids    = np.array([""] * len(parsed), dtype=object)

    # 保存
    print(f"保存向量库 → {OUTPUT_NPZ} …")
    np.savez_compressed(
        OUTPUT_NPZ,
        embeddings   = raw_embs.astype(np.float32),
        taxon_ids    = taxon_ids,
        sci_names    = sci_names,
        common_names = common_names,
        taxon_paths  = taxon_paths,
        kingdoms     = kingdoms,
        families     = families,
        genera       = genera,
    )

    import os
    size_mb = os.path.getsize(OUTPUT_NPZ) / 1024 / 1024
    print(f"✓ 完成：{len(parsed)} 个物种，{size_mb:.1f} MB  [{OUTPUT_NPZ}]")

    # 验证几条
    print("\n前 3 条示例：")
    for i in range(min(3, len(parsed))):
        print(f"  [{i}] {sci_names[i]}  |  {common_names[i]}  |  {taxon_paths[i]}")

    print("\n使用方式：")
    print(f"  在 infer.py 中将 INDEX_NPZ 改为 '{OUTPUT_NPZ}'")


if __name__ == "__main__":
    main()