"""
修复 manifest 与磁盘不一致 (Botanical Garden pipeline · step 11)
================================================================
背景：bg_10 一次补图时把 manifest 的 source_id 写成了浮点（123.0），导致已登记的图被当作"新候选"
重复下载，随后被 sha 去重逻辑按同名文件删掉 → manifest 有行、磁盘无文件。
本脚本：1) 规范化 source_id；2) 对磁盘缺失的行，按 manifest 里记录的 url 重下；
3) 重下失败的行从 manifest 移除（并写 repair_failed.csv），保证 manifest ⇔ 磁盘一致。
"""
import os
import re
import hashlib
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm

GH = "./BotanicalGarden/gh_images"
HEADERS = {"User-Agent": "bg-image-fetcher/1.0 (jiayuanj@umich.edu)"}
MAX_SIDE = 640


def fetch(row):
    path = f"{GH}/images/{row['accepted_name'].replace(' ', '_')}/{row['file']}"
    try:
        r = requests.get(row["url"], headers=HEADERS, timeout=(5, 20))
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if min(img.size) < 200:
            return row.name, None, "too_small"
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = BytesIO()
        img.save(buf, "JPEG", quality=92)
        data = buf.getvalue()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return row.name, hashlib.sha1(data).hexdigest(), ""
    except Exception as e:
        return row.name, None, type(e).__name__


def main():
    mp = f"{GH}/manifest.csv"
    m = pd.read_csv(mp, dtype={"source_id": str})
    m["source_id"] = m["source_id"].str.replace(r"\.0$", "", regex=True)
    m["path"] = [f"{GH}/images/{n.replace(' ', '_')}/{f}" for n, f in zip(m.accepted_name, m.file)]
    miss = m[[not os.path.exists(p) for p in m.path]]
    print(f"manifest {len(m)} 行；磁盘缺失 {len(miss)} 行（{miss.accepted_name.nunique()} 个物种）", flush=True)

    ok, fail, sha_diff = 0, [], 0
    with ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(fetch, r) for _, r in miss.iterrows()]
        for f in tqdm(as_completed(futs), total=len(futs), desc="重下", unit="img"):
            idx, sha, err = f.result()
            if sha:
                ok += 1
                if sha != m.at[idx, "sha1"]:
                    sha_diff += 1
                    m.at[idx, "sha1"] = sha
            else:
                fail.append((idx, err))
    print(f"重下成功 {ok}（sha 与原记录不同 {sha_diff}）；失败 {len(fail)}")
    if fail:
        bad = m.loc[[i for i, _ in fail]].assign(error=[e for _, e in fail])
        bad.drop(columns=["path"]).to_csv(f"{GH}/repair_failed.csv", index=False)
        print(pd.Series([e for _, e in fail]).value_counts().to_dict())
        m = m.drop(index=[i for i, _ in fail])
    m.drop(columns=["path"]).to_csv(mp, index=False)
    m["exists"] = [os.path.exists(p) for p in m.path]
    print(f"修复后 manifest {len(m)} 行；磁盘存在 {m.exists.sum()}")


if __name__ == "__main__":
    main()
