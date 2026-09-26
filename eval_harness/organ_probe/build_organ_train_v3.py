"""organ_train v3 —— 干净 bark + 新 stem 类，共 5 类。

起因：v1/v2 的 bark 标签来自 Pl@ntNet-300K 的 `organ` 字段，PlantNet 没有 "stem" 类，
草本茎 / 木本嫩枝 / 真树干树皮全塞进 bark → bark 头 AUC-PR 只有 0.60。

v3 分类与来源（全部开放下载，无需登录）：
  leaf / flower / fruit  ── 复用 organ_train.npz（PlantNet，标签干净，不重编码）
  bark                   ── BarkNet 1.0 (Zenodo 11508014, 按树种 zip) + BARK-KR (Zenodo 4749062)
  stem                   ── PlantCLEF 2015 训练包 + 2016 测试包里 <Content>Stem</Content> 的图 + NTU-Tree (HF)
  PlantNet 的 bark 桶     ── 整个丢弃（就是被污染的数据）

下载策略：能 partial 就 partial（BarkNet 按种）；只能整包的（BARK-KR / PlantCLEF）
下完 filter 出需要的部分，随即删掉原始大包（--keep-archives 保留）。

输出：organ_train_v3.npz / organ_eval_v3.npz
      keys: emb[N,768] float32(已归一) · organs[N] · source[N] · species[N]

分阶段，中间产物在 _v3_cache/，可断点续跑：
  python build_organ_train_v3.py --stage fetch       # 下载 + 解包 + filter + 删大包
  python build_organ_train_v3.py --stage manifest    # 建清单 + 取样（打印计数）
  python build_organ_train_v3.py --stage encode      # 编码 + 存 npz
  python build_organ_train_v3.py --stage all
  python build_organ_train_v3.py --dry-run
  选项：--keep-archives  --bark-species N（每种 BarkNet 只留 N 张，默认 400）

依赖：huggingface_hub, pyarrow, open_clip, torch, PIL, numpy
"""
import os, io, sys, json, glob, shutil, tarfile, zipfile, random, argparse, subprocess, collections, time
import xml.etree.ElementTree as ET
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "_v3_cache")
RAW = os.path.join(CACHE, "raw")
ARCH = os.path.join(CACHE, "_archives")
W = os.path.join(ROOT, "demo", "weights", "open_clip_model.safetensors")
OLD_TRAIN = os.path.join(HERE, "organ_train.npz")
random.seed(0)

CLASSES = ["leaf", "flower", "fruit", "bark"]   # v3.1: 砍掉 stem（无干净数据集），只洗 bark
CAP = {"bark": 5000}                            # 每类取样上限；bark 来自 BarkNet+BARK-KR+NTU-Tree
EVAL_FRAC = 0.20
IMG_EXT = (".jpg", ".jpeg", ".png")

BARKNET_ZENODO = "11508014"
BARKKR_ZENODO = "4749062"
NTU_TREE_HF = "liswei/NTU-Tree"
PLANTCLEF_TARS = [
    ("plantclef2015tr", "https://lab.plantnet.org/LifeCLEF/PlantCLEF2015/TrainingPackage/PlantCLEF2015TrainingData.tar.gz"),
    ("plantclef2016te", "https://lab.plantnet.org/LifeCLEF/PlantCLEF2016/PlantCLEF2016Test.tar.gz"),
]
KEEP_ARCHIVES = False
BARK_PER_SPECIES = 400


def sh(cmd):
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def curl(url, out, tries=8):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"  cached {os.path.basename(out)} ({os.path.getsize(out)/1e9:.1f} GB)", flush=True)
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    for k in range(1, tries + 1):
        r = subprocess.run(
            ["curl", "-fL", "--retry", "5", "--retry-all-errors", "--retry-delay", "10",
             "--retry-max-time", "0", "-C", "-", "--no-progress-meter", "-o", out + ".part", url])
        if r.returncode == 0:
            os.rename(out + ".part", out)
            print(f"  ↓ {os.path.basename(out)} {os.path.getsize(out)/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)
            return out
        sz = os.path.getsize(out + ".part") if os.path.exists(out + ".part") else 0
        print(f"  !! curl exit {r.returncode} on {os.path.basename(out)} (have {sz/1e9:.2f} GB), 第 {k}/{tries} 次续传 …", flush=True)
        time.sleep(15)
    raise RuntimeError(f"下载失败（{tries} 次）：{url}")


def zget(record_id):
    p = curl(f"https://zenodo.org/api/records/{record_id}", os.path.join(CACHE, f"zenodo_{record_id}.json"))
    return json.load(open(p)).get("files", [])


def rm(p):
    if KEEP_ARCHIVES:
        return
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    elif os.path.exists(p):
        os.remove(p)
    print(f"  rm {os.path.basename(p)}", flush=True)


def imgs_in(*dirs):
    out = []
    for d in dirs:
        for e in IMG_EXT:
            out += glob.glob(os.path.join(d, "**", "*" + e), recursive=True)
            out += glob.glob(os.path.join(d, "**", "*" + e.upper()), recursive=True)
    return sorted(set(out))


# ─────────────────────────── fetch: bark ───────────────────────────
def _extract_images(zip_path, dest, cap=None, prefix=""):
    """把 zip 里的图片解到 dest（平铺，带 prefix 防重名）；cap=随机上限。
    先试 unzip CLI（容错好），失败退回 Python zipfile 逐条 try。返回落地张数。"""
    os.makedirs(dest, exist_ok=True)
    tmp = dest + "_x"
    n = 0
    os.makedirs(tmp, exist_ok=True)
    print("  $ unzip -q -o", os.path.basename(zip_path), "-d", os.path.basename(tmp), flush=True)
    subprocess.run(["unzip", "-q", "-o", zip_path, "-d", tmp], check=False)  # 允许 warning 退出码
    got = imgs_in(tmp)
    if got:
        random.shuffle(got)
        for src in (got[:cap] if cap else got):
            shutil.move(src, os.path.join(dest, f"{prefix}{n:05d}_{os.path.basename(src)}"))
            n += 1
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
        print("    unzip 没出图，退回 zipfile 逐条", flush=True)
        try:
            zf = zipfile.ZipFile(zip_path)
        except Exception as e2:
            print(f"    !! zip 打不开：{e2}", flush=True); return 0
        members = [m for m in zf.namelist() if m.lower().endswith(IMG_EXT)]
        random.shuffle(members)
        for m in members:
            if cap and n >= cap:
                break
            try:
                data = zf.read(m)
            except Exception:
                continue
            with open(os.path.join(dest, f"{prefix}{n:05d}_{os.path.basename(m)}"), "wb") as o:
                o.write(data)
            n += 1
    return n


def fetch_barknet():
    d = os.path.join(RAW, "barknet")
    if os.path.exists(os.path.join(d, ".done")):
        print(f"barknet: done ({len(imgs_in(d))} imgs)", flush=True); return
    os.makedirs(d, exist_ok=True)
    for f in zget(BARKNET_ZENODO):
        name = f["key"]
        if not name.lower().endswith(".zip"):
            continue
        sp = os.path.splitext(name)[0]
        spdir = os.path.join(d, sp)
        if os.path.exists(spdir + ".done"):
            continue
        try:
            z = curl(f["links"]["self"], os.path.join(ARCH, "barknet", name))
        except Exception as e:
            print(f"  !! 跳过 {name}：{e}", flush=True); continue
        print(f"  extract {name} → {sp}/ (cap {BARK_PER_SPECIES})", flush=True)
        got = _extract_images(z, spdir, cap=BARK_PER_SPECIES, prefix=f"{sp}_")
        print(f"    {sp}: {got} imgs", flush=True)
        open(spdir + ".done", "w").close()
        rm(z)
    rm(os.path.join(ARCH, "barknet"))
    open(os.path.join(d, ".done"), "w").close()
    print(f"barknet: {len(imgs_in(d))} imgs across {len(glob.glob(d + '/*/'))} species", flush=True)


def fetch_barkkr():
    d = os.path.join(RAW, "barkkr")
    if os.path.exists(os.path.join(d, ".done")):
        print(f"bark-kr: done ({len(imgs_in(d))} imgs)", flush=True); return
    os.makedirs(d, exist_ok=True)
    for f in zget(BARKKR_ZENODO):
        name = f["key"]
        z = curl(f["links"]["self"], os.path.join(ARCH, "barkkr", name))
        print(f"  extract {name}", flush=True)
        if name.lower().endswith(".zip"):
            got = _extract_images(z, d, cap=CAP["bark"], prefix="kr_")
            print(f"    bark-kr: {got} imgs", flush=True)
        elif name.lower().endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(z) as tf:
                for m in tf:
                    if m.isfile() and m.name.lower().endswith(IMG_EXT):
                        with open(os.path.join(d, "kr_" + os.path.basename(m.name)), "wb") as o:
                            o.write(tf.extractfile(m).read())
        rm(z)
    rm(os.path.join(ARCH, "barkkr"))
    open(os.path.join(d, ".done"), "w").close()
    print(f"bark-kr: {len(imgs_in(d))} imgs", flush=True)


# ─────────────────────────── fetch: stem ───────────────────────────
def _xml_content(txt):
    try:
        root = ET.fromstring(txt)
    except Exception:
        return ""
    for el in root.iter():
        if el.tag.split("}")[-1].lower() in ("content", "type", "organ") and el.text:
            return el.text.strip()
    return ""


def fetch_plantclef_stem():
    d = os.path.join(RAW, "plantclef_stem")
    done = os.path.join(d, ".done")
    if os.path.exists(done):
        print(f"plantclef stem: done ({len(imgs_in(d))} imgs)", flush=True); return
    os.makedirs(d, exist_ok=True)
    for tag, url in PLANTCLEF_TARS:
        marker = os.path.join(d, f".{tag}.done")
        if os.path.exists(marker):
            continue
        tar = curl(url, os.path.join(ARCH, f"{tag}.tar.gz"))
        print(f"  [{tag}] pass1: 扫 xml 找 <Content>Stem> …", flush=True)
        stem_ids = set()
        with tarfile.open(tar, "r:*") as tf:
            for m in tf:
                if m.isfile() and m.name.lower().endswith(".xml"):
                    if _xml_content(tf.extractfile(m).read()).lower() == "stem":
                        stem_ids.add(os.path.splitext(os.path.basename(m.name))[0])
        print(f"  [{tag}] stem xml 命中 {len(stem_ids)}；pass2 抽 jpg …", flush=True)
        want = {f"{i}{e}" for i in stem_ids for e in (".jpg", ".JPG", ".jpeg", ".png")}
        n = 0
        with tarfile.open(tar, "r:*") as tf:
            for m in tf:
                if m.isfile() and os.path.basename(m.name) in want:
                    with open(os.path.join(d, f"{tag}_{os.path.basename(m.name).lower()}"), "wb") as o:
                        o.write(tf.extractfile(m).read())
                    n += 1
        print(f"  [{tag}] extracted {n}", flush=True)
        rm(tar)
        open(marker, "w").close()
    open(done, "w").close()
    print(f"plantclef stem: {len(imgs_in(d))} imgs", flush=True)


def fetch_ntu_tree():
    d = os.path.join(RAW, "ntu_tree")
    done = os.path.join(d, ".done")
    if os.path.exists(done):
        print(f"ntu-tree: done ({len(imgs_in(d))} imgs)", flush=True); return
    os.makedirs(d, exist_ok=True)
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import pyarrow.parquet as pq
        files = [s for s in HfApi().list_repo_files(NTU_TREE_HF, repo_type="dataset") if s.endswith(".parquet")]
        n = 0
        for rel in files:
            p = hf_hub_download(NTU_TREE_HF, rel, repo_type="dataset")
            for batch in pq.ParquetFile(p).iter_batches(batch_size=128):
                col = batch.to_pydict().get("image", [])
                for im in col:
                    b = im.get("bytes") if isinstance(im, dict) else None
                    if b:
                        with open(os.path.join(d, f"ntu_{n:05d}.jpg"), "wb") as o:
                            o.write(b)
                        n += 1
        print(f"ntu-tree: {n} imgs", flush=True)
    except Exception as e:
        print(f"  !! NTU-Tree 失败（{e}），跳过（只是 stem 补充）", flush=True)
    open(done, "w").close()


# ─────────────────────────── manifest ───────────────────────────
def build_manifest():
    man = []
    for p in imgs_in(os.path.join(RAW, "barknet")):
        man.append([p, "bark", "barknet"])
    for p in imgs_in(os.path.join(RAW, "barkkr")):
        man.append([p, "bark", "barkkr"])
    for p in imgs_in(os.path.join(RAW, "ntu_tree")):
        man.append([p, "bark", "ntu"])           # NTU-Tree 实为树皮特写，归 bark
    # plantclef_stem 已弃用（~50% 是树干=bark，无法干净当 stem）

    by = collections.defaultdict(list)
    for row in man:
        by[row[1]].append(row)
    picked = []
    for lab, rows in by.items():
        random.shuffle(rows)
        cap = CAP.get(lab)
        picked += rows if cap is None else rows[:cap]
    random.shuffle(picked)
    cnt = collections.Counter((l, s) for _, l, s in picked)
    print("new-class 取样：", flush=True)
    for (l, s), n in sorted(cnt.items()):
        print(f"    {l:6} {s:16} {n}", flush=True)
    json.dump(picked, open(os.path.join(CACHE, "manifest.json"), "w"))
    return picked


# ─────────────────────────── encode ───────────────────────────
def encode_new(manifest):
    import torch, open_clip
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    torch.set_num_threads(4)
    print("加载 open_clip ViT-L-14 …", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=W)
    model.eval()

    def _load(p):
        # BarkNet 图 3000² 太大 → 先缩到 ~640 再 preprocess，省内存省解码
        im = Image.open(p)
        im.draft("RGB", (640, 640))            # JPEG 快速降采样解码
        im = im.convert("RGB")
        im.thumbnail((640, 640), Image.BILINEAR)
        return preprocess(im)

    ckpt = os.path.join(CACHE, "_encode_partial.npy")

    @torch.no_grad()
    def enc(paths, bs=16):
        embs, t0 = [], time.time()
        start = 0
        if os.path.exists(ckpt):
            prev = np.load(ckpt)
            if len(prev) <= len(paths):
                embs.append(prev); start = len(prev)
                print(f"  续跑：已有 {start}/{len(paths)}", flush=True)
        for i in range(start, len(paths), bs):
            batch = []
            for p in paths[i:i + bs]:
                try:
                    batch.append(_load(p))
                except Exception:
                    batch.append(torch.zeros(3, 224, 224))
            e = model.encode_image(torch.stack(batch)).float()
            embs.append((e / e.norm(dim=-1, keepdim=True)).numpy())
            done = i + len(batch)
            if (i // bs) % 20 == 0:
                np.save(ckpt, np.concatenate(embs, 0))   # checkpoint
                el = time.time() - t0
                print(f"  {done}/{len(paths)}  {el:.0f}s  eta {el/max(done-start,1)*(len(paths)-done):.0f}s", flush=True)
        out = np.concatenate(embs, 0)
        np.save(ckpt, out)
        return out

    emb = enc([p for p, _, _ in manifest])
    return (emb,
            np.array([l for _, l, _ in manifest]),
            np.array([s for _, _, s in manifest]),
            np.array([""] * len(manifest)))


def load_old_lff():
    z = np.load(OLD_TRAIN, allow_pickle=True)
    org = z["organs"].astype(str)
    m = np.isin(org, ["leaf", "flower", "fruit"])
    sp = z["species"].astype(str) if "species" in z.files else np.array([""] * len(org))
    print(f"复用 organ_train.npz 的 leaf/flower/fruit：{int(m.sum())} 行（不重编码）", flush=True)
    return z["emb"][m].astype(np.float32), org[m], np.array(["plantnet"] * int(m.sum())), sp[m]


def split_save(emb, org, src, sp):
    rng = np.random.default_rng(0)
    ev = np.zeros(len(emb), bool)
    for k in sorted(set(zip(org.tolist(), src.tolist()))):
        g = np.where((org == k[0]) & (src == k[1]))[0]
        rng.shuffle(g)
        ev[g[:max(1, int(len(g) * EVAL_FRAC))]] = True
    for tag, mask, out in [("train", ~ev, "organ_train_v3.npz"), ("eval", ev, "organ_eval_v3.npz")]:
        np.savez(os.path.join(HERE, out), emb=emb[mask], organs=org[mask], source=src[mask], species=sp[mask])
        print(f"{tag:5} → {out}  {int(mask.sum())} 行  {dict(sorted(collections.Counter(org[mask].tolist()).items()))}", flush=True)


# ─────────────────────────── main ───────────────────────────
def main():
    global KEEP_ARCHIVES, BARK_PER_SPECIES
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fetch", "manifest", "encode", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--bark-species", type=int, default=BARK_PER_SPECIES, help="每种 BarkNet 只留 N 张")
    a = ap.parse_args()
    KEEP_ARCHIVES = a.keep_archives
    BARK_PER_SPECIES = a.bark_species
    os.makedirs(RAW, exist_ok=True)

    if a.dry_run:
        print("将下载（下完 filter 后删大包，--keep-archives 保留）：")
        print(f"  BarkNet  Zenodo {BARKNET_ZENODO}  按树种 zip，每种留 {BARK_PER_SPECIES} 张")
        print(f"  BARK-KR  Zenodo {BARKKR_ZENODO}   ~10.2 GB 单 zip")
        print(f"  PlantCLEF 2015训练(17.7G) + 2016测试(1.7G)  只留 <Content>Stem>")
        print(f"  NTU-Tree HF {NTU_TREE_HF}  (parquet, 全作 stem)")
        print(f"  复用 {OLD_TRAIN} 的 leaf/flower/fruit")
        for x in ("barknet", "barkkr", "plantclef_stem", "ntu_tree"):
            print(f"  已缓存 {x:16} {len(imgs_in(os.path.join(RAW, x)))} 张")
        return

    if a.stage in ("fetch", "all"):
        for fn in (fetch_barknet, fetch_barkkr, fetch_plantclef_stem, fetch_ntu_tree):
            try:
                fn()
            except Exception as e:
                print(f"!! {fn.__name__} 失败：{e}（继续下一个源）", flush=True)
    if a.stage == "fetch":
        for x in ("barknet", "barkkr", "plantclef_stem", "ntu_tree"):
            print(f"  {x}: {len(imgs_in(os.path.join(RAW, x)))} 张", flush=True)
        return
    man = build_manifest() if a.stage in ("manifest", "all") else json.load(open(os.path.join(CACHE, "manifest.json")))
    if a.stage == "manifest":
        return
    en, oo, so, sp0 = encode_new(man)
    eo, ol, sl, spl = load_old_lff()
    split_save(np.concatenate([eo, en]), np.concatenate([ol, oo]),
               np.concatenate([sl, so]), np.concatenate([spl, sp0]))
    ck = os.path.join(CACHE, "_encode_partial.npy")
    if os.path.exists(ck):
        os.remove(ck)
    print("\n下一步：python organ_tagger_v3.py（CLS=[leaf,flower,fruit,bark]，train/test=organ_*_v3.npz）。"
          "确认无误后可 rm -rf _v3_cache/raw。", flush=True)


if __name__ == "__main__":
    main()
