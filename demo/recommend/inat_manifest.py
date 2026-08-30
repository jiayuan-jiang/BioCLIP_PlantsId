r"""iNat Open Data 元数据 -> 抽 top-N 植物种 x <=K 张的下载清单。

前置：下好并解开元数据（--meta 目录里应有 photos.csv.gz / observations.csv.gz /
taxa.csv.gz / observers.csv.gz）：
  aws s3 cp --no-sign-request s3://inaturalist-open-data/metadata/inaturalist-open-data-latest.tar.gz .
  tar xzf inaturalist-open-data-latest.tar.gz

用法：
  pip install duckdb
  python inat_manifest.py --meta D:\inat\metadata --out D:\inat\manifest.parquet \
      --n-species 20000 --per-species 100

脚本会先打印各 CSV 的表头 + license 取值分布，再跑 join。列名/取值对不上时按打印结果改。
"""
import argparse, os, sys, duckdb

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n-species", type=int, default=20000)
ap.add_argument("--per-species", type=int, default=100)
ap.add_argument("--per-obs", type=int, default=3)
ap.add_argument("--plantae-id", type=int, default=47126, help="iNat Plantae taxon_id")
ap.add_argument("--probe-only", action="store_true", help="只打印 schema，不跑 join")
A = ap.parse_args()
M = A.meta.replace("\\", "/")

con = duckdb.connect()
con.execute("PRAGMA threads=8; PRAGMA memory_limit='12GB';")
def csv(n):
    p = f"{M}/{n}.csv.gz"
    if not os.path.exists(p):
        sys.exit(f"缺文件：{p}")
    return f"read_csv_auto('{p}', header=true, sample_size=200000)"

# ── 0. schema 自检 ───────────────────────────────────────────────
print("=== CSV 表头 ===")
COLS = {}
for n in ("photos", "observations", "taxa", "observers"):
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {csv(n)} LIMIT 0").fetchall()]
    COLS[n] = set(cols)
    print(f"  {n:13} {cols}")

def need(tbl, *cands):
    for c in cands:
        if c in COLS[tbl]:
            return c
    sys.exit(f"{tbl}.csv 里找不到列 {cands}（实际列见上），改 --meta 或脚本")

P_ID   = need("photos", "photo_id")
P_OBS  = need("photos", "observation_uuid")
P_OBSR = need("photos", "observer_id")
P_EXT  = need("photos", "extension")
P_LIC  = need("photos", "license")
P_W    = "width" if "width" in COLS["photos"] else None
P_H    = "height" if "height" in COLS["photos"] else None
O_UUID = need("observations", "observation_uuid")
O_TAX  = need("observations", "taxon_id")
O_QG   = need("observations", "quality_grade")
T_ID   = need("taxa", "taxon_id")
T_ANC  = need("taxa", "ancestry")
T_NAME = need("taxa", "name")
T_RANK = need("taxa", "rank")
OBR_ID = need("observers", "observer_id")
OBR_LG = need("observers", "login")

print("\n=== photos.license 取值分布（前 20）===")
print(con.execute(f"""
  SELECT {P_LIC} AS license, count(*) n FROM {csv('photos')}
  GROUP BY 1 ORDER BY n DESC LIMIT 20
""").fetchdf().to_string(index=False))

print("\n=== observations.quality_grade 取值 ===")
print(con.execute(f"SELECT {O_QG} qg, count(*) n FROM {csv('observations')} GROUP BY 1 ORDER BY n DESC").fetchdf().to_string(index=False))

if A.probe_only:
    sys.exit(0)

# license 规整：只留 a-z0-9，判 CC0 / CC-BY / CC-BY-NC（排除含 nd/sa 的），兼容 URL 形式
LIC_KEEP = (f"( lic LIKE '%cc0%' OR lic LIKE '%publicdomain%' "
            f"  OR (lic LIKE '%by%' AND lic NOT LIKE '%nd%' AND lic NOT LIKE '%sa%') )")
PX = f"COALESCE({P_W},0)*COALESCE({P_H},0)" if P_W and P_H else "0"

# ── 1. 植物种 ────────────────────────────────────────────────────
con.execute(f"""
CREATE TABLE plant_taxa AS
SELECT {T_ID} AS taxon_id, {T_NAME} AS name FROM {csv('taxa')}
WHERE lower({T_RANK})='species'
  AND ('/' || CAST({T_ANC} AS VARCHAR) || '/') LIKE '%/{A.plantae_id}/%';
""")
npt = con.execute("SELECT count(*) FROM plant_taxa").fetchone()[0]
print(f"\n① 植物种 {npt:,}", flush=True)
if npt == 0:
    sys.exit("植物种为 0 —— 检查 --plantae-id（默认 47126）或 taxa.ancestry 格式")

# ── 2. research-grade 植物观测 ──────────────────────────────────
con.execute(f"""
CREATE TABLE obs AS
SELECT {O_UUID} AS observation_uuid, {O_TAX} AS taxon_id FROM {csv('observations')}
WHERE lower({O_QG})='research' AND {O_TAX} IN (SELECT taxon_id FROM plant_taxa);
""")

# ── 3. top-N 种 ─────────────────────────────────────────────────
con.execute(f"""
CREATE TABLE top_taxa AS
SELECT o.taxon_id, t.name AS scientific_name, count(*) AS n_obs
FROM obs o JOIN plant_taxa t USING (taxon_id)
GROUP BY 1,2 ORDER BY n_obs DESC LIMIT {A.n_species};
""")

# ── 4. join 照片 + observers，按种/按观测抽样 ───────────────────
con.execute(f"""
CREATE TABLE manifest AS
WITH cand AS (
  SELECT p.{P_ID} AS photo_id, p.{P_OBS} AS observation_uuid, p.{P_EXT} AS extension,
         p.{P_LIC} AS license_raw,
         regexp_replace(lower(COALESCE(CAST(p.{P_LIC} AS VARCHAR),'')), '[^a-z0-9]', '', 'g') AS lic,
         {PX} AS px, tt.taxon_id, tt.scientific_name, tt.n_obs,
         obr.{OBR_LG} AS observer_login
  FROM {csv('photos')} p
  JOIN obs o        ON o.observation_uuid = p.{P_OBS}
  JOIN top_taxa tt  ON tt.taxon_id = o.taxon_id
  LEFT JOIN {csv('observers')} obr ON obr.{OBR_ID} = p.{P_OBSR}
),
filt AS (SELECT * FROM cand WHERE {LIC_KEEP}),
per_obs AS (
  SELECT *, row_number() OVER (PARTITION BY observation_uuid ORDER BY px DESC, random()) AS rn_o FROM filt
),
per_sp AS (
  SELECT *, row_number() OVER (
    PARTITION BY taxon_id
    ORDER BY (lic LIKE '%cc0%') DESC, (lic NOT LIKE '%nc%') DESC, px DESC, random()
  ) AS rn_s
  FROM per_obs WHERE rn_o <= {A.per_obs}
)
SELECT photo_id, taxon_id, scientific_name, n_obs, extension, license_raw AS license, observer_login,
       'https://inaturalist-open-data.s3.amazonaws.com/photos/' || photo_id || '/medium.' || extension AS url
FROM per_sp WHERE rn_s <= {A.per_species};
""")

con.execute(f"COPY manifest TO '{A.out.replace(chr(92), '/')}' (FORMAT parquet);")
n, ns = con.execute("SELECT count(*), count(DISTINCT taxon_id) FROM manifest").fetchone()
print(f"\n✓ {A.out}   {n:,} 张 / {ns:,} 种   （约 {n*60/1e6:.0f} GB medium 下载量）")
print(con.execute("SELECT license, count(*) FROM manifest GROUP BY 1 ORDER BY 2 DESC").fetchdf().to_string(index=False))
