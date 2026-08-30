"""iNat Open Data 元数据 → 抽 top-20k 植物种 × ≤100 张的下载清单。

前置：下好并解开元数据（在 --meta 目录里应有 photos.csv.gz / observations.csv.gz /
taxa.csv.gz / observers.csv.gz）：
  aws s3 cp --no-sign-request s3://inaturalist-open-data/metadata/inaturalist-open-data-latest.tar.gz .
  tar xzf inaturalist-open-data-latest.tar.gz

用法：
  pip install duckdb
  python inat_manifest.py --meta D:\inat\metadata --out D:\inat\manifest.parquet \
      --n-species 20000 --per-species 100

⚠ 先核对 CSV 表头列名与 license 取值（脚本按当前 schema 写，iNat 偶尔改）。
"""
import argparse, os, duckdb

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n-species", type=int, default=20000)
ap.add_argument("--per-species", type=int, default=100)
ap.add_argument("--per-obs", type=int, default=3)
ap.add_argument("--licenses", default="CC0,CC-BY,CC-BY-NC")
A = ap.parse_args()
M = A.meta.replace("\\", "/")
lic = "(" + ",".join(f"'{x.strip()}'" for x in A.licenses.split(",")) + ")"
PLANTAE = 47126  # iNat taxon_id

con = duckdb.connect()
con.execute("PRAGMA threads=8; PRAGMA memory_limit='12GB';")
csv = lambda n: f"read_csv_auto('{M}/{n}.csv.gz', header=true)"

print("① 植物种…", flush=True)
con.execute(f"""
CREATE TABLE plant_taxa AS
SELECT taxon_id, name FROM {csv('taxa')}
WHERE rank='species' AND active
  AND (ancestry LIKE '{PLANTAE}/%' OR ancestry LIKE '%/{PLANTAE}/%' OR taxon_id={PLANTAE});
""")
print("  ", con.execute("SELECT count(*) FROM plant_taxa").fetchone()[0], "种", flush=True)

print("② research-grade 植物观测…", flush=True)
con.execute(f"""
CREATE TABLE obs AS
SELECT observation_uuid, taxon_id FROM {csv('observations')}
WHERE quality_grade='research'
  AND taxon_id IN (SELECT taxon_id FROM plant_taxa);
""")

print(f"③ 按观测数取 top {A.n_species}…", flush=True)
con.execute(f"""
CREATE TABLE top_taxa AS
SELECT o.taxon_id, t.name AS scientific_name, count(*) AS n_obs
FROM obs o JOIN plant_taxa t USING (taxon_id)
GROUP BY 1,2 ORDER BY n_obs DESC LIMIT {A.n_species};
""")

print("④ join 照片 + observers，按种抽样…", flush=True)
con.execute(f"""
CREATE TABLE manifest AS
WITH cand AS (
  SELECT p.photo_id, p.observation_uuid, p.extension, p.license,
         COALESCE(p.width,0)*COALESCE(p.height,0) AS px,
         tt.taxon_id, tt.scientific_name, tt.n_obs,
         obr.login AS observer_login
  FROM {csv('photos')} p
  JOIN obs o        USING (observation_uuid)
  JOIN top_taxa tt  ON tt.taxon_id = o.taxon_id
  LEFT JOIN {csv('observers')} obr ON obr.observer_id = p.observer_id
  WHERE p.license IN {lic}
),
per_obs AS (
  SELECT *, row_number() OVER (PARTITION BY observation_uuid ORDER BY px DESC, random()) AS rn_o
  FROM cand
),
per_sp AS (
  SELECT *, row_number() OVER (
    PARTITION BY taxon_id
    ORDER BY (license='CC0') DESC, (license='CC-BY') DESC, px DESC, random()
  ) AS rn_s
  FROM per_obs WHERE rn_o <= {A.per_obs}
)
SELECT photo_id, taxon_id, scientific_name, n_obs, extension, license, observer_login,
       'https://inaturalist-open-data.s3.amazonaws.com/photos/' || photo_id || '/medium.' || extension AS url
FROM per_sp WHERE rn_s <= {A.per_species};
""")

con.execute(f"COPY manifest TO '{A.out.replace(chr(92),'/')}' (FORMAT parquet);")
n, ns = con.execute("SELECT count(*), count(DISTINCT taxon_id) FROM manifest").fetchone()
print(f"\n✓ {A.out}  {n:,} 张 / {ns:,} 种  （约 {n*60/1e6:.0f} GB medium 下载量）", flush=True)
print(con.execute("SELECT license, count(*) FROM manifest GROUP BY 1 ORDER BY 2 DESC").fetchdf().to_string(), flush=True)
