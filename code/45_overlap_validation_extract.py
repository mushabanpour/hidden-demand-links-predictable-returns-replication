"""
45_overlap_validation_extract.py -- POI-level detail of the same-category cross-shopping lists, for
validating customer overlap (repeated customers, censoring). One pass over the raw shards per month,
parsing only the same-category list; POIs are selected and de-duplicated exactly as in 01_extract_month.py.

  val_SAME/<month>   brand f x listed brand b:
                       act_num  sum_p w_p * pct_pb               (as rel_SAME: customer-weighted)
                       act_poi  sum_p pct_pb                     (POI-equal weights)
                       n_list   number of f's POIs that list b
                       thr_lst  sum_p w_p * thr_p over f's POIs with a full (20-entry) list that list b
  val_brand/<month>  brand f:
                       n_poi, w_sum
                       n_full, w_full  POIs (and their customers) whose list has 20 entries (censored lists)
                       thr_w   sum_p w_p * thr_p over POIs with full lists, thr_p = smallest listed share
                       len_w   sum_p w_p * (number of listed brands)
Usage: python3 45_overlap_validation_extract.py all | 2019-01 [2019-02 ...]
"""
import sys, glob, os, time
import duckdb

BASE = os.environ.get("SG_BASE", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SPEND = f"{BASE}/spend"
OUT = f"{BASE}/revision/data/extract"
COL = "RELATED_CROSS_SHOPPING_SAME_CATEGORY_BRANDS_PCT"
FULL = 20


def entries(col):
    return (f"""(CASE WHEN {col} LIKE '{{"key_value"%'
                 THEN json_transform({col}, '{{"key_value":[{{"key":"VARCHAR","value":"DOUBLE"}}]}}').key_value
                 ELSE map_entries(json_transform({col}, '"MAP(VARCHAR, DOUBLE)"')) END)""")


def run_month(month, con):
    fs = sorted(glob.glob(f"{SPEND}/{month}--*.snappy.parquet"))
    if not fs:
        print(month, "no files"); return
    if all(os.path.exists(f"{OUT}/{t}/{month}.parquet") for t in ("val_SAME", "val_brand")):
        return
    fl = "[" + ",".join(f"'{f}'" for f in fs) + "]"
    src = f"read_parquet({fl}, filename=true, file_row_number=true)"
    t0 = time.time()
    for tb in ("poi", "rel", "plist"):
        con.execute(f"DROP TABLE IF EXISTS {tb}")
    con.execute(f"""
      CREATE TEMP TABLE poi AS
        SELECT filename AS fn, file_row_number AS rn,
          CASE WHEN BRANDS IS NULL OR BRANDS IN ('', '[]') THEN NULL
               WHEN BRANDS LIKE '[{{%' THEN json_extract_string(BRANDS, '$[0].safegraph_brand_name')
               ELSE BRANDS END AS brand,
          CAST(RAW_NUM_CUSTOMERS AS DOUBLE) AS w, PLACEKEY
        FROM {src} WHERE RAW_NUM_CUSTOMERS > 0
        QUALIFY row_number() OVER (PARTITION BY PLACEKEY ORDER BY filename, file_row_number) = 1""")
    con.execute(f"""
      CREATE TEMP TABLE rel AS
        WITH x AS (SELECT filename AS fn, file_row_number AS rn, unnest({entries(COL)}) AS e
                   FROM {src} WHERE {COL} IS NOT NULL AND {COL} NOT IN ('{{}}', ''))
        SELECT p.fn, p.rn, p.brand AS f, p.w, x.e.key AS b, x.e.value AS pct
        FROM x JOIN poi p ON p.fn = x.fn AND p.rn = x.rn
        WHERE p.brand IS NOT NULL AND x.e.value IS NOT NULL""")
    con.execute("""
      CREATE TEMP TABLE plist AS
        SELECT fn, rn, any_value(f) AS f, any_value(w) AS w, count(*) AS len, min(pct) AS thr FROM rel GROUP BY fn, rn""")

    def write(sql, table):
        os.makedirs(f"{OUT}/{table}", exist_ok=True)
        con.execute(f"COPY ({sql}) TO '{OUT}/{table}/{month}.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)")

    write(f"""SELECT '{month}' AS month, r.f AS brand, r.b, sum(r.w * r.pct) AS act_num, sum(r.pct) AS act_poi,
                     count(*) AS n_list, sum(CASE WHEN l.len >= {FULL} THEN r.w * l.thr ELSE 0 END) AS thr_lst
              FROM rel r JOIN plist l USING (fn, rn) GROUP BY r.f, r.b""", "val_SAME")
    write(f"""SELECT '{month}' AS month, p.brand, count(*) AS n_poi, sum(p.w) AS w_sum,
                     sum(CASE WHEN l.len >= {FULL} THEN 1 ELSE 0 END) AS n_full,
                     sum(CASE WHEN l.len >= {FULL} THEN p.w ELSE 0 END) AS w_full,
                     sum(CASE WHEN l.len >= {FULL} THEN p.w * l.thr ELSE 0 END) AS thr_w,
                     sum(p.w * coalesce(l.len, 0)) AS len_w
              FROM poi p LEFT JOIN plist l USING (fn, rn) WHERE p.brand IS NOT NULL GROUP BY p.brand""", "val_brand")
    print(month, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    months = sorted({os.path.basename(f)[:7] for f in glob.glob(f"{SPEND}/*.snappy.parquet")}) if sys.argv[1:] == ["all"] else sys.argv[1:]
    con = duckdb.connect(); con.execute("SET threads=7; SET memory_limit='4500MB'; SET preserve_insertion_order=false")
    for m in months:
        run_month(m, con)
